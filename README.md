# torch-convnd

Full N-dimensional dense convolution for PyTorch. Use `Conv4d` or `ConvNd` in an
ordinary `nn.Module`, with independent weights along every spatial axis, autograd,
AMP, available framework transforms, compilation and export. No training framework is required.

```python
import torch
from torch_convnd import Conv4d

layer = Conv4d(2, 8, kernel_size=3, padding=1)
x = torch.randn(1, 2, 4, 4, 4, 8, requires_grad=True)
y = layer(x)                       # [1, 8, 4, 4, 4, 8]
y.square().mean().backward()
```

Four axes can represent `(x, y, z, time)`; the operator applies PyTorch's
cross-correlation convention without imposing causality or separable weights.
The generic API supports 1-62 spatial dimensions.

## Install

Install your desired PyTorch distribution first. Python 3.10+ and PyTorch 1.11+
are required. Install directly from the public repository:

```sh
python -m pip install "git+https://github.com/lyehe/torch-convnd.git"
```

From a checkout:

```sh
python -m pip install .                    # portable use; PyTorch is the only runtime dependency
python -m pip install ".[native]"         # optional native build dependencies
python -m pip install ".[onnx]"           # optional ONNX export and CPU runtime
python -m pip install ".[legacy,onnx]"    # older PyTorch wheels needing NumPy <2
```

For another project's dependency file, use a built wheel or a pinned VCS URL for
this repository. This checkout has not been published to PyPI. See the
[installation guide](docs/installation.md) for deployment and toolchain details.

Compatibility is selected by available APIs, not PyTorch version strings. The
version contract is tested across 18 release lines from 1.11 through 2.14.
Native acceleration needs the newer custom-op/autocast APIs; older releases keep
the same public layers through portable operations. `torch.func`, `torch.compile`
and exporter features remain conditional on what the installed framework provides.

Importing the package never compiles. `backend="auto"` selects vendor/native
kernels when their prerequisites are available and uses PyTorch operations when
native build tools are missing. `backend="torch"` explicitly avoids compilation;
`backend="native"` requires the addon and reports missing prerequisites.

## Integrate

```python
from torch import nn
from torch_convnd import Conv4d, ConvNd

model = nn.Sequential(
    Conv4d(2, 8, 3, padding="same", backend="torch"),
    nn.GELU(),
    Conv4d(8, 4, 1, backend="torch"),
)
layer5 = ConvNd(2, 8, 3, dimensions=5, padding=1)
```

- Inputs: `[batch, channels, *spatial]` or `[channels, *spatial]`.
- Weights: `[out_channels, in_channels // groups, *kernel]`.
- Stock-style groups, bias, per-axis stride/dilation, numeric/`same`/`valid`
  padding, and zeros/reflect/replicate/circular modes.
- CPU/CUDA real and complex types, higher derivatives, transforms and serialization.
- Standard state-dictionary keys for the 1D/2D/3D wrappers; typed public API and `py.typed`.

See the [integration guide](docs/integration.md), [API reference](docs/api.md),
and [framework-independent training example](examples/integration.py).

## Compatibility and performance

| Backend | Use |
| --- | --- |
| `auto` | Vendor kernels for ranks 1-3 and CPU 4D; native higher-rank CPU/NVIDIA CUDA when available; portable fallback otherwise. |
| `torch` | Compiler-free PyTorch kernels/decompositions, including portable higher-rank execution. |
| `native` | Tiled patch extraction plus GEMM; requires a C++ compiler, and a matching toolkit for CUDA. |

The native patch budget defaults to 64 MiB and is not a total memory limit.
Vendor paths manage their own workspace. Native CUDA 4D can be faster while using
more memory than the portable decomposition; see the matched
[performance records](docs/PERFORMANCE.md).

The [support matrix](docs/support.md) separates validated environments from CI
coverage and hardware that has not been tested. Apple MPS and AMD ROCm use
portable operations where their PyTorch operators are available. Native GPU
kernels target NVIDIA CUDA. No prebuilt platform binaries are bundled.

## Develop and verify

```sh
python -m pip install -e ".[dev,docs,onnx]"
pre-commit install --hook-type pre-commit --hook-type pre-push --install-hooks
pre-commit run --all-files
python -m pytest -q -m portable             # no native compiler required
python -m pytest -q                         # full correctness suite; builds native addon
python -m mypy
python -m mkdocs build --strict
```

On an older framework, run the version contract instead of the modern feature suite:

```sh
python -m pytest tests/test_version_compatibility.py tests/test_native_compatibility.py -q
```

Tests use stock 2D/3D layers, an independent N-D reference, generated geometries,
finite differences and feature-combination regressions. CI also installs and
checks built wheels in isolated Python. GPU cases skip without the required
hardware; CUDA CI runs manually on a trusted runner.

[Correctness evidence](docs/PARITY.md) · [Contributing](CONTRIBUTING.md) ·
[Release process](docs/releasing.md) · [Changelog](CHANGELOG.md)

CI covers Linux/Windows/macOS and minimum/current PyTorch across Python 3.10-3.14.
Documentation and PyPI deployment workflows are included; their external targets
must be configured before publishing. Nothing is published by installing this project.

MIT licensed. Diagnose an integration with `torch-convnd-info --json`.
