# torch-convnd

[![CI](https://github.com/lyehe/torch-convnd/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lyehe/torch-convnd/actions/workflows/ci.yml)
[![Documentation](https://github.com/lyehe/torch-convnd/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/lyehe/torch-convnd/actions/workflows/docs.yml)

Full N-dimensional dense convolution for PyTorch, including 4D, 5D and 6D.
Use `Conv4d` or `ConvNd` in an ordinary `nn.Module`, with independent weights along
every spatial axis and standard PyTorch training. Autograd, AMP, serialization,
framework transforms, compilation and export follow the installed framework's
capabilities. No training framework is required.

```python
import torch
from torch_convnd import Conv4d

layer = Conv4d(2, 8, kernel_size=3, padding=1, backend="torch")
x = torch.randn(1, 2, 4, 4, 4, 8, requires_grad=True)
y = layer(x)                       # [1, 8, 4, 4, 4, 8]
y.square().mean().backward()
```

Four axes can represent `(x, y, z, time)`; the operator applies PyTorch's
cross-correlation convention without imposing causality or separable weights.
This example uses the compiler-free backend. Use `backend="auto"` to select
vendor/native acceleration when available.

## Install

Install your desired PyTorch distribution first. Python 3.10+ and PyTorch 1.11+
are required. Install directly from the public repository:

```sh
python -m pip install "git+https://github.com/lyehe/torch-convnd.git"
```

Extras also work when installing directly from GitHub:

```sh
python -m pip install "torch-convnd[native] @ git+https://github.com/lyehe/torch-convnd.git"
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

Compatibility is selected by available APIs. The version contract is tested
across 18 PyTorch release lines from 1.11 through 2.14.
Native acceleration needs the newer custom-op/autocast APIs; older releases keep
the same public layers through portable operations. `torch.func`, `torch.compile`
and exporter features remain conditional on what the installed framework provides.

Importing the package never compiles. `backend="auto"` selects vendor/native
kernels when their prerequisites are available and uses PyTorch operations when
native build tools are missing. `backend="torch"` explicitly avoids compilation;
`backend="native"` requires the addon and reports missing prerequisites.

## Conv5D, Conv6D, and higher ranks

Use `ConvNd` with `dimensions` equal to the number of spatial axes. Batch and
channels are additional axes: a batched 5D convolution takes a 7-axis tensor,
and a batched 6D convolution takes an 8-axis tensor.

```python
import torch
from torch_convnd import ConvNd

conv5 = ConvNd(2, 8, 3, dimensions=5, padding=1, backend="torch")
x5 = torch.randn(1, 2, 4, 4, 4, 4, 4, requires_grad=True)
y5 = conv5(x5)  # [1, 8, 4, 4, 4, 4, 4]

conv6 = ConvNd(2, 8, 3, dimensions=6, padding=1, backend="torch")
x6 = torch.randn(1, 2, 3, 3, 3, 3, 3, 3, requires_grad=True)
y6 = conv6(x6)  # [1, 8, 3, 3, 3, 3, 3, 3]

(y5.square().mean() + y6.square().mean()).backward()
```

The generic layer accepts 1-62 spatial dimensions. `kernel_size`, `stride`,
`padding` and `dilation` accept a scalar or a tuple with one entry per spatial
axis; padding also accepts `"same"` and `"valid"`. For example, a 6D layer can
use `kernel_size=(3, 3, 1, 1, 1, 1)` and `padding=(1, 1, 0, 0, 0, 0)`.

A dense size-3 kernel has 243 spatial taps in 5D and 729 in 6D. Start with small
grids and choose per-axis kernel sizes to match the interactions you need.

## Integrate

```python
from torch import nn
from torch_convnd import Conv4d

model = nn.Sequential(
    Conv4d(2, 8, 3, padding="same", backend="torch"),
    nn.GELU(),
    Conv4d(8, 4, 1, backend="torch"),
)
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

For older PyTorch wheels, install the `legacy` extra to preserve NumPy 1.x
interoperability, and run the version contract:

```sh
python -m pip install -e ".[dev,docs,legacy,onnx]"
python -m pytest tests/test_version_compatibility.py tests/test_native_compatibility.py -q
```

Documentation dependencies are selected for the Python version: Python 3.10
retains the compatible PyMdown stack, while Python 3.11+ uses the newer
mkdocstrings integration. CI checks documentation on Python 3.10 and 3.12.

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
