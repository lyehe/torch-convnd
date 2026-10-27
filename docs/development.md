# Development and testing

```sh
python -m pip install -e ".[dev,docs,onnx]"
pre-commit install --hook-type pre-commit --hook-type pre-push --install-hooks
```

Pre-commit checks Python syntax, TOML/YAML/JSON, case/merge conflicts, file size,
whitespace, line endings, Ruff lint and formatting. The pre-push hook runs portable
correctness tests. Full native compilation/GPU tests remain explicit commands and
CI jobs. Immutable historical report JSON is excluded from formatting hooks.
See [pre-commit](https://pre-commit.com/) for hook environment behavior.

The pre-push runner uses the project's `.venv` so Git clients do not need an
activated shell. For another environment, set `TORCH_CONVND_TEST_PYTHON` to its
Python executable. Hook dependencies remain small; PyTorch is taken from the
development environment rather than installed into each isolated hook environment.

```sh
pre-commit run --all-files
python -m pytest -q -m portable
python -m pytest -q --junitxml=.artifacts/correctness.xml
python -m mypy
python -m mkdocs build --strict
```

## Correctness criteria

- Generated rank-1 through rank-5 geometries combine groups, stride, dilation,
  padding and small workspace tiles. Both native and portable results and all
  input/weight/bias gradients use an independent tap-wise reference.
- The broader suite compares stock Conv2d/Conv3d across devices, dtypes, geometry,
  AMP, higher derivatives, transforms, compilers, serialization and ONNX execution.
- Finite differences check derivatives independently of another autodiff formula.
- Integration regressions cover compiler-free imports/training, automatic fallback,
  explicit-native errors, checkpointing, state-dictionary transfer and type checking.
- Built wheel checks run in isolated Python with a required `site-packages` import.
  Distribution checks validate metadata, every bundled source, native files,
  typing metadata and version/tag agreement.

Hypothesis uses reproducible generated examples with no wall-time deadline for
these small reference problems. A failing example is minimized. Numerical parity
uses dtype-specific tolerances rather than bitwise equality; full criteria and
archived measurements are in [correctness](PARITY.md).

Tests honor `CUDA_VISIBLE_DEVICES`. GPU cases skip when compatible hardware/tools
are unavailable; skips do not prove GPU support. The portable marker avoids
native compilation and is appropriate before pushing on a compiler-free machine.

The complete feature suite targets modern PyTorch. To validate an older release,
run `pytest tests/test_version_compatibility.py tests/test_native_compatibility.py`.
This contract compares independent values/gradients, dtypes, padding, higher
derivatives, serialization and ONNX execution without importing missing APIs.
Native checks skip with an explicit reason when the framework lacks the required
custom-op APIs. The CI `versions` job checks 18 release lines on Python 3.10.

## Archive a run

```sh
python benchmarks/summarize_tests.py --junit .artifacts/correctness.xml --output .artifacts/correctness.json
```

The report retains test cases, numerical errors, versions and source hashes.
Keep fresh paths for new measurements. Published historical raw records live
under `docs/assets/reports`; scratch benchmark output belongs in `.artifacts`.

## Structure

`src/torch_convnd` contains the public frontend, modules, backend policy, build
and prerequisite checks, differentiation, export, diagnostics, and native sources.
`tests` holds correctness/integration checks, `examples` consuming-model examples,
`tools` distribution validation, `benchmarks` measurement scripts, and `docs`
the MkDocs documentation. Generated builds, environments, caches and test outputs
are ignored. The Python wheel contains the package and native sources only.

## Documentation

```sh
python -m mkdocs serve
```

The site includes API documentation from the public package. Strict builds catch
broken links and missing references. It can be built locally without publishing;
the GitHub Pages workflow only deploys after repository configuration is enabled.
