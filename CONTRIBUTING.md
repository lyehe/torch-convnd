# Contributing

Install the desired PyTorch build, then:

```sh
python -m pip install -e ".[dev,docs,onnx]"
pre-commit install --hook-type pre-commit --hook-type pre-push --install-hooks
```

Keep public imports in `torch_convnd`, preserve stock-compatible state dictionaries,
and update the public stub when an API changes. Build tools and exporters remain
optional runtime dependencies. Avoid compiling or allocating GPU tensors on import.

Use the [development guide](docs/development.md) for tests and project structure.
For numerical changes, add a regression using a stock or independent reference
and compare gradients. Performance changes need matched precision, representative
shapes, raw samples and memory measurements. Do not update historical report JSON
to make a current result appear to match an old run.

Before proposing a change:

```sh
pre-commit run --all-files
python -m pytest -q
python -m mypy
python -m mkdocs build --strict
```

Report the platform, Python/PyTorch versions, input/kernel shapes, dtype, backend
and `torch-convnd-info --json` output with failures. GPU cases need actual GPU
execution; an unavailable-device skip does not qualify as validation.

See the [release guide](docs/releasing.md) for distribution and publication checks.
