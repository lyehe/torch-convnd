# Benchmarks

Run from the repository with the package installed:

```sh
python benchmarks/compare_backends.py --device cpu --output .artifacts/cpu_benchmark.json
python benchmarks/compare_backends.py --device cuda:0 --output .artifacts/cuda_benchmark.json
```

Use fresh output paths. The scripts compare values/gradients before recording raw
timings. Match precision flags, shape, workspace, device and competing workload
conditions before attributing a speed difference to code.

Historical records are preserved under [docs/assets/reports](../docs/assets/reports)
and described in the [performance report](../docs/PERFORMANCE.md). New scratch
outputs are ignored. Archive a named record under the documentation assets only
when its code/environment provenance and interpretation are clear.

`summarize_tests.py` archives JUnit and measured numerical errors with source
hashes. It reports what actually ran and retains skips rather than calling them
successful accelerator validation.
