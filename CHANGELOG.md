# Changelog

## 0.5.0

- Extend portable training and serialization to PyTorch 1.11+ using feature
  detection rather than version-string branches or unconditional modern imports.
- Use legacy/unified autocast APIs as available, promote unsupported legacy CPU
  FP16 operations, and handle complex inputs through real-valued convolutions.
- Provide a pure Tensor TorchScript adapter when custom operator libraries are absent.
- Gate native autograd, vmap registration and FP32-output mixed GEMM independently.
  Older capable native builds retain FP32 promotion when mixed GEMM is unavailable.
- Select compatible ONNX export options/opsets and defer modern exporter dependencies.
- Add a core/native/ONNX correctness contract and an 18-release PyTorch CI matrix.
- Add repository metadata and source installation links for lyehe/torch-convnd.

## 0.4.0

- Make PyTorch the sole mandatory runtime dependency; move native tools into extras.
- Add automatic portable fallback for missing native prerequisites and CUDA after
  a CPU-only addon load; keep explicit native failures actionable.
- Avoid unsupported 3D tap kernels on non-CPU/CUDA higher-rank portable devices.
- Ship a typed public API, a single version source, compiler-free diagnostics,
  and framework-independent model/training examples.
- Isolate native caches by sources, installation, compiler settings and PyTorch
  version so rebuilding does not overwrite a library held by another process.
- Add generated independent-reference correctness checks, compiler-free integration
  regressions, checkpoint/state transfer tests and installed-wheel validation tools.
- Add Python pre-commit/pre-push checks, cross-platform/version CI, manual CUDA CI,
  gated Trusted Publishing and documentation deployment workflows.
- Publishable MkDocs guides and API reference; preserve historical reports as assets.

## 0.3.0

- Fix script/trace forward-mode AD and lazy native initialization in compiled graphs.
- Add standard ONNX export and serialized/dynamic-shape export checks.
- Select vendor kernels for lower ranks/CPU 4D and native kernels for CUDA 4D.
- Reduce CUDA mixed-precision backward copies using FP32-output GEMM.
- Validate 997 tests and CPU/CUDA installed-wheel checks on the original Windows stack.

## 0.2.0

- Expand stock Conv2d/Conv3d constructor, dtype, transform and serialization coverage.
- A later audit found the serialized forward-AD defect fixed in 0.3.0.

## 0.1.0

- Initial standalone full ConvNd CPU/CUDA addon and dense 4D benchmarks.
