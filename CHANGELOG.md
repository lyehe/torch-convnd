# Changelog

## Unreleased

- Add NVIDIA's partial convolution at every rank and backend: `PartialConvNd`,
  `PartialConv1d`-`PartialConv4d` and `partial_conv_nd`, with `multi_channel`,
  stock-compatible state dictionaries, grouped mask counts, masks padded like the
  input, and counts computed in float32 outside autocast. Tracing, `torch.compile` and
  ONNX export are supported; TorchScript is refused explicitly.
- Add `padding_mode="partial"` to `ConvNd`, the fixed-rank layers and `conv_nd`:
  partial-convolution border rescaling with unchanged weights and call signature.
- Return `PartialConvOutput(output, mask)` from `partial_conv_nd` and, when requested,
  from partial layers; `return_mask` set at construction is a default each call can
  override. Masks follow a documented contract: any real dtype, broadcastable batch
  and spatial sizes, and validated shapes.
- Add `PartialSequential`, which carries the mask through partial layers and
  position-wise modules, refuses modules that would lose or misplace it, keeps
  mask-free layers on the fast path, and reports the last layer's update layout.
- Without a mask, copying padding modes and unpadded layers run as an ordinary
  convolution, and zero padding uses exact per-axis counts computed on the device
  instead of a counting convolution: no host copy or synchronization, and symbolic
  sizes stay symbolic under `torch.compile` and `torch.export`. Eager calls cache short
  per-axis vectors and small finished scales per geometry and CUDA stream; functorch
  transforms, compilers, CUDA graph capture and fake or mode-dispatched tensors bypass
  the cache. Rescaling and bias share one fused pass, grouped counts are applied
  through views, and modules build the update mask only when they return it. Windows
  beyond float16's range rescale in float32, and counts leave autocast on every device
  type that supports it. Both ONNX exporters handle partial layers, including dynamic
  sizes. The version-compatibility contract covers partial convolution and its ONNX
  export on every tested PyTorch release.

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
