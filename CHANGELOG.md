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
- Build the native addon on Windows on Arm: the MSVC environment now targets the
  interpreter's platform, using the ARM64 toolset there, instead of always x64.
- Report the CPU model, ATen's CPU capability and whether the GIL is enabled in
  `torch-convnd-info`.
- Compile every layer kind as one graph (`torch.compile(fullgraph=True)`), with static
  or dynamic sizes, from PyTorch 2.2. Automatic layers that may use native kernels
  failed on every release, because Dynamo traced into the toolchain probe: its
  constant mark sat on the cache wrapper, which Dynamo unwraps. Older Dynamo
  releases also broke the graph on `any`/`all` over generators, the functorch probe
  (before 2.5), list item assignment (2.3), `Tensor.is_complex()` (2.3), comparing a
  module with `None`, `Tensor.element_size()` and `inference_mode` (2.2). Compiled
  graphs before 2.5 keep the portable path, which every transform supports, since
  the native operator has no vmap rule there. Dynamo 2.0-2.1 cannot trace
  `zip(strict=True)`, so those releases compile with graph breaks.
- Check the native kernels under AddressSanitizer and UBSan in CI
  (`tools/sanitize_native.py`).
- Run the native CPU kernels on PyTorch's intra-op threads. Under PyTorch's default
  OpenMP backend they ran serially, since `at::parallel_for` in an extension compiled
  without OpenMP flags is a plain loop, and decoded each element's position by
  integer division on every axis. Each thread now decodes its first element and steps
  through the rest, and axes with one sample, a one-tap kernel and no padding are
  skipped on CPU and CUDA, so CUDA problems with at most eight other axes use the
  per-rank kernels at any rank. Values are bitwise unchanged. On four threads, the
  automatic CPU choice at ranks 5 and 6 went from 14-17 times slower than the PyTorch
  backend to 5.3-5.4 times faster.
- Check partial-convolution arguments once per call, and reuse the all-ones counting
  kernel across eager CPU and CUDA calls (one per stream; compilers, functorch
  transforms, fake tensors and CUDA graph capture build their own). Eager training
  steps of small partial layers are about 2-3% faster.
- Test partial convolution at ranks 5 to 42: against NVIDIA's arithmetic in every
  padding mode, and natively against the portable backend beyond CUDA's per-rank
  kernels. Also test inductor compilation of rank-5 native and partial layers, ONNX
  Runtime execution at rank 5 with partial layers, and two-process data-parallel
  gradients over Gloo where the PyTorch build provides a Gloo transport (PyTorch
  2.10's Windows wheels do not).
- Keep minidumps of interpreter crashes in Windows CI and upload them with the built
  addon when a job fails (`tools/crash_dumps.ps1`); each job first confirms that its
  runner writes dumps. Runner processes inherit `SEM_NOGPFAULTERRORBOX`, which keeps
  Windows Error Reporting from writing dumps, so the test session clears it there.
- Explain the Windows CI illegal-instruction crash: on GitHub's Emerald Rapids
  Windows runners, oneDNN's AMX bfloat16 kernels fault at their first tile
  instruction. CI keeps oneDNN below AMX there (`ONEDNN_MAX_CPU_ISA`).
- Export bias-free 3-D convolutions that ONNX Runtime accepts. onnxscript's `conv3d`
  translation, used by PyTorch 2.6's Dynamo exporter, writes a `[C_out, 2]` zero bias
  copied from its complex variant; `export_onnx` now supplies a correct one, which
  also repairs stock `Conv3d` layers in the exported model.
- Script layers on Python 3.14 with PyTorch 2.9, whose TorchScript reads
  `__annotations__` from module instances; the scriptable adapter declares it
  explicitly. Stock modules such as `Linear` still need PyTorch 2.10 there.
- Document and detect in the version contract that PyTorch 2.4.1 on Windows
  computes wrong grouped, dilated float64 `conv3d` gradients on AMD CPUs.
- Check Apple MPS values, gradients and masked partial convolution against float64
  CPU references wherever MPS is available, including hosted macOS runners.
- Broaden CI to PyTorch 2.15 and Python 3.15. The version contract runs every
  release line from 1.11 on the oldest and the newest Python it supports, and the
  portable and native floors on Linux ARM64, Windows, macOS, Windows on Arm and
  Intel macOS. The full suite adds Python 3.11, 3.15 and free-threaded 3.14t, Linux
  ARM64 and Windows on Arm. Hosted CI compiles, loads and registers the CUDA kernels
  without a GPU for PyTorch 2.4 and 2.15 (`tools/check_cuda_build.py`), runs weekly,
  and adds a PyTorch nightly canary to scheduled and manual runs. Uploaded
  correctness artifacts now include the JUnit and JSON reports, which sat in a
  dot-directory that uploads skipped.
- Releases attest build provenance and publish a GitHub release with the checked
  wheel, sdist and changelog section (`tools/release_notes.py`). The manual CUDA
  workflow takes the PyTorch version, CUDA wheel index and Python as inputs.

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
