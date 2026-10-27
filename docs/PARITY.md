# Conformance and verification boundaries

## Current v0.5.0 validation

The [version contract](assets/reports/compatibility_0_5_0.json) passes across 18
PyTorch release lines from 1.11 through 2.14: **778 passing checks, no failures**.
The 122 skips correspond to unavailable native custom-op APIs and `torch.func`
on older frameworks; they are not counted as verified implementations.

The full Windows CPU/RTX 4090 run passed **1,060 tests without skips** on
PyTorch 2.10.0+cu130. The newer PyTorch 2.14.1 CPU run passed **602 tests**, with
458 CUDA-dependent skips and no failures. Raw reports:
[2.10 CPU/CUDA](assets/reports/correctness_0_5_0_windows_torch210.json),
[2.14 CPU](assets/reports/correctness_0_5_0_windows_torch214.json).

The core contract uses independent values and input/weight/bias gradients,
multiple real/complex dtypes, groups, stride/dilation, padding, serialization,
second derivatives, forward AD and actual ONNX Runtime inference. Native checks
run where the required APIs are present. Compatibility uses feature probes,
without a hard-coded PyTorch version table in the execution path.

CI repeats the 18-release contract on Linux and the modern feature matrix on
Linux/Windows/macOS. Local version evidence is Windows; remote outcomes are
recorded separately in the repository's Actions checks.

## Historical v0.4.0 validation

The final Windows run passed **1,010 tests with no failures or skips** on
Python 3.10 / PyTorch 2.10.0+cu130, CPU and RTX 4090. A separate Python 3.13 /
PyTorch 2.14.1 CPU environment passed **552 tests**, with **458 CUDA-dependent
cases skipped** and no failures.

Raw cases, numerical comparisons, versions and source hashes:
[minimum CUDA/CPU stack](assets/reports/correctness_0_4_0_windows_torch210.json),
[newer CPU stack](assets/reports/correctness_0_4_0_windows_torch214.json).
Each run writes separate numerical measurement files beside its JUnit output,
so concurrent compatibility runs cannot overwrite one another's measurements.

The built 0.4.0 wheel also passed isolated native training, script/trace JVP,
and ONNX inference on both stacks. The
[distribution record](assets/reports/distribution_0_4_0.json) retains its hash,
typed-package check and installed-wheel import verification.

The additions cover generated independent-reference geometries/gradients,
compiler-free import/training, automatic fallback, checkpointed updates, stock
state transfer, diagnostics and native cache isolation while another process
holds a loaded library. Public typing, pre-commit checks, strict documentation
builds and actionlint validation also pass. Linux/macOS are defined in CI but
have not been executed from this workspace.

## Historical v0.3.0 baseline

Version 0.3.0 closes the confirmed scripted/traced forward-AD failure in v0.2.0.
A shared composite dispatcher keeps transform context at serialized operator
boundaries. Both `torch.func.jvp` and `forward_ad` dual tensors now retain their
correct nonzero tangents after script/trace save and reload.

The [v0.3.0 machine report](assets/reports/parity_0_3_0_final.json) records the
full suite, numerical comparisons, environment and implementation hashes. The
[v0.2.0 audit](assets/reports/scripted_forward_ad_audit_0_2_0.json) is retained
as historical evidence of the defect, rather than a current limitation.

**Result: 997 passed, 0 failed, 0 skipped in 227.50 seconds.** This includes 55
native tests, 643 feature-parity tests, 80 compiler/export tests, 156 serialized
transform combinations, 57 ONNX checks and 6 fresh-process loading/compiler checks.
The numerical archive retains 2,256 output/gradient comparisons.

The built wheel was also installed and tested with isolated Python (`-I`),
loading from `site-packages` on CPU-only and CUDA builds. All 16 checks passed:
native training, AOT compilation, script/trace JVP after save/load, AMP, complex
training and 4D ONNX Runtime inference. The
[wheel validation record](assets/reports/wheel_validation_0_3_0.json)
includes its SHA256 and confirms that packaged sources match this suite.

## Verified matrix

Validation platform: Windows, Python 3.10, PyTorch 2.10.0+cu130, CPU and physical
GPU 0 (NVIDIA RTX 4090). Tests disable TF32 for both implementations and configure
deterministic GEMM before CUDA initialization. CUDA Inductor uses
triton-windows 3.6.0.post26. ONNX Runtime CPU is version 1.23.2.

| Coverage | Ranks / devices |
| --- | --- |
| Forward, input/weight/bias gradients | Stock 2D/3D, CPU/CUDA, all three backends |
| Per-axis geometry, groups/depthwise, no bias, unbatched inputs | Stock 2D/3D, CPU/CUDA, all backends |
| Numeric/`same`/`valid` padding; zeros/reflect/replicate/circular | Stock 2D/3D, CPU/CUDA, all backends |
| FP16/BF16/FP32/FP64, complex32/64/128 | Stock 2D/3D, CPU/CUDA, all backends |
| FP16/BF16 autocast and scaled training updates | Stock 2D/3D, CPU/CUDA, all backends |
| Noncontiguous tensors, channels-last, empty batches | 2D/3D CPU/CUDA; additional native N-D cases |
| Higher derivatives, complex gradgradcheck and finite differences | 2D/3D and independent 4D checks |
| vmap, nested vmap, mapped parameters, per-sample gradients | 2D/3D CPU/CUDA; independent raw-op rule |
| JVP/VJP, grad, Jacobians, Hessians, linearize, dual tensors | 2D/3D CPU/CUDA; independent 4D checks |
| Script **and** trace save/reload combined with JVP, dual tensors, vmap and grad | 2D/3D/4D CPU/CUDA, all backends |
| Script combined with complex forward AD | 2D/3D CPU/CUDA, all backends |
| Fullgraph AOTAutograd, including AMP and func.grad | 2D/3D CPU/CUDA; native 4D optimizer checks |
| Default Inductor forward/backward | 2D/3D CPU/CUDA, all backends |
| Default Inductor 4D FP32 and AMP training | CPU/CUDA, all backends |
| Dynamic batch/spatial `torch.export` save/reload | 2D/3D/4D CPU/CUDA, all backends |
| Fresh-process script/trace loading with compiler calls blocked | Native 4D saved graphs; portable transforms and CPU auto execution |
| First-use native initialization inside compiled graphs | Fresh-process AOTAutograd and Inductor, CPU/CUDA |
| Standard ONNX Conv plus activation, groups, every padding mode | 2D/3D/4D, all backends, CPU runtime |
| ONNX dynamic shapes and script/trace models | 2D/3D/4D, CPU runtime |
| ONNX floating dtype retention | FP16/BF16/FP32/FP64; runtime execution for FP16/FP32 |
| ONNX native export without a compiler | Fresh-process 4D export and ONNX checker |

The native suite additionally covers ranks 1-8, a nontrivial rank-9 kernel,
rank-62 indexing, forced small workspace tiles, large integer metadata,
deterministic fold, CUDA stream/device handling and finite-difference checks.
The independent reference contracts each kernel tap with spatial slices; it does
not call the package implementation. Constructor tests check initialization,
state dictionaries, positional arguments, attributes and invalid configurations.

## Numerical criteria

Comparisons check elementwise error and relative L2 error, including input,
weight and bias gradients under shared cotangents. Low-precision accumulation
order can change rounding; bitwise equality is not the criterion.

| Dtype | Absolute tolerance | Relative tolerance | Maximum relative L2 |
| --- | --- | --- | --- |
| FP64 | 1e-9 | 1e-9 | 1e-10 |
| complex128 | 2e-9 | 2e-9 | 1e-10 |
| FP32 | 2e-5 | 2e-5 | 1e-5 |
| complex64 | 5e-5 | 5e-5 | 2e-5 |
| FP16 | 0.005 | 0.005 | 0.004 |
| BF16 | 0.05 | 0.03 | 0.02 |
| complex32 | 0.02 | 0.02 | 0.012 |

The machine report retains every measured comparison. AMP gradients are grouped
by compute precision even when parameters store gradients in FP32. Independent
native, finite-difference, compiler and ONNX tests have their own appropriate
tolerances. Complex32 remains experimental in PyTorch.

## Boundaries

- This matrix does not establish every possible feature combination or
  performance parity for every shape. Higher ranks retain generic fallbacks.
- Linux and other PyTorch versions/accelerators have not been run locally. The
  checked-in Windows/Linux CPU CI workflow defines future checks; no successful
  remote CI run is claimed. Non-native devices use portable PyTorch operations,
  subject to the underlying device's operator support.
- ONNX exports use standard real `Conv` nodes. Complex interfaces need an explicit
  real/imaginary model interface. Local ONNX Runtime CPU executes FP16/FP32
  spatial ranks 2-4; FP64/BF16 graphs pass checking but that provider lacks their
  Conv kernels. Other providers and higher runtime ranks remain unverified.
- Empty vmap batches follow the stock PyTorch 2.10 inner-batch quirk on the default
  rank-2/3 vendor path. The raw native rule retains the logical inner batch size;
  this difference is explicitly tested rather than treated as exact parity.
- Native rank-2/3 output strides can follow the requested channels-last layout
  even when a vendor would choose another layout. Default auto paths use vendor
  behavior directly. Values and gradients remain the comparison criterion.
- Native execution uses ATen GEMM and custom patch kernels, rather than cuDNN.
  cuDNN-specific settings apply to vendor paths. The native deterministic path
  uses gather for input gradients; GEMM requires PyTorch's deterministic setup.
- Transposed, sparse, quantized and lazy convolution are separate operator families.
- Saved TorchScript/export graphs need `import torch_convnd` in a fresh process.
  Import registers metadata without building; native execution builds lazily.
  Compiled graphs also load native kernels on first execution without a manual
  preload. Host compiler requirements for Inductor still apply.

## Reproduce

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
.venv/Scripts/python.exe -m pytest -q --disable-warnings --junitxml=.artifacts/parity_0_3_0_final.xml
```

GPU/toolkit and optional Triton availability determine skips on other machines;
skipped tests are not evidence of support on those machines.

Reference APIs: [Conv2d](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Conv2d.html),
[Conv3d](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Conv3d.html),
[custom operators](https://docs.pytorch.org/docs/2.10/library.html),
[ONNX export](https://docs.pytorch.org/docs/2.10/onnx.html), and
[ONNX Conv schema](https://onnx.ai/onnx/operators/onnx__Conv.html).
