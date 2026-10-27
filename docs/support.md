# Backends and compatibility

The minimum runtime is Python 3.10 / PyTorch 1.11. The version contract checks
every minor release line from 1.11 to 2.14, with native tests enabled by actual
capabilities. A separate modern matrix covers Linux/Windows/macOS and selected
Python versions 3.10-3.14. Source and CI live at
[lyehe/torch-convnd](https://github.com/lyehe/torch-convnd).

There is no promise that an absent framework API is emulated, or that an untested
future release is already verified. The package has no hard upper version pin;
optional APIs are probed and portable execution avoids modern custom-op dependencies.

| Feature | Availability policy |
| --- | --- |
| Dense forward/backward, higher derivatives, padding, state dictionaries | Portable core verified on 1.11-2.14 |
| Script/trace | Compatible operator when available; pure Tensor script adapter otherwise |
| `torch.func` / `torch.compile` | Only where the framework supplies the API |
| Native autograd/fake kernels | Requires detected registration and unified-autocast APIs; CPU checked from 2.4 |
| Native raw-op vmap | Registration is conditional; public transforms retain tensor decompositions |
| FP32-output mixed CUDA GEMM | Requires the actual ATen overloads; FP32-promotion fallback otherwise |
| ONNX | Legacy/modern export selected from the actual signature and supported opsets |

## Backend policy

| Device / rank | Automatic choice |
| --- | --- |
| CPU and NVIDIA CUDA, ranks 1-3 | Stock PyTorch convolution kernels |
| CPU rank 4 | Batched full Conv3d taps |
| NVIDIA CUDA rank 4, CPU/CUDA ranks 5-62 | Native addon if build prerequisites are available; portable otherwise |
| AMD ROCm | PyTorch vendor/decomposition operations; no native CUDA addon |
| Apple MPS | Vendor operations for supported lower ranks; generic indexed decomposition for higher ranks |
| Other devices / Meta | Portable PyTorch operations and shape inference, subject to device support |

`backend="torch"` always avoids addon compilation. `backend="native"` explicitly
requires CPU or NVIDIA CUDA kernels and fails with a prerequisite message when
necessary. An automatic path can use portable CUDA after a CPU-only addon was
loaded. Tools and environment probes are cached for a process.

Native and portable kernels retain the same full weight tensor. No backend is
claimed to be fastest for every shape or device. The [measurements](PERFORMANCE.md)
show speed and memory tradeoffs, including native workspace costs.

## Verification status

The [correctness report](PARITY.md) retains historical baselines and the current
compatibility evidence. Local hardware is Windows CPU and RTX 4090.
Python 3.13 / PyTorch 2.14.1 CPU is additionally exercised in a separate environment.
Linux/macOS, Apple MPS, AMD ROCm and other GPUs have not been executed locally.
The CUDA workflow is manual and requires a trusted self-hosted runner with a
matching toolkit. CI does not claim distributed multi-process or CUDA Graph coverage.

## Export and reference differences

ONNX export preserves FP16, BF16, FP32 and FP64 graph dtypes. Local ONNX Runtime
1.23.2 CPU executes FP16/FP32 spatial ranks 2-4; its FP64/BF16 Conv kernels are
unavailable. Other versions/providers may differ. Complex tensor interfaces are
outside the [ONNX Conv schema](https://onnx.ai/onnx/operators/onnx__Conv.html);
provide an explicit real/imaginary model interface when required.

Default rank-2/3 vendor paths inherit PyTorch's empty-vmap behavior and layout
selection. The raw native rule preserves the logical inner batch for empty maps;
native output strides can differ while values/gradients agree. cuDNN settings
apply to vendor paths; the native addon uses ATen GEMM and its own patch kernels.

Transposed, quantized, sparse and lazy convolutions are separate operator families
and are not included. Dense kernel volume still grows as the product of all
per-axis kernel sizes; higher rank alone does not remove that computational cost.
