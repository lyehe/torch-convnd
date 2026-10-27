# Measured backend and kernel improvements

Version 0.3.0 selects vendor convolution for ranks 1-3 and CPU rank 4, retaining
the native addon for NVIDIA CUDA rank 4 and higher CPU/CUDA ranks. These choices
come from the comparisons below, rather than assuming custom kernels are always
faster. Both the native backend and portable backend remain selectable explicitly.

## Method

Windows, PyTorch 2.10.0+cu130, CUDA 13, NVIDIA RTX 4090 (physical GPU 0),
dual Intel Xeon Gold 6248R host with four PyTorch CPU threads. The machine is
shared: GPU 0 also serves the desktop; another workload uses GPU 1. These are
operator measurements, rather than full network training results.

CPU/GPU benchmark processes were run sequentially for the records below. Each
case checks matched outputs and input/weight/bias gradients before timing. Timings
are synchronized wall-time medians over 30 iterations after 5 warmups. Both
matmul and cuDNN TF32 are disabled. AMP cases use FP32 parameters with FP16/BF16
compute. Kernels are size 3 on each spatial axis, padding 1, groups 1, batch 1,
and a 64 MiB native patch budget. Build and compilation times are excluded.

Raw samples, environment, memory and source hashes:
[CPU](assets/reports/backends_0_3_0_cpu_release.json),
[CUDA](assets/reports/backends_0_3_0_cuda_release.json).
Timing differences between `auto` and its selected backend also reflect
wall-clock variation and measurement order; these runs do not establish small
percentage improvements with statistical confidence.

## Forward plus backward time

CPU times in milliseconds:

| Input shape / output channels | Forced native | PyTorch backend | Auto |
| --- | --- | --- | --- |
| `[1,4,32,32]` / 8 | 8.52 | 0.40 | 0.40 |
| `[1,4,8,8,8]` / 8 | 15.11 | 0.47 | 0.47 |
| `[1,2,4,4,4,8]` / 4 | 26.21 | 2.51 | 2.43 |

Auto improves over forcing the native addon by approximately 11-33 times on
these CPU cases. The rank-4 portable path batches one axis through vendor Conv3d,
retaining unrestricted full 4D weights. Lower ranks call vendor kernels directly.

CUDA times in milliseconds. Small 4D is `[1,4,8,8,8,16]` with 8 output channels;
medium 4D is `[1,8,16,16,16,32]` with 16 output channels.

| Workload | Forced native | PyTorch backend | Auto |
| --- | --- | --- | --- |
| 2D FP32, `[1,4,32,32]` / 8 | 1.01 | 0.60 | 0.61 |
| 3D FP32, `[1,4,8,8,8]` / 8 | 1.06 | 0.55 | 0.58 |
| Small 4D FP32 | 1.03 | 2.06 | 1.01 |
| Medium 4D FP32 | 2.64 | 3.87 | 2.53 |
| Small 4D FP16 AMP | 1.30 | 2.77 | 1.28 |
| Medium 4D FP16 AMP | 2.37 | 7.52 | 2.44 |
| Medium 4D BF16 AMP | 2.54 | 3.15 | 2.54 |

The medium FP32 auto case is about 1.5 times faster than the portable 4D path;
the FP16 AMP case is about 3.1 times faster. The BF16 case has a smaller gain.
Shape, workspace, hardware, cuDNN settings and precision can change the ranking;
auto is a measured policy, not runtime autotuning or a universal optimum.

## Memory and mixed-precision backward

CUDA backward now multiplies FP16/BF16 operands into FP32 outputs and accumulates
gradients in FP32. This avoids full FP32 copies of the input, weights and output
cotangent. One FP32 patch allocation is reused for low-precision patches and
FP32 input-gradient columns. CPU native low-precision backward retains promotion.

Medium 4D AMP extra allocated CUDA tensor memory fell from **90.10 MiB to
78.06 MiB**, approximately 13%, between the
[initial](assets/reports/backends_0_3_0_cuda_initial.json) and
[mixed-GEMM](assets/reports/backends_0_3_0_cuda_mixed_gemm.json) records.
Small timing changes across those runs are subject to machine variation.

The faster native 4D path still uses more memory than the portable path:

| Medium 4D compute | Auto/native extra MiB | PyTorch backend extra MiB |
| --- | --- | --- |
| FP32 | 84.04 | 44.50 |
| FP16 AMP | 78.06 | 34.83 |
| BF16 AMP | 78.06 | 36.55 |

These figures are peak extra allocated tensor storage for the timed operation,
including the loss and gradients, above already allocated inputs/parameters.
They exclude optimizer state and CUDA allocator reservations. The native patch
budget does not cap total memory; vendor paths manage their own workspace.
Choose `backend="torch"` or reduce the native patch budget when memory is the
binding constraint. Forward AD and `torch.func` use differentiable operations
whose speed/memory is not represented by these ordinary-training measurements.

The native rank-1 through rank-8 fast indexing path, guarded 32-bit division,
64-bit fallback, output-tile GEMM and patch recomputation optimizations are
retained. No dimension is factored into separable learned kernels.

## Reproduce

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
.venv/Scripts/python.exe benchmarks/compare_backends.py --device cuda:0 --iterations 30 --output benchmarks/results/new_cuda.json
.venv/Scripts/python.exe benchmarks/compare_backends.py --device cpu --iterations 30 --output benchmarks/results/new_cpu.json
```

Use fresh output paths. The legacy `torch-convnd-benchmark` CLI explicitly
forces the native backend and retains its original Conv3d-tap reference for
historical comparisons. Earlier v0.1/v0.2 records used different GPU selections,
workspace settings and/or implementations; they are not matched before/after
comparisons with this run.
