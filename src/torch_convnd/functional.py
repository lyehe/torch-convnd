"""Public full N-dimensional convolution, validation and layout handling."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ._compat import HAS_LIBRARY, is_autocast_enabled
from ._dispatch import BACKENDS, convolution, ensure_compatible_operator
from ._geometry import DEFAULT_WORKSPACE_BYTES, _tuple
from ._padding import apply_padding, padding_pairs


def conv_nd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: str | int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    *,
    padding_mode: str = "zeros",
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    backend: str = "auto",
) -> Tensor:
    """Apply full dense convolution with Conv2d/Conv3d-style semantics.

    Supports batched/unbatched inputs, zero/reflect/replicate/circular padding,
    'same'/'valid', real and complex dtypes, AMP, higher derivatives and torch.func.
    ``backend="auto"`` uses vendor kernels for ranks 1-3 and CPU rank 4, with
    native kernels for NVIDIA CUDA rank 4 and higher CPU/CUDA ranks. ``native``
    and ``torch`` select the native and portable implementations explicitly.
    Transforms preserve differentiable operations on every backend. Low-precision
    native backward accumulates in FP32.

    ``workspace_bytes`` caps the reusable native patch buffer, excluding inputs,
    outputs, casts, gradient storage and GEMM workspace. Transform decompositions
    additionally allocate indexing tensors. Backward recomputes patches.
    """
    rank = weight.ndim - 2
    if backend not in BACKENDS:
        raise ValueError(f"Invalid backend {backend!r}; expected 'auto', 'native' or 'torch'")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive")
    if not 1 <= rank <= 62:
        message = "weight must have output/input channels and 1-62 spatial dimensions"
        raise ValueError(message)
    if input.ndim not in {rank + 1, rank + 2}:
        raise ValueError(f"expected {rank + 1} unbatched or {rank + 2} batched dimensions")
    unbatched = input.ndim == rank + 1
    if unbatched:
        input = input.unsqueeze(0)
    stride, dilation = _tuple(stride, rank, "stride"), _tuple(dilation, rank, "dilation")
    padding = padding if isinstance(padding, str) else _tuple(padding, rank, "padding")
    if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
        raise ValueError(f"Invalid padding_mode {padding_mode!r}")
    if min(*stride, *dilation) <= 0 or (not isinstance(padding, str) and min(padding) < 0):
        raise RuntimeError("stride and dilation must be positive; padding must be nonnegative")
    if groups <= 0:
        raise RuntimeError("groups must be positive")
    if (
        input.shape[1] <= 0
        or weight.shape[0] <= 0
        or input.shape[1] % groups
        or weight.shape[0] % groups
        or weight.shape[1] != input.shape[1] // groups
    ):
        raise RuntimeError("input/weight channels must match and be divisible by groups")
    if any(size <= 0 for size in (*input.shape[2:], *weight.shape[2:])):
        raise RuntimeError("spatial and kernel dimensions must be positive")
    if input.device != weight.device:
        raise RuntimeError("input and weight must have the same device and dtype")
    castable = {torch.float16, torch.bfloat16, torch.float32}
    amp = input.device.type in {"cpu", "cuda"} and is_autocast_enabled(input.device.type)
    if input.dtype != weight.dtype and not (
        amp and input.dtype in castable and weight.dtype in castable
    ):
        raise RuntimeError("input and weight must have the same device and dtype")
    if bias is not None and (
        bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
        or bias.device != input.device
        or (
            bias.dtype != input.dtype
            and not (amp and bias.dtype in castable and input.dtype in castable)
        )
    ):
        raise RuntimeError("bias must match output channels, device and dtype")
    pairs = padding_pairs(padding, weight.shape[2:], stride, dilation)
    if padding_mode != "zeros" or any(left != right for left, right in pairs):
        input = apply_padding(input, pairs, padding_mode)
        implicit_padding = (0,) * rank
    else:
        implicit_padding = tuple(left for left, _ in pairs)
    parameters = stride, implicit_padding, dilation, groups, workspace_bytes, backend
    operation = convolution
    if torch.jit.is_tracing() and HAS_LIBRARY:
        ensure_compatible_operator()
        operation = torch.ops.torch_convnd.compatible_conv_nd
    output = operation(input, weight, bias, *parameters)
    return output.squeeze(0) if unbatched else output
