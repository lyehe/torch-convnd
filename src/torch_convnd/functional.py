"""Public full N-dimensional convolution, validation and layout handling."""

import contextlib
import functools
import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from ._compat import (
    HAS_LIBRARY,
    autocast_disabled,
    current_stream_id,
    is_autocast_enabled,
    is_capturing,
    is_compiling,
    is_onnx_exporting,
    plain_tensor,
)
from ._dispatch import BACKENDS, convolution, ensure_compatible_operator
from ._geometry import DEFAULT_WORKSPACE_BYTES, _tuple, any_true, has_rank
from ._padding import apply_padding, padding_pairs

# Tuples, not sets: Dynamo before PyTorch 2.2 cannot trace set membership.
_CASTABLE = (torch.float16, torch.bfloat16, torch.float32)
_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}
if hasattr(torch, "complex32"):  # absent in PyTorch 1.11
    _REAL[torch.complex32] = torch.float16


_PADDING_MODES = ("zeros", "reflect", "replicate", "circular", "partial")


def _outside_inference():
    """Leave inference mode, so eager caches serve both modes.

    Compiled counts are never cached, and Dynamo before PyTorch 2.3 cannot trace
    the context. Leaving inference mode enables grad mode, so enter it before no_grad.
    """
    return contextlib.nullcontext() if is_compiling() else torch.inference_mode(False)


class PartialConvOutput(NamedTuple):
    """A partial convolution's output and its update mask, the next layer's input mask."""

    output: Tensor
    mask: Tensor


def _check_bias(input: Tensor, weight: Tensor, bias: Tensor | None, amp: bool) -> None:
    if bias is not None and (
        bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
        or bias.device != input.device
        or (
            bias.dtype != input.dtype
            and not (amp and bias.dtype in _CASTABLE and input.dtype in _CASTABLE)
        )
    ):
        raise RuntimeError("bias must match output channels, device and dtype")


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
    ``padding_mode="partial"`` is zero padding with NVIDIA's partial-convolution
    rescaling: each window is scaled by its size over its in-bounds taps, as
    :func:`partial_conv_nd` does without a mask.
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
    if not 1 <= rank <= 62:
        message = "weight must have output/input channels and 1-62 spatial dimensions"
        raise ValueError(message)
    if not has_rank(input.ndim, rank):
        raise ValueError(f"expected {rank + 1} unbatched or {rank + 2} batched dimensions")
    unbatched = input.ndim == rank + 1
    if unbatched:
        input = input.unsqueeze(0)
    stride, dilation = _tuple(stride, rank, "stride"), _tuple(dilation, rank, "dilation")
    padding = padding if isinstance(padding, str) else _tuple(padding, rank, "padding")
    _check_options(padding_mode, workspace_bytes, backend)
    if padding_mode == "partial":
        # The mask-free partial convolution; it validates and pads with zeros itself.
        output, _ = _partial_conv_nd(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            mask=None,
            multi_channel=False,
            padding_mode="zeros",
            workspace_bytes=workspace_bytes,
            backend=backend,
            need_update=False,
        )
        return output.squeeze(0) if unbatched else output
    amp = _check_arguments(input, weight, stride, padding, dilation, groups)
    _check_bias(input, weight, bias, amp)
    pairs = padding_pairs(padding, weight.shape[2:], stride, dilation)
    output = _execute(
        input, weight, bias, stride, pairs, dilation, groups, padding_mode, workspace_bytes, backend
    )
    return output.squeeze(0) if unbatched else output


def _check_options(padding_mode, workspace_bytes, backend):
    """conv_nd's checks of the options that select padding and kernels."""
    if backend not in BACKENDS:
        raise ValueError(f"Invalid backend {backend!r}; expected 'auto', 'native' or 'torch'")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive")
    if padding_mode not in _PADDING_MODES:
        raise ValueError(f"Invalid padding_mode {padding_mode!r}")


def _check_arguments(input, weight, stride, padding, dilation, groups):
    """conv_nd's geometry, device and dtype checks of a batched input; returns whether
    autocast applies on its device."""
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
    if any_true([size <= 0 for size in (*input.shape[2:], *weight.shape[2:])]):
        raise RuntimeError("spatial and kernel dimensions must be positive")
    if input.device != weight.device:
        raise RuntimeError("input and weight must have the same device and dtype")
    amp = input.device.type in {"cpu", "cuda"} and is_autocast_enabled(input.device.type)
    if input.dtype != weight.dtype and not (
        amp and input.dtype in _CASTABLE and weight.dtype in _CASTABLE
    ):
        raise RuntimeError("input and weight must have the same device and dtype")
    return amp


def _execute(
    input, weight, bias, stride, pairs, dilation, groups, padding_mode, workspace_bytes, backend
):
    """Pad and dispatch a batched convolution whose arguments conv_nd has validated."""
    if padding_mode != "zeros" or any_true([left != right for left, right in pairs]):
        input = apply_padding(input, pairs, padding_mode)
        implicit_padding = (0,) * len(pairs)
    else:
        implicit_padding = tuple(left for left, _ in pairs)
    parameters = stride, implicit_padding, dilation, groups, workspace_bytes, backend
    operation = convolution
    if torch.jit.is_tracing() and HAS_LIBRARY:
        ensure_compatible_operator()
        operation = torch.ops.torch_convnd.compatible_conv_nd
    return operation(input, weight, bias, *parameters)


def partial_conv_nd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: str | int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    *,
    mask: Tensor | None = None,
    multi_channel: bool = False,
    padding_mode: str = "zeros",
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    backend: str = "auto",
) -> PartialConvOutput:
    """Apply NVIDIA's partial convolution; return the output and its update mask.

    Each output sums the valid samples of its window, rescaled by the window size
    over the valid count, as in NVIDIA's partialconv (Liu et al., 2018)::

        count  = sum(mask over the window and, with multi_channel, the group's channels)
        update = clamp(count, 0, 1)
        output = conv(input * mask) * window / count * update           # without bias
        output = (conv(input * mask) * window / count * update + bias) * update

    Mask contract:

    - Values: 1 marks a valid sample and 0 a hole; fractional values follow the
      formula above. Any real dtype on the input's device. No gradient flows through
      the counts (only through ``input * mask``, as in NVIDIA's code).
    - Layout: the input's number of dimensions. Without ``multi_channel`` the mask has
      one channel shared by every input channel; with it, one channel per input
      channel (or one, broadcast) and counts per group. The batch and spatial sizes
      equal the input's or are 1 (broadcast).
    - Padding: the mask is padded like the input. Zero padding adds holes; reflect,
      replicate and circular padding copy validity with the values. ``mask=None``
      means every sample is valid, so zero padding gives partial-convolution padding
      (also available as ``conv_nd(..., padding_mode="partial")``).
    - Result: ``PartialConvOutput(output, mask)``, where ``mask`` is the update mask:
      one channel, or ``out_channels`` with ``multi_channel``, in the layout the next
      partial layer of the same kind accepts. It is real, may be a broadcast view
      (clone it before writing in place), and is 1 wherever a window saw a valid sample.

    Counts are computed in float32, or float64 for float64 inputs, outside autocast, so
    they stay exact under AMP; the rescale runs in the output dtype, or in float32 when
    the window exceeds that dtype's range. Geometry, dtype, AMP, padding and backend
    options follow :func:`conv_nd`.
    """
    if padding_mode == "partial":
        message = "partial_conv_nd already rescales zero padding; use padding_mode='zeros'"
        raise ValueError(message)
    output, update = _partial_conv_nd(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        mask=mask,
        multi_channel=multi_channel,
        padding_mode=padding_mode,
        workspace_bytes=workspace_bytes,
        backend=backend,
        need_update=True,
    )
    assert update is not None
    return PartialConvOutput(output, update)


def _mask_free_update_is_full(spatial, kernel, stride, padding, dilation, padding_mode):
    """Whether a mask-free partial convolution leaves every output sample valid.

    Pure geometry, so no tensor is inspected and nothing synchronizes. Only zero
    padding wider than a kernel's reach can leave a window with no in-bounds tap.
    Traced sizes, which are tensors, answer False, which asks the caller to keep the
    update mask; symbolic sizes under torch.compile become guards.
    """
    if padding_mode != "zeros":
        return True
    if any_true([type(size) is not int for size in spatial]):
        return False
    rank = len(kernel)
    steps, spacings = _tuple(stride, rank, "stride"), _tuple(dilation, rank, "dilation")
    pads = padding if isinstance(padding, str) else _tuple(padding, rank, "padding")
    pairs = padding_pairs(pads, kernel, steps, spacings)
    for size, taps, step, spacing, (left, right) in zip(
        spatial, kernel, steps, spacings, pairs, strict=True
    ):
        if not _axis_is_full(size, taps, step, spacing, left, right):
            return False
    return True


def _windows(size, kernel, stride, dilation, left, right):
    """Output windows along one padded axis."""
    return (size + left + right - dilation * (kernel - 1) - 1) // stride + 1


def _valid_taps(size, kernel, stride, dilation, left, right):
    """Taps of each output window along one zero-padded axis that land inside the input."""
    counts = []
    for position in range(_windows(size, kernel, stride, dilation, left, right)):
        origin = position * stride - left
        first = max(0, -(origin // dilation))  # ceil(-origin / dilation), at least 0
        last = min(kernel - 1, (size - 1 - origin) // dilation)
        counts.append(max(0, last - first + 1))
    return counts


def _axis_is_full(size, kernel, stride, dilation, left, right):
    """Whether every output window along one zero-padded axis has an in-bounds tap.

    Taps at most an axis length apart cannot straddle the input, so a window then
    misses it only by lying wholly in the padding, as the first or the last would.
    """
    length = _windows(size, kernel, stride, dilation, left, right)
    if length <= 0:
        return False
    if dilation > size:
        return min(_valid_taps(size, kernel, stride, dilation, left, right)) > 0
    return left <= dilation * (kernel - 1) and (length - 1) * stride - left <= size - 1


def _scale(counts, window, ratio_dtype, update_dtype):
    """NVIDIA's ratio (window / count * update) and update (clamp(count, 0, 1)).

    The division never sees zero: an empty window has update 0, hence ratio 0.
    """
    update = counts.clamp(0, 1)
    ratio = update * window / counts.clamp_min(torch.finfo(counts.dtype).tiny)
    return ratio.to(ratio_dtype), update.to(update_dtype)


def _axis_counts(size, kernel, stride, dilation, left, right, dtype, device):
    """_valid_taps as integer arithmetic on the device.

    No host values are copied, so nothing synchronizes and symbolic sizes stay symbolic.
    """
    length = _windows(size, kernel, stride, dilation, left, right)
    with _outside_inference(), torch.no_grad():
        origin = torch.arange(-left, (length - 1) * stride - left + 1, stride, device=device)
        first = (-torch.div(origin, dilation, rounding_mode="floor")).clamp_min(0)
        last = torch.div(size - 1 - origin, dilation, rounding_mode="floor")
        return (last.clamp_max(kernel - 1) - first + 1).clamp_min(0).to(dtype)


@functools.lru_cache(maxsize=512)
def _cached_axis_counts(size, kernel, stride, dilation, left, right, dtype, device, stream):
    """_axis_counts, kept per device stream."""
    return _axis_counts(size, kernel, stride, dilation, left, right, dtype, device)


def _mask_free_scale(
    spatial,
    kernel,
    stride,
    dilation,
    pairs,
    exact,
    ratio_dtype,
    update_dtype,
    device,
    *,
    cache=False,
    stream=None,
):
    """Ratio and update of an all-valid zero-padded input, and whether no window is empty.

    With zero padding and no holes a window's valid taps factor over the axes, so the
    counts are exact integer products of per-axis counts and need no convolution.
    ``cache`` keeps short per-axis vectors for the caller's ``stream``.
    """
    rank, full = len(spatial), True
    with _outside_inference(), torch.no_grad():
        counts = torch.ones((1, 1) + (1,) * rank, dtype=exact, device=device)
        for axis, (size, taps, step, spacing, (left, right)) in enumerate(
            zip(spatial, kernel, stride, dilation, pairs, strict=True)
        ):
            geometry = size, taps, step, spacing, left, right
            if cache and _windows(*geometry) <= _CACHED_AXIS:
                vector = _cached_axis_counts(*geometry, exact, device, stream)
            else:
                vector = _axis_counts(*geometry, exact, device)
            # Built whole: Dynamo before PyTorch 2.4 cannot trace list item assignment.
            shape = (1, 1) + (1,) * axis + (-1,) + (1,) * (rank - axis - 1)
            counts = counts * vector.reshape(shape)
            full = full and _axis_is_full(*geometry)
        ratio, update = _scale(counts, math.prod(kernel), ratio_dtype, update_dtype)
    return ratio, update, full


# Eager calls keep counts on the device, so later calls launch nothing for them. Keys
# include the current stream: a cached tensor is only read, and only freed, in the
# order of the stream that wrote it. Per-axis vectors are kept for axes of up to
# _CACHED_AXIS windows (32 KiB in float64) and finished scales for outputs of up to
# _SMALL_SCALE positions (1 MiB per float32 tensor), so ragged sizes cannot pin much
# memory. Cached tensors are never returned to callers or modified.
_SMALL_SCALE = 1 << 18
_CACHED_AXIS = 1 << 12


@functools.lru_cache(maxsize=32)
def _cached_small_scale(*geometry, stream):
    return _mask_free_scale(*geometry, cache=True, stream=stream)


def _count(
    valid, channels, kernel, stride, pairs, dilation, padding_mode, workspace_bytes, backend
):
    """Window sums of a validity map: a depthwise all-ones convolution outside autocast.

    The geometry is the data convolution's, which conv_nd has already validated.
    ``channels`` is the map's channel count as a plain int, so exporters that trace
    sizes still see a counting kernel of static shape.
    """
    ones = _counting_kernel(valid, channels, kernel)
    with autocast_disabled(valid.device.type):
        return _execute(
            valid,
            ones,
            None,
            stride,
            pairs,
            dilation,
            channels,
            padding_mode,
            workspace_bytes,
            backend,
        )


def _counting_kernel(valid, channels, kernel):
    """_count's all-ones kernel; eager CPU and CUDA calls keep one per stream.

    Compilers, functorch transforms, fake or mode-dispatched tensors and CUDA graph
    capture build their own, as they do the cached counts.
    """
    device = valid.device
    if (
        is_compiling()
        or not plain_tensor(valid)
        or device.type not in {"cpu", "cuda"}
        or is_capturing(device)
    ):
        return torch.ones((channels, 1, *kernel), dtype=valid.dtype, device=device)
    return _cached_ones(channels, kernel, valid.dtype, device, current_stream_id(device))


@functools.lru_cache(maxsize=64)
def _cached_ones(channels, kernel, dtype, device, stream):
    """_counting_kernel's eager kernel, kept per device stream."""
    with _outside_inference(), torch.no_grad():
        return torch.ones((channels, 1, *kernel), dtype=dtype, device=device)


def _partial_conv_nd(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    *,
    mask,
    multi_channel,
    padding_mode,
    workspace_bytes,
    backend,
    need_update,
):
    """partial_conv_nd; ``need_update=False`` skips materializing the update mask."""
    rank = weight.ndim - 2
    if not 1 <= rank <= 62:
        message = "weight must have output/input channels and 1-62 spatial dimensions"
        raise ValueError(message)
    if not has_rank(input.ndim, rank):
        raise ValueError(f"expected {rank + 1} unbatched or {rank + 2} batched dimensions")
    if not isinstance(multi_channel, bool):
        raise TypeError("multi_channel must be a bool")
    if padding_mode == "partial":
        raise ValueError("partial convolution already rescales zero padding; use 'zeros'")
    if mask is not None:
        if mask.ndim != input.ndim:
            raise ValueError(f"mask must have the input's {input.ndim} dimensions")
        if mask.dtype.is_complex:
            raise ValueError("mask must be real")
        if mask.device != input.device:
            raise ValueError("mask must be on the input's device")
        # Explicit comparisons: traced shapes are tensors, which set lookups compare by id.
        batch = 1 if input.ndim == rank + 1 else input.shape[0]
        if input.ndim == rank + 2 and mask.shape[0] != 1 and mask.shape[0] != batch:
            raise ValueError(f"mask batch must be 1 or {batch}")
        for size, wanted in zip(mask.shape[-rank:], input.shape[-rank:], strict=True):
            if size != 1 and size != wanted:
                message = (
                    f"mask spatial shape {tuple(mask.shape[-rank:])} must equal or broadcast "
                    f"to the input's {tuple(input.shape[-rank:])}"
                )
                raise ValueError(message)
    unbatched = input.ndim == rank + 1
    if unbatched:
        input = input.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)
    channels, spatial = input.shape[1], tuple(input.shape[2:])
    if groups <= 0 or channels % groups or weight.shape[0] % groups:
        raise RuntimeError("input/weight channels must match and be divisible by groups")
    amp = input.device.type in {"cpu", "cuda"} and is_autocast_enabled(input.device.type)
    _check_bias(input, weight, bias, amp)
    real = _REAL.get(input.dtype, input.dtype)
    # A weight's shape is fixed, so plain ints are right even where tracing would
    # record sizes, and they keep the counting kernel's shape static for exporters.
    per_group, kernel = channels // groups, tuple(int(size) for size in weight.shape[2:])
    exact = torch.float64 if real == torch.float64 else torch.float32
    options = {"padding_mode": padding_mode, "workspace_bytes": workspace_bytes, "backend": backend}
    update_channels = weight.shape[0] if multi_channel else 1
    steps, spacings = _tuple(stride, rank, "stride"), _tuple(dilation, rank, "dilation")
    pads = padding if isinstance(padding, str) else _tuple(padding, rank, "padding")
    pairs = padding_pairs(pads, kernel, steps, spacings)
    full, valid = False, None
    if mask is None:
        if padding_mode != "zeros" or not any_true([left or right for left, right in pairs]):
            # Every tap reads a real or copied sample: an ordinary convolution.
            output = conv_nd(input, weight, bias, stride, padding, dilation, groups, **options)
            update = None
            if need_update:
                level = _REAL.get(output.dtype, output.dtype)
                ones = torch.ones((), dtype=level, device=input.device)
                update = ones.expand(output.shape[0], update_channels, *output.shape[2:])
            if unbatched:
                return output.squeeze(0), None if update is None else update.squeeze(0)
            return output, update
        window = math.prod(kernel)
        capturing = not is_compiling() and is_capturing(input.device)
        if any_true([type(size) is not int for size in spatial]) or capturing:
            # Symbolic sizes (dynamic export, tracing) or CUDA graph capture: count with
            # a convolution, which needs no host values.
            valid = torch.ones((1, 1, *spatial), dtype=exact, device=input.device)
    else:
        width = channels if multi_channel else 1
        if mask.shape[1] != 1 and mask.shape[1] != width:
            message = (
                f"mask must have 1 or {channels} channels"
                if multi_channel
                else "mask must have 1 channel; use multi_channel=True for per-channel masks"
            )
            raise ValueError(message)
        # One cast serves both uses when the input's real dtype is the counting dtype.
        cast = mask.to(real)
        valid = (cast if real == exact else mask.to(exact)).detach()
        valid = valid.expand(mask.shape[0], width, *spatial)
        if multi_channel:
            # Each output channel counts its own group's channels, as a grouped
            # all-ones convolution would; summing them first keeps the count depthwise.
            shape = (mask.shape[0], groups, per_group)
            if rank == 62:
                # A group axis would pass PyTorch's 64 dimensions: sum over flat space.
                valid = valid.reshape(*shape, -1).sum(2).reshape(*shape[:2], *spatial)
            else:
                valid = valid.reshape(*shape, *spatial).sum(2)
        window = math.prod(kernel) * (per_group if multi_channel else 1)
    # conv_nd's checks, once: the data and count convolutions share the checked geometry.
    _check_options(padding_mode, workspace_bytes, backend)
    _check_arguments(input, weight, steps, pads, spacings, groups)
    data = input if mask is None else input * cast
    output = _execute(
        data, weight, None, steps, pairs, spacings, groups, padding_mode, workspace_bytes, backend
    )
    produced = output.dtype
    level = _REAL.get(produced, produced)
    # Only float16's range is small enough for a window (many channels per group) to
    # exceed; the ratio would overflow, so rescale in float32 and round once at the end.
    # Checking the dtype first also keeps traced windows, which are integer tensors, from
    # meeting a float beyond int64's range.
    widened = level == torch.float16 and window > torch.finfo(level).max
    if widened:
        level = torch.float32
        output = output.to(torch.complex64 if produced.is_complex else level)
    ratio_dtype = output.dtype if output.dtype.is_complex else level
    cached = False
    if valid is None:
        geometry = spatial, kernel, steps, spacings, pairs, exact, ratio_dtype, level
        device = input.device
        if is_compiling() or not plain_tensor(input) or device.type not in {"cpu", "cuda"}:
            # Compilers, functorch transforms and fake or mode-dispatched tensors build
            # their own counts, so nothing they create outlives them; other devices'
            # streams are not modeled, so nothing is kept for them either.
            ratio, update, full = _mask_free_scale(*geometry, device)
        else:
            stream = current_stream_id(device)
            if math.prod(output.shape[2:]) <= _SMALL_SCALE:
                ratio, update, full = _cached_small_scale(*geometry, device, stream=stream)
                cached = True
            else:
                ratio, update, full = _mask_free_scale(*geometry, device, cache=True, stream=stream)
    else:
        # Per-group counts for multi-channel masks, one shared count otherwise.
        channels = groups if multi_channel and mask is not None else 1
        counts = _count(
            valid, channels, kernel, steps, pairs, spacings, padding_mode, workspace_bytes, backend
        )
        ratio, update = _scale(counts, window, ratio_dtype, level)
    batch, out_channels = output.shape[0], output.shape[1]
    grouped = update.shape[1] > 1
    if grouped and rank == 62:
        # A group axis would pass PyTorch's 64 dimensions: copy the counts per channel.
        ratio, update = (t.repeat_interleave(out_channels // groups, 1) for t in (ratio, update))
        grouped = False
    if grouped:
        # Per-group counts: view the output by group instead of copying them per channel.
        output = output.reshape(batch, groups, out_channels // groups, *output.shape[2:])
        ratio, update = ratio.unsqueeze(2), update.unsqueeze(2)
    if bias is None:
        output = output * ratio
    else:
        view = bias.to(output.dtype).reshape(
            (1, *output.shape[1 : 3 if grouped else 2]) + (1,) * rank
        )
        # NVIDIA's ((raw - bias) * ratio + bias) * update, with raw computed without bias.
        # Old ONNX exporters, which trace, have no addcmul; the unfused form exports anywhere.
        if torch.jit.is_tracing() and is_onnx_exporting():
            output = view + output * ratio
        else:
            output = torch.addcmul(view, output, ratio)
        if not full:
            output = output * update
    if grouped:
        output = output.reshape(batch, out_channels, *output.shape[3:])
    if widened:
        output = output.to(produced)
        update = update.to(_REAL.get(produced, produced))
    # Expanding with -1 keeps spatial sizes symbolic for exporters that trace them.
    keep = (-1,) * rank
    if not need_update:
        update = None
    elif cached:
        # Callers may write the update mask; never hand out the cached tensor.
        update = update.clone().expand(batch, update_channels, *keep)
    elif grouped:
        update = update.expand(batch, groups, out_channels // groups, *keep).flatten(1, 2)
    else:
        update = update.expand(batch, update_channels, *keep)
    if unbatched:
        return output.squeeze(0), None if update is None else update.squeeze(0)
    return output, update
