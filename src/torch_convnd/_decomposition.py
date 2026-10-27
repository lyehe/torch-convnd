"""Differentiable portable convolution implementations."""

import math

import torch
from torch.nn import functional as F


def conv4d_taps(input, weight, bias, stride, padding, dilation, groups):
    """Compute unrestricted 4D weights with batched vendor Conv3d kernels.

    This is an exact decomposition over the first spatial axis, not a separable
    kernel. Low-precision partial results accumulate in FP32 before the final
    cast. Only ordinary-sized metadata reaches this path.
    """
    extent = (input.shape[2] + 2 * padding[0] - dilation[0] * (weight.shape[2] - 1) - 1) // stride[
        0
    ] + 1
    if extent <= 0:
        raise RuntimeError("kernel exceeds padded input")
    padded = F.pad(input, (0, 0, 0, 0, 0, 0, padding[0], padding[0])) if padding[0] else input
    accumulator = None
    output_dtype = input.dtype
    for tap in range(weight.shape[2]):
        start = tap * dilation[0]
        frames = padded[:, :, start : start + extent * stride[0] : stride[0]]
        frames = frames.permute(0, 2, 1, 3, 4, 5).reshape(
            input.shape[0] * extent, input.shape[1], *input.shape[3:]
        )
        result = F.conv3d(
            frames, weight[:, :, tap], None, stride[1:], padding[1:], dilation[1:], groups
        )
        result = result.reshape(input.shape[0], extent, weight.shape[0], *result.shape[2:])
        result = result.permute(0, 2, 1, 3, 4, 5)
        output_dtype = result.dtype
        partial = result.float() if result.dtype in {torch.float16, torch.bfloat16} else result
        accumulator = partial if accumulator is None else accumulator + partial
    if bias is not None:
        accumulator = accumulator + bias.reshape(1, -1, 1, 1, 1, 1).to(accumulator.dtype)
    return accumulator.to(output_dtype)


def composite_conv(input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
    """Use tiled differentiable indexing for torch.func and non-native devices.

    All temporary tensors have low rank, even for a rank-62 convolution. Patch
    values are tiled; integer indexing tensors and transform intermediates are
    additional workspace, as with other PyTorch decompositions.
    """
    batch, channels = input.shape[:2]
    spatial = input.shape[2:]
    kernel = weight.shape[2:]
    output_shape = tuple(
        (size + 2 * pad - dil * (tap - 1) - 1) // step + 1
        for size, pad, dil, tap, step in zip(
            spatial, padding, dilation, kernel, stride, strict=True
        )
    )
    if any(size <= 0 for size in output_shape):
        raise RuntimeError("kernel exceeds padded input")
    volume, kernel_volume = math.prod(spatial), math.prod(kernel)
    output_volume = math.prod(output_shape)
    rows = channels * kernel_volume
    bytes_per_position = max(4, input.element_size()) * rows
    if workspace_bytes < bytes_per_position:
        raise RuntimeError("workspace_bytes is too small for one output position")
    tile = min(output_volume, workspace_bytes // bytes_per_position)
    row = torch.arange(rows, device=input.device).unsqueeze(1)
    channel_offset = row.div(kernel_volume, rounding_mode="floor") * volume
    flat_input = input.reshape(batch, channels * volume)
    flat_weight = weight.reshape(groups, weight.shape[0] // groups, rows // groups)
    chunks = []
    zero = input.new_zeros(())
    for start in range(0, output_volume, tile):
        length = min(tile, output_volume - start)
        position = torch.arange(start, start + length, device=input.device).unsqueeze(0)
        tap = row.remainder(kernel_volume)
        offset = channel_offset.expand(rows, length)
        valid = torch.ones((rows, length), device=input.device, dtype=torch.bool)
        multiplier = 1
        for axis in range(len(spatial) - 1, -1, -1):
            coordinate = (
                position.remainder(output_shape[axis]) * stride[axis]
                - padding[axis]
                + tap.remainder(kernel[axis]) * dilation[axis]
            )
            valid = valid & (coordinate >= 0) & (coordinate < spatial[axis])
            offset = offset + coordinate * multiplier
            multiplier *= spatial[axis]
            position = position.div(output_shape[axis], rounding_mode="floor")
            tap = tap.div(kernel[axis], rounding_mode="floor")
        values = flat_input.index_select(1, offset.clamp(0, channels * volume - 1).reshape(-1))
        columns = torch.where(valid, values.reshape(batch, rows, length), zero)
        columns = columns.reshape(batch, groups, rows // groups, length)
        chunks.append(torch.matmul(flat_weight, columns).reshape(batch, weight.shape[0], length))
    output = torch.cat(chunks, dim=-1).reshape(batch, weight.shape[0], *output_shape)
    if bias is not None:
        output = output + bias.reshape(1, -1, *((1,) * len(spatial)))
    return output
