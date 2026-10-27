"""Pure Tensor TorchScript fallback for releases without torch.library."""

from typing import Optional

import torch
from torch import Tensor

from ._script import ScriptableConvNd


def scripted_convolution(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],  # noqa: UP045
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
    workspace_bytes: int,
) -> Tensor:
    """Express tiled indexing/matmul in the older TorchScript syntax subset."""
    cast_half = input.device.type == "cpu" and input.dtype == torch.float16
    if cast_half:
        input, weight = input.float(), weight.float()
        if bias is not None:
            bias = bias.float()
    rank = input.dim() - 2
    batch, channels = input.size(0), input.size(1)
    spatial, kernel, output_shape = [0] * rank, [0] * rank, [0] * rank
    volume, kernel_volume, output_volume = 1, 1, 1
    for axis in range(rank):
        spatial[axis], kernel[axis] = input.size(axis + 2), weight.size(axis + 2)
        extent = spatial[axis] + 2 * padding[axis] - dilation[axis] * (kernel[axis] - 1) - 1
        output_shape[axis] = extent // stride[axis] + 1
        if output_shape[axis] <= 0:
            raise RuntimeError("kernel exceeds padded input")
        volume *= spatial[axis]
        kernel_volume *= kernel[axis]
        output_volume *= output_shape[axis]
    rows = channels * kernel_volume
    tile = min(output_volume, workspace_bytes // (max(4, input.element_size()) * rows))
    if tile <= 0:
        raise RuntimeError("workspace_bytes is too small for one output position")
    row = torch.arange(rows, device=input.device).unsqueeze(1)
    channel_offset = torch.div(row, kernel_volume, rounding_mode="floor") * volume
    flat_input = input.reshape(batch, channels * volume)
    flat_weight = weight.reshape(groups, weight.size(0) // groups, rows // groups)
    chunks = torch.jit.annotate(list[Tensor], [])
    zero = input.new_zeros(())
    for start in range(0, output_volume, tile):
        length = min(tile, output_volume - start)
        position = torch.arange(start, start + length, device=input.device).unsqueeze(0)
        tap = row.remainder(kernel_volume)
        offset = channel_offset.expand(rows, length)
        valid = torch.ones((rows, length), dtype=torch.bool, device=input.device)
        multiplier = 1
        for reverse_axis in range(rank):
            axis = rank - 1 - reverse_axis
            coordinate = (
                position.remainder(output_shape[axis]) * stride[axis]
                - padding[axis]
                + tap.remainder(kernel[axis]) * dilation[axis]
            )
            valid = valid & (coordinate >= 0) & (coordinate < spatial[axis])
            offset = offset + coordinate * multiplier
            multiplier *= spatial[axis]
            position = torch.div(position, output_shape[axis], rounding_mode="floor")
            tap = torch.div(tap, kernel[axis], rounding_mode="floor")
        values = flat_input.index_select(1, offset.clamp(0, channels * volume - 1).reshape(-1))
        columns = torch.where(valid, values.reshape(batch, rows, length), zero)
        columns = columns.reshape(batch, groups, rows // groups, length)
        chunks.append(torch.matmul(flat_weight, columns).reshape(batch, weight.size(0), length))
    output = torch.cat(chunks, dim=-1).reshape([batch, weight.size(0)] + output_shape)
    if bias is not None:
        output = output + bias.to(output.dtype).reshape([1, -1] + [1] * rank)
    return output.half() if cast_half else output


class LegacyScriptableConvNd(ScriptableConvNd):
    """Keep the public module/parameters while avoiding unavailable custom operators."""

    def _real(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # noqa: UP045
        return scripted_convolution(
            input,
            weight,
            bias,
            self.stride,
            self.native_padding,
            self.dilation,
            self.groups,
            self.workspace_bytes,
        )
