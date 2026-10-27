"""Scriptable module adapter using the same native operators and parameters."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScriptableConvNd(nn.Module):
    """Keep TorchScript's restricted syntax separate from eager transform dispatch."""

    __constants__ = [
        "dimensions",
        "stride",
        "dilation",
        "groups",
        "workspace_bytes",
        "mode",
        "left",
        "right",
        "native_padding",
        "explicit_padding",
        "reverse_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "padding",
        "padding_mode",
        "transposed",
        "output_padding",
        "backend",
    ]

    def __init__(self, layer):
        super().__init__()
        self.weight = layer.weight
        if layer.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = layer.bias
        self.dimensions = layer.dimensions
        self.in_channels, self.out_channels = layer.in_channels, layer.out_channels
        self.kernel_size = layer.kernel_size
        self.padding = layer.padding
        self.padding_mode = layer.padding_mode
        self.transposed = layer.transposed
        self.output_padding = layer.output_padding
        self.training = layer.training
        self.backend = layer.backend
        self.stride = list(layer.stride)
        self.dilation = list(layer.dilation)
        self.groups = layer.groups
        self.workspace_bytes = layer.workspace_bytes
        self.mode = layer.padding_mode
        self.reverse_padding = list(layer._reversed_padding_repeated_twice)
        pairs = list(
            reversed(list(zip(self.reverse_padding[::2], self.reverse_padding[1::2], strict=True)))
        )
        self.left = [left for left, _ in pairs]
        self.right = [right for _, right in pairs]
        self.explicit_padding = self.mode != "zeros" or self.left != self.right
        self.native_padding = [0] * self.dimensions if self.explicit_padding else self.left

    def _pad(self, input: Tensor) -> Tensor:
        if not self.explicit_padding:
            return input
        if self.mode == "zeros":
            return F.pad(input, self.reverse_padding)
        for axis in range(self.dimensions):
            left, right = self.left[axis], self.right[axis]
            if left != 0 or right != 0:
                size = input.size(axis + 2)
                if self.mode == "reflect" and (left >= size or right >= size):
                    raise RuntimeError("Reflection padding must be less than the input dimension")
                if self.mode == "circular" and (left > size or right > size):
                    raise RuntimeError("Padding value causes wrapping around more than once")
                coordinates = torch.arange(-left, size + right, device=input.device)
                if self.mode == "replicate":
                    indices = coordinates.clamp(0, size - 1)
                elif self.mode == "circular":
                    indices = coordinates.remainder(size)
                else:
                    indices = torch.where(coordinates < 0, -coordinates, coordinates)
                    indices = torch.where(indices >= size, 2 * (size - 1) - indices, indices)
                input = input.index_select(axis + 2, indices)
        return input

    def _real(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # noqa: UP045
        return torch.ops.torch_convnd.compatible_conv_nd(
            input,
            weight,
            bias,
            self.stride,
            self.native_padding,
            self.dilation,
            self.groups,
            self.workspace_bytes,
            self.backend,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Run native convolution with batched/unbatched and complex support."""
        unbatched = input.dim() == self.dimensions + 1
        if unbatched:
            input = input.unsqueeze(0)
        elif input.dim() != self.dimensions + 2:
            raise ValueError("Incorrect input rank")
        input = self._pad(input)
        output = self._real(input, self.weight, self.bias)
        return output.squeeze(0) if unbatched else output
