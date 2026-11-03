"""A small downstream application checked by mypy against the shipped API."""

import torch
from torch import Tensor, nn

from torch_convnd import (
    Conv3d,
    Conv4d,
    ConvNd,
    PartialConv4d,
    PartialConvNd,
    PartialConvOutput,
    PartialSequential,
    __version__,
    conv_nd,
    get_runtime_info,
    partial_conv_nd,
)

four = Conv4d(2, 4, (3, 3, 3, 3), padding="same", backend="torch")
five = ConvNd(2, 4, 3, dimensions=5, device=torch.device("cpu"))
three = Conv3d(2, 4, 3, 1, 1, 1, 1, False, "zeros", "cpu", torch.float32)
bordered = Conv3d(2, 4, 3, padding=1, padding_mode="partial")
output: Tensor = four.forward(torch.randn(1, 2, 4, 4, 4, 4))
functional: Tensor = conv_nd(torch.randn(1, 2, 4, 4, 4, 4), four.weight, four.bias, padding=1)
partial = PartialConv4d(2, 4, 3, padding=1, multi_channel=True)
volume, holes = torch.randn(1, 2, 4, 4, 4, 4), torch.ones(1, 2, 4, 4, 4, 4)
with_mask: PartialConvOutput = partial.forward(volume, holes, return_mask=True)
only_output: Tensor = partial.forward(volume, holes, return_mask=False)
either: Tensor | PartialConvOutput = partial.forward(volume, mask_in=holes)
next_mask: Tensor = with_mask.mask
generic_partial = PartialConvNd(2, 4, 3, dimensions=5, backend="torch")
result: PartialConvOutput = partial_conv_nd(
    volume, four.weight, padding=1, mask=torch.ones(1, 1, 4, 4, 4, 4)
)
values, update = result
network = PartialSequential(partial, nn.ReLU(), PartialConv4d(4, 1, 3, padding=1))
chained: PartialConvOutput = network.forward(volume, holes)
version: str = __version__
information: dict[str, object] = get_runtime_info()
