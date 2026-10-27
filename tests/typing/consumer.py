"""A small downstream application checked by mypy against the shipped API."""

import torch
from torch import Tensor

from torch_convnd import Conv3d, Conv4d, ConvNd, __version__, conv_nd, get_runtime_info

four = Conv4d(2, 4, (3, 3, 3, 3), padding="same", backend="torch")
five = ConvNd(2, 4, 3, dimensions=5, device=torch.device("cpu"))
three = Conv3d(2, 4, 3, 1, 1, 1, 1, False, "zeros", "cpu", torch.float32)
output: Tensor = four.forward(torch.randn(1, 2, 4, 4, 4, 4))
functional: Tensor = conv_nd(torch.randn(1, 2, 4, 4, 4, 4), four.weight, four.bias, padding=1)
version: str = __version__
information: dict[str, object] = get_runtime_info()
