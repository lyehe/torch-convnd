"""Partial convolution on the compiled native backend matches the portable backend."""

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from torch_convnd import partial_conv_nd

_CUDA_AVAILABLE = CUDA_HOME is not None and torch.cuda.is_available()
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA GPU/toolkit is unavailable"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("multi_channel", [False, True])
def test_native_partial_matches_portable_with_gradients(device, rank, multi_channel):
    generator = torch.Generator().manual_seed(rank)
    options = {"dtype": torch.float64, "device": device}
    input = torch.randn((2, 4) + (5,) * rank, generator=generator).to(**options)
    weight = torch.randn((4, 2) + (3,) * rank, generator=generator).to(**options)
    bias = torch.randn(4, generator=generator).to(**options)
    channels = 4 if multi_channel else 1
    mask = (torch.rand((2, channels) + (5,) * rank, generator=generator) > 0.3).to(**options)
    results = []
    for backend in ("native", "torch"):
        leaves = [value.detach().requires_grad_() for value in (input, weight, bias)]
        output, update = partial_conv_nd(
            *leaves,
            stride=(1,) * (rank - 1) + (2,),
            padding=1,
            groups=2,
            mask=mask,
            multi_channel=multi_channel,
            backend=backend,
        )
        gradients = torch.autograd.grad(output.square().sum(), leaves)
        results.append((output, update, *gradients))
    for native, portable in zip(*results, strict=True):
        torch.testing.assert_close(native, portable, rtol=1e-10, atol=1e-10)
