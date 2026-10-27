"""Native training contract on releases that provide the required custom-op APIs."""

import pytest
import torch
from test_version_compatibility import reference

from torch_convnd import ConvNd, conv_nd
from torch_convnd._compat import HAS_NATIVE_AUTOGRAD

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(not HAS_NATIVE_AUTOGRAD, reason="PyTorch lacks native custom-op APIs"),
]


@pytest.mark.parametrize("rank", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16, torch.complex64])
def test_native_version_training(rank, dtype):
    torch.manual_seed(507)
    layer = ConvNd(2, 4, 2, dimensions=rank, groups=2, dtype=dtype, backend="native")
    input = torch.randn((1, 2) + (3,) * rank).to(dtype).requires_grad_()
    high = torch.complex128 if dtype.is_complex else torch.float64
    converted = tuple(
        t.detach().to(high).requires_grad_() for t in (input, layer.weight, layer.bias)
    )
    actual = layer(input)
    expected = reference(*converted, groups=2)
    tolerance = 0.05 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected.to(dtype), atol=tolerance, rtol=tolerance)
    cotangent = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, converted, cotangent.to(high))
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target.to(dtype), atol=tolerance, rtol=tolerance)


def test_native_version_second_derivatives(deterministic_algorithms):
    input = torch.randn(1, 1, 2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1, 1, 1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda x, w: conv_nd(x, w, backend="native"), (input, weight), fast_mode=True
    )
