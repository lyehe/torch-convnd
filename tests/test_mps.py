"""Apple MPS execution against float64 CPU references, where the machine exposes MPS."""

import pytest
import torch
from test_version_compatibility import partial_reference, reference

from torch_convnd import PartialConvNd, conv_nd, partial_conv_nd

MPS = getattr(torch.backends, "mps", None)
pytestmark = [
    pytest.mark.portable,
    pytest.mark.skipif(not (MPS and MPS.is_available()), reason="MPS is unavailable"),
]
TOLERANCE = {"atol": 2e-4, "rtol": 2e-4}


def on_mps(*values):
    """float32 MPS leaves and float64 CPU leaves holding the same values."""
    low = [value.float().to("mps").requires_grad_() for value in values]
    high = [value.double().requires_grad_() for value in values]
    return low, high


def assert_matches(actual, expected, leaves, references):
    torch.testing.assert_close(actual.cpu().double(), expected, **TOLERANCE)
    cotangent = torch.randn(expected.shape, dtype=torch.float64)
    first = torch.autograd.grad(actual, leaves, cotangent.float().to("mps"))
    second = torch.autograd.grad(expected, references, cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value.cpu().double(), target, **TOLERANCE)


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_mps_geometry_gradients(rank, backend):
    """Vendor ranks, the MPS rank-3 decomposition and the generic higher ranks."""
    torch.manual_seed(1101 + rank)
    geometry = {
        "stride": (2,) + (1,) * (rank - 1),
        "padding": (1,) * rank,
        "dilation": (1,) + (2,) * (rank - 1),
        "groups": 2,
    }
    values = (
        torch.randn((2, 4) + (5,) * rank),
        torch.randn((6, 2) + (2,) * rank),
        torch.randn(6),
    )
    leaves, references = on_mps(*values)
    actual = conv_nd(*leaves, backend=backend, workspace_bytes=4096, **geometry)
    assert actual.device.type == "mps"
    assert_matches(actual, reference(*references, **geometry), leaves, references)


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("multi_channel", [False, True])
def test_mps_partial_convolution(rank, multi_channel):
    """Masked partial convolution keeps float32 counts on a device without float64."""
    generator = torch.Generator().manual_seed(1201 + rank)
    input = torch.randn((2, 4) + (5,) * rank, generator=generator)
    weight = torch.randn((4, 2) + (3,) * rank, generator=generator)
    bias = torch.randn(4, generator=generator)
    shape = (2, 4 if multi_channel else 1) + (5,) * rank
    mask = (torch.rand(shape, generator=generator) > 0.3).float()
    leaves, references = on_mps(input, weight, bias)
    options = {"padding": 1, "groups": 2, "multi_channel": multi_channel}
    actual, update = partial_conv_nd(*leaves, mask=mask.to("mps"), **options)
    expected, expected_update = partial_reference(
        references[0], mask.double(), *references[1:], (1,) * rank, 2, multi_channel
    )
    torch.testing.assert_close(update.cpu().double(), expected_update.expand_as(update.cpu()))
    assert_matches(actual, expected, leaves, references)
    layer = PartialConvNd(
        4, 4, 3, padding=1, groups=2, dimensions=rank, multi_channel=multi_channel
    ).to("mps")
    layer.load_state_dict({"weight": weight, "bias": bias})
    with torch.no_grad():
        output = layer(input.to("mps"), mask.to("mps"))
    torch.testing.assert_close(output.cpu().double(), expected.detach(), **TOLERANCE)
