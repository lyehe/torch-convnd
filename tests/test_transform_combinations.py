"""Regression coverage for feature intersections that isolated feature tests missed."""

import io

import pytest
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import CUDA_HOME

from torch_convnd import Conv2d, Conv3d, Conv4d

CUDA = CUDA_HOME is not None and torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="no CUDA"))]
FACTORIES = {2: Conv2d, 3: Conv3d, 4: Conv4d}


def serialized(layer, example, mode):
    """Test reloaded graphs, not just the in-memory script adapter."""
    compiled = torch.jit.script(layer) if mode == "script" else torch.jit.trace(layer, example)
    buffer = io.BytesIO()
    torch.jit.save(compiled, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer, map_location=example.device)


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
@pytest.mark.parametrize("mode", ["script", "trace"])
@pytest.mark.parametrize("transform", ["jvp", "dual", "vmap", "grad"])
def test_serialized_transform_parity(rank, device, backend, mode, transform):
    """Script/trace plus forward/reverse transforms retain the correct derivatives."""
    from test_convnd import reference_nd

    layer = FACTORIES[rank](2, 3, 2, device=device, dtype=torch.float64, backend=backend)
    input = torch.randn((1, 2) + (3,) * rank, device=device, dtype=torch.float64)
    compiled = serialized(layer, input, mode)

    def reference(value):
        if rank <= 3:
            return getattr(F, f"conv{rank}d")(value, layer.weight, layer.bias)
        return reference_nd(value, layer.weight, layer.bias)

    if transform == "jvp":
        tangent = torch.randn_like(input)
        actual = torch.func.jvp(compiled, (input,), (tangent,))
        expected = torch.func.jvp(reference, (input,), (tangent,))
        for observed, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=1e-9, rtol=1e-9)
    elif transform == "dual":
        tangent = torch.randn_like(input)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(input, tangent)
            actual = torch.autograd.forward_ad.unpack_dual(compiled(dual))
            expected = torch.autograd.forward_ad.unpack_dual(reference(dual))
            assert actual.tangent is not None
            torch.testing.assert_close(actual.primal, expected.primal, atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(actual.tangent, expected.tangent, atol=1e-9, rtol=1e-9)
    elif transform == "vmap":
        values = input.unsqueeze(0).expand(3, *input.shape)
        torch.testing.assert_close(
            torch.vmap(compiled)(values), torch.vmap(reference)(values), atol=1e-9, rtol=1e-9
        )
    else:
        actual = torch.func.grad(lambda x: compiled(x).sin().square().sum())(input)
        expected = torch.func.grad(lambda x: reference(x).sin().square().sum())(input)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_scripted_complex_forward_ad(rank, device, backend):
    """Complex tangents survive the script boundary without an opaque-AD fallback."""
    layer = FACTORIES[rank](2, 3, 2, device=device, dtype=torch.complex128, backend=backend)
    input = torch.randn((1, 2) + (3,) * rank, device=device, dtype=torch.complex128)
    tangent = torch.randn_like(input)
    compiled = torch.jit.script(layer)
    actual = torch.func.jvp(compiled, (input,), (tangent,))

    def reference(x):
        return getattr(F, f"conv{rank}d")(x, layer.weight, layer.bias)

    expected = torch.func.jvp(reference, (input,), (tangent,))
    for observed, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, target, atol=2e-9, rtol=2e-9)
