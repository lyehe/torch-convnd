"""Compiler, tracing and export checks against both stock spatial ranks."""

import importlib.util

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from torch_convnd import Conv2d, Conv3d, Conv4d
from torch_convnd.convnd import _build_environment

CUDA = CUDA_HOME is not None and torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="no CUDA"))]


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_inductor_cpu_forward_backward(rank, backend):
    """The default CPU compiler preserves values and gradients, with fullgraph enabled."""
    torch._dynamo.reset()
    factory = Conv2d if rank == 2 else Conv3d
    layer = factory(2, 3, 2, padding="same", dtype=torch.float64, backend=backend)
    reference = getattr(torch.nn, f"Conv{rank}d")(2, 3, 2, padding="same", dtype=torch.float64)
    reference.load_state_dict(layer.state_dict())
    input = torch.randn((1, 2) + (4,) * rank, dtype=torch.float64, requires_grad=True)
    with _build_environment(False):
        compiled = torch.compile(layer, fullgraph=True)
        actual = compiled(input)
        expected = reference(input)
        torch.testing.assert_close(actual, expected)
        first = torch.autograd.grad(actual.square().sum(), (input, *layer.parameters()))
        second = torch.autograd.grad(expected.square().sum(), (input, *reference.parameters()))
    for observed, target in zip(first, second, strict=True):
        torch.testing.assert_close(observed, target)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_trace_save_load_and_different_spatial_size(rank, device, tmp_path):
    """Traced modules serialize and retain dynamic input spatial sizes."""
    factory = Conv2d if rank == 2 else Conv3d
    layer = factory(2, 3, 2, padding="same", padding_mode="reflect", device=device)
    input = torch.randn((1, 2) + (4,) * rank, device=device)
    traced = torch.jit.trace(layer, input)
    path = tmp_path / f"conv{rank}d.pt"
    traced.save(str(path))
    restored = torch.jit.load(str(path), map_location=device)
    larger = torch.randn((2, 2) + (5,) * rank, device=device)
    torch.testing.assert_close(restored(larger), layer(larger))


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_export_dynamic_batch_and_spatial_sizes(rank, device, backend, tmp_path):
    """Exported graphs support dynamic batch/spatial shapes and round-trip serialization."""
    torch._dynamo.reset()
    factory = Conv2d if rank == 2 else Conv3d
    layer = factory(2, 3, 2, padding="same", device=device, backend=backend)
    input = torch.randn((2, 2) + (4,) * rank, device=device)
    dimensions = {0: torch.export.Dim("batch", min=1, max=4)}
    dimensions.update(
        {axis: torch.export.Dim(f"axis{axis}", min=3, max=8) for axis in range(2, rank + 2)}
    )
    program = torch.export.export(layer, (input,), dynamic_shapes=({**dimensions},))
    path = tmp_path / f"conv{rank}d.pt2"
    torch.export.save(program, path)
    restored = torch.export.load(path).module()
    larger = torch.randn((3, 2) + (6,) * rank, device=device)
    torch.testing.assert_close(restored(larger), layer(larger))


@pytest.mark.skipif(
    not CUDA or importlib.util.find_spec("triton") is None,
    reason="CUDA Inductor requires a matching Triton installation",
)
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_inductor_cuda_forward_backward(rank, amp, backend):
    """The default CUDA backend agrees with the stock reference, including AMP."""
    torch._dynamo.reset()
    factory = Conv2d if rank == 2 else Conv3d
    layer = factory(2, 3, 2, padding="same", device="cuda", backend=backend)
    reference = getattr(torch.nn, f"Conv{rank}d")(2, 3, 2, padding="same", device="cuda")
    reference.load_state_dict(layer.state_dict())
    input = torch.randn((1, 2) + (4,) * rank, device="cuda", requires_grad=True)
    with _build_environment(True):
        compiled = torch.compile(layer, fullgraph=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            actual, expected = compiled(input), reference(input)
        tolerance = 0.005 if amp else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
        first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
        second = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for observed, target in zip(first, second, strict=True):
        torch.testing.assert_close(observed, target, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", [False, True])
def test_inductor_four_dimensional_training(backend, device, amp):
    """The 4D paths compile without graph breaks and retain parameter/input gradients."""
    if device == "cuda" and importlib.util.find_spec("triton") is None:
        pytest.skip("CUDA Inductor requires a matching Triton installation")
    torch._dynamo.reset()
    layer = Conv4d(2, 3, 2, padding=1, device=device, backend=backend)
    reference = Conv4d(2, 3, 2, padding=1, device=device, backend="torch")
    reference.load_state_dict(layer.state_dict())
    input = torch.randn((1, 2, 3, 3, 3, 3), device=device, requires_grad=True)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with _build_environment(device == "cuda"):
        compiled = torch.compile(layer, fullgraph=True)
        with torch.autocast(device_type=device, dtype=dtype, enabled=amp):
            actual, expected = compiled(input), reference(input)
        tolerance = 0.03 if amp and device == "cpu" else 0.005 if amp else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
        first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
        second = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for observed, target in zip(first, second, strict=True):
        torch.testing.assert_close(observed, target, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
@pytest.mark.parametrize("device", DEVICES)
def test_export_four_dimensional_dynamic_shapes(backend, device, tmp_path):
    """Vendor-slice and native ND graphs both retain dynamic batch and all four axes."""
    torch._dynamo.reset()
    layer = Conv4d(2, 3, 2, device=device, backend=backend)
    input = torch.randn((2, 2, 4, 4, 4, 4), device=device)
    dimensions = {0: torch.export.Dim("batch", min=1, max=4)}
    dimensions.update({axis: torch.export.Dim(f"axis{axis}", min=3, max=7) for axis in range(2, 6)})
    program = torch.export.export(layer, (input,), dynamic_shapes=(dimensions,))
    path = tmp_path / "conv4d.pt2"
    torch.export.save(program, path)
    restored = torch.export.load(path).module()
    changed = torch.randn((3, 2, 5, 5, 5, 5), device=device)
    torch.testing.assert_close(restored(changed), layer(changed), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_script_save_load_forward_backward(rank, device, mode, tmp_path):
    """Scripted modules retain native execution, all padding modes and parameter gradients."""
    factory = Conv2d if rank == 2 else Conv3d
    layer = factory(2, 3, 2, padding="same", padding_mode=mode, device=device, dtype=torch.float64)
    reference = getattr(torch.nn, f"Conv{rank}d")(
        2, 3, 2, padding="same", padding_mode=mode, device=device, dtype=torch.float64
    )
    reference.load_state_dict(layer.state_dict())
    scripted = torch.jit.script(layer)
    path = tmp_path / "scripted.pt"
    scripted.save(str(path))
    restored = torch.jit.load(str(path), map_location=device)
    input = torch.randn((2,) + (4,) * rank, device=device, dtype=torch.float64, requires_grad=True)
    actual, expected = restored(input), reference(input)
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual.square().sum(), (input, *restored.parameters()))
    second = torch.autograd.grad(expected.square().sum(), (input, *reference.parameters()))
    for observed, target in zip(first, second, strict=True):
        torch.testing.assert_close(observed, target)
    for name in ["in_channels", "out_channels", "kernel_size", "padding_mode", "groups"]:
        assert getattr(scripted, name) == getattr(layer, name)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_script_dtype_parity(rank, device, dtype):
    """The script adapter keeps mixed and complex precision through serialization-ready code."""
    factory = Conv2d if rank == 2 else Conv3d
    layer = factory(2, 3, 2, padding="same", device=device, dtype=dtype)
    reference = getattr(torch.nn, f"Conv{rank}d")(
        2, 3, 2, padding="same", device=device, dtype=dtype
    )
    reference.load_state_dict(layer.state_dict())
    input = torch.randn((1, 2) + (3,) * rank, device=device, dtype=dtype, requires_grad=True)
    actual, expected = torch.jit.script(layer)(input), reference(input)
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for observed, target in zip(first, second, strict=True):
        torch.testing.assert_close(observed, target, atol=tolerance, rtol=tolerance)
