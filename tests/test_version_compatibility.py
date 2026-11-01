"""Core adoption contract exercised independently on old and current PyTorch."""

import io
import itertools

import pytest
import torch

from torch_convnd import (
    Conv4d,
    ConvNd,
    PartialConvNd,
    PartialSequential,
    conv_nd,
    export_onnx,
    get_runtime_info,
    partial_conv_nd,
)
from torch_convnd._compat import HAS_NATIVE_AUTOGRAD

pytestmark = pytest.mark.portable


@pytest.fixture(autouse=True)
def portable_auto(monkeypatch):
    from torch_convnd import _dispatch

    monkeypatch.setattr(_dispatch, "native_unavailable_reason", lambda _: "portable-only check")


def reference(input, weight, bias, stride=None, padding=None, dilation=None, groups=1):
    """Independent tap contraction, including geometry and blocked channel groups."""
    rank = input.ndim - 2
    stride, padding, dilation = (
        stride or (1,) * rank,
        padding or (0,) * rank,
        dilation or (1,) * rank,
    )
    padded = torch.nn.functional.pad(input, tuple(v for p in reversed(padding) for v in (p, p)))
    shape = tuple(
        (n + 2 * p - d * (k - 1) - 1) // s + 1
        for n, p, d, k, s in zip(
            input.shape[2:], padding, dilation, weight.shape[2:], stride, strict=True
        )
    )
    result = None
    for tap in itertools.product(*(range(k) for k in weight.shape[2:])):
        slices = tuple(
            slice(k * d, k * d + n * s, s)
            for k, d, n, s in zip(tap, dilation, shape, stride, strict=True)
        )
        patch = padded[(slice(None), slice(None), *slices)].reshape(
            input.shape[0], groups, input.shape[1] // groups, -1
        )
        coefficient = weight[(slice(None), slice(None), *tap)].reshape(
            groups, weight.shape[0] // groups, -1
        )
        term = torch.einsum("goc,bgcp->bgop", coefficient, patch)
        result = term if result is None else result + term
    result = result.reshape(input.shape[0], weight.shape[0], *shape)
    return result if bias is None else result + bias.reshape(1, -1, *((1,) * rank))


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_version_geometry_gradients(rank, backend):
    torch.manual_seed(505)
    geometry = {
        "stride": (2,) + (1,) * (rank - 1),
        "padding": (1,) * rank,
        "dilation": (1,) + (2,) * (rank - 1),
        "groups": 2,
    }
    input = torch.randn((1, 2) + (3,) * rank, dtype=torch.float64, requires_grad=True)
    weight = torch.randn((4, 1) + (2,) * rank, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(4, dtype=torch.float64, requires_grad=True)
    actual = conv_nd(input, weight, bias, backend=backend, workspace_bytes=4096, **geometry)
    expected = reference(input, weight, bias, **geometry)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    cotangent = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (input, weight, bias), cotangent)
    second = torch.autograd.grad(expected, (input, weight, bias), cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target, atol=1e-9, rtol=1e-9)


def partial_reference(input, mask, weight, bias, padding, groups, multi_channel):
    """NVIDIA's partial convolution built from the independent reference above."""
    rank, per_group = input.ndim - 2, weight.shape[1]
    width = input.shape[1] if multi_channel else 1
    if mask is None:
        mask = torch.ones((input.shape[0], width) + tuple(input.shape[2:]), dtype=input.dtype)
    updater = torch.ones(
        (weight.shape[0] if multi_channel else 1, per_group if multi_channel else 1)
        + tuple(weight.shape[2:]),
        dtype=input.dtype,
    )
    count = reference(mask, updater, None, padding=padding, groups=groups if multi_channel else 1)
    window = updater[0].numel()
    update = count.clamp(0, 1)
    ratio = torch.where(count > 0, window / count.clamp_min(1e-300), torch.zeros_like(count))
    raw = reference(input * mask, weight, None, padding=padding, groups=groups)
    view = bias.reshape(1, -1, *((1,) * rank))
    return (raw * ratio * update + view) * update, update


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("multi_channel", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_version_partial_convolution(rank, multi_channel, masked):
    """Values, update masks and gradients on the mask-free and masked paths."""
    generator = torch.Generator().manual_seed(707 + rank)
    input = torch.randn((2, 4) + (4,) * rank, dtype=torch.float64, generator=generator)
    weight = torch.randn((4, 2) + (3,) * rank, dtype=torch.float64, generator=generator)
    bias = torch.randn(4, dtype=torch.float64, generator=generator)
    mask = None
    if masked:
        shape = (2, 4 if multi_channel else 1) + (4,) * rank
        mask = (torch.rand(shape, generator=generator) > 0.3).double()
    leaves = [value.clone().requires_grad_() for value in (input, weight, bias)]
    actual, update = partial_conv_nd(
        *leaves, padding=1, groups=2, mask=mask, multi_channel=multi_channel, backend="torch"
    )
    expected, expected_update = partial_reference(
        *leaves[:1], mask, *leaves[1:], (1,) * rank, 2, multi_channel
    )
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(update, expected_update.expand_as(update))
    cotangent = torch.randn(actual.shape, dtype=torch.float64, generator=generator)
    first = torch.autograd.grad(actual, leaves, cotangent)
    second = torch.autograd.grad(expected, leaves, cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target, atol=1e-9, rtol=1e-9)
    layer = PartialConvNd(
        4,
        4,
        3,
        padding=1,
        groups=2,
        dimensions=rank,
        multi_channel=multi_channel,
        backend="torch",
        dtype=torch.float64,
    )
    layer.load_state_dict({"weight": weight, "bias": bias})
    torch.testing.assert_close(layer(input, mask), expected.detach(), atol=1e-9, rtol=1e-9)


def test_version_partial_padding_and_chaining():
    """padding_mode='partial' and PartialSequential follow the same reference."""
    generator = torch.Generator().manual_seed(808)
    input = torch.randn((2, 4) + (4,) * 4, dtype=torch.float64, generator=generator)
    weight = torch.randn((4, 2) + (3,) * 4, dtype=torch.float64, generator=generator)
    bias = torch.randn(4, dtype=torch.float64, generator=generator)
    padding = (1,) * 4
    expected, _ = partial_reference(input, None, weight, bias, padding, 2, False)
    options = {"padding": 1, "groups": 2, "dimensions": 4, "backend": "torch"}
    bordered = ConvNd(4, 4, 3, padding_mode="partial", dtype=torch.float64, **options)
    bordered.load_state_dict({"weight": weight, "bias": bias})
    torch.testing.assert_close(bordered(input), expected, atol=1e-9, rtol=1e-9)
    functional = conv_nd(input, weight, bias, padding=1, groups=2, padding_mode="partial")
    torch.testing.assert_close(functional, expected, atol=1e-9, rtol=1e-9)
    layers = [PartialConvNd(4, 4, 3, dtype=torch.float64, **options) for _ in range(2)]
    for layer in layers:
        layer.load_state_dict({"weight": weight, "bias": bias})
    mask = (torch.rand((2, 1) + (4,) * 4, generator=generator) > 0.3).double()
    output, final = PartialSequential(layers[0], torch.nn.ReLU(), layers[1])(input, mask)
    hidden, update = partial_reference(input, mask, weight, bias, padding, 2, False)
    expected, expected_final = partial_reference(
        hidden.relu(), update, weight, bias, padding, 2, False
    )
    torch.testing.assert_close(output, expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(final, expected_final)


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_version_dtypes(rank, dtype):
    layer = ConvNd(2, 3, 2, dimensions=rank, dtype=dtype, backend="torch")
    input = torch.randn((1, 2) + (3,) * rank, dtype=torch.float64).to(dtype).requires_grad_()
    high = torch.complex128 if input.is_complex() else torch.float64
    converted = tuple(
        t.detach().to(high).requires_grad_() for t in (input, layer.weight, layer.bias)
    )
    actual = layer(input)
    expected = reference(*converted)
    tolerance = 0.05 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected.to(dtype), atol=tolerance, rtol=tolerance)
    cotangent = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, converted, cotangent.to(high))
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target.to(dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_version_padding(mode):
    layer = Conv4d(1, 2, 2, padding="same", padding_mode=mode, backend="torch", dtype=torch.float64)
    input = torch.randn(1, 1, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    padded = input
    for axis in range(2, 6):
        moved = padded.movedim(axis, -1)
        shape = moved.shape
        frames = moved.reshape(-1, 1, shape[-1])
        frames = torch.nn.functional.pad(
            frames, (0, 1), mode="constant" if mode == "zeros" else mode
        )
        padded = frames.reshape(*shape[:-1], shape[-1] + 1).movedim(-1, axis)
    torch.testing.assert_close(layer(input), reference(padded, layer.weight, layer.bias))


@pytest.mark.parametrize("mode", ["script", "trace"])
def test_version_serialization(mode):
    layer = Conv4d(2, 3, 2, backend="torch", dtype=torch.float64)
    input = torch.randn(1, 2, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    compiled = torch.jit.script(layer) if mode == "script" else torch.jit.trace(layer, input)
    buffer = io.BytesIO()
    torch.jit.save(compiled, buffer)
    buffer.seek(0)
    restored = torch.jit.load(buffer)
    actual = restored(input)
    expected = reference(input, layer.weight, layer.bias)
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual.square().mean(), (input, *restored.parameters()))
    second = torch.autograd.grad(expected.square().mean(), (input, *layer.parameters()))
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target)


def test_version_second_derivatives():
    input = torch.randn(1, 1, 2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1, 1, 1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda x, w: conv_nd(x, w, backend="torch"), (input, weight), fast_mode=True
    )


def test_version_forward_ad():
    layer = Conv4d(1, 2, 2, dtype=torch.float64, backend="torch")
    input = torch.randn(1, 1, 3, 3, 3, 3, dtype=torch.float64)
    tangent = torch.randn_like(input)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(input, tangent)
        actual = torch.autograd.forward_ad.unpack_dual(layer(dual))
        expected = torch.autograd.forward_ad.unpack_dual(reference(dual, layer.weight, layer.bias))
        torch.testing.assert_close(actual.primal, expected.primal)
        torch.testing.assert_close(actual.tangent, expected.tangent)


@pytest.mark.skipif(not hasattr(torch, "func"), reason="torch.func is absent in this PyTorch")
def test_version_func_jvp():
    layer = Conv4d(1, 2, 2, dtype=torch.float64, backend="torch")
    input = torch.randn(1, 1, 3, 3, 3, 3, dtype=torch.float64)
    tangent = torch.randn_like(input)
    actual = torch.func.jvp(layer, (input,), (tangent,))
    expected = torch.func.jvp(
        lambda x: reference(x, layer.weight, layer.bias), (input,), (tangent,)
    )
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target)


def test_version_native_guard_and_diagnostics():
    info = get_runtime_info()
    assert info["torch"] == str(torch.__version__)
    assert info["features"]["native_autograd"] is HAS_NATIVE_AUTOGRAD
    if not HAS_NATIVE_AUTOGRAD:
        from torch_convnd import load_native

        with pytest.raises(RuntimeError, match="Native.*unavailable"):
            load_native(with_cuda=False)


def test_version_onnx_runtime(tmp_path):
    pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")
    layer = Conv4d(2, 3, 2, backend="torch").eval()
    input = torch.randn(1, 2, 3, 3, 3, 3)
    path = tmp_path / "version.onnx"
    export_onnx(layer, (input,), path, external_data=False)
    options = runtime.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = runtime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = torch.from_numpy(session.run(None, {session.get_inputs()[0].name: input.numpy()})[0])
    torch.testing.assert_close(actual, layer(input), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("masked", [False, True], ids=["mask_free", "masked"])
def test_version_partial_onnx_runtime(masked, tmp_path):
    """Partial layers export with every argument and their update masks, on every exporter."""
    pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")
    generator = torch.Generator().manual_seed(909)
    layer = PartialConvNd(
        4,
        6,
        3,
        padding=1,
        groups=2,
        dimensions=2,
        multi_channel=True,
        return_mask=True,
        backend="torch",
    ).eval()
    input = torch.randn((2, 4, 6, 7), generator=generator)
    mask = (torch.rand((2, 4, 6, 7), generator=generator) > 0.3).float()
    arguments = (input, mask) if masked else (input,)
    path = tmp_path / "partial.onnx"
    export_onnx(layer, arguments, path, external_data=False)
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_inputs()]
    feed = {name: value.numpy() for name, value in zip(names, arguments, strict=True)}
    output, update = session.run(None, feed)
    with torch.no_grad():
        expected = layer(*arguments)
    torch.testing.assert_close(torch.from_numpy(output), expected.output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.from_numpy(update), expected.mask.contiguous())
