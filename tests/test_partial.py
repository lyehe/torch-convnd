"""Partial convolution against an independent transcription of NVIDIA's partialconv."""

import functools
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn
from torch.nn import functional as F

import torch_convnd.functional as functional_module
from torch_convnd import (
    ConvNd,
    PartialConv2d,
    PartialConv3d,
    PartialConv4d,
    PartialConvNd,
    conv_nd,
    partial_conv_nd,
)
from torch_convnd._compat import is_autocast_enabled

pytestmark = pytest.mark.portable
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")


def stock_conv(input, weight, stride, padding, dilation, groups):
    """Stock convolution for ranks 1-3; rank 4 sums stock 3D convolutions over axis 0."""
    rank = weight.ndim - 2
    if rank <= 3:
        operation = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
        return operation(input, weight, None, stride, padding, dilation, groups)
    padded = F.pad(input, (0, 0) * (rank - 1) + (padding[0], padding[0]))
    taps, step, spacing = weight.shape[2], stride[0], dilation[0]
    length = (padded.shape[2] - spacing * (taps - 1) - 1) // step + 1
    frames = [
        sum(
            stock_conv(
                padded[:, :, frame * step + tap * spacing],
                weight[:, :, tap],
                stride[1:],
                padding[1:],
                dilation[1:],
                groups,
            )
            for tap in range(taps)
        )
        for frame in range(length)
    ]
    return torch.stack(frames, 2)


def nvidia_reference(input, mask, weight, bias, stride, padding, dilation, groups, multi_channel):
    """NVIDIA's PartialConv2d/3d arithmetic, with grouped counts, at any rank.

    NVIDIA divides by ``count + 1e-8``; the exact quotient is used so comparisons can
    be tight (the epsilon changes results by about 1e-8 relative). NVIDIA's own
    multi-channel updater ignores groups; with groups=1 this is its arithmetic.
    """
    out_channels, per_group, kernel = weight.shape[0], weight.shape[1], weight.shape[2:]
    width = per_group * groups if multi_channel else 1
    # Masks and counts are real even for complex values.
    level = input.real.dtype if input.is_complex() else input.dtype
    if mask is None:
        mask = torch.ones(input.shape[0], width, *input.shape[2:], dtype=level)
    updater = torch.ones(
        out_channels if multi_channel else 1, per_group if multi_channel else 1, *kernel
    ).to(level)
    slide = updater.shape[1] * math.prod(kernel)
    with torch.no_grad():
        count = stock_conv(
            mask.to(level), updater, stride, padding, dilation, groups if multi_channel else 1
        )
        update = count.clamp(0, 1)
        ratio = slide / torch.where(count > 0, count, torch.ones_like(count)) * update
    raw = stock_conv(input * mask.to(level), weight, stride, padding, dilation, groups)
    if bias is None:
        return raw * ratio, update
    view = bias.view(1, -1, *([1] * (weight.ndim - 2)))
    return ((raw + view - view) * ratio + view) * update, update


def copy_pad(tensor, pairs, mode):
    """F.pad's copying modes for any number of spatial axes, by indexing each axis."""
    for axis, (left, right) in enumerate(pairs, start=2):
        if not left and not right:
            continue
        size = tensor.shape[axis]
        index = torch.arange(-left, size + right)
        if mode == "reflect":
            index = index.abs()
            index = torch.where(index >= size, 2 * (size - 1) - index, index)
        elif mode == "replicate":
            index = index.clamp(0, size - 1)
        else:
            index = index % size
        tensor = tensor.index_select(axis, index)
    return tensor


def binary_mask(shape, generator, holes=0.3):
    """Random holes plus one large hole, so some windows see no valid sample."""
    mask = (torch.rand(shape, generator=generator) > holes).double()
    corner = tuple(slice(0, min(size, size // 2 + 2)) for size in shape[2:])
    mask[(slice(0, 1), slice(None), *corner)] = 0
    return mask


GEOMETRIES = [
    {"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
    {"stride": 2, "padding": 1, "dilation": 1, "groups": 2},
    {"stride": 1, "padding": 2, "dilation": 2, "groups": 1},
]


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("multi_channel", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=["plain", "strided_grouped", "dilated"])
@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_matches_nvidia_arithmetic_with_gradients(rank, multi_channel, use_bias, geometry, backend):
    """Outputs, update masks and every parameter gradient follow NVIDIA's formulation."""
    generator = torch.Generator().manual_seed(rank * 100 + 7)
    groups = geometry["groups"]
    spatial = (7, 6, 5, 6)[:rank]
    input = torch.randn((2, 4, *spatial), dtype=torch.float64, generator=generator)
    weight = torch.randn((6, 4 // groups) + (3,) * rank, dtype=torch.float64, generator=generator)
    bias = torch.randn(6, dtype=torch.float64, generator=generator) if use_bias else None
    mask = binary_mask((2, 4 if multi_channel else 1, *spatial), generator)
    stride = (geometry["stride"],) * rank
    padding = (geometry["padding"],) * rank
    dilation = (geometry["dilation"],) * rank
    leaves = [input.requires_grad_(), weight.requires_grad_()] + (
        [bias.requires_grad_()] if use_bias else []
    )
    output, update = partial_conv_nd(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        mask=mask,
        multi_channel=multi_channel,
        backend=backend,
    )
    expected, expected_update = nvidia_reference(
        input, mask, weight, bias, stride, padding, dilation, groups, multi_channel
    )
    assert output.shape == expected.shape and output.dtype == torch.float64
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(update, expected_update.expand_as(update), rtol=0, atol=0)
    assert update.shape[1] == (6 if multi_channel else 1)
    assert (expected_update == 0).any(), "the large hole must empty some windows"
    probe = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    actual = torch.autograd.grad((output * probe).sum(), leaves)
    reference = torch.autograd.grad((expected * probe).sum(), leaves)
    for left, right in zip(actual, reference, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("rank", [5, 9])
@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("multi_channel", [False, True])
def test_high_ranks_match_nvidia_arithmetic_in_every_padding_mode(rank, mode, multi_channel):
    """Ranks beyond the stock layers follow NVIDIA's formulation, padding included."""
    generator = torch.Generator().manual_seed(rank * 10 + 3)
    size, kernel, padding = [1] * rank, [1] * rank, [0] * rank
    for axis, length, taps in ((0, 5, 3), (rank // 2, 4, 2), (rank - 1, 6, 3)):
        size[axis], kernel[axis], padding[axis] = length, taps, 1
    input = torch.randn((2, 4, *size), dtype=torch.float64, generator=generator)
    weight = torch.randn((6, 2, *kernel), dtype=torch.float64, generator=generator)
    bias = torch.randn(6, dtype=torch.float64, generator=generator)
    mask = binary_mask((2, 4 if multi_channel else 1, *size), generator)
    leaves = [input.requires_grad_(), weight.requires_grad_(), bias.requires_grad_()]
    output, update = partial_conv_nd(
        input,
        weight,
        bias,
        padding=padding,
        groups=2,
        mask=mask,
        multi_channel=multi_channel,
        padding_mode=mode,
        backend="torch",
    )
    ones = (1,) * rank
    if mode == "zeros":
        expected, expected_update = nvidia_reference(
            input, mask, weight, bias, ones, padding, ones, 2, multi_channel
        )
    else:
        pairs = [(value, value) for value in padding]
        expected, expected_update = nvidia_reference(
            copy_pad(input, pairs, mode),
            copy_pad(mask, pairs, mode),
            weight,
            bias,
            ones,
            (0,) * rank,
            ones,
            2,
            multi_channel,
        )
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(update, expected_update.expand_as(update), rtol=0, atol=0)
    probe = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    actual = torch.autograd.grad((output * probe).sum(), leaves)
    reference = torch.autograd.grad((expected * probe).sum(), leaves)
    for left, right in zip(actual, reference, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("multi_channel", [False, True])
def test_rank_limit_matches_the_rank_one_reference(multi_channel):
    """Masked counts, per group too, reach PyTorch's 64-axis limit."""
    generator = torch.Generator().manual_seed(62)
    trivial = (1,) * 61
    input = torch.randn((2, 4, *trivial, 7), dtype=torch.float64, generator=generator)
    weight = torch.randn((6, 2, *trivial, 3), dtype=torch.float64, generator=generator)
    bias = torch.randn(6, dtype=torch.float64, generator=generator)
    mask = binary_mask((2, 4 if multi_channel else 1, 7), generator)
    leaves = [input.requires_grad_(), weight.requires_grad_(), bias.requires_grad_()]
    output, update = partial_conv_nd(
        input,
        weight,
        bias,
        padding=(0,) * 61 + (1,),
        groups=2,
        mask=mask.reshape(2, -1, *trivial, 7),
        multi_channel=multi_channel,
        backend="torch",
    )
    expected, expected_update = nvidia_reference(
        input.reshape(2, 4, 7),
        mask,
        weight.reshape(6, 2, 3),
        bias,
        (1,),
        (1,),
        (1,),
        2,
        multi_channel,
    )
    assert output.shape == (2, 6, *trivial, 7)
    torch.testing.assert_close(output.reshape(2, 6, 7), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(update.reshape(2, -1, 7), expected_update, rtol=0, atol=0)
    assert (expected_update == 0).any(), "the large hole must empty some windows"
    probe = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    actual = torch.autograd.grad((output.reshape(2, 6, 7) * probe).sum(), leaves)
    reference = torch.autograd.grad((expected * probe).sum(), leaves)
    for left, right in zip(actual, reference, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("multi_channel", [False, True])
def test_no_mask_is_partial_convolution_padding(rank, multi_channel):
    """Interior windows equal ordinary convolution; border windows are rescaled."""
    generator = torch.Generator().manual_seed(3)
    input = torch.randn((1, 3) + (6,) * rank, dtype=torch.float64, generator=generator)
    weight = torch.randn((2, 3) + (3,) * rank, dtype=torch.float64, generator=generator)
    bias = torch.randn(2, dtype=torch.float64, generator=generator)
    output, update = partial_conv_nd(
        input, weight, bias, padding=1, multi_channel=multi_channel, backend="torch"
    )
    expected, _ = nvidia_reference(
        input, None, weight, bias, (1,) * rank, (1,) * rank, (1,) * rank, 1, multi_channel
    )
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(update, torch.ones_like(update))
    plain = conv_nd(input, weight, bias, padding=1, backend="torch")
    interior = (slice(None), slice(None)) + (slice(1, -1),) * rank
    torch.testing.assert_close(output[interior], plain[interior], rtol=1e-12, atol=1e-12)
    corner = (0, 0) + (0,) * rank
    window, valid = 3**rank, 2**rank
    torch.testing.assert_close(
        output[corner] - bias[0], (plain[corner] - bias[0]) * window / valid, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
@pytest.mark.parametrize("rank", [2, 3])
def test_padding_modes_pad_validity_with_the_values(mode, rank):
    """Copying padding modes copy validity: all-valid is ordinary, holes are padded alike."""
    generator = torch.Generator().manual_seed(11)
    input = torch.randn((2, 2) + (5,) * rank, dtype=torch.float64, generator=generator)
    weight = torch.randn((3, 2) + (3,) * rank, dtype=torch.float64, generator=generator)
    bias = torch.randn(3, dtype=torch.float64, generator=generator)
    plain = conv_nd(input, weight, bias, padding=1, padding_mode=mode, backend="torch")
    output, update = partial_conv_nd(
        input, weight, bias, padding=1, padding_mode=mode, backend="torch"
    )
    torch.testing.assert_close(output, plain, rtol=1e-12, atol=1e-12)
    assert torch.equal(update, torch.ones_like(update))
    mask = binary_mask((2, 1) + (5,) * rank, generator)
    output, _ = partial_conv_nd(
        input, weight, bias, padding=1, padding_mode=mode, mask=mask, backend="torch"
    )
    pads = (1, 1) * rank
    expected, _ = nvidia_reference(
        F.pad(input, pads, mode=mode),
        F.pad(mask, pads, mode=mode),
        weight,
        bias,
        (1,) * rank,
        (0,) * rank,
        (1,) * rank,
        1,
        False,
    )
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)


def test_same_padding_with_an_even_kernel_pads_both_paths_identically():
    """Asymmetric 'same' padding rescales the windows the asymmetric zeros cut."""
    generator = torch.Generator().manual_seed(5)
    input = torch.randn(1, 2, 6, 7, dtype=torch.float64, generator=generator)
    weight = torch.randn(3, 2, 4, 3, dtype=torch.float64, generator=generator)
    mask = binary_mask((1, 2, 6, 7), generator)
    output, update = partial_conv_nd(
        input, weight, padding="same", mask=mask, multi_channel=True, backend="torch"
    )
    pads = (1, 1, 1, 2)  # PyTorch pads the larger half after the input
    expected, expected_update = nvidia_reference(
        F.pad(input, pads), F.pad(mask, pads), weight, None, (1, 1), (0, 0), (1, 1), 1, True
    )
    assert output.shape == input.shape[:1] + (3,) + input.shape[2:]
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(update, expected_update, rtol=0, atol=0)


def test_complex_inputs_are_partial_in_each_component():
    """The operator is linear in the input, so complex values split into real parts."""
    generator = torch.Generator().manual_seed(9)
    real, imaginary = (torch.randn(1, 2, 6, 6, generator=generator) for _ in range(2))
    weight = torch.randn(3, 2, 3, 3, generator=generator)
    mask = binary_mask((1, 1, 6, 6), generator).float()
    complex_weight = torch.complex(weight, torch.zeros_like(weight))
    output, _ = partial_conv_nd(
        torch.complex(real, imaginary), complex_weight, padding=1, mask=mask, backend="torch"
    )
    parts = [partial_conv_nd(part, weight, padding=1, mask=mask)[0] for part in (real, imaginary)]
    torch.testing.assert_close(output, torch.complex(*parts), rtol=1e-5, atol=1e-5)


def test_counts_leave_autocast_and_stay_exact(monkeypatch):
    """Mask counts above bfloat16's exact-integer range are computed in float32."""
    calls = []
    original = functional_module._execute

    def spy(input, weight, *args, **kwargs):
        calls.append((input.dtype, is_autocast_enabled("cpu")))
        return original(input, weight, *args, **kwargs)

    monkeypatch.setattr(functional_module, "_execute", spy)
    layer = PartialConv3d(64, 4, 5, padding=2, multi_channel=True, backend="torch")
    input = torch.randn(1, 64, 5, 5, 5)
    mask = (torch.rand(1, 64, 5, 5, 5) > 0.2).float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(input, mask)
    assert output.dtype == torch.bfloat16
    data_call, count_call = calls
    assert count_call == (torch.float32, False) and data_call[1] is True
    with torch.no_grad():
        expected, _ = nvidia_reference(
            input.double(),
            mask.double(),
            layer.weight.double(),
            layer.bias.double(),
            (1,) * 3,
            (2,) * 3,
            (1,) * 3,
            1,
            True,
        )
    torch.testing.assert_close(output.double(), expected, rtol=3e-2, atol=3e-2)


def test_module_api_state_and_options():
    """Stock state dictionaries load; return_mask, unbatched input and repr behave."""
    stock = nn.Conv2d(3, 5, 3, padding=1)
    layer = PartialConv2d(3, 5, 3, padding=1, backend="torch", return_mask=True)
    assert layer.state_dict().keys() == stock.state_dict().keys()
    layer.load_state_dict(stock.state_dict())
    input = torch.randn(2, 3, 6, 6)
    mask = (torch.rand(2, 1, 6, 6) > 0.4).float()
    output, update = layer(input, mask)
    expected, expected_update = partial_conv_nd(
        input, stock.weight, stock.bias, padding=1, mask=mask, backend="torch"
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(update, expected_update)
    single, single_update = layer(input[1], mask[1])
    torch.testing.assert_close(single, output[1])
    torch.testing.assert_close(single_update, update[1])
    layer.return_mask = False
    assert isinstance(layer(input), torch.Tensor)
    assert "multi_channel=False, return_mask=False" in repr(layer)
    generic = PartialConvNd(3, 5, 3, dimensions=2, padding=1, backend="torch")
    generic.load_state_dict(stock.state_dict())
    torch.testing.assert_close(generic(input, mask), expected)
    assert isinstance(PartialConv4d(1, 1, 1), ConvNd)


def test_masked_update_feeds_the_next_layer():
    """A stacked partial network shrinks holes the way NVIDIA's inpainting stack does."""
    first = PartialConv2d(1, 4, 3, padding=1, multi_channel=True, return_mask=True)
    second = PartialConv2d(4, 2, 3, padding=1, multi_channel=True, return_mask=True)
    mask = torch.ones(1, 1, 9, 9)
    mask[..., 2:7, 2:7] = 0
    hidden, update = first(torch.randn(1, 1, 9, 9), mask)
    _, final = second(hidden, update)
    assert update.shape == (1, 4, 9, 9) and final.shape == (1, 2, 9, 9)
    assert update[0, 0].sum() > mask.sum() and final[0, 0].sum() > update[0, 0].sum()
    assert final[0, 0, 4, 4] == 0  # each 3x3 layer fills one ring; the center survives two


def test_invalid_masks_and_options_are_reported():
    weight = torch.randn(2, 4, 3, 3)
    input = torch.randn(1, 4, 5, 5)
    with pytest.raises(ValueError, match="dimensions"):
        partial_conv_nd(input, weight, mask=torch.ones(1, 5, 5))
    with pytest.raises(ValueError, match="multi_channel=True"):
        partial_conv_nd(input, weight, mask=torch.ones(1, 4, 5, 5))
    with pytest.raises(ValueError, match="1 or 4 channels"):
        partial_conv_nd(input, weight, mask=torch.ones(1, 3, 5, 5), multi_channel=True)
    with pytest.raises(ValueError, match="real"):
        partial_conv_nd(input, weight, mask=torch.ones(1, 1, 5, 5, dtype=torch.complex64))
    with pytest.raises(ValueError, match="batch"):
        partial_conv_nd(input.expand(2, -1, -1, -1), weight, mask=torch.ones(3, 1, 5, 5))
    with pytest.raises(TypeError, match="bool"):
        partial_conv_nd(input, weight, multi_channel=1)
    with pytest.raises(TypeError, match="bools"):
        PartialConv2d(4, 2, 3, return_mask="yes")
    with pytest.raises(RuntimeError, match="bias"):
        partial_conv_nd(input, weight, torch.randn(3))


def test_tracing_and_compilation_match_eager_but_scripting_is_refused():
    """Plain-convolution TorchScript adapters must never stand in for a partial layer."""
    layer = PartialConv2d(2, 3, 3, padding=1, backend="torch").eval()
    input, mask = torch.randn(1, 2, 6, 6), (torch.rand(1, 1, 6, 6) > 0.3).float()
    expected = layer(input, mask)
    traced = torch.jit.trace(layer, (input, mask))
    torch.testing.assert_close(traced(input, mask), expected)
    if hasattr(torch, "compile"):
        torch._dynamo.reset()
        compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(input, mask), expected)
    with pytest.raises(NotImplementedError, match="TorchScript"):
        torch.jit.script(layer)


@pytest.mark.parametrize("dynamic", [False, True])
def test_mask_free_compilation_with_static_and_symbolic_shapes(dynamic):
    """Per-axis counts stay symbolic: dynamic shapes share one graph, with no count conv."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    graphs = []

    def counting(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    layer = PartialConv3d(2, 3, 3, padding=1, backend="torch").eval()
    torch._dynamo.reset()
    compiled = torch.compile(layer, backend=counting, fullgraph=True, dynamic=dynamic)
    # Spatial sizes only: the underlying convolution specializes the batch, as stock does.
    shapes = ((2, 2, 5, 6, 7), (2, 2, 6, 5, 8), (2, 2, 9, 9, 9))
    for shape in shapes:
        input = torch.randn(shape)
        torch.testing.assert_close(compiled(input), layer(input))
    assert len(graphs) == (1 if dynamic else len(shapes))
    targets = [str(node.target) for node in graphs[0].graph.nodes if node.op == "call_function"]
    assert sum("conv" in target for target in targets) == 1, "only the data convolution"


def test_mask_free_layers_export_strictly_with_dynamic_sizes():
    export = getattr(torch, "export", None)
    if export is None or not hasattr(export, "Dim"):
        pytest.skip("torch.export is unavailable")
    layer = PartialConv2d(2, 3, 3, padding=1, backend="torch").eval()
    sizes = {2: export.Dim("height", min=4, max=32), 3: export.Dim("width", min=4, max=32)}
    program = export.export(layer, (torch.randn(2, 2, 6, 7),), dynamic_shapes=(sizes,), strict=True)
    for shape in ((2, 2, 6, 7), (2, 2, 9, 5)):
        input = torch.randn(shape)
        torch.testing.assert_close(program.module()(input), layer(input))


@pytest.mark.parametrize("rank", [2, 3, 5])
def test_onnx_export_runs_in_onnx_runtime(rank, tmp_path):
    """Masked and mask-free partial layers export as standard ONNX nodes."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from torch_convnd import export_onnx

    factory = {
        2: PartialConv2d,
        3: PartialConv3d,
        5: functools.partial(PartialConvNd, dimensions=5),
    }[rank]
    layer = factory(4, 6, 3, padding=1, groups=2, multi_channel=True, backend="torch").eval()
    input = torch.randn((2, 4) + (5,) * rank)
    mask = (torch.rand((2, 4) + (5,) * rank) > 0.3).float()
    path = tmp_path / "partial.onnx"
    export_onnx(layer, (input, mask), path, external_data=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_inputs()]
    actual = session.run(None, dict(zip(names, (input.numpy(), mask.numpy()), strict=True)))[0]
    np.testing.assert_allclose(actual, layer(input, mask).detach().numpy(), atol=2e-5, rtol=2e-5)
    assert all(node.domain in {"", "ai.onnx"} for node in onnx.load(path).graph.node)


def test_mask_free_export_keeps_dynamic_spatial_shapes(tmp_path):
    """Symbolic sizes count with a convolution instead of specializing per-axis counts."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from torch_convnd import export_onnx

    layer = PartialConv2d(2, 3, 3, padding=1, backend="torch").eval()
    dimensions = {
        0: torch.export.Dim("batch", min=1, max=4),
        2: torch.export.Dim("height", min=4, max=12),
        3: torch.export.Dim("width", min=4, max=12),
    }
    path = tmp_path / "dynamic.onnx"
    export_onnx(
        layer, (torch.randn(2, 2, 6, 6),), path, dynamic_shapes=(dimensions,), external_data=False
    )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    changed = torch.randn(3, 2, 9, 7)
    actual = session.run(None, {session.get_inputs()[0].name: changed.numpy()})[0]
    np.testing.assert_allclose(actual, layer(changed).detach().numpy(), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("masked", [False, True], ids=["mask_free", "masked"])
def test_legacy_onnx_export_keeps_dynamic_sizes(masked, tmp_path):
    """The TorchScript-based exporter passes every argument positionally and traces sizes."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from torch_convnd import export_onnx

    torch.manual_seed(43)
    layer = PartialConv2d(
        4, 6, 3, padding=1, groups=2, multi_channel=True, return_mask=True, backend="torch"
    ).eval()
    names = ["input", "mask"][: 1 + masked]

    def arguments(shape):
        return (torch.randn(shape), (torch.rand(shape) > 0.3).float())[: len(names)]

    axes = {name: {0: "batch", 2: "height", 3: "width"} for name in names}
    path = tmp_path / "legacy.onnx"
    export_onnx(
        layer, arguments((2, 4, 7, 9)), path, dynamo=False, input_names=names, dynamic_axes=axes
    )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for shape in ((2, 4, 7, 9), (3, 4, 10, 6)):
        values = arguments(shape)
        feed = {name: value.numpy() for name, value in zip(names, values, strict=True)}
        output, update = session.run(None, feed)
        with torch.no_grad():
            expected = layer(*values)
        np.testing.assert_allclose(output, expected.output.numpy(), atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(update, expected.mask.numpy())


@CUDA
def test_cuda_graph_capture_needs_no_cached_counts():
    """Capture with a cold cache counts on the device, without a host-to-device copy."""
    layer = PartialConv2d(4, 4, 3, padding=1, backend="torch").cuda().eval()
    input = torch.randn(2, 4, 16, 16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(2):
            layer(input)
    torch.cuda.current_stream().wait_stream(stream)
    functional_module._cached_small_scale.cache_clear()
    functional_module._cached_axis_counts.cache_clear()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        captured = layer(input)
    graph.replay()
    torch.cuda.synchronize()
    with torch.no_grad():
        torch.testing.assert_close(captured, layer(input))


@CUDA
@pytest.mark.parametrize("rank", [2, 4])
def test_cuda_matches_the_float64_reference(rank):
    generator = torch.Generator().manual_seed(21)
    input = torch.randn((2, 4) + (5,) * rank, generator=generator)
    weight = torch.randn((4, 2) + (3,) * rank, generator=generator)
    bias = torch.randn(4, generator=generator)
    mask = binary_mask((2, 4) + (5,) * rank, generator).float()
    output, update = partial_conv_nd(
        input.cuda(),
        weight.cuda(),
        bias.cuda(),
        padding=1,
        groups=2,
        mask=mask.cuda(),
        multi_channel=True,
        backend="torch",
    )
    expected, expected_update = nvidia_reference(
        input.double(),
        mask.double(),
        weight.double(),
        bias.double(),
        (1,) * rank,
        (1,) * rank,
        (1,) * rank,
        2,
        True,
    )
    torch.testing.assert_close(output.cpu().double(), expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(update.cpu().double(), expected_update, rtol=0, atol=0)
    with pytest.raises(ValueError, match="device"):
        partial_conv_nd(input.cuda(), weight.cuda(), mask=mask)


@settings(max_examples=600, deadline=None)
@given(
    st.integers(1, 12),
    st.integers(1, 5),
    st.integers(1, 3),
    st.integers(1, 5),
    st.integers(0, 8),
    st.integers(0, 8),
)
def test_axis_tap_counts_match_brute_force(size, kernel, stride, dilation, left, right):
    """Per-axis counts, on the host and the device, and the emptiness check count every tap."""
    extent = dilation * (kernel - 1) + 1
    geometry = size, kernel, stride, dilation, left, right
    if size + left + right < extent:
        assert not functional_module._axis_is_full(*geometry)
        return
    expected = [
        sum(0 <= position * stride - left + tap * dilation < size for tap in range(kernel))
        for position in range((size + left + right - extent) // stride + 1)
    ]
    assert functional_module._valid_taps(*geometry) == expected
    counts = functional_module._axis_counts(*geometry, torch.float64, torch.device("cpu"))
    assert counts.tolist() == expected
    assert functional_module._axis_is_full(*geometry) == (min(expected) > 0)


@st.composite
def partial_geometries(draw):
    """Small generated layers; padding can exceed a kernel's reach and empty windows."""
    rank = draw(st.integers(1, 3))
    groups = draw(st.integers(1, 2))
    channels, outputs = groups * draw(st.integers(1, 2)), groups * draw(st.integers(1, 2))
    kernel = tuple(draw(st.integers(1, 4)) for _ in range(rank))
    stride = tuple(draw(st.integers(1, 3)) for _ in range(rank))
    dilation = tuple(draw(st.integers(1, 2)) for _ in range(rank))
    padding = tuple(draw(st.integers(0, 4)) for _ in range(rank))
    spatial = tuple(
        draw(st.integers(max(1, d * (k - 1) + 1 - 2 * p), 7))
        for k, d, p in zip(kernel, dilation, padding, strict=True)
    )
    kind = draw(st.sampled_from(["none", "binary", "fractional"]))
    multi, bias = draw(st.booleans()), draw(st.booleans())
    seed = draw(st.integers(0, 2**16))
    return (
        rank,
        groups,
        channels,
        outputs,
        kernel,
        stride,
        dilation,
        padding,
        spatial,
        kind,
        multi,
        bias,
        seed,
    )


@settings(max_examples=250, deadline=None)
@given(partial_geometries())
def test_generated_geometries_match_nvidia_arithmetic(geometry):
    """Values, update masks and gradients on generated layers, masks and fast paths."""
    (
        rank,
        groups,
        channels,
        outputs,
        kernel,
        stride,
        dilation,
        padding,
        spatial,
        kind,
        multi,
        use_bias,
        seed,
    ) = geometry
    generator = torch.Generator().manual_seed(seed)
    input = torch.randn((2, channels, *spatial), dtype=torch.float64, generator=generator)
    weight = torch.randn(
        (outputs, channels // groups, *kernel), dtype=torch.float64, generator=generator
    )
    bias = torch.randn(outputs, dtype=torch.float64, generator=generator) if use_bias else None
    mask = None
    if kind != "none":
        mask = torch.rand((2, channels if multi else 1, *spatial), generator=generator)
        mask = (mask if kind == "fractional" else (mask > 0.4).double()).double()
    leaves = [input.requires_grad_(), weight.requires_grad_()] + (
        [bias.requires_grad_()] if use_bias else []
    )
    actual, update = partial_conv_nd(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        mask=mask,
        multi_channel=multi,
        backend="torch",
    )
    expected, expected_update = nvidia_reference(
        input, mask, weight, bias, stride, padding, dilation, groups, multi
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(update, expected_update.expand_as(update), rtol=0, atol=1e-15)
    probe = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    for left, right in zip(
        torch.autograd.grad((actual * probe).sum(), leaves),
        torch.autograd.grad((expected * probe).sum(), leaves),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("padding", [0, 1, 3, "same", "valid"])
@pytest.mark.parametrize("multi_channel", [False, True])
def test_mask_free_paths_equal_the_counted_all_valid_mask(mode, padding, multi_channel):
    """The plain-convolution and separable-count shortcuts change nothing but the work."""
    generator = torch.Generator().manual_seed(17)
    stride = (1, 1, 1) if padding == "same" else (1, 2, 1)
    dilation = (1, 1, 1) if padding == "same" else (1, 1, 2)
    input = torch.randn(2, 4, 6, 7, 8, dtype=torch.float64, generator=generator)
    weight = torch.randn(6, 2, 3, 4, 2, dtype=torch.float64, generator=generator)
    bias = torch.randn(6, dtype=torch.float64, generator=generator)
    ones = torch.ones(2, 4 if multi_channel else 1, 6, 7, 8, dtype=torch.float64)
    options = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": 2,
        "padding_mode": mode,
        "multi_channel": multi_channel,
        "backend": "torch",
    }
    leaves = [value.detach().requires_grad_() for value in (input, weight, bias)]
    fast, fast_update = partial_conv_nd(*leaves, **options)
    counted, counted_update = partial_conv_nd(*leaves, mask=ones, **options)
    torch.testing.assert_close(fast, counted, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(fast_update, counted_update, rtol=0, atol=0)
    for left, right in zip(
        torch.autograd.grad(fast.square().sum(), leaves),
        torch.autograd.grad(counted.square().sum(), leaves),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
    if mode == "zeros" and padding == 3:
        assert (fast_update == 0).any(), "padding beyond the kernel's reach empties windows"


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("mode", ["zeros", "reflect"])
def test_finite_difference_first_and_second_derivatives(masked, mode):
    generator = torch.Generator().manual_seed(23)
    input = torch.randn(1, 2, 4, 5, dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn(2, 1, 3, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    bias = torch.randn(2, dtype=torch.float64, generator=generator, requires_grad=True)
    mask = (torch.rand(1, 2, 4, 5, generator=generator) > 0.3).double() if masked else None

    def function(input, weight, bias):
        output, _ = partial_conv_nd(
            input,
            weight,
            bias,
            1,
            1,
            1,
            2,
            mask=mask,
            multi_channel=True,
            padding_mode=mode,
            backend="torch",
        )
        return output

    assert torch.autograd.gradcheck(function, (input, weight, bias))
    assert torch.autograd.gradgradcheck(function, (input, weight, bias))


def test_vmap_and_forward_mode_follow_linearity_in_the_input():
    """For a fixed mask the layer is affine in its input: vmap and jvp must agree."""
    func = pytest.importorskip("torch.func")
    generator = torch.Generator().manual_seed(29)
    input = torch.randn(3, 2, 6, 6, generator=generator)
    tangent = torch.randn(3, 2, 6, 6, generator=generator)
    weight, bias = torch.randn(4, 2, 3, 3, generator=generator), torch.randn(4, generator=generator)
    mask = (torch.rand(3, 1, 6, 6, generator=generator) > 0.3).float()
    for masked in (False, True):
        current = mask if masked else None
        batched, _ = partial_conv_nd(input, weight, bias, padding=1, mask=current, backend="torch")

        def single(sample, sample_mask=None):
            return partial_conv_nd(
                sample, weight, bias, padding=1, mask=sample_mask, backend="torch"
            )[0]

        mapped = func.vmap(single)(input, current) if masked else func.vmap(single)(input)
        torch.testing.assert_close(mapped, batched, rtol=1e-5, atol=1e-5)
        _, derivative = func.jvp(
            lambda value, current=current: partial_conv_nd(
                value, weight, bias, padding=1, mask=current, backend="torch"
            )[0],
            (input,),
            (tangent,),
        )
        linear, _ = partial_conv_nd(tangent, weight, None, padding=1, mask=current, backend="torch")
        torch.testing.assert_close(derivative, linear, rtol=1e-5, atol=1e-5)


def test_empty_batches_unbatched_multichannel_and_validation():
    weight = torch.randn(4, 2, 3, 3)
    empty, update = partial_conv_nd(
        torch.randn(0, 2, 5, 5), weight, mask=torch.ones(0, 2, 5, 5), multi_channel=True, padding=1
    )
    assert empty.shape == (0, 4, 5, 5) and update.shape == (0, 4, 5, 5)
    layer = PartialConv2d(2, 4, 3, padding=1, multi_channel=True, return_mask=True)
    single, single_update = layer(torch.randn(2, 5, 5), torch.ones(2, 5, 5))
    assert single.shape == (4, 5, 5) and single_update.shape == (4, 5, 5)
    sample = torch.randn(2, 5, 5)
    plain, plain_update = partial_conv_nd(sample, weight, padding=1, padding_mode="reflect")
    torch.testing.assert_close(plain, conv_nd(sample, weight, padding=1, padding_mode="reflect"))
    assert plain_update.shape == (1, 5, 5) and bool((plain_update == 1).all())
    reflecting = PartialConv2d(2, 4, 3, padding=1, padding_mode="reflect", bias=False)
    reflecting.weight.data.copy_(weight)
    torch.testing.assert_close(reflecting(sample), plain)
    with pytest.raises(ValueError, match="1-62 spatial"):
        partial_conv_nd(torch.randn(1, 2), torch.randn(4, 2))
    with pytest.raises(ValueError, match="batched dimensions"):
        partial_conv_nd(torch.randn(5, 5), weight)
    with pytest.raises(RuntimeError, match="divisible by groups"):
        partial_conv_nd(torch.randn(1, 3, 5, 5), torch.randn(4, 1, 3, 3), groups=2)
    with pytest.raises(ValueError, match="batched dimensions"):
        layer(torch.randn(1, 1, 2, 5, 5))


def test_cached_mask_free_scale_survives_inference_mode_and_changes_nothing():
    """Scales cached during inference serve later training calls unchanged, per dtype."""
    functional_module._cached_small_scale.cache_clear()
    layer = PartialConv3d(2, 3, 3, padding=1, backend="torch")
    input = torch.randn(1, 2, 5, 6, 7)
    with torch.inference_mode():
        inferred = layer(input)
    trained = layer(input.requires_grad_())
    trained.sum().backward()
    torch.testing.assert_close(trained.detach(), inferred)
    information = functional_module._cached_small_scale.cache_info()
    assert information.misses == 1 and information.hits == 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert layer(input).dtype == torch.bfloat16
    assert functional_module._cached_small_scale.cache_info().misses == 2
    geometry = (5, 6, 7), (3, 3, 3), (1,) * 3, (1,) * 3, ((1, 1),) * 3, torch.float32
    uncached = functional_module._mask_free_scale(*geometry, torch.float32, torch.float32, "cpu")
    cached = functional_module._cached_small_scale(
        *geometry, torch.float32, torch.float32, input.device, stream=None
    )
    for left, right in zip(uncached, cached, strict=True):
        assert left == right if isinstance(left, bool) else torch.equal(left, right)
    assert not any(value.is_inference() for value in cached[:2])
    corner_ratio = cached[0][(0, 0, 0, 0, 0)]
    assert corner_ratio == 27 / 8 and cached[2] is True


def test_return_mask_only_changes_what_is_returned():
    """The module skips materializing the update mask without changing the output."""
    torch.manual_seed(31)
    layer = PartialConv2d(4, 6, 3, padding=1, groups=2, multi_channel=True, backend="torch")
    input, mask = torch.randn(2, 4, 7, 7), (torch.rand(2, 4, 7, 7) > 0.3).float()
    output = layer(input, mask)
    layer.return_mask = True
    with_mask, update = layer(input, mask)
    assert torch.equal(output, with_mask) and update.shape == (2, 6, 7, 7)
    from_keyword, _ = layer(input, mask_in=mask)
    assert torch.equal(from_keyword, output)
    with pytest.raises(TypeError, match="not both"):
        layer(input, mask, mask_in=mask)


def test_writing_a_returned_update_never_reaches_the_cache():
    """Returned update masks are fresh tensors, so in-place edits stay local."""
    functional_module._cached_small_scale.cache_clear()
    input, weight, bias = torch.randn(1, 2, 6, 6), torch.randn(3, 2, 3, 3), torch.randn(3)
    expected, _ = partial_conv_nd(input, weight, bias, padding=3)
    _, update = partial_conv_nd(input, weight, bias, padding=3)
    update.zero_()
    again, again_update = partial_conv_nd(input, weight, bias, padding=3)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
    assert bool(again_update.any())


def test_fake_tensor_forwards_leave_the_cache_untouched():
    """Fake-tensor and other dispatch-mode calls never store values for eager calls."""
    fake = pytest.importorskip("torch._subclasses.fake_tensor")
    functional_module._cached_small_scale.cache_clear()
    functional_module._cached_axis_counts.cache_clear()
    input, weight, bias = torch.randn(1, 2, 6, 6), torch.randn(3, 2, 3, 3), torch.randn(3)
    with fake.FakeTensorMode() as mode:
        shaped, _ = partial_conv_nd(
            mode.from_tensor(input), mode.from_tensor(weight), mode.from_tensor(bias), padding=1
        )
    assert shaped.shape == (1, 3, 6, 6)
    assert functional_module._cached_small_scale.cache_info().currsize == 0
    assert functional_module._cached_axis_counts.cache_info().currsize == 0
    eager, _ = partial_conv_nd(input, weight, bias, padding=1)
    expected, _ = nvidia_reference(input, None, weight, bias, (1, 1), (1, 1), (1, 1), 1, False)
    torch.testing.assert_close(eager, expected)


def test_counting_kernels_are_reused_only_by_plain_eager_calls():
    """Masked eager calls share one counting kernel; fake tensors, transforms and
    compilers build their own, so nothing they create outlives them."""
    fake = pytest.importorskip("torch._subclasses.fake_tensor")
    functional_module._cached_ones.cache_clear()
    input, weight = torch.randn(2, 2, 6, 6), torch.randn(3, 2, 3, 3)
    mask = (torch.rand(2, 1, 6, 6) > 0.3).float()

    def masked(input, mask):
        return partial_conv_nd(input, weight, padding=1, mask=mask, backend="torch").output

    with fake.FakeTensorMode() as mode:
        partial_conv_nd(
            mode.from_tensor(input),
            mode.from_tensor(weight),
            padding=1,
            mask=mode.from_tensor(mask),
            backend="torch",
        )
    torch.func.vmap(masked)(input.unsqueeze(1), mask.unsqueeze(1))
    if hasattr(torch, "compile"):
        torch._dynamo.reset()
        torch.compile(masked, backend="aot_eager", fullgraph=True)(input, mask)
    assert functional_module._cached_ones.cache_info().currsize == 0
    first, second = masked(input, mask), masked(input, mask)
    information = functional_module._cached_ones.cache_info()
    assert (information.misses, information.hits, information.currsize) == (1, 1, 1)
    expected, _ = nvidia_reference(input, mask, weight, None, (1, 1), (1, 1), (1, 1), 1, False)
    torch.testing.assert_close(first, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(first, second)


def test_functorch_transforms_leave_the_cache_untouched():
    """Transforms wrap tensors that look plain; nothing created inside them is kept."""
    if not hasattr(torch, "func"):
        pytest.skip("torch.func is unavailable")
    functional_module._cached_small_scale.cache_clear()
    functional_module._cached_axis_counts.cache_clear()
    torch.manual_seed(47)
    layer = PartialConv2d(2, 3, 3, padding=1, backend="torch")
    input = torch.randn(4, 2, 8, 8, requires_grad=True)
    functionalized = torch.func.functionalize(layer)(input)
    gradient = torch.func.grad(lambda value: layer(value).sum())(input)
    batched = torch.func.vmap(layer)(input.unsqueeze(1))
    _, tangent = torch.func.jvp(layer, (input.detach(),), (input.detach(),))
    assert functional_module._cached_small_scale.cache_info().currsize == 0
    assert functional_module._cached_axis_counts.cache_info().currsize == 0
    expected = layer(input)
    (expected_gradient,) = torch.autograd.grad(expected.sum(), input)
    torch.testing.assert_close(functionalized, expected)
    torch.testing.assert_close(batched.squeeze(1), expected)
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(tangent, expected - layer.bias.view(1, -1, 1, 1))
    # Training afterwards sees ordinary tensors; a poisoned cache made this step raise.
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer(input).square().mean().backward()
    optimizer.step()
    assert not torch._is_functional_tensor(layer(input))


@CUDA
def test_cached_scales_are_kept_per_stream():
    """A scale cached by a busy side stream is never read early from another stream."""
    functional_module._cached_small_scale.cache_clear()
    functional_module._cached_axis_counts.cache_clear()
    torch.manual_seed(53)
    input = torch.randn(2, 4, 32, 32, device="cuda")
    weight = torch.randn(4, 4, 3, 3, device="cuda")
    expected, _ = partial_conv_nd(input.double(), weight.double(), padding=1)
    partial_conv_nd(input.half(), weight.half(), padding=1)  # warms float32 axis counts
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(1_000_000_000)
        partial_conv_nd(input, weight, padding=1)  # its scale queues behind the sleep
    output, _ = partial_conv_nd(input, weight, padding=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.double(), expected, rtol=1e-5, atol=1e-5)


def test_long_axes_keep_no_per_axis_counts():
    """Only short axes are cached, so ragged long signals pin no device memory."""
    functional_module._cached_small_scale.cache_clear()
    functional_module._cached_axis_counts.cache_clear()
    weight, bias = torch.randn(1, 1, 3), torch.randn(1)
    for length in (300_000, 300_001):  # above both cache limits
        output, _ = partial_conv_nd(torch.ones(1, 1, length), weight, bias, padding=1)
    assert functional_module._cached_small_scale.cache_info().currsize == 0
    assert functional_module._cached_axis_counts.cache_info().currsize == 0
    torch.testing.assert_close(output[0, 0, 0], weight[0, 0, 1:].sum() * 3 / 2 + bias[0])
    torch.testing.assert_close(output[0, 0, 1], weight.sum() + bias[0])


def test_counts_leave_autocast_on_every_device_type():
    """Counting leaves autocast on any device type it supports, not only CPU and CUDA."""
    from torch_convnd._compat import autocast_available, autocast_disabled

    if not (hasattr(torch, "get_autocast_dtype") and autocast_available("mps")):
        pytest.skip("this PyTorch has no MPS autocast")
    with torch.autocast("mps", dtype=torch.float16):
        with autocast_disabled("mps"):
            assert not torch.is_autocast_enabled("mps")
        assert torch.is_autocast_enabled("mps")
    assert not autocast_available("meta")


def test_large_outputs_cache_only_per_axis_counts():
    """Finished scales are cached for small outputs; large ones keep only 1-D vectors."""
    functional_module._cached_small_scale.cache_clear()
    functional_module._cached_axis_counts.cache_clear()
    input = torch.randn(1, 1, 72, 72, 72)  # 373,248 output positions, above the limit
    weight, bias = torch.randn(1, 1, 3, 3, 3), torch.randn(1)
    for _ in range(2):
        output, _ = partial_conv_nd(input, weight, bias, padding=1)
    assert functional_module._cached_small_scale.cache_info().currsize == 0
    assert functional_module._cached_axis_counts.cache_info().currsize == 1  # three equal axes
    expected, _ = nvidia_reference(
        input.double(),
        None,
        weight.double(),
        bias.double(),
        (1,) * 3,
        (1,) * 3,
        (1,) * 3,
        1,
        False,
    )
    torch.testing.assert_close(output.double(), expected, rtol=1e-5, atol=1e-5)


def test_float16_windows_beyond_range_rescale_in_float32():
    """A per-channel window over 65504 would overflow the float16 ratio."""
    torch.manual_seed(37)
    layer = PartialConv2d(8192, 1, 3, padding=1, multi_channel=True, backend="torch")
    input = torch.randn(1, 8192, 4, 4)
    mask = torch.zeros(1, 8192, 4, 4)
    mask[0, 0, 1, 1] = 1  # a single valid sample: ratio 73,728 in its windows
    expected = layer.double()(input.double(), mask.double())
    layer.half().return_mask = True
    output, update = layer(input.half(), mask.half())
    assert output.dtype == update.dtype == torch.float16 and bool(torch.isfinite(output).all())
    torch.testing.assert_close(output.double(), expected, rtol=2e-2, atol=2e-1)


def test_complex_weights_and_bias_follow_the_complex_reference():
    generator = torch.Generator().manual_seed(41)

    def complex_tensor(*shape):
        real, imaginary = (
            torch.randn(shape, generator=generator, dtype=torch.float64) for _ in "ri"
        )
        return torch.complex(real, imaginary)

    input, weight, bias = complex_tensor(2, 2, 6, 6), complex_tensor(3, 2, 3, 3), complex_tensor(3)
    mask = binary_mask((2, 1, 6, 6), generator)
    for current in (None, mask):
        output, update = partial_conv_nd(
            input, weight, bias, padding=1, mask=current, backend="torch"
        )
        expected, expected_update = nvidia_reference(
            input, current, weight, bias, (1, 1), (1, 1), (1, 1), 1, False
        )
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        assert not update.is_complex()
        torch.testing.assert_close(update, expected_update.real.expand_as(update))


@CUDA
def test_cuda_compilation_of_the_mask_free_path_is_one_graph():
    """The capture check stays out of compiled graphs: fullgraph compiles on CUDA."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    layer = PartialConv3d(2, 3, 3, padding=1, backend="torch").cuda().eval()
    torch._dynamo.reset()
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    input = torch.randn(1, 2, 5, 6, 7, device="cuda")
    torch.testing.assert_close(compiled(input), layer(input))
