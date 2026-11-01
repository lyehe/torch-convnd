"""The partial-convolution API: partial padding mode, mask contract, results, chaining."""

import numpy as np
import pytest
import torch
from torch import nn

import torch_convnd.functional as functional_module
from torch_convnd import (
    Conv2d,
    Conv3d,
    ConvNd,
    PartialConv2d,
    PartialConv3d,
    PartialConvNd,
    PartialConvOutput,
    PartialSequential,
    conv_nd,
    partial_conv_nd,
)

pytestmark = pytest.mark.portable
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
GEOMETRIES = [
    {"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
    {"stride": 2, "padding": 2, "dilation": 1, "groups": 2},
    {"stride": 1, "padding": 2, "dilation": 2, "groups": 1},
    {"stride": 1, "padding": "same", "dilation": 1, "groups": 2},
    {"stride": 1, "padding": "valid", "dilation": 1, "groups": 1},
]


# 1. padding_mode="partial" on ordinary layers.


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "geometry", GEOMETRIES, ids=["plain", "strided", "dilated", "same", "valid"]
)
def test_partial_padding_mode_is_the_mask_free_partial_convolution(rank, geometry):
    """Values and gradients equal a PartialConvNd without a mask, bit for bit."""
    torch.manual_seed(rank)
    options = {**geometry, "dimensions": rank, "backend": "torch", "dtype": torch.float64}
    kernel = 4 if geometry["padding"] == "same" else 3  # an even kernel pads asymmetrically
    bordered = ConvNd(4, 6, kernel, padding_mode="partial", **options)
    partial = PartialConvNd(4, 6, kernel, **options)
    partial.load_state_dict(bordered.state_dict())
    input = torch.randn((2, 4) + (7,) * rank, dtype=torch.float64, requires_grad=True)
    expected = partial(input)
    actual = bordered(input)
    assert torch.equal(actual, expected)
    first = torch.autograd.grad(actual.square().sum(), (input, *bordered.parameters()))
    second = torch.autograd.grad(expected.square().sum(), (input, *partial.parameters()))
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    functional = conv_nd(
        input, bordered.weight, bordered.bias, padding_mode="partial", backend="torch", **geometry
    )
    assert torch.equal(functional, expected)
    assert "padding_mode='partial'" in repr(bordered)


def test_partial_padding_differs_from_zeros_only_where_windows_leave_the_input():
    torch.manual_seed(1)
    input = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    weight = torch.randn(3, 2, 3, 3, dtype=torch.float64)
    zeros = conv_nd(input, weight, padding=1, backend="torch")
    partial = conv_nd(input, weight, padding=1, padding_mode="partial", backend="torch")
    torch.testing.assert_close(partial[..., 1:-1, 1:-1], zeros[..., 1:-1, 1:-1])
    torch.testing.assert_close(partial[..., 0, 0], zeros[..., 0, 0] * 9 / 4)
    torch.testing.assert_close(partial[..., 0, 3], zeros[..., 0, 3] * 9 / 6)
    unbatched = conv_nd(input[0], weight, padding=1, padding_mode="partial", backend="torch")
    assert torch.equal(unbatched, partial[0])


@pytest.mark.parametrize(
    ("weight_shape", "arguments"),
    [
        ((4, 4, 3, 3), {"stride": 0}),
        ((4, 4, 3, 3), {"padding": -1}),
        ((4, 4, 3, 3), {"padding": 1, "dilation": 0}),
        ((4, 4, 3, 3), {"padding": 1, "dilation": 4}),
        ((4, 4, 3, 3), {"padding": 1, "groups": 3}),
        ((4, 4, 3, 3), {"padding": "same", "stride": 2}),
        ((4, 4, 3, 3), {"padding": "sideways"}),
        ((4, 3, 3, 3), {"padding": 1}),
    ],
    ids=["stride", "padding", "dilation", "extent", "groups", "same", "name", "channels"],
)
def test_partial_padding_rejects_what_zero_padding_rejects(weight_shape, arguments):
    input, weight = torch.randn(1, 4, 6, 6), torch.randn(weight_shape)
    with pytest.raises((RuntimeError, ValueError)) as zeros:
        conv_nd(input, weight, backend="torch", **arguments)
    with pytest.raises(zeros.type) as partial:
        conv_nd(input, weight, padding_mode="partial", backend="torch", **arguments)
    assert str(partial.value) == str(zeros.value)


def test_partial_padding_keeps_stock_state_positions_dtypes_and_amp():
    stock = nn.Conv3d(2, 4, 3, padding=1)
    bordered = Conv3d(2, 4, 3, 1, 1, 1, 1, True, "partial", backend="torch")
    assert bordered.state_dict().keys() == stock.state_dict().keys()
    bordered.load_state_dict(stock.state_dict())
    input = torch.randn(1, 2, 5, 6, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low = bordered(input)
    assert low.dtype == torch.bfloat16
    torch.testing.assert_close(low.float(), bordered(input), rtol=3e-2, atol=3e-2)


def test_partial_mode_is_refused_where_it_would_be_wrong():
    with pytest.raises(NotImplementedError, match="padding_mode='partial'"):
        torch.jit.script(Conv2d(2, 3, 3, padding=1, padding_mode="partial"))
    scripted = torch.jit.script(Conv2d(2, 3, 3, padding=1, padding_mode="reflect"))
    assert scripted(torch.randn(1, 2, 6, 6)).shape == (1, 3, 6, 6)
    with pytest.raises(ValueError, match="already rescales zero padding"):
        PartialConv2d(2, 3, 3, padding=1, padding_mode="partial")
    with pytest.raises(ValueError, match="already rescales zero padding"):
        partial_conv_nd(torch.randn(1, 2, 5, 5), torch.randn(3, 2, 3, 3), padding_mode="partial")
    with pytest.raises(ValueError, match="Invalid padding_mode"):
        Conv2d(2, 3, 3, padding_mode="partially")
    mutated = PartialConv2d(2, 3, 3, padding=1)
    mutated.padding_mode = "partial"  # set after construction, past the constructor's check
    with pytest.raises(ValueError, match="already rescales zero padding"):
        mutated(torch.randn(1, 2, 5, 5))


@pytest.mark.parametrize("dynamic", [False, True])
def test_partial_padding_compiles_as_one_graph(dynamic):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    layer = Conv3d(2, 3, 3, padding=1, padding_mode="partial", backend="torch").eval()
    torch._dynamo.reset()
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True, dynamic=dynamic)
    for shape in ((1, 2, 5, 6, 7), (2, 2, 6, 5, 8)):
        input = torch.randn(shape)
        torch.testing.assert_close(compiled(input), layer(input))


@pytest.mark.parametrize("dynamic", [False, True])
def test_partial_padding_exports_to_onnx(dynamic, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from torch_convnd import export_onnx

    layer = Conv2d(2, 3, 3, padding=1, padding_mode="partial", backend="torch").eval()
    options = {"external_data": False}
    if dynamic:
        options["dynamic_shapes"] = (
            {
                0: torch.export.Dim("batch", min=1, max=4),
                2: torch.export.Dim("height", min=4, max=12),
                3: torch.export.Dim("width", min=4, max=12),
            },
        )
    path = tmp_path / "partial_padding.onnx"
    export_onnx(layer, (torch.randn(2, 2, 6, 6),), path, **options)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    changed = torch.randn((3, 2, 9, 7) if dynamic else (2, 2, 6, 6))
    actual = session.run(None, {session.get_inputs()[0].name: changed.numpy()})[0]
    np.testing.assert_allclose(actual, layer(changed).detach().numpy(), atol=2e-5, rtol=2e-5)


@CUDA
def test_partial_padding_compiles_on_cuda():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    layer = Conv3d(2, 3, 3, padding=1, padding_mode="partial", backend="torch").cuda().eval()
    torch._dynamo.reset()
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    input = torch.randn(1, 2, 5, 6, 7, device="cuda")
    torch.testing.assert_close(compiled(input), layer(input))


# 2. The mask contract.


def test_masks_broadcast_over_batch_and_spatial_axes_of_size_one():
    torch.manual_seed(2)
    input, weight = torch.randn(3, 2, 6, 7, dtype=torch.float64), torch.randn(4, 2, 3, 3).double()
    row = (torch.rand(1, 1, 1, 7) > 0.4).double()  # one row pattern for every sample and row
    output, update = partial_conv_nd(input, weight, padding=1, mask=row)
    expanded, expanded_update = partial_conv_nd(
        input, weight, padding=1, mask=row.expand(3, 1, 6, 7).contiguous()
    )
    assert torch.equal(output, expanded) and torch.equal(update, expanded_update)
    assert update.shape == (3, 1, 6, 7)
    with pytest.raises(ValueError, match="spatial shape"):
        partial_conv_nd(input, weight, padding=1, mask=torch.ones(1, 1, 5, 7))


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int64, torch.float16, torch.float64]
)
def test_masks_of_any_real_dtype_mark_the_same_samples(dtype):
    torch.manual_seed(9)
    input, weight = torch.randn(2, 3, 7, 6), torch.randn(4, 3, 3, 3)
    holes = torch.rand(2, 1, 7, 6) > 0.4
    expected = partial_conv_nd(input, weight, padding=1, mask=holes.float())
    actual = partial_conv_nd(input, weight, padding=1, mask=holes.to(dtype))
    assert torch.equal(actual.output, expected.output) and torch.equal(actual.mask, expected.mask)
    assert actual.mask.dtype == torch.float32


@pytest.mark.parametrize("multi_channel", [False, True])
def test_mask_gradients_flow_only_through_the_masked_input(multi_channel):
    """The counts are constants, so the mask's gradient is the masked input's alone."""
    torch.manual_seed(10)
    input = torch.randn(2, 4, 6, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 2, 3, 3, dtype=torch.float64)
    bias = torch.randn(6, dtype=torch.float64)
    # Fractional but nonzero values, so input.grad = mask * g can be divided by the mask.
    mask = 0.5 + 0.5 * torch.rand(2, 4 if multi_channel else 1, 6, 7, dtype=torch.float64)
    mask.requires_grad_()
    output, _ = partial_conv_nd(
        input, weight, bias, padding=1, groups=2, mask=mask, multi_channel=multi_channel
    )
    probe = torch.randn_like(output)
    input_grad, mask_grad = torch.autograd.grad((output * probe).sum(), (input, mask))
    # With g the gradient of input * mask: input.grad = mask * g and mask.grad = input * g,
    # summed over the channels a shared mask covers.
    expected = input.detach() * input_grad / mask.detach()
    if not multi_channel:
        expected = expected.sum(1, keepdim=True)
    torch.testing.assert_close(mask_grad, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("multi_channel", [False, True])
def test_update_mask_layout_feeds_the_same_kind_of_layer(multi_channel):
    torch.manual_seed(3)
    first = PartialConv2d(2, 4, 3, padding=1, multi_channel=multi_channel)
    second = PartialConv2d(4, 3, 3, padding=1, multi_channel=multi_channel)
    mask = (torch.rand(2, 2 if multi_channel else 1, 8, 8) > 0.3).float()
    hidden, update = first(torch.randn(2, 2, 8, 8), mask, return_mask=True)
    assert update.shape == (2, 4 if multi_channel else 1, 8, 8)
    assert update.dtype == hidden.dtype and not update.requires_grad
    output, final = second(hidden, update, return_mask=True)
    assert final.shape == (2, 3 if multi_channel else 1, 8, 8) and output.shape == (2, 3, 8, 8)


# 3. Results.


def test_functional_returns_a_named_output_and_mask():
    result = partial_conv_nd(torch.randn(1, 2, 5, 5), torch.randn(3, 2, 3, 3), padding=1)
    assert isinstance(result, PartialConvOutput) and isinstance(result, tuple)
    output, mask = result
    assert output is result.output and mask is result.mask
    assert result.output.shape == (1, 3, 5, 5) and result.mask.shape == (1, 1, 5, 5)


def test_return_mask_is_a_default_each_call_can_override():
    torch.manual_seed(4)
    input, mask = torch.randn(1, 2, 6, 6), (torch.rand(1, 1, 6, 6) > 0.3).float()
    quiet = PartialConv2d(2, 3, 3, padding=1)
    loud = PartialConv2d(2, 3, 3, padding=1, return_mask=True)
    loud.load_state_dict(quiet.state_dict())
    plain = quiet(input, mask)
    assert isinstance(plain, torch.Tensor) and not isinstance(plain, tuple)
    asked = quiet(input, mask, return_mask=True)
    assert isinstance(asked, PartialConvOutput) and torch.equal(asked.output, plain)
    assert isinstance(loud(input, mask), PartialConvOutput)
    declined = loud(input, mask, return_mask=False)
    assert isinstance(declined, torch.Tensor) and torch.equal(declined, plain)
    with pytest.raises(TypeError, match="return_mask"):
        quiet(input, mask, return_mask=1)


# 4. PartialSequential.


def manual(layers, input, mask):
    """Thread the mask by hand: the reference for PartialSequential."""
    for layer in layers:
        if isinstance(layer, PartialConvNd):
            input, mask = layer(input, mask, return_mask=True)
        else:
            input = layer(input)
    return input, mask


@pytest.mark.parametrize("multi_channel", [False, True])
def test_partial_sequential_threads_masks_like_manual_chaining(multi_channel):
    torch.manual_seed(5)
    layers = [
        PartialConv2d(2, 4, 3, padding=1, multi_channel=multi_channel),
        nn.ReLU(),
        nn.Dropout(0.0),
        PartialConv2d(4, 4, 3, padding=1, groups=2, multi_channel=multi_channel),
        nn.GroupNorm(2, 4),
        PartialConv2d(4, 1, 3, padding=1, multi_channel=multi_channel),
    ]
    network = PartialSequential(*layers)
    input = torch.randn(2, 2, 9, 9, requires_grad=True)
    mask = (torch.rand(2, 2 if multi_channel else 1, 9, 9) > 0.4).float()
    result = network(input, mask)
    expected_output, expected_mask = manual(layers, input, mask)
    assert isinstance(result, PartialConvOutput)
    assert torch.equal(result.output, expected_output) and torch.equal(result.mask, expected_mask)
    first = torch.autograd.grad(result.output.sum(), (input, *network.parameters()))
    second = torch.autograd.grad(expected_output.sum(), (input, *network.parameters()))
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    nested = PartialSequential(PartialSequential(*layers[:3]), PartialSequential(*layers[3:]))
    nested_output, nested_mask = nested(input, mask)
    assert torch.equal(nested_output, expected_output) and torch.equal(nested_mask, expected_mask)
    assert isinstance(network[1:], PartialSequential)
    # GroupNorm needs a batch axis, so the unbatched check stops before it.
    single, single_mask = PartialSequential(*layers[:4])(input[0], mask[0])
    batch_of_one, batch_of_one_mask = manual(layers[:4], input[:1], mask[:1])
    assert torch.equal(single, batch_of_one[0]) and torch.equal(single_mask, batch_of_one_mask[0])


def test_partial_sequential_without_a_mask_stays_on_the_mask_free_path(monkeypatch):
    torch.manual_seed(6)
    layers = [
        PartialConv3d(2, 3, 3, padding=1, return_mask=True),  # the chain overrides it
        nn.SiLU(),
        PartialConv3d(3, 2, 3, padding=1),
    ]
    network = PartialSequential(*layers)
    input = torch.randn(1, 2, 5, 6, 7)
    expected = layers[2](layers[1](layers[0](input, return_mask=False)))
    seen = []
    original = functional_module._partial_conv_nd

    def spy(*args, mask=None, need_update, **kwargs):
        seen.append((mask is None, need_update))
        return original(*args, mask=mask, need_update=need_update, **kwargs)

    monkeypatch.setattr("torch_convnd.modules._partial_conv_nd", spy)
    output, mask = network(input)
    assert seen == [(True, False), (True, False)] and torch.equal(output, expected)
    assert mask.shape == (1, 1, 5, 6, 7) and bool((mask == 1).all())


@pytest.mark.parametrize("multi_channel", [False, True])
def test_partial_sequential_reports_the_same_mask_layout_on_every_path(multi_channel):
    """All-valid results take the last partial layer's update layout, as counted ones do."""
    torch.manual_seed(11)
    layers = [
        PartialConv2d(2, 4, 3, padding=1, multi_channel=multi_channel),
        nn.ReLU(),
        PartialConv2d(4, 3, 3, padding=1, multi_channel=multi_channel),
    ]
    input = torch.randn(2, 2, 6, 5)
    output, mask = PartialSequential(*layers)(input)
    expected_output, expected_mask = manual(layers, input, None)
    assert torch.equal(output, expected_output)
    assert mask.shape == expected_mask.shape == (2, 3 if multi_channel else 1, 6, 5)
    assert torch.equal(mask, expected_mask)
    # Batch sizes may pick different CPU kernels, so compare against a batch of one.
    unbatched_output, unbatched_mask = PartialSequential(*layers)(input[0])
    batch_of_one, batch_of_one_mask = manual(layers, input[:1], None)
    assert torch.equal(unbatched_output, batch_of_one[0])
    assert torch.equal(unbatched_mask, batch_of_one_mask[0])


def test_partial_sequential_keeps_masks_that_padding_alone_creates():
    """Padding beyond a kernel's reach empties windows, so the mask must be carried."""
    torch.manual_seed(7)
    layers = [PartialConv2d(1, 2, 3, padding=3), PartialConv2d(2, 1, 3, padding=1)]
    input = torch.randn(1, 1, 6, 6)
    output, mask = PartialSequential(*layers)(input)
    expected_output, expected_mask = manual(layers, input, None)
    assert torch.equal(output, expected_output) and torch.equal(mask, expected_mask)
    hidden, carried = layers[0](input, return_mask=True)
    assert bool((carried == 0).any()), "the outer ring of windows sees only padding"
    assert not torch.allclose(output, layers[1](hidden)), "dropping the mask changes the result"


def test_partial_sequential_refuses_modules_that_would_lose_the_mask():
    layer = PartialConv2d(1, 2, 3, padding=1)
    input, mask = torch.randn(1, 1, 8, 8), (torch.rand(1, 1, 8, 8) > 0.3).float()
    pooled = PartialSequential(layer, nn.MaxPool2d(2), PartialConv2d(2, 1, 3, padding=1))
    with pytest.raises(ValueError, match="while a mask is carried"):
        pooled(input, mask)
    # A nested container without partial layers checks against the carried mask's axes.
    with pytest.raises(ValueError, match="while a mask is carried"):
        PartialSequential(layer, PartialSequential(nn.MaxPool2d(2)))(input, mask)
    with pytest.raises(ValueError, match="while a mask is carried"):
        PartialSequential(layer, nn.Flatten())(input, mask)
    # Without a mask, resampling is fine: every sample stays valid.
    output, final = pooled(input)
    assert output.shape == (1, 1, 4, 4) and final.shape == (1, 1, 4, 4)
    with pytest.raises(ValueError, match="lost the 2 spatial axes"):
        PartialSequential(layer, nn.Flatten())(input)
    # A plain container would drop the mask between its partial layers, with or without
    # a mask so far, so it is refused either way.
    hidden = nn.Sequential(PartialConv2d(2, 2, 3, padding=1))
    for arguments in ((input, mask), (input,)):
        with pytest.raises(TypeError, match="use PartialSequential"):
            PartialSequential(layer, hidden)(*arguments)
    with pytest.raises(NotImplementedError, match="TorchScript"):
        torch.jit.script(PartialSequential(layer))


def test_partial_sequential_validity_checks_for_copying_padding_and_traced_sizes():
    """Copying padding never empties a window; traced sizes conservatively keep the mask."""
    check = functional_module._mask_free_update_is_full
    assert check((5, 6), (3, 3), 1, 1, 1, "reflect")
    assert not check((torch.tensor(5), torch.tensor(6)), (3, 3), 1, 1, 1, "zeros")
    torch.manual_seed(13)
    layer = PartialConv2d(2, 3, 3, padding=1, padding_mode="reflect")
    input = torch.randn(1, 2, 6, 6)
    output, mask = PartialSequential(layer)(input)
    assert torch.equal(output, layer(input)) and bool((mask == 1).all())


def test_partial_sequential_without_partial_layers_passes_the_mask_through():
    input, mask = torch.randn(2, 3, 5, 5), (torch.rand(2, 1, 5, 5) > 0.5).float()
    output, carried = PartialSequential(nn.ReLU())(input, mask)
    assert torch.equal(output, input.relu()) and carried is mask
    output, ones = PartialSequential()(input)
    assert output is input and ones.shape == (2, 1, 5, 5) and bool((ones == 1).all())
    with pytest.raises(ValueError, match="while a mask is carried"):
        PartialSequential(nn.MaxPool2d(2))(input, mask)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("masked", [False, True], ids=["mask_free", "masked"])
@pytest.mark.parametrize("reach", ["within", "beyond"])
def test_partial_sequential_compiles_as_one_graph(reach, masked, dynamic):
    """Padding beyond the kernel's reach makes even a mask-free chain carry a mask."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    torch.manual_seed(8)
    last = {"padding": 2, "dilation": 2} if reach == "within" else {"padding": 3}
    network = PartialSequential(
        PartialConv2d(2, 4, 3, padding=1, multi_channel=True),
        nn.ReLU(),
        PartialConv2d(4, 2, 3, multi_channel=True, **last),
    ).eval()
    graphs = []

    def counting(graph, example_inputs):
        graphs.append(graph)
        return torch._dynamo.lookup_backend("aot_eager")(graph, example_inputs)

    torch._dynamo.reset()
    compiled = torch.compile(network, backend=counting, fullgraph=True, dynamic=dynamic)
    sizes, first = (7, 9, 12), None
    for size in sizes:
        input = torch.randn(1, 2, size, size)
        mask = (torch.rand(1, 2, size, size) > 0.3).float() if masked else None
        expected, actual = network(input, mask), compiled(input, mask)
        torch.testing.assert_close(actual.output, expected.output)
        torch.testing.assert_close(actual.mask, expected.mask)
        first = len(graphs) if first is None else first
    # Validity checks become guards on symbolic sizes, so later sizes reuse the graphs.
    # (Returning the named tuple also traces its _make once, as a separate empty graph.)
    assert len(graphs) - first == (0 if dynamic else len(sizes) - 1)


def test_partial_sequential_exports_to_onnx(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from torch_convnd import export_onnx

    torch.manual_seed(8)
    network = PartialSequential(
        PartialConv2d(2, 4, 3, padding=1, multi_channel=True),
        nn.ReLU(),
        PartialConv2d(4, 2, 3, padding=1, multi_channel=True),
    ).eval()
    input, mask = torch.randn(1, 2, 7, 7), (torch.rand(1, 2, 7, 7) > 0.3).float()
    expected = network(input, mask)
    path = tmp_path / "chain.onnx"
    export_onnx(network, (input, mask), path, external_data=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_inputs()]
    output, final = session.run(None, dict(zip(names, (input.numpy(), mask.numpy()), strict=True)))
    np.testing.assert_allclose(output, expected.output.detach().numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(final, expected.mask.numpy(), atol=0, rtol=0)
