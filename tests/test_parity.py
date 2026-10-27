"""Behavioral parity against torch.nn.Conv2d and Conv3d on CPU and CUDA."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import CUDA_HOME

from torch_convnd import Conv2d, Conv3d, Conv4d, ConvNd, conv_nd

CUDA = CUDA_HOME is not None and torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="no CUDA"))]
FACTORIES = {2: Conv2d, 3: Conv3d}
MEASUREMENTS = []


@pytest.fixture(scope="module", autouse=True)
def numerical_measurements(request):
    """Retain observed numerical errors alongside the test result archive."""
    MEASUREMENTS.clear()
    yield
    junit = request.config.getoption("xmlpath")
    destination = (
        Path(junit).with_suffix(".measurements.json")
        if junit
        else Path(__file__).resolve().parents[1] / ".artifacts" / "parity_measurements.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(MEASUREMENTS, indent=2) + "\n", encoding="utf-8")


def tolerances(dtype):
    """Require tight full-precision agreement and bounded low-precision rounding error."""
    return {
        torch.float64: (1e-9, 1e-9, 1e-10),
        torch.complex128: (2e-9, 2e-9, 1e-10),
        torch.float32: (2e-5, 2e-5, 1e-5),
        torch.complex64: (5e-5, 5e-5, 2e-5),
        torch.float16: (0.005, 0.005, 0.004),
        torch.bfloat16: (0.05, 0.03, 0.02),
        torch.complex32: (0.02, 0.02, 0.012),
    }[dtype]


def assert_numerical(actual, expected, dtype=None):
    """Check elementwise agreement and whole-tensor relative L2 error."""
    atol, rtol, relative = tolerances(dtype or actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if actual.numel():
        target = expected.to(torch.complex128 if expected.is_complex() else torch.float64)
        observed = actual.to(target.dtype)
        error = (observed - target).norm() / target.norm().clamp_min(1e-20)
        MEASUREMENTS.append(
            {
                "dtype": str(dtype or actual.dtype),
                "device": str(actual.device),
                "shape": list(actual.shape),
                "relative_l2_error": error.item(),
                "max_absolute_error": (observed - target).abs().max().item(),
                "allowed_relative_l2_error": relative,
            }
        )
        assert error.item() <= relative


def assert_module_parity(ours, reference, input, *, dtype=None):
    """Compare outputs and input/weight/bias vector-Jacobian products."""
    actual, expected = ours(input), reference(input)
    assert_numerical(actual, expected, dtype)
    cotangent_dtype = torch.complex64 if actual.dtype == torch.complex32 else actual.dtype
    cotangent = (
        torch.randn_like(actual, dtype=cotangent_dtype) / math.sqrt(max(actual.numel(), 1))
    ).to(actual.dtype)
    observed_parameters = (input, *ours.parameters())
    reference_parameters = (input, *reference.parameters())
    observed = torch.autograd.grad(actual, observed_parameters, cotangent)
    target = torch.autograd.grad(expected, reference_parameters, cotangent)
    for first, second in zip(observed, target, strict=True):
        assert_numerical(first, second, dtype or actual.dtype)


def pair(
    rank, device, *, dtype=torch.float64, in_channels=4, out_channels=6, backend="auto", **kwargs
):
    """Create independent modules with identical initialized parameters."""
    ours = FACTORIES[rank](
        in_channels, out_channels, device=device, dtype=dtype, backend=backend, **kwargs
    )
    reference = getattr(torch.nn, f"Conv{rank}d")(
        in_channels, out_channels, device=device, dtype=dtype, **kwargs
    )
    reference.load_state_dict(ours.state_dict())
    return ours, reference


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["basic", "stride_dilation", "groups", "depthwise", "no_bias"])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_geometry_and_batch_parity(rank, device, case, unbatched, backend):
    """Standard, grouped/depthwise, strided/dilated and bias-free variants agree."""
    torch.manual_seed(123)
    kwargs = {"kernel_size": (2,) + (3,) * (rank - 1), "padding": 1}
    if case == "stride_dilation":
        kwargs.update(stride=(2,) + (1,) * (rank - 1), dilation=(1,) + (2,) * (rank - 1))
    if case == "groups":
        kwargs["groups"] = 2
    if case == "depthwise":
        kwargs["groups"] = 4
    if case == "no_bias":
        kwargs["bias"] = False
    ours, reference = pair(
        rank, device, out_channels=8 if case == "depthwise" else 6, backend=backend, **kwargs
    )
    shape = (4,) + (6,) * rank if unbatched else (2, 4) + (6,) * rank
    input = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
    assert_module_parity(ours, reference, input)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("padding", ["same", "valid", 1])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_padding_parity(rank, device, mode, padding, unbatched, backend):
    """All padding modes and asymmetric 'same' behavior match both reference layers."""
    ours, reference = pair(
        rank,
        device,
        kernel_size=(2,) + (3,) * (rank - 1),
        padding=padding,
        padding_mode=mode,
        dilation=(1,) + (2,) * (rank - 1),
        backend=backend,
    )
    shape = (4,) + (6,) * rank if unbatched else (1, 4) + (6,) * rank
    input = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
    assert_module_parity(ours, reference, input)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex32,
        torch.complex64,
        torch.complex128,
    ],
)
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_dtype_forward_backward_parity(rank, device, dtype, backend):
    """Native mixed precision and real-convolution complex arithmetic match Conv2d/3d."""
    torch.manual_seed(210)
    ours, reference = pair(
        rank,
        device,
        kernel_size=2,
        padding=1,
        dtype=dtype,
        in_channels=2,
        out_channels=4,
        groups=2,
        backend=backend,
    )
    input = torch.randn((1, 2) + (4,) * rank, device=device, dtype=dtype, requires_grad=True)
    assert_module_parity(ours, reference, input)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_autocast_parity(rank, device, dtype, backend):
    """AMP honors the selected low precision while returning FP32 parameter gradients."""
    ours, reference = pair(
        rank, device, kernel_size=2, padding="same", dtype=torch.float32, backend=backend
    )
    input = torch.randn((1, 4) + (4,) * rank, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        actual, expected = ours(input), reference(input)
    assert actual.dtype == expected.dtype == dtype
    assert_numerical(actual, expected)
    cotangent = torch.randn_like(actual) / math.sqrt(actual.numel())
    first = torch.autograd.grad(actual, (input, *ours.parameters()), cotangent)
    second = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for observed, target in zip(first, second, strict=True):
        assert observed.dtype == target.dtype == torch.float32
        assert_numerical(observed, target, dtype)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_amp_gradscaler_training_parity(rank, device):
    """Scaled FP16 training updates agree with the stock convolution for two steps."""
    ours, reference = pair(rank, device, kernel_size=3, padding=1, dtype=torch.float32)
    input = torch.randn((1, 4) + (4,) * rank, device=device)
    optimizers = [torch.optim.SGD(layer.parameters(), lr=0.01) for layer in (ours, reference)]
    scalers = [torch.amp.GradScaler(device, init_scale=128) for _ in range(2)]
    for _ in range(2):
        for layer, optimizer, scaler in zip((ours, reference), optimizers, scalers, strict=True):
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                loss = layer(input).float().square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    for first, second in zip(ours.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_channels_last_parity(rank, device):
    """Non-default memory layouts preserve values, gradients and the output format."""
    layout = torch.channels_last if rank == 2 else torch.channels_last_3d
    ours, reference = pair(rank, device, kernel_size=3, padding=1)
    ours, reference = ours.to(memory_format=layout), reference.to(memory_format=layout)
    input = torch.randn((2, 4) + (4,) * rank, device=device, dtype=torch.float64)
    input = input.to(memory_format=layout).requires_grad_()
    assert_module_parity(ours, reference, input)
    assert ours(input).stride() == reference(input).stride()


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_second_derivative_parity(rank, device):
    """Second derivatives of a nonlinear loss agree for input, weight and bias."""
    ours, reference = pair(rank, device, kernel_size=2, in_channels=2, out_channels=2)
    input = torch.randn(
        (1, 2) + (3,) * rank, device=device, dtype=torch.float64, requires_grad=True
    )
    parameters = (input, *ours.parameters())
    targets = (input, *reference.parameters())
    first = torch.autograd.grad(ours(input).sin().square().sum(), parameters, create_graph=True)
    second = torch.autograd.grad(reference(input).sin().square().sum(), targets, create_graph=True)
    directions = [torch.randn_like(value) for value in first]
    observed = torch.autograd.grad(
        sum((a * d).sum() for a, d in zip(first, directions, strict=True)), parameters
    )
    expected = torch.autograd.grad(
        sum((a * d).sum() for a, d in zip(second, directions, strict=True)), targets
    )
    for actual, target in zip(observed, expected, strict=True):
        assert_numerical(actual, target)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_gradgrad_finite_differences(rank, device, deterministic_algorithms):
    """Finite differences provide an independent check of registered derivative formulas."""
    input = torch.randn(
        (1, 1) + (3,) * rank, device=device, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(
        (1, 1) + (2,) * rank, device=device, dtype=torch.float64, requires_grad=True
    )
    bias = torch.randn(1, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda x, w, b: conv_nd(x, w, b, padding=1), (input, weight, bias), fast_mode=True
    )


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "transform",
    [
        "vmap_input",
        "vmap_parameters",
        "nested_vmap",
        "jvp",
        "vjp",
        "grad",
        "jacrev",
        "jacfwd",
        "hessian",
        "linearize",
    ],
)
def test_torch_func_parity(rank, device, transform):
    """Vectorization, forward/reverse differentiation and Hessians match stock operations."""
    torch.manual_seed(52)
    input = torch.randn((1, 1) + (3,) * rank, device=device, dtype=torch.float64)
    weight = torch.randn((1, 1) + (2,) * rank, device=device, dtype=torch.float64)
    bias = torch.randn(1, device=device, dtype=torch.float64)
    stock = getattr(F, f"conv{rank}d")

    def evaluate(function):
        if transform == "vmap_input":
            return torch.vmap(lambda a: function(a, weight, bias))(
                input.unsqueeze(0).expand(3, *input.shape)
            )
        if transform == "vmap_parameters":
            return torch.vmap(lambda a, b: function(input, a, b))(
                weight.unsqueeze(0).expand(3, *weight.shape), bias.unsqueeze(0).expand(3, 1)
            )
        if transform == "nested_vmap":
            return torch.vmap(torch.vmap(lambda a: function(a, weight, bias)))(
                input.expand(2, 3, *input.shape)
            )
        if transform == "jvp":
            return torch.func.jvp(
                function,
                (input, weight, bias),
                (torch.ones_like(input), torch.ones_like(weight), torch.ones_like(bias)),
            )
        if transform == "vjp":
            output, backward = torch.func.vjp(function, input, weight, bias)
            return backward(torch.ones_like(output))
        if transform == "linearize":
            output, linear = torch.func.linearize(lambda a: function(a, weight, bias), input)
            return output, linear(torch.ones_like(input))

        def loss(a):
            return function(a, weight, bias).sin().square().sum()

        if transform == "grad":
            return torch.func.grad(loss)(input)
        if transform == "hessian":
            return torch.func.hessian(loss)(input)
        return getattr(torch.func, transform)(lambda a: function(a, weight, bias))(input)

    actual, expected = evaluate(conv_nd), evaluate(stock)
    if isinstance(actual, tuple):
        for first, second in zip(actual, expected, strict=True):
            assert_numerical(first, second)
    else:
        assert_numerical(actual, expected)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_low_level_forward_ad_parity(rank, device):
    """The forward_ad dual-tensor API propagates tangents without torch.func."""
    input = torch.randn((1, 1) + (3,) * rank, device=device, dtype=torch.float64)
    weight = torch.randn((1, 1) + (2,) * rank, device=device, dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(input, torch.ones_like(input))
        actual = torch.autograd.forward_ad.unpack_dual(conv_nd(dual, weight))
        expected = torch.autograd.forward_ad.unpack_dual(getattr(F, f"conv{rank}d")(dual, weight))
        assert_numerical(actual.primal, expected.primal)
        assert_numerical(actual.tangent, expected.tangent)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["plain", "padding", "amp", "func_grad"])
def test_compiled_aot_parity(rank, device, case):
    """Compiled execution agrees with both eager implementations, including transforms."""
    # Each matrix case is a separate program; do not exhaust a shared frame's cache limit.
    torch._dynamo.reset()
    kwargs = {"kernel_size": 2, "padding": "same"}
    if case == "padding":
        kwargs["padding_mode"] = "circular"
    ours, reference = pair(rank, device, dtype=torch.float32, **kwargs)
    input = torch.randn((1, 4) + (4,) * rank, device=device, requires_grad=True)
    if case == "func_grad":
        actual = torch.compile(
            torch.func.grad(lambda x: ours(x).square().sum()), backend="aot_eager", fullgraph=True
        )(input)
        expected = torch.func.grad(lambda x: reference(x).square().sum())(input)
        assert_numerical(actual, expected)
        return
    compiled = torch.compile(ours, backend="aot_eager", fullgraph=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=case == "amp"):
        actual, expected = compiled(input), reference(input)
    assert_numerical(actual, expected)
    cotangent = torch.randn_like(actual) / math.sqrt(actual.numel())
    observed = torch.autograd.grad(actual, (input, *ours.parameters()), cotangent)
    target = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for first, second in zip(observed, target, strict=True):
        assert_numerical(first, second, actual.dtype)


@pytest.mark.parametrize("rank", [2, 3])
def test_constructor_and_state_dict_parity(rank):
    """Standard positional arguments, initialization, attributes and state names match."""
    arguments = (4, 6, 3, 1, 1, 1, 2, False, "replicate", None, torch.float64)
    torch.manual_seed(42)
    ours = FACTORIES[rank](*arguments)
    torch.manual_seed(42)
    reference = getattr(torch.nn, f"Conv{rank}d")(*arguments)
    torch.testing.assert_close(ours.weight, reference.weight, atol=0, rtol=0)
    assert ours.bias is None and "bias" in ours._parameters
    assert ours.state_dict().keys() == reference.state_dict().keys()
    for name in [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "transposed",
        "output_padding",
    ]:
        assert getattr(ours, name) == getattr(reference, name)
    reference.load_state_dict(ours.state_dict())
    ours.load_state_dict(reference.state_dict())


@pytest.mark.parametrize("rank", [2, 3])
def test_meta_shape_parity(rank):
    """Shape inference works without allocating data on a native device."""
    ours, reference = pair(rank, "meta", kernel_size=2, padding="same")
    input = torch.empty((2, 4) + (5,) * rank, device="meta", dtype=torch.float64)
    assert ours(input).shape == reference(input).shape


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"padding": "invalid"},
        {"padding": "same", "stride": 2},
        {"padding_mode": "invalid"},
        {"groups": 3},
    ],
)
def test_constructor_rejection_parity(rank, kwargs):
    """Invalid public configurations are rejected by both module families."""
    for factory in [FACTORIES[rank], getattr(torch.nn, f"Conv{rank}d")]:
        with pytest.raises(ValueError):
            factory(4, 6, 3, **kwargs)


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_four_dimensional_padding_and_transforms(mode):
    """Added features operate over all four spatial axes, including asymmetric padding."""
    from test_convnd import reference_nd

    layer = Conv4d(2, 3, (2, 3, 2, 3), padding="same", padding_mode=mode, dtype=torch.float64)
    input = torch.randn(1, 2, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    pairs = [(0, 1), (1, 1), (0, 1), (1, 1)]
    padded = input
    # Use one-axis F.pad on a temporary 3D view as an independent rank-4 reference.
    for axis, (left, right) in enumerate(pairs, start=2):
        moved = padded.movedim(axis, -1)
        shape = moved.shape
        frames = moved.reshape(-1, 1, shape[-1])
        frames = F.pad(frames, (left, right), mode="constant" if mode == "zeros" else mode)
        padded = frames.reshape(*shape[:-1], shape[-1] + left + right).movedim(-1, axis)
    expected = reference_nd(padded, layer.weight, layer.bias)
    actual = layer(input)
    assert_numerical(actual, expected)
    cotangent = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, (input, *layer.parameters()), cotangent)
    for observed, target in zip(first, second, strict=True):
        assert_numerical(observed, target)


def test_generic_module_unbatched_four_dimensional():
    """The generic and fixed-rank APIs share newly supported unbatched semantics."""
    layer = ConvNd(2, 3, 3, dimensions=4, padding="same", dtype=torch.float64)
    input = torch.randn(2, 3, 3, 3, 3, dtype=torch.float64)
    torch.testing.assert_close(layer(input), layer(input.unsqueeze(0)).squeeze(0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.complex128]
)
def test_four_dimensional_dtype_independent_reference(device, dtype):
    """New dtypes agree with an independent high-precision four-axis convolution."""
    from test_convnd import reference_nd

    layer = Conv4d(2, 3, 2, device=device, dtype=dtype)
    input = torch.randn(1, 2, 3, 3, 3, 3, device=device, dtype=dtype, requires_grad=True)
    high = torch.complex128 if dtype.is_complex else torch.float64
    reference_tensors = tuple(
        value.detach().to(high).requires_grad_() for value in (input, layer.weight, layer.bias)
    )
    actual = layer(input)
    expected = reference_nd(*reference_tensors)
    assert_numerical(actual, expected.to(dtype))
    cotangent = torch.randn_like(actual) / math.sqrt(actual.numel())
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, reference_tensors, cotangent.to(high))
    for observed, target in zip(first, second, strict=True):
        assert_numerical(observed, target.to(dtype))


@pytest.mark.parametrize("device", DEVICES)
def test_four_dimensional_forward_ad(device):
    """The transform decomposition preserves all four independently weighted axes."""
    from test_convnd import reference_nd

    input = torch.randn(1, 2, 3, 3, 3, 3, device=device, dtype=torch.float64)
    weight = torch.randn(3, 2, 2, 2, 2, 2, device=device, dtype=torch.float64)
    arguments = (input, weight)
    tangents = tuple(torch.randn_like(value) for value in arguments)
    actual = torch.func.jvp(conv_nd, arguments, tangents)
    expected = torch.func.jvp(reference_nd, arguments, tangents)
    for observed, target in zip(actual, expected, strict=True):
        assert_numerical(observed, target)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_functional_call_per_sample_parameter_gradients(rank, device):
    """The common vmap(grad(functional_call)) training pattern matches stock layers."""
    ours, reference = pair(rank, device, kernel_size=2, in_channels=2, out_channels=3)
    input = torch.randn((3, 2) + (3,) * rank, device=device, dtype=torch.float64)

    def evaluate(layer):
        parameters = dict(layer.named_parameters())

        def loss(values, sample):
            return torch.func.functional_call(layer, values, (sample,)).square().mean()

        return torch.vmap(torch.func.grad(loss), in_dims=(None, 0))(parameters, input)

    actual, expected = evaluate(ours), evaluate(reference)
    for key in actual:
        assert_numerical(actual[key], expected[key])


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mapped", ["input", "weight", "all"])
def test_raw_operator_vmap_rule(rank, device, mapped):
    """The registered raw-operator batching rule also agrees with explicit application."""
    input = torch.randn((3, 1, 2) + (3,) * rank, device=device, dtype=torch.float64)
    weight = torch.randn((3, 3, 2) + (2,) * rank, device=device, dtype=torch.float64)
    bias = torch.randn(3, 3, device=device, dtype=torch.float64)

    def operation(x, w, b):
        return torch.ops.torch_convnd.conv_nd.default(
            x, w, b, [1] * rank, [0] * rank, [1] * rank, 1, 65536
        )

    arguments = (
        (input, weight[0], bias[0])
        if mapped == "input"
        else ((input[0], weight, bias[0]) if mapped == "weight" else (input, weight, bias))
    )
    dimensions = (
        (0, None, None)
        if mapped == "input"
        else ((None, 0, None) if mapped == "weight" else (0, 0, 0))
    )
    actual = torch.vmap(operation, in_dims=dimensions)(*arguments)
    expected = torch.stack(
        [
            operation(
                *(
                    value[index] if dim is not None else value
                    for value, dim in zip(arguments, dimensions, strict=True)
                )
            )
            for index in range(3)
        ]
    )
    assert_numerical(actual, expected)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_complex_gradgrad_finite_differences(rank, device, deterministic_algorithms):
    """Complex double backward respects PyTorch's conjugate-gradient convention."""
    input = torch.randn(
        (1, 1) + (2,) * rank, device=device, dtype=torch.complex128, requires_grad=True
    )
    weight = torch.randn(
        (1, 1) + (1,) * rank, device=device, dtype=torch.complex128, requires_grad=True
    )
    assert torch.autograd.gradgradcheck(conv_nd, (input, weight), fast_mode=True)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("device", DEVICES)
def test_empty_vmap_public_and_raw(rank, device):
    """Zero-sized mapping dimensions retain correct output shapes on both APIs."""
    input = torch.randn((0, 1, 2) + (3,) * rank, device=device)
    weight = torch.randn((3, 2) + (2,) * rank, device=device)
    expected = torch.vmap(lambda a: getattr(F, f"conv{rank}d")(a, weight))(input)
    public = torch.vmap(lambda a: conv_nd(a, weight))(input)
    raw = torch.vmap(
        lambda a: torch.ops.torch_convnd.conv_nd.default(
            a, weight, None, [1] * rank, [0] * rank, [1] * rank, 1, 65536
        )
    )(input)
    logical_shape = (0, 1, 3) + (2,) * rank
    assert public.shape == expected.shape
    assert raw.shape == logical_shape
    # PyTorch 2.10 collapses the inner batch size to zero for an empty vmap.
    # Preserve the mathematical shape and explicitly recognize that reference quirk.
    assert expected.shape in {torch.Size(logical_shape), torch.Size((0, 0, 3) + (2,) * rank)}
    mapped_weight = weight.unsqueeze(0).expand(0, *weight.shape)
    shared_input = torch.randn((1, 2) + (3,) * rank, device=device)
    raw_weight = torch.vmap(
        lambda w: torch.ops.torch_convnd.conv_nd.default(
            shared_input, w, None, [1] * rank, [0] * rank, [1] * rank, 1, 65536
        )
    )(mapped_weight)
    assert raw_weight.shape == logical_shape
