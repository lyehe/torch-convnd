"""Generated geometry checks against an independent tap-by-tap reference."""

import itertools
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torch_convnd import conv_nd


@st.composite
def geometries(draw):
    rank = draw(st.integers(1, 5))
    groups = draw(st.integers(1, 2))
    channels = groups * draw(st.integers(1, 2))
    output_channels = groups * draw(st.integers(1, 2))
    spatial = tuple(draw(st.integers(3, 5)) for _ in range(rank))
    kernel = tuple(draw(st.integers(1, 2)) for _ in range(rank))
    stride = tuple(draw(st.integers(1, 2)) for _ in range(rank))
    dilation = tuple(draw(st.integers(1, 2)) for _ in range(rank))
    padding = tuple(draw(st.integers(0, 1)) for _ in range(rank))
    return rank, groups, channels, output_channels, spatial, kernel, stride, dilation, padding


def independent_convolution(input, weight, bias, stride, padding, dilation, groups):
    """Contract each tap separately; no package code or vendor convolution is used."""
    padded = torch.nn.functional.pad(
        input, tuple(p for pad in reversed(padding) for p in (pad, pad))
    )
    output_shape = tuple(
        (size + 2 * pad - dil * (tap - 1) - 1) // step + 1
        for size, pad, dil, tap, step in zip(
            input.shape[2:], padding, dilation, weight.shape[2:], stride, strict=True
        )
    )
    result = None
    for tap in itertools.product(*(range(size) for size in weight.shape[2:])):
        slices = tuple(
            slice(index * dil, index * dil + extent * step, step)
            for index, dil, extent, step in zip(tap, dilation, output_shape, stride, strict=True)
        )
        patch = padded[(slice(None), slice(None), *slices)].reshape(
            input.shape[0], groups, input.shape[1] // groups, -1
        )
        coefficient = weight[(slice(None), slice(None), *tap)].reshape(
            groups, weight.shape[0] // groups, input.shape[1] // groups
        )
        term = torch.einsum("goc,bgcp->bgop", coefficient, patch)
        result = term if result is None else result + term
    output = result.reshape(input.shape[0], weight.shape[0], *output_shape)
    return output + bias.reshape(1, -1, *((1,) * len(output_shape)))


def check_geometry(geometry, backend):
    rank, groups, channels, output_channels, spatial, kernel, stride, dilation, padding = geometry
    generator = torch.Generator().manual_seed(604 + rank)
    input = torch.randn((1, channels, *spatial), generator=generator, dtype=torch.float64)
    weight = torch.randn(
        (output_channels, channels // groups, *kernel), generator=generator, dtype=torch.float64
    )
    bias = torch.randn(output_channels, generator=generator, dtype=torch.float64)
    tensors = tuple(value.requires_grad_() for value in (input, weight, bias))
    # Deliberately force several tiles, including a short last tile on many shapes.
    budget = max(1, channels * math.prod(kernel)) * 8 * 3
    actual = conv_nd(
        *tensors, stride, padding, dilation, groups, backend=backend, workspace_bytes=budget
    )
    expected = independent_convolution(*tensors, stride, padding, dilation, groups)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    first = torch.autograd.grad(actual, tensors, cotangent)
    second = torch.autograd.grad(expected, tensors, cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target, atol=1e-9, rtol=1e-9)


@pytest.mark.portable
@settings(max_examples=24, deadline=None, derandomize=True)
@given(geometries())
def test_portable_generated_geometry(geometry):
    check_geometry(geometry, "torch")


@pytest.mark.native
@settings(max_examples=24, deadline=None, derandomize=True)
@given(geometries())
def test_native_generated_geometry(geometry):
    check_geometry(geometry, "native")
