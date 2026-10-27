"""Explicit backend policy and transform-safe dispatch for eager and scripted calls."""

import threading

import torch
from torch.nn import functional as F

from . import _build
from ._compat import (
    HAS_LIBRARY,
    HAS_UNIFIED_AUTOCAST,
    autocast_dtype,
    is_autocast_enabled,
    is_compiling,
    is_onnx_exporting,
    transforms_active,
)
from ._decomposition import composite_conv, conv4d_taps
from ._layout import restore_native_layout
from ._toolchain import native_unavailable_reason

BACKENDS = {"auto", "native", "torch"}
_COMPATIBLE_LIBRARY = None
_REGISTRATION_LOCK = threading.Lock()


def _transforms_active():
    return transforms_active()


def _portable_kernel(input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
    """Use the efficient 4D decomposition when metadata permits bounded padding."""
    if (
        weight.ndim == 6
        and input.device.type in {"cpu", "cuda", "meta"}
        and padding[0] <= input.shape[2]
        and all(value <= 2**31 - 1 for value in (*stride, *padding, *dilation))
    ):
        return conv4d_taps(input, weight, bias, stride, padding, dilation, groups)
    return composite_conv(input, weight, bias, stride, padding, dilation, groups, workspace_bytes)


def _portable_conv(input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
    """Cast AMP operands once, rather than once per 3D kernel slice."""
    if input.device.type in {"cpu", "cuda"} and is_autocast_enabled(input.device.type):
        dtype = autocast_dtype(input.device.type)
        castable = {torch.float16, torch.bfloat16, torch.float32}
        input = input.to(dtype) if input.dtype in castable else input
        weight = weight.to(dtype) if weight.dtype in castable else weight
        bias = bias.to(dtype) if bias is not None and bias.dtype in castable else bias
        with torch.autocast(device_type=input.device.type, enabled=False):
            return _portable_kernel(
                input, weight, bias, stride, padding, dilation, groups, workspace_bytes
            )
    return _portable_kernel(input, weight, bias, stride, padding, dilation, groups, workspace_bytes)


def real_conv(
    input, weight, bias, stride, padding, dilation, groups, workspace_bytes, backend="auto"
):
    """Choose vendor kernels for ranks 1-3 and native N-D kernels where appropriate.

    ``native`` forces the N-D implementation in ordinary execution. Transforms
    always retain differentiable operations; unsupported devices use the portable
    implementation with ``auto``. This avoids silently losing forward tangents.
    """
    # Early CPU kernels do not implement FP16 convolution or GEMM. Casts remain
    # differentiable, and preserve the user's output/parameter gradient dtype.
    if input.device.type == "cpu" and input.dtype == torch.float16 and not HAS_UNIFIED_AUTOCAST:
        with torch.autocast(device_type="cpu", enabled=False):
            return real_conv(
                input.float(),
                weight.float(),
                None if bias is None else bias.float(),
                stride,
                padding,
                dilation,
                groups,
                workspace_bytes,
                backend,
            ).to(input.dtype)
    rank = weight.ndim - 2
    if backend != "native" and rank <= 3 and not (input.device.type == "mps" and rank == 3):
        operation = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
        return operation(input, weight, bias, stride, padding, dilation, groups)
    native_device = input.device.type == "cpu" or (
        input.device.type == "cuda" and torch.version.hip is None
    )
    # Measured policy: CPU 4D benefits from vendor slices; NVIDIA CUDA benefits
    # from the native kernel for FP32 and mixed precision on the tested workloads.
    prefer_vendor_4d = backend == "auto" and rank == 4 and input.device.type == "cpu"
    if (
        _transforms_active()
        or backend == "torch"
        or prefer_vendor_4d
        or input.device.type == "meta"
        or (backend == "auto" and not native_device)
    ):
        return _portable_conv(
            input, weight, bias, stride, padding, dilation, groups, workspace_bytes
        )
    if not native_device:
        raise RuntimeError("The native backend supports CPU and NVIDIA CUDA; use backend='torch'")
    if backend == "auto" and (
        (_build._LOADED and input.device.type == "cuda" and not _build._WITH_CUDA)
        or (not _build._LOADED and native_unavailable_reason(input.device.type) is not None)
    ):
        return _portable_conv(
            input, weight, bias, stride, padding, dilation, groups, workspace_bytes
        )
    if not _build._LOADED and not is_compiling():
        _build.load_native()
    output = torch.ops.torch_convnd.conv_nd.default(
        input, weight, bias, stride, padding, dilation, groups, workspace_bytes
    )
    return restore_native_layout(output, input, weight)


def _complex_parts(input, weight, bias, parameters):
    """Gauss's three real convolutions, shared by eager and serialized execution."""
    real = real_conv(input.real, weight.real, None, *parameters)
    imaginary = real_conv(input.imag, weight.imag, None, *parameters)
    cross = real_conv(input.real + input.imag, weight.real + weight.imag, None, *parameters)
    real_output, imaginary_output = real - imaginary, cross - real - imaginary
    if bias is not None:
        shape = (1, -1, *((1,) * (input.ndim - 2)))
        real_output = real_output + bias.real.reshape(shape)
        imaginary_output = imaginary_output + bias.imag.reshape(shape)
    return torch.complex(real_output, imaginary_output)


def convolution(
    input, weight, bias, stride, padding, dilation, groups, workspace_bytes, backend="auto"
):
    """Dispatch real or complex operands without duplicating logic in graph adapters."""
    parameters = stride, padding, dilation, groups, workspace_bytes, backend
    if is_onnx_exporting():
        # Legacy exporters execute examples rather than fake kernels. Export the
        # same mathematics without requiring an optional native compiler.
        backend = "torch"
        parameters = stride, padding, dilation, groups, workspace_bytes, backend
    if input.is_complex():
        if input.device.type in {"cpu", "cuda"}:
            with torch.autocast(device_type=input.device.type, enabled=False):
                return _complex_parts(input, weight, bias, parameters)
        return _complex_parts(input, weight, bias, parameters)
    return real_conv(input, weight, bias, *parameters)


def ensure_compatible_operator():
    """Register one composite operator for serialized graphs, without compiling.

    Scripted graphs cannot call the eager Python frontend. The composite
    registration preserves their transform context before reaching an opaque
    native primitive. Its ordinary execution still uses the same backend policy.
    Keep the library alive for the lifetime of the process.
    """
    global _COMPATIBLE_LIBRARY
    with _REGISTRATION_LOCK:
        if _COMPATIBLE_LIBRARY is not None or not HAS_LIBRARY:
            return
        from . import _operators

        _operators.ensure_native_definitions()
        library = _operators._LIBRARY
        library.define(
            "compatible_conv_nd(Tensor input, Tensor weight, Tensor? bias, int[] stride, "
            "int[] padding, int[] dilation, int groups, int workspace_bytes, str backend) -> Tensor"
        )
        library.impl("compatible_conv_nd", convolution, "CompositeImplicitAutograd")
        _COMPATIBLE_LIBRARY = library
