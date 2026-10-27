"""Native schemas and differentiation metadata, independent of compiler availability."""

import threading

import torch

from ._compat import HAS_LIBRARY, HAS_NATIVE_AUTOGRAD

_LIBRARY = None
_LOCK = threading.Lock()


def _load_for(input):
    """Allow a compiled graph's first real call to initialize its native kernels."""
    from ._build import load_native

    if input.device.type not in {"cpu", "cuda"} or (
        input.device.type == "cuda" and torch.version.hip is not None
    ):
        raise RuntimeError("Native operators support CPU and NVIDIA CUDA")
    load_native(with_cuda=True if input.device.type == "cuda" else None)


def _lazy_forward(input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
    _load_for(input)
    return torch.ops.torch_convnd.conv_nd.default(
        input, weight, bias, stride, padding, dilation, groups, workspace_bytes
    )


def _lazy_backward(
    grad_output, input, weight, stride, padding, dilation, groups, workspace_bytes, output_mask
):
    _load_for(input)
    return torch.ops.torch_convnd.conv_nd_backward.default(
        grad_output, input, weight, stride, padding, dilation, groups, workspace_bytes, output_mask
    )


def ensure_native_definitions():
    """Make native operations traceable/exportable before their kernels are compiled."""
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is not None or not HAS_LIBRARY:
            return
        # One owning library also works before FRAGMENT libraries were introduced.
        library = torch.library.Library("torch_convnd", "DEF")
        library.define(
            "conv_nd(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
            "int[] dilation, int groups, int workspace_bytes) -> Tensor"
        )
        library.define(
            "conv_nd_backward(Tensor grad_output, Tensor input, Tensor weight, int[] stride, "
            "int[] padding, int[] dilation, int groups, int workspace_bytes, "
            "bool[3] output_mask) -> (Tensor?, Tensor?, Tensor?)"
        )
        # Fake/Meta calls infer shapes without a compiler. Real compiled/exported
        # calls reach these fallbacks only until the CPU/CUDA kernels are loaded.
        library.impl("conv_nd", _lazy_forward, "CompositeExplicitAutograd")
        library.impl("conv_nd_backward", _lazy_backward, "CompositeExplicitAutograd")
        from ._autograd import register_training

        if HAS_NATIVE_AUTOGRAD:
            register_training()
        _LIBRARY = library
