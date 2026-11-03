"""Feature detection at PyTorch boundaries; no version-string dispatch."""

import contextlib

import torch

_library = getattr(torch, "library", None)
HAS_LIBRARY = callable(getattr(_library, "Library", None))
HAS_NATIVE_AUTOGRAD = all(
    callable(getattr(_library, name, None)) for name in ("register_fake", "register_autograd")
) and callable(getattr(torch, "get_autocast_dtype", None))
HAS_NATIVE_VMAP = callable(getattr(_library, "register_vmap", None))
HAS_UNIFIED_AUTOCAST = callable(getattr(torch, "get_autocast_dtype", None))


def has_fp32_output_gemm() -> bool:
    """Check generated ATen overloads before compiling the optional mixed CUDA path."""
    for name in ("bmm", "baddbmm"):
        operation = getattr(torch.ops.aten, name, None)
        if operation is None or not hasattr(operation, "overloads"):
            return False
        if "dtype_out" not in operation.overloads():
            return False
    return True


# Resolved at import: Dynamo before PyTorch 2.3 cannot compare a module with None.
_COMPILER_IS_COMPILING = callable(getattr(getattr(torch, "compiler", None), "is_compiling", None))
_HAS_DYNAMO = hasattr(torch, "compile")


def is_compiling() -> bool:
    if _COMPILER_IS_COMPILING:
        return torch.compiler.is_compiling()
    if not _HAS_DYNAMO:
        return False
    # PyTorch 2.0-2.2: every compilation has imported Dynamo already.
    from torch import _dynamo

    return _dynamo.is_compiling()


def assume_constant_result(function):
    if not HAS_NATIVE_AUTOGRAD:
        return function
    compiler = getattr(torch, "compiler", None)
    decorator = getattr(compiler, "assume_constant_result", None)
    if decorator is None and hasattr(torch, "compile"):
        from torch import _dynamo

        decorator = getattr(_dynamo, "assume_constant_result", None)
    return decorator(function) if decorator is not None else function


def is_autocast_enabled(device_type: str) -> bool:
    if HAS_UNIFIED_AUTOCAST:
        return torch.is_autocast_enabled(device_type)
    if device_type == "cpu":
        return torch.is_autocast_cpu_enabled()
    return torch.is_autocast_enabled() if device_type == "cuda" else False


def plain_tensor(tensor) -> bool:
    """A real eager tensor: no subclass, no Python dispatch or function mode, no transform.

    Values derived under fake, meta-tracing or other modes, or inside functorch
    transforms (whose wrapped tensors look like plain ones), must not outlive them.
    """
    dispatch = getattr(torch._C, "_len_torch_dispatch_stack", None)
    function = getattr(torch._C, "_is_torch_function_mode_enabled", None)
    return (
        type(tensor) is torch.Tensor
        and (dispatch is None or dispatch() == 0)
        and (function is None or not function())
        and not transforms_active()
    )


def current_stream_id(device: torch.device):
    """A hashable identity of the current CUDA stream on this device; None elsewhere."""
    if device.type != "cuda":
        return None
    raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)  # avoids building a Stream
    return raw(device.index) if raw is not None else torch.cuda.current_stream(device).cuda_stream


def is_capturing(device: torch.device) -> bool:
    """Whether CUDA graph capture is recording on this device's current stream."""
    capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return device.type == "cuda" and capturing is not None and capturing()


@assume_constant_result
def _autocast_probe(device_type: str) -> bool:
    amp = getattr(torch, "amp", None)
    probe = getattr(getattr(amp, "autocast_mode", None), "is_autocast_available", None)
    return HAS_UNIFIED_AUTOCAST and probe is not None and probe(device_type)


def autocast_available(device_type: str) -> bool:
    """Whether autocast supports this device type; probed only beyond CPU and CUDA."""
    return device_type in {"cpu", "cuda"} or _autocast_probe(device_type)


def autocast_disabled(device_type: str):
    """Leave an active autocast region on this device type; a no-op context otherwise."""
    if autocast_available(device_type) and is_autocast_enabled(device_type):
        return torch.autocast(device_type, enabled=False)
    return contextlib.nullcontext()


def autocast_dtype(device_type: str):
    if HAS_UNIFIED_AUTOCAST:
        return torch.get_autocast_dtype(device_type)
    getter = torch.get_autocast_cpu_dtype if device_type == "cpu" else torch.get_autocast_gpu_dtype
    return getter()


def transforms_active() -> bool:
    functorch_active = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)
    return functorch_active() or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0


def is_onnx_exporting() -> bool:
    """Whether the legacy, tracing exporter runs; Dynamo-based export never sets this."""
    if is_compiling():
        # Dynamo before PyTorch 2.3 cannot compare the torch.onnx module with None.
        return False
    onnx = getattr(torch, "onnx", None)
    return bool(onnx is not None and onnx.is_in_onnx_export())
