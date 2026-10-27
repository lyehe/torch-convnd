"""Feature detection at PyTorch boundaries; no version-string dispatch."""

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


def is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return compiler.is_compiling()
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and dynamo.is_compiling())


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


def autocast_dtype(device_type: str):
    if HAS_UNIFIED_AUTOCAST:
        return torch.get_autocast_dtype(device_type)
    getter = torch.get_autocast_cpu_dtype if device_type == "cpu" else torch.get_autocast_gpu_dtype
    return getter()


def transforms_active() -> bool:
    functorch_active = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)
    return functorch_active() or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0


def is_onnx_exporting() -> bool:
    onnx = getattr(torch, "onnx", None)
    return bool(onnx is not None and onnx.is_in_onnx_export())
