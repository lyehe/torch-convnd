"""Compiler-free runtime diagnostics suitable for application support reports."""

import argparse
import json
import platform
from typing import Any

import torch

from . import _build
from ._compat import HAS_LIBRARY, HAS_NATIVE_AUTOGRAD, HAS_NATIVE_VMAP, has_fp32_output_gemm
from ._toolchain import native_unavailable_reason
from ._version import __version__


def get_runtime_info() -> dict[str, Any]:
    """Describe versions and backend prerequisites without compiling or allocating GPU tensors."""
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    cpu_reason = native_unavailable_reason("cpu")
    cuda_reason = native_unavailable_reason("cuda") if cuda else "CUDA runtime is unavailable."
    return {
        "torch_convnd": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "rocm_runtime": torch.version.hip,
        "cuda_available": cuda,
        "mps_available": bool(mps and mps.is_available()),
        "native_loaded": _build._LOADED,
        "native_cuda_loaded": _build._WITH_CUDA,
        "features": {
            "custom_operator_library": HAS_LIBRARY,
            "native_autograd": HAS_NATIVE_AUTOGRAD,
            "native_vmap": HAS_NATIVE_VMAP,
            "fp32_output_gemm": has_fp32_output_gemm(),
            "torch_func": hasattr(torch, "func"),
            "torch_compile": hasattr(torch, "compile"),
        },
        "native_cpu_unavailable_reason": cpu_reason,
        "native_cuda_unavailable_reason": cuda_reason,
        "portable_backend": "Available where the required PyTorch operations are supported.",
    }


def main() -> None:
    """Print a JSON report; this command never builds the optional addon."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit compact machine-readable JSON")
    args = parser.parse_args()
    print(json.dumps(get_runtime_info(), indent=None if args.json else 2))


if __name__ == "__main__":
    main()
