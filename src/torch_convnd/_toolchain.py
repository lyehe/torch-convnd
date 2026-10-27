"""Read-only, cached prerequisite checks; never compile or mutate the environment."""

import importlib.util
import os
import shutil
import sys
from functools import cache

import torch

from ._compat import HAS_NATIVE_AUTOGRAD, assume_constant_result


def msvc_environment():
    """Locate the MSVC helper across PyTorch/setuptools layouts."""
    try:
        from torch.utils.cpp_extension import _get_vc_env
    except ImportError:
        from setuptools._distutils._msvccompiler import _get_vc_env
    return _get_vc_env("x86_amd64")


@assume_constant_result
@cache
def native_unavailable_reason(device_type: str) -> str | None:
    """Return a missing prerequisite, or None when a native build can be attempted.

    Toolchain configuration is treated as constant for a process. This also lets
    Dynamo select a portable graph without tracing filesystem/compiler probes.
    A successful probe is not a promise that a compiler/toolkit build will succeed.
    """
    if not HAS_NATIVE_AUTOGRAD:
        return "This PyTorch lacks the native custom-op/autocast APIs; use backend='torch'."
    if device_type not in {"cpu", "cuda"}:
        return "Native kernels support CPU and NVIDIA CUDA; use backend='torch'."
    if device_type == "cuda" and (torch.version.cuda is None or torch.version.hip is not None):
        return "Native CUDA requires an NVIDIA CUDA build of PyTorch."
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return "Native build dependencies are missing; install torch-convnd[native]."
    if device_type == "cuda" and (
        CUDA_HOME is None
        or not (
            os.path.isfile(
                os.path.join(CUDA_HOME, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
            )
        )
    ):
        return "CUDA toolkit/nvcc is missing; install a toolkit compatible with PyTorch."
    if importlib.util.find_spec("ninja") is None and shutil.which("ninja") is None:
        return "Ninja is missing; install torch-convnd[native]."
    compiler = os.environ.get("CXX", "cl" if sys.platform == "win32" else "c++")
    if shutil.which(compiler) is not None:
        return None
    if sys.platform == "win32" and "CXX" not in os.environ:
        try:
            environment = msvc_environment()
            path = environment.get("PATH", environment.get("path", ""))
            if shutil.which("cl", path=path) is not None:
                return None
        except Exception as error:
            # A failed metadata probe must not prevent portable execution.
            return f"MSVC compiler environment is unavailable: {error}"
    return f"C++ compiler {compiler!r} is unavailable; install a C++17 compiler."
