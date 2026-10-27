"""Lazy compilation and native extension lifecycle."""

import hashlib
import os
import shutil
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

from ._compat import HAS_NATIVE_AUTOGRAD, has_fp32_output_gemm

_BUILD_LOCK = threading.Lock()
_LOADED = False
_WITH_CUDA = False


def _native_build_name(source_dir: Path, with_cuda: bool) -> str:
    """Separate immutable cache entries across sources, installations and Torch ABIs.

    Reusing one DLL path makes a second Windows process unable to rebuild while
    another process holds the old library. Include source paths too: identical
    sources installed at another path produce different Ninja build commands.
    """
    digest = hashlib.sha256()
    configuration = (
        str(source_dir.resolve()),
        sys.prefix,
        torch.__version__,
        str(with_cuda),
        os.environ.get("CXX", ""),
        os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
    )
    for value in configuration:
        digest.update(value.encode("utf-8") + b"\0")
    for path in sorted(source_dir.iterdir()):
        if path.suffix in {".cpp", ".cu", ".h"}:
            digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
    kind = "cuda" if with_cuda else "cpu"
    return f"torch_convnd_{kind}_{digest.hexdigest()[:16]}"


@contextmanager
def _build_environment(with_cuda: bool):
    # Locate a venv-installed ninja even when Python was invoked without activation.
    changes = {}
    if sys.platform == "win32" and shutil.which("cl") is None:
        from ._toolchain import msvc_environment

        changes.update({key.upper(): value for key, value in msvc_environment().items()})
        changes["DISTUTILS_USE_SDK"] = "1"
    changes["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + changes.get("PATH", os.environ["PATH"])
    )
    if "MAX_JOBS" not in os.environ:
        changes["MAX_JOBS"] = "2"
    if with_cuda and "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        capabilities = {
            torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
        }
        changes["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{a}.{b}" for a, b in sorted(capabilities))
    previous = {key: os.environ.get(key) for key in changes}
    os.environ.update(changes)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_native(*, with_cuda: bool | None = None, verbose: bool = False) -> None:
    """Build and cache the optional addon using PyTorch's C++ extension tooling.

    A C++ compiler and ninja are required. CUDA builds additionally require nvcc
    matching the installed PyTorch build. Set ``with_cuda=False`` for a CPU-only
    build. Build selection cannot be changed after loading in the same process.
    Importing :mod:`torch_convnd` alone never invokes the compiler.
    """
    if not HAS_NATIVE_AUTOGRAD:
        raise RuntimeError(
            "Native custom-op APIs are unavailable in this PyTorch; use backend='torch'"
        )
    from torch.utils.cpp_extension import CUDA_HOME, load

    from ._operators import ensure_native_definitions

    ensure_native_definitions()

    global _LOADED, _WITH_CUDA
    requested_cuda = (
        torch.version.cuda is not None and CUDA_HOME is not None and torch.cuda.is_available()
        if with_cuda is None
        else with_cuda
    )
    with _BUILD_LOCK:
        if _LOADED:
            if requested_cuda and not _WITH_CUDA:
                message = "CPU-only addon already loaded; use a fresh process for CUDA"
                raise RuntimeError(message)
            return
        from ._toolchain import native_unavailable_reason

        reason = native_unavailable_reason("cuda" if requested_cuda else "cpu")
        if reason is not None:
            raise RuntimeError(f"Cannot load native torch-convnd kernels: {reason}")
        source_dir = Path(__file__).parent / "csrc"
        sources = [str(source_dir / "convnd.cpp")]
        cflags = ["/O2"] if sys.platform == "win32" else ["-O3"]
        if requested_cuda:
            sources.append(str(source_dir / "convnd_cuda.cu"))
            cflags.append("-DWITH_CUDA")
            if has_fp32_output_gemm():
                sources.append(str(source_dir / "convnd_mixed.cpp"))
                cflags.append("-DTORCH_CONVND_FP32_OUTPUT_GEMM")
        with _build_environment(requested_cuda):
            load(
                name=_native_build_name(source_dir, requested_cuda),
                sources=sources,
                extra_cflags=cflags,
                extra_cuda_cflags=["-O3", "-DWITH_CUDA"],
                with_cuda=requested_cuda,
                is_python_module=False,
                verbose=verbose,
            )
        _WITH_CUDA = requested_cuda
        _LOADED = True
