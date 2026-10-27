"""Keep caller GPU selection and configure deterministic CUDA matrix multiplication."""

import os

import pytest
import torch

from torch_convnd import load_native


def pytest_configure(config):
    """Configure deterministic matrix multiplication before the first CUDA operation."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(scope="session")
def native_runtime():
    """Build once, only when a test requiring the addon is selected."""
    from torch.utils.cpp_extension import CUDA_HOME

    load_native(with_cuda=CUDA_HOME is not None and torch.cuda.is_available())


@pytest.fixture(autouse=True)
def require_native(request):
    """Portable checks remain runnable on machines with no C++/CUDA compiler."""
    if request.node.get_closest_marker("portable") is None:
        request.getfixturevalue("native_runtime")


@pytest.fixture(scope="session", autouse=True)
def precision_policy():
    """Compare both implementations with matching precision and bounded CPU threads."""
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    threads = torch.get_num_threads()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(threads, 4))
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
        torch.set_num_threads(threads)


@pytest.fixture
def deterministic_algorithms():
    """Use the deterministic path for exact finite-difference reentrancy checks."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)
