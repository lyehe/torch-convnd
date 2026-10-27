"""Downstream usage and missing-toolchain regressions, with no compiler required."""

import io
import json
import subprocess
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torch_convnd import Conv2d, Conv3d, Conv4d, ConvNd, __version__, get_runtime_info

pytestmark = pytest.mark.portable


@pytest.mark.parametrize("rank", [4, 5])
def test_auto_missing_toolchain_preserves_values_and_gradients(rank, monkeypatch):
    """Auto falls back before invoking a compiler, including higher-rank training."""
    from torch_convnd import _build, _dispatch

    monkeypatch.setattr(_build, "_LOADED", False)
    monkeypatch.setattr(_dispatch, "native_unavailable_reason", lambda _: "no test compiler")

    def unexpected_build(*args, **kwargs):
        raise AssertionError("auto fallback must not invoke native compilation")

    monkeypatch.setattr(_build, "load_native", unexpected_build)
    # CPU rank 5 selects native by default; rank 4 exercises the vendor-slice path.
    layer = ConvNd(2, 4, 2, dimensions=rank, groups=2, dtype=torch.float64)
    reference = ConvNd(2, 4, 2, dimensions=rank, groups=2, dtype=torch.float64, backend="torch")
    reference.load_state_dict(layer.state_dict())
    input = torch.randn((1, 2) + (3,) * rank, dtype=torch.float64, requires_grad=True)
    actual, expected = layer(input), reference(input)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target)


def test_compile_auto_without_toolchain(monkeypatch):
    """Dynamo chooses a portable full graph when native prerequisites are absent."""
    from torch_convnd import _build, _dispatch

    monkeypatch.setattr(_build, "_LOADED", False)
    monkeypatch.setattr(_dispatch, "native_unavailable_reason", lambda _: "no test compiler")
    layer = ConvNd(1, 2, 2, dimensions=5)
    input = torch.randn(1, 1, 3, 3, 3, 3, 3, requires_grad=True)
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(input), layer(input))
    compiled(input).square().mean().backward()
    assert input.grad is not None and torch.isfinite(input.grad).all()


def test_explicit_native_missing_tools_is_actionable(monkeypatch):
    """An explicit native request fails clearly rather than changing the requested backend."""
    from torch_convnd import _build, _toolchain

    monkeypatch.setattr(_build, "_LOADED", False)
    monkeypatch.setattr(_toolchain, "native_unavailable_reason", lambda _: "Ninja is missing")
    with pytest.raises(RuntimeError, match="Cannot load native.*Ninja is missing"):
        _build.load_native(with_cuda=False)


@pytest.mark.parametrize("factory,rank", [(Conv2d, 2), (Conv3d, 3)])
def test_stock_state_dict_round_trip(factory, rank):
    """Existing projects can exchange parameters with stock layers without renaming keys."""
    stock = getattr(torch.nn, f"Conv{rank}d")(4, 6, 3, padding=1, groups=2)
    layer = factory(4, 6, 3, padding=1, groups=2, backend="torch")
    buffer = io.BytesIO()
    torch.save(stock.state_dict(), buffer)
    buffer.seek(0)
    layer.load_state_dict(torch.load(buffer, weights_only=True))
    input = torch.randn((2, 4) + (4,) * rank)
    torch.testing.assert_close(layer(input), stock(input))
    stock.load_state_dict(layer.state_dict())


def test_checkpoint_and_optimizer_match_ordinary_training():
    """Activation checkpointing recomputes full 4D kernels and updates the same parameters."""
    layer = Conv4d(2, 4, 2, backend="torch")
    reference = Conv4d(2, 4, 2, backend="torch")
    reference.load_state_dict(layer.state_dict())
    input = torch.randn(2, 2, 3, 3, 3, 3, requires_grad=True)
    actual = checkpoint(layer, input, use_reentrant=False)
    expected = reference(input)
    first = torch.optim.SGD(layer.parameters(), lr=0.1)
    second = torch.optim.SGD(reference.parameters(), lr=0.1)
    actual.square().mean().backward()
    expected.square().mean().backward()
    first.step()
    second.step()
    for value, target in zip(layer.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(value, target)


def test_diagnostics_do_not_compile(monkeypatch):
    """Applications can report environment metadata without initializing native kernels."""
    from torch_convnd import _build

    def unexpected_build(*args, **kwargs):
        raise AssertionError("diagnostics must not compile")

    monkeypatch.setattr(_build, "load_native", unexpected_build)
    information = get_runtime_info()
    assert information["torch_convnd"] == __version__
    assert information["torch"] == torch.__version__
    json.dumps(information)


def test_diagnostics_module_cli_is_machine_readable():
    """The standard module entry point is usable without editable-install PATH setup."""
    result = subprocess.run(
        [sys.executable, "-m", "torch_convnd", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    information = json.loads(result.stdout)
    assert information["torch_convnd"] == __version__
    assert information["native_loaded"] is False


def test_import_and_portable_training_need_no_build_dependencies():
    """A fresh process remains usable even when every native build dependency is blocked."""
    program = """
import importlib.abc
import sys
import torch

class NoBuildTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ninja', 'setuptools', 'packaging'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoBuildTools())
from torch_convnd import ConvNd, _build
layer = ConvNd(1, 2, 2, dimensions=5)
input = torch.randn(1, 1, 3, 3, 3, 3, 3, requires_grad=True)
layer(input).square().mean().backward()
assert input.grad is not None
assert not _build._LOADED
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
