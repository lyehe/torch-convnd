"""Fresh-process registration and operation without the optional native compiler."""

import subprocess
import sys
import textwrap

import pytest
import torch

from torch_convnd import Conv4d


@pytest.mark.parametrize("mode", ["script", "trace"])
def test_serialized_graph_loads_in_fresh_process_without_build(mode, tmp_path):
    """Import registers serialized operators; transforms need no compiled kernels."""
    layer = Conv4d(2, 3, 2, backend="native")
    input = torch.randn(1, 2, 3, 3, 3, 3)
    compiled = torch.jit.script(layer) if mode == "script" else torch.jit.trace(layer, input)
    path = tmp_path / "module.pt"
    compiled.save(str(path))
    program = textwrap.dedent("""
        import sys
        import torch
        from torch_convnd import Conv4d, _build

        def unexpected_build(*args, **kwargs):
            raise AssertionError("portable execution must not invoke the compiler")

        _build.load_native = unexpected_build
        assert not _build._LOADED
        assert "onnx" not in sys.modules
        layer = torch.jit.load(sys.argv[1])
        reference = Conv4d(2, 3, 2, backend="torch")
        reference.load_state_dict(layer.state_dict())
        input = torch.randn(1, 2, 3, 3, 3, 3)
        tangent = torch.randn_like(input)
        actual = torch.func.jvp(layer, (input,), (tangent,))
        expected = torch.func.jvp(reference, (input,), (tangent,))
        assert actual[1].norm() > 0
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
        Conv4d(2, 3, 2)(input).sum().backward()
        assert not _build._LOADED
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("compile_backend", ["aot_eager", "inductor"])
def test_compiled_native_first_use_builds_lazily(device, compile_backend):
    """Compiled graphs initialize kernels at execution, without manual preload."""
    import importlib.util

    from torch.utils.cpp_extension import CUDA_HOME

    if device == "cuda" and (not torch.cuda.is_available() or CUDA_HOME is None):
        pytest.skip("CUDA GPU/toolkit is unavailable")
    if device == "cuda" and compile_backend == "inductor":
        if importlib.util.find_spec("triton") is None:
            pytest.skip("CUDA Inductor requires a matching Triton installation")
    program = textwrap.dedent("""
        import sys
        import torch
        from torch_convnd import Conv4d, _build
        from torch_convnd.convnd import _build_environment

        torch.set_num_threads(2)
        device = sys.argv[1]
        layer = Conv4d(2, 3, 2, backend="native", device=device)
        reference = Conv4d(2, 3, 2, backend="torch", device=device)
        reference.load_state_dict(layer.state_dict())
        input = torch.randn(1, 2, 3, 3, 3, 3, device=device, requires_grad=True)
        assert not _build._LOADED
        with _build_environment(device == "cuda"):
            compiled = torch.compile(layer, backend=sys.argv[2], fullgraph=True)
            actual = compiled(input)
        assert _build._LOADED
        expected = reference(input)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        first = torch.autograd.grad(actual.square().mean(), (input, *layer.parameters()))
        second = torch.autograd.grad(expected.square().mean(), (input, *reference.parameters()))
        for value, target in zip(first, second):
            torch.testing.assert_close(value, target, atol=2e-5, rtol=2e-5)
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, device, compile_backend],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_cache_isolated_while_another_process_holds_library(tmp_path):
    """A second installation can build while the original CPU library remains loaded."""
    import shutil
    from pathlib import Path

    import torch_convnd

    copy_root = tmp_path / "second_installation"
    shutil.copytree(Path(torch_convnd.__file__).parent / "csrc", copy_root / "csrc")
    child = textwrap.dedent("""
        import sys
        from pathlib import Path
        import torch
        from torch_convnd import Conv4d, _build

        _build.__file__ = str(Path(sys.argv[1]) / "_build.py")
        _build.load_native(with_cuda=False)
        layer = Conv4d(1, 1, 2, bias=False, dtype=torch.float64, backend="native")
        with torch.no_grad():
            layer.weight.fill_(1)
        input = torch.ones(1, 1, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        output = layer(input)
        torch.testing.assert_close(output, torch.full_like(output, 16))
        output.sum().backward()
        torch.testing.assert_close(layer.weight.grad, torch.full_like(layer.weight, 16))
        assert input.grad.sum() == 256
    """)
    parent = textwrap.dedent("""
        import subprocess
        import sys
        from torch_convnd import load_native

        load_native(with_cuda=False)
        subprocess.run([sys.executable, "-c", sys.argv[1], sys.argv[2]], check=True)
    """)
    result = subprocess.run(
        [sys.executable, "-c", parent, child, str(copy_root)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
