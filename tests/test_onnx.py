"""Export whole models to standard ONNX nodes and execute them in ONNX Runtime."""

import numpy as np
import pytest
import torch

from torch_convnd import Conv2d, Conv3d, Conv4d, export_onnx

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

FACTORIES = {2: Conv2d, 3: Conv3d, 4: Conv4d}


def session_for(path):
    """Keep runtime thread use bounded and require a standard, checked graph."""
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_onnx_network_runtime(rank, backend, mode, tmp_path):
    """Convolution plus activations/bias/groups exports without a custom operator plugin."""
    layer = FACTORIES[rank](4, 6, 2, groups=2, padding="same", padding_mode=mode, backend=backend)
    model = torch.nn.Sequential(layer, torch.nn.ReLU()).eval()
    input = torch.randn((2, 4) + (4,) * rank)
    path = tmp_path / "network.onnx"
    export_onnx(model, (input,), path, external_data=False)
    session = session_for(path)
    actual = session.run(None, {session.get_inputs()[0].name: input.numpy()})[0]
    np.testing.assert_allclose(actual, model(input).detach().numpy(), atol=2e-5, rtol=2e-5)
    graph = onnx.load(path)
    assert all(node.domain in {"", "ai.onnx"} for node in graph.graph.node)


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_onnx_dynamic_shapes(rank, tmp_path):
    """Native ND nodes retain dynamic batch and spatial dimensions after translation."""
    layer = FACTORIES[rank](2, 3, 2, padding="same", backend="native").eval()
    input = torch.randn((2, 2) + (4,) * rank)
    dimensions = {0: torch.export.Dim("batch", min=1, max=4)}
    dimensions.update(
        {axis: torch.export.Dim(f"axis{axis}", min=3, max=7) for axis in range(2, rank + 2)}
    )
    path = tmp_path / "dynamic.onnx"
    export_onnx(layer, (input,), path, dynamic_shapes=(dimensions,), external_data=False)
    session = session_for(path)
    changed = torch.randn((3, 2) + (5,) * rank)
    actual = session.run(None, {session.get_inputs()[0].name: changed.numpy()})[0]
    np.testing.assert_allclose(actual, layer(changed).detach().numpy(), atol=2e-5, rtol=2e-5)


def test_complex_export_is_rejected_explicitly(tmp_path):
    """Do not silently discard components outside the standard ONNX Conv type schema."""
    layer = Conv2d(2, 3, 2, dtype=torch.complex64)
    input = torch.randn(1, 2, 3, 3, dtype=torch.complex64)
    with pytest.raises(ValueError, match="real tensors"):
        export_onnx(layer, (input,), tmp_path / "complex.onnx")


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("mode", ["script", "trace"])
def test_serialized_onnx_runtime(rank, mode, tmp_path):
    """Saved-graph interfaces select the legacy translator rather than failing in Dynamo."""
    layer = FACTORIES[rank](2, 3, 2, padding="same", backend="native").eval()
    input = torch.randn((1, 2) + (3,) * rank)
    compiled = torch.jit.script(layer) if mode == "script" else torch.jit.trace(layer, input)
    path = tmp_path / "serialized.onnx"
    export_onnx(compiled, (input,), path, external_data=False)
    session = session_for(path)
    actual = session.run(None, {session.get_inputs()[0].name: input.numpy()})[0]
    np.testing.assert_allclose(actual, layer(input).detach().numpy(), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_onnx_preserves_floating_dtype(rank, dtype, tmp_path):
    """Export all standard Conv floating types without narrowing to FP32 silently."""
    layer = FACTORIES[rank](2, 3, 2, dtype=dtype, backend="native").eval()
    input = torch.randn((1, 2) + (3,) * rank, dtype=dtype)
    path = tmp_path / "typed.onnx"
    export_onnx(layer, (input,), path, external_data=False)
    onnx.checker.check_model(str(path))
    graph = onnx.load(path)
    expected_type = {
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.float64: onnx.TensorProto.DOUBLE,
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
    }[dtype]
    assert graph.graph.input[0].type.tensor_type.elem_type == expected_type
    assert graph.graph.output[0].type.tensor_type.elem_type == expected_type
    if dtype == torch.float16:
        # The local CPU provider implements FP16/FP32 Conv, but not FP64/BF16.
        session = session_for(path)
        actual = session.run(None, {session.get_inputs()[0].name: input.numpy()})[0]
        reference = FACTORIES[rank](2, 3, 2, dtype=dtype, backend="torch").eval()
        reference.load_state_dict(layer.state_dict())
        np.testing.assert_allclose(
            actual, reference(input).detach().numpy(), atol=0.005, rtol=0.005
        )


def test_bfloat16_legacy_opset_is_rejected_explicitly(tmp_path):
    """BF16 needs the Conv schema from opset 22 rather than an invalid legacy graph."""
    layer = Conv2d(2, 3, 2, dtype=torch.bfloat16)
    input = torch.randn(1, 2, 3, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="opset 22"):
        export_onnx(layer, (input,), tmp_path / "legacy.onnx", opset_version=20)


def test_native_onnx_export_needs_no_compiler(tmp_path):
    """Metadata alone lets a fresh process export a native 4D graph."""
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import sys
        import torch
        import onnx
        from torch_convnd import Conv4d, export_onnx, _build

        def unexpected_build(*args, **kwargs):
            raise AssertionError("ONNX export must not invoke the native compiler")

        _build.load_native = unexpected_build
        layer = Conv4d(2, 3, 2, backend="native").eval()
        input = torch.randn(1, 2, 3, 3, 3, 3)
        export_onnx(layer, (input,), sys.argv[1], external_data=False)
        onnx.checker.check_model(sys.argv[1])
        assert not _build._LOADED
        assert not torch._C._dispatch_has_kernel_for_dispatch_key("torch_convnd::conv_nd", "CPU")
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "no_compiler.onnx")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
