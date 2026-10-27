"""Optional ONNX export using standard Conv nodes, without custom runtime plugins."""

from collections.abc import Sequence
from functools import cache
from inspect import signature
from typing import TypeVar

import torch


def _translation_table():
    """Import exporter dependencies lazily and retain the input floating dtype."""
    from onnxscript import BFLOAT16, DOUBLE, FLOAT, FLOAT16
    from onnxscript import opset22 as op

    floating = TypeVar("floating", FLOAT16, BFLOAT16, FLOAT, DOUBLE)

    def native_conv(
        input: floating,
        weight: floating,
        bias: floating | None,
        stride: Sequence[int],
        padding: Sequence[int],
        dilation: Sequence[int],
        groups: int,
        workspace_bytes: int,
    ) -> floating:
        attributes = {
            "strides": stride,
            "pads": [*padding, *padding],
            "dilations": dilation,
            "group": groups,
        }
        if bias is None:
            return op.Conv(input, weight, **attributes)
        return op.Conv(input, weight, bias, **attributes)

    def compatible_conv(
        input: floating,
        weight: floating,
        bias: floating | None,
        stride: Sequence[int],
        padding: Sequence[int],
        dilation: Sequence[int],
        groups: int,
        workspace_bytes: int,
        backend: str,
    ) -> floating:
        return native_conv(input, weight, bias, stride, padding, dilation, groups, workspace_bytes)

    return {
        torch.ops.torch_convnd.conv_nd.default: native_conv,
        torch.ops.torch_convnd.compatible_conv_nd.default: compatible_conv,
    }


@cache
def _register_legacy_symbolics(opset_version):
    """Support traced/scripted graphs when the caller chooses the legacy exporter."""
    from torch.onnx import symbolic_helper

    def symbolic(graph, input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
        inputs = [input, weight]
        if not symbolic_helper._is_none(bias):
            inputs.append(bias)
        return graph.op(
            "Conv",
            *inputs,
            strides_i=stride,
            pads_i=[*padding, *padding],
            dilations_i=dilation,
            group_i=groups,
        )

    native = symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "i")(symbolic)

    @symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "i", "s")
    def compatible(
        graph, input, weight, bias, stride, padding, dilation, groups, workspace_bytes, backend
    ):
        return symbolic(
            graph, input, weight, bias, stride, padding, dilation, groups, workspace_bytes
        )

    torch.onnx.register_custom_op_symbolic("torch_convnd::conv_nd", native, opset_version)
    torch.onnx.register_custom_op_symbolic(
        "torch_convnd::compatible_conv_nd", compatible, opset_version
    )


def export_onnx(model, args, path=None, *, opset_version=None, dynamo=None, **kwargs):
    """Export real-valued modules with standard ONNX convolution operators.

    Install ``torch-convnd[onnx]`` first. Forward all other options, including
    ``dynamic_shapes`` and ``input_names``, to :func:`torch.onnx.export`. Complex
    tensor interfaces are outside the ONNX Conv schema, as for stock Conv2d/3d.
    Execution support for a rank/dtype depends on the chosen ONNX runtime provider.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise ImportError("Install ONNX support with: pip install 'torch-convnd[onnx]'") from error

    from ._dispatch import ensure_compatible_operator

    parameters = signature(torch.onnx.export).parameters
    modern_exporter = "custom_translation_table" in parameters
    if dynamo is None:
        dynamo = modern_exporter and not isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )
    if dynamo and not modern_exporter:
        raise ValueError("This PyTorch lacks custom Dynamo ONNX translations; use dynamo=False")
    if opset_version is None:
        constants = getattr(torch.onnx, "_constants", None)
        maximum = getattr(constants, "ONNX_MAX_OPSET", None)
        if maximum is None:
            from torch.onnx import symbolic_helper

            maximum = getattr(symbolic_helper, "_onnx_main_opset", 16)
        opset_version = 22 if dynamo else min(20, maximum)

    tensors = (args,) if isinstance(args, torch.Tensor) else args
    if isinstance(model, torch.nn.Module):
        tensors = (*tensors, *model.parameters())
    if any(isinstance(value, torch.Tensor) and value.is_complex() for value in tensors):
        raise ValueError(
            "ONNX Conv accepts real tensors; export an explicit real/imaginary interface"
        )
    if opset_version < 22 and any(
        isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16 for value in tensors
    ):
        raise ValueError("BF16 ONNX Conv requires opset 22 or newer; export the eager module")
    ensure_compatible_operator()
    if dynamo:
        try:
            import onnxscript  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Install ONNX support with: pip install 'torch-convnd[onnx]'"
            ) from error
        translations = _translation_table()
        translations.update(kwargs.pop("custom_translation_table", {}))
        kwargs["custom_translation_table"] = translations
    else:
        _register_legacy_symbolics(opset_version)
        if kwargs.pop("dynamic_shapes", None) is not None:
            raise ValueError("The legacy exporter uses dynamic_axes rather than dynamic_shapes")
    if "external_data" not in parameters and "external_data" in kwargs:
        external = kwargs.pop("external_data")
        if "use_external_data_format" in parameters:
            kwargs["use_external_data_format"] = external
    if "dynamo" in parameters:
        kwargs["dynamo"] = dynamo
    # PyTorch's progress glyphs are not encodable in the default Windows console.
    kwargs.setdefault("verbose", False)
    return torch.onnx.export(model, args, path, opset_version=opset_version, **kwargs)
