"""Native operator metadata, derivative formulas and batching rules."""

import torch

from ._compat import HAS_NATIVE_VMAP


def _fake_forward(input, weight, bias, stride, padding, dilation, groups, workspace_bytes):
    shape = [input.shape[0], weight.shape[0]]
    shape.extend(
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(
            input.shape[2:], weight.shape[2:], stride, padding, dilation, strict=True
        )
    )
    return input.new_empty(shape)


def _fake_backward(
    grad_output, input, weight, stride, padding, dilation, groups, workspace_bytes, output_mask
):
    return (
        torch.empty_like(input, memory_format=torch.contiguous_format) if output_mask[0] else None,
        torch.empty_like(weight, memory_format=torch.contiguous_format) if output_mask[1] else None,
        input.new_empty(weight.shape[0]) if output_mask[2] else None,
    )


def _setup_context(ctx, inputs, output):
    input, weight, bias, *parameters = inputs
    ctx.save_for_backward(input, weight)
    ctx.parameters = parameters
    ctx.has_bias = bias is not None


def _backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    mask = [
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[1],
        ctx.has_bias and ctx.needs_input_grad[2],
    ]
    gradients = torch.ops.torch_convnd.conv_nd_backward.default(
        grad_output, input, weight, *ctx.parameters, mask
    )
    return (*gradients, None, None, None, None, None)


def _setup_backward_context(ctx, inputs, output):
    grad_output, input, weight, *parameters = inputs
    ctx.save_for_backward(grad_output, input, weight)
    ctx.parameters = parameters[:-1]


def _backward_backward(ctx, grad_input_grad, grad_weight_grad, grad_bias_grad):
    grad_output, input, weight = ctx.saved_tensors
    parameters = ctx.parameters
    output_terms = []
    input_grad = weight_grad = None
    with torch.autocast(device_type=input.device.type, enabled=False):
        if grad_input_grad is not None:
            if ctx.needs_input_grad[0]:
                output_terms.append(
                    torch.ops.torch_convnd.conv_nd.default(
                        grad_input_grad, weight, None, *parameters
                    )
                )
            if ctx.needs_input_grad[2]:
                weight_grad = torch.ops.torch_convnd.conv_nd_backward.default(
                    grad_output, grad_input_grad, weight, *parameters, [False, True, False]
                )[1]
        if grad_weight_grad is not None:
            if ctx.needs_input_grad[0]:
                output_terms.append(
                    torch.ops.torch_convnd.conv_nd.default(
                        input, grad_weight_grad, None, *parameters
                    )
                )
            if ctx.needs_input_grad[1]:
                input_grad = torch.ops.torch_convnd.conv_nd_backward.default(
                    grad_output, input, grad_weight_grad, *parameters, [True, False, False]
                )[0]
        if grad_bias_grad is not None and ctx.needs_input_grad[0]:
            output_terms.append(
                grad_bias_grad.reshape(1, -1, *((1,) * (input.ndim - 2))).expand_as(grad_output)
            )
    grad_output_grad = None
    for term in output_terms:
        grad_output_grad = term if grad_output_grad is None else grad_output_grad + term
    return grad_output_grad, input_grad, weight_grad, None, None, None, None, None, None


def _vmap_native(
    info, in_dims, input, weight, bias, stride, padding, dilation, groups, workspace_bytes
):
    if in_dims[0] is not None and in_dims[1] is None and in_dims[2] is None:
        mapped_input = input.movedim(in_dims[0], 0)
        output = torch.ops.torch_convnd.conv_nd.default(
            mapped_input.flatten(0, 1),
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            workspace_bytes,
        )
        return output.reshape(info.batch_size, mapped_input.shape[1], *output.shape[1:]), 0
    if info.batch_size == 0:
        input_shape, weight_shape = list(input.shape), list(weight.shape)
        if in_dims[0] is not None:
            del input_shape[in_dims[0]]
        if in_dims[1] is not None:
            del weight_shape[in_dims[1]]
        spatial = [
            (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
            for size, pad, dil, kernel, step in zip(
                input_shape[2:], padding, dilation, weight_shape[2:], stride, strict=True
            )
        ]
        return input.new_empty((0, input_shape[0], weight_shape[0], *spatial)), 0
    # A rule executes below its vmap interpreter; re-entering vmap here is unsafe
    # for factory operations on some PyTorch versions. Redispatch unwrapped slices.
    results = []
    for index in range(info.batch_size):
        x = input.select(in_dims[0], index) if in_dims[0] is not None else input
        w = weight.select(in_dims[1], index) if in_dims[1] is not None else weight
        b = bias.select(in_dims[2], index) if in_dims[2] is not None else bias
        results.append(
            torch.ops.torch_convnd.conv_nd.default(
                x, w, b, stride, padding, dilation, groups, workspace_bytes
            )
        )
    return torch.stack(results), 0


def register_training() -> None:
    torch.library.register_fake("torch_convnd::conv_nd")(_fake_forward)
    torch.library.register_fake("torch_convnd::conv_nd_backward")(_fake_backward)
    torch.library.register_autograd(
        "torch_convnd::conv_nd", _backward, setup_context=_setup_context
    )
    torch.library.register_autograd(
        "torch_convnd::conv_nd_backward",
        _backward_backward,
        setup_context=_setup_backward_context,
    )
    if HAS_NATIVE_VMAP:
        torch.library.register_vmap("torch_convnd::conv_nd", _vmap_native)
