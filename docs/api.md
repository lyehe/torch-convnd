# Public API

Import from `torch_convnd`. Private modules and `torch.ops.torch_convnd` are
implementation interfaces; consumers should use the modules/functions below.

::: torch_convnd.ConvNd

`Conv1d`, `Conv2d`, `Conv3d`, and `Conv4d` use the same constructor without the
required `dimensions` keyword. Their positional arguments follow stock Conv3d:
channels, kernel, stride, padding, dilation, groups, bias, padding mode, device,
and dtype. The additional `backend` and `workspace_bytes` arguments are keywords.

::: torch_convnd.Conv4d

::: torch_convnd.conv_nd

## Partial convolution

`PartialConvNd` and `PartialConv1d` through `PartialConv4d` implement NVIDIA's
[partial convolution](https://github.com/NVIDIA/partialconv) at any rank: the stock
constructor plus the keyword options `multi_channel` and `return_mask`, and
`forward(input, mask=None)`, which returns the output or, when the update mask is
requested at construction or with the call's `return_mask=` keyword,
`PartialConvOutput(output, mask)`. `PartialSequential`
carries the mask through a chain, and `partial_conv_nd` states the arithmetic and the
mask contract. For border correction without a mask, `ConvNd`, the fixed-rank layers
and `conv_nd` accept `padding_mode="partial"`. State dictionaries match `ConvNd` and the
stock layers, so pretrained weights load directly. TorchScript is refused rather than
scripting a plain convolution; tracing, `torch.compile` and `export_onnx` are supported.

::: torch_convnd.PartialConvNd

::: torch_convnd.PartialSequential

::: torch_convnd.PartialConvOutput

::: torch_convnd.partial_conv_nd

## Runtime

::: torch_convnd.load_native

::: torch_convnd.export_onnx

::: torch_convnd.get_runtime_info

`torch_convnd.__version__` is the installed package version. Build metadata reads
the same version source without importing PyTorch.
