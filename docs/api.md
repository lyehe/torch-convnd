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

::: torch_convnd.load_native

::: torch_convnd.export_onnx

::: torch_convnd.get_runtime_info

`torch_convnd.__version__` is the installed package version. Build metadata reads
the same version source without importing PyTorch.
