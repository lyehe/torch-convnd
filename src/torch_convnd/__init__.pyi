from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ._version import __version__ as __version__

_Spatial = int | Sequence[int]
_Padding = str | _Spatial
_Backend = Literal["auto", "native", "torch"]
_Device = str | torch.device | None

class ConvNd(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter | None
    dimensions: int
    in_channels: int
    out_channels: int
    groups: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    padding: str | tuple[int, ...]
    padding_mode: str
    workspace_bytes: int
    backend: _Backend
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Spatial,
        stride: _Spatial = ...,
        padding: _Padding = ...,
        dilation: _Spatial = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: str = ...,
        device: _Device = ...,
        dtype: torch.dtype | None = ...,
        *,
        dimensions: int,
        workspace_bytes: int = ...,
        backend: _Backend = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self) -> str: ...

class _FixedConvNd(ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Spatial,
        stride: _Spatial = ...,
        padding: _Padding = ...,
        dilation: _Spatial = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: str = ...,
        device: _Device = ...,
        dtype: torch.dtype | None = ...,
        *,
        workspace_bytes: int = ...,
        backend: _Backend = ...,
    ) -> None: ...

class Conv1d(_FixedConvNd): ...
class Conv2d(_FixedConvNd): ...
class Conv3d(_FixedConvNd): ...
class Conv4d(_FixedConvNd): ...

def conv_nd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: _Spatial = ...,
    padding: _Padding = ...,
    dilation: _Spatial = ...,
    groups: int = ...,
    *,
    padding_mode: str = ...,
    workspace_bytes: int = ...,
    backend: _Backend = ...,
) -> Tensor: ...
def load_native(*, with_cuda: bool | None = ..., verbose: bool = ...) -> None: ...
def export_onnx(
    model: Any,
    args: tuple[Any, ...] | Tensor,
    path: str | PathLike[str] | None = ...,
    *,
    opset_version: int | None = ...,
    dynamo: bool | None = ...,
    **kwargs: Any,
) -> Any: ...
def get_runtime_info() -> dict[str, Any]: ...
