"""Trainable convolution modules with PyTorch-compatible constructor arguments."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ._geometry import DEFAULT_WORKSPACE_BYTES, _tuple
from ._padding import padding_pairs
from .functional import conv_nd


class ConvNd(nn.Module):
    """Trainable full N-dimensional convolution with PyTorch-style parameters."""

    backend = "auto"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        *,
        dimensions: int,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        backend: str = "auto",
    ):
        super().__init__()
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or not 1 <= dimensions <= 62
        ):
            message = "dimensions must be an integer from 1 to 62"
            raise ValueError(message)
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuple(kernel_size, dimensions, "kernel_size")
        self.stride = _tuple(stride, dimensions, "stride")
        self.padding = (
            padding if isinstance(padding, str) else _tuple(padding, dimensions, "padding")
        )
        self.dilation = _tuple(dilation, dimensions, "dilation")
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = False
        self.output_padding = (0,) * dimensions
        self.workspace_bytes = workspace_bytes
        if backend not in {"auto", "native", "torch"}:
            raise ValueError(f"Invalid backend {backend!r}")
        self.backend = backend
        if groups <= 0 or in_channels <= 0 or out_channels <= 0:
            message = "channels and groups must be positive"
            raise ValueError(message)
        if in_channels % groups or out_channels % groups:
            message = "channels must be divisible by groups"
            raise ValueError(message)
        if min(*self.kernel_size, *self.stride, *self.dilation) <= 0 or (
            not isinstance(self.padding, str) and min(self.padding) < 0
        ):
            message = "kernel, stride and dilation must be positive; padding nonnegative"
            raise ValueError(message)
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(f"Invalid padding_mode {padding_mode!r}")
        pairs = padding_pairs(self.padding, self.kernel_size, self.stride, self.dilation)
        self._reversed_padding_repeated_twice = [v for pair in reversed(pairs) for v in pair]
        if workspace_bytes <= 0:
            message = "workspace_bytes must be positive"
            raise ValueError(message)
        options = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size, **options)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **options))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the same initialization rule as PyTorch convolution modules."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels // self.groups * math.prod(self.kernel_size)
            nn.init.uniform_(self.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, input: Tensor) -> Tensor:
        """Convolve batched or unbatched channels-first tensors."""
        if input.ndim not in {self.dimensions + 1, self.dimensions + 2}:
            message = (
                f"expected {self.dimensions + 1} unbatched or "
                f"{self.dimensions + 2} batched dimensions, got {input.ndim}"
            )
            raise ValueError(message)
        return conv_nd(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            padding_mode=self.padding_mode,
            workspace_bytes=self.workspace_bytes,
            backend=self.backend,
        )

    def extra_repr(self) -> str:
        """Describe the kernel geometry in module summaries."""
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"dimensions={self.dimensions}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, "
            f"padding_mode={self.padding_mode!r}, backend={self.backend!r}"
        )

    def __prepare_scriptable__(self):
        """Expose equivalent native execution in TorchScript-compatible syntax."""
        from ._compat import HAS_LIBRARY
        from ._dispatch import ensure_compatible_operator
        from ._script import ScriptableConvNd

        if not HAS_LIBRARY:
            from ._legacy_script import LegacyScriptableConvNd

            return LegacyScriptableConvNd(self)
        ensure_compatible_operator()
        return ScriptableConvNd(self)


class _FixedConvNd(ConvNd):
    """Share the standard convolution constructor across fixed spatial ranks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        *,
        workspace_bytes=DEFAULT_WORKSPACE_BYTES,
        backend="auto",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            dimensions=self._spatial_dimensions,
            workspace_bytes=workspace_bytes,
            backend=backend,
        )


class Conv1d(_FixedConvNd):
    """Full 1D convolution with the torch.nn.Conv1d constructor."""

    _spatial_dimensions = 1


class Conv2d(_FixedConvNd):
    """Full 2D convolution with the torch.nn.Conv2d constructor."""

    _spatial_dimensions = 2


class Conv3d(_FixedConvNd):
    """Full 3D convolution with the torch.nn.Conv3d constructor."""

    _spatial_dimensions = 3


class Conv4d(_FixedConvNd):
    """Full 4D convolution; input axes may represent x, y, z and time."""

    _spatial_dimensions = 4
