"""Trainable convolution modules with PyTorch-compatible constructor arguments."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ._geometry import DEFAULT_WORKSPACE_BYTES, _tuple, has_rank
from ._padding import padding_pairs
from .functional import (
    _PADDING_MODES,
    _REAL,
    PartialConvOutput,
    _mask_free_update_is_full,
    _partial_conv_nd,
    conv_nd,
)


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
        if padding_mode not in _PADDING_MODES:
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

    def _check_input(self, input: Tensor) -> None:
        if not has_rank(input.ndim, self.dimensions):
            message = (
                f"expected {self.dimensions + 1} unbatched or "
                f"{self.dimensions + 2} batched dimensions, got {input.ndim}"
            )
            raise ValueError(message)

    def forward(self, input: Tensor) -> Tensor:
        """Convolve batched or unbatched channels-first tensors."""
        self._check_input(input)
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
        if self.padding_mode == "partial":
            message = (
                "padding_mode='partial' does not support TorchScript; use eager execution, "
                "torch.compile, torch.jit.trace or export_onnx"
            )
            raise NotImplementedError(message)
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


class PartialConvNd(ConvNd):
    """NVIDIA's partial convolution at any spatial rank, with ConvNd's parameters.

    ``forward(input, mask=None)`` convolves only valid samples and rescales each
    window by its valid fraction; :func:`partial_conv_nd` states the formula and the
    mask contract. ``multi_channel`` (fixed per layer) selects one mask channel per
    input channel instead of one shared channel. ``return_mask`` sets whether calls
    also return the update mask, as ``PartialConvOutput(output, mask)``; each call can
    override it. For border correction without a mask, an ordinary layer with
    ``padding_mode="partial"`` gives the same result. State dictionaries match ConvNd
    and the stock layers.
    """

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
        multi_channel: bool = False,
        return_mask: bool = False,
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
            dimensions=dimensions,
            workspace_bytes=workspace_bytes,
            backend=backend,
        )
        if not isinstance(multi_channel, bool) or not isinstance(return_mask, bool):
            raise TypeError("multi_channel and return_mask must be bools")
        if padding_mode == "partial":
            message = "PartialConvNd already rescales zero padding; use padding_mode='zeros'"
            raise ValueError(message)
        self.multi_channel = multi_channel
        self.return_mask = return_mask

    def forward(  # type: ignore[override]
        self,
        input: Tensor,
        mask: Tensor | None = None,
        mask_in: Tensor | None = None,
        return_mask: bool | None = None,
    ) -> Tensor | PartialConvOutput:
        """Convolve valid samples; with ``return_mask`` also return the update mask.

        ``return_mask`` (None: the layer's setting) chooses whether this call returns
        ``PartialConvOutput(output, mask)`` or the output alone. ``mask_in`` is NVIDIA's
        keyword for ``mask``. The update mask may be a broadcast view; clone it before
        writing to it in place. Pass both options by keyword; they are not keyword-only
        because the legacy ONNX exporter binds every parameter positionally.
        """
        if mask_in is not None:
            if mask is not None:
                raise TypeError("pass mask or mask_in, not both")
            mask = mask_in
        if return_mask is None:
            return_mask = self.return_mask
        elif not isinstance(return_mask, bool):
            raise TypeError("return_mask must be a bool or None")
        self._check_input(input)
        output, update = _partial_conv_nd(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            mask=mask,
            multi_channel=self.multi_channel,
            need_update=return_mask,
            padding_mode=self.padding_mode,
            workspace_bytes=self.workspace_bytes,
            backend=self.backend,
        )
        if not return_mask:
            return output
        assert update is not None
        return PartialConvOutput(output, update)

    def _keeps_every_sample_valid(self, input: Tensor) -> bool:
        """Whether this layer, without a mask, leaves every output sample valid."""
        return _mask_free_update_is_full(
            tuple(input.shape[-self.dimensions :]),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.padding_mode,
        )

    def extra_repr(self) -> str:
        """Describe the kernel geometry and mask options in module summaries."""
        return (
            f"{super().extra_repr()}, multi_channel={self.multi_channel}, "
            f"return_mask={self.return_mask}"
        )

    def __prepare_scriptable__(self):
        """Refuse TorchScript rather than script ConvNd's plain convolution."""
        message = (
            "PartialConvNd does not support TorchScript; use eager execution, "
            "torch.compile, torch.jit.trace or export_onnx"
        )
        raise NotImplementedError(message)


class _FixedPartialConvNd(PartialConvNd):
    """Share the partial convolution constructor across fixed spatial ranks."""

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
        multi_channel=False,
        return_mask=False,
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
            multi_channel=multi_channel,
            return_mask=return_mask,
        )


class PartialConv1d(_FixedPartialConvNd):
    """Partial 1D convolution with the torch.nn.Conv1d constructor."""

    _spatial_dimensions = 1


class PartialConv2d(_FixedPartialConvNd):
    """Partial 2D convolution; NVIDIA's PartialConv2d at the stock constructor."""

    _spatial_dimensions = 2


class PartialConv3d(_FixedPartialConvNd):
    """Partial 3D convolution; NVIDIA's PartialConv3d at the stock constructor."""

    _spatial_dimensions = 3


class PartialConv4d(_FixedPartialConvNd):
    """Partial 4D convolution; input axes may represent x, y, z and time."""

    _spatial_dimensions = 4


def _partial_rank(container: nn.Module) -> int | None:
    """Spatial rank of the first partial layer inside a module, or None."""
    for module in container.modules():
        if isinstance(module, PartialConvNd):
            return module.dimensions
    return None


def _spatial(shape: Sequence[int], rank: int) -> tuple[int, ...]:
    """The trailing ``rank`` sizes of a shape."""
    return tuple(shape[max(len(shape) - rank, 0) :])


def _all_valid(output: Tensor, rank: int, channels: int) -> Tensor:
    """The update mask of an output whose samples are all valid: a broadcast view of ones."""
    if not has_rank(output.ndim, rank):
        message = (
            f"the {output.ndim}-dimensional output has lost the {rank} spatial axes "
            "a validity mask describes"
        )
        raise ValueError(message)
    level = _REAL.get(output.dtype, output.dtype)
    spatial = _spatial(output.shape, rank)
    shape = (channels, *spatial)
    if output.ndim == rank + 2:
        shape = (output.shape[0], *shape)
    return torch.ones((), dtype=level, device=output.device).expand(shape)


class PartialSequential(nn.Sequential):
    """Run modules in order, carrying the validity mask through partial layers.

    ``forward(input, mask=None)`` returns ``PartialConvOutput(output, mask)``.

    - Partial layers, including those in nested ``PartialSequential`` containers, read
      the current mask and replace it with their update mask.
    - Other modules receive only the tensor and are taken to act position by position
      (activations, normalization, dropout, 1x1 convolutions), so the mask passes
      unchanged. While a mask is carried they must keep the spatial shape; one that
      changes it raises, since its effect on validity is undefined, so resample the
      mask around it yourself. A module that is not a ``PartialSequential`` but holds
      partial layers raises, because the mask cannot reach them.
    - Without a mask, layers stay on the mask-free path while their geometry keeps
      every sample valid; an all-valid result is reported as ones (a broadcast view).

    The final mask has the last partial layer's update layout whichever path ran, and
    the output must keep the spatial axes it describes: the partial layers', or a
    batched input's when the container holds none.
    """

    def forward(  # type: ignore[override]
        self, input: Tensor, mask: Tensor | None = None
    ) -> PartialConvOutput:
        """Run every module; return the output and the final validity mask."""
        rank = _partial_rank(self)
        layout = (max(input.ndim - 2, 0) if rank is None else rank, 1)
        output, mask, layout = self._carry(input, mask, layout)
        return PartialConvOutput(output, _all_valid(output, *layout) if mask is None else mask)

    def _carry(self, input: Tensor, mask: Tensor | None, layout: tuple[int, int]):
        """Run the modules; ``layout`` is the mask's spatial rank and channel count."""
        for module in self:
            if isinstance(module, PartialSequential):
                input, mask, layout = module._carry(input, mask, layout)
            elif isinstance(module, PartialConvNd):
                layout = module.dimensions, module.out_channels if module.multi_channel else 1
                if mask is None and module._keeps_every_sample_valid(input):
                    input = module(input, return_mask=False)
                else:
                    input, mask = module(input, mask, return_mask=True)
            elif _partial_rank(module) is not None:
                message = (
                    f"{type(module).__name__} holds partial layers the mask cannot reach; "
                    "use PartialSequential"
                )
                raise TypeError(message)
            elif mask is None:
                input = module(input)
            else:
                # Keep only the shape: holding the tensor would pin it during later layers.
                shape, rank = input.shape, layout[0]
                input = module(input)
                if input.ndim != len(shape) or _spatial(input.shape, rank) != _spatial(shape, rank):
                    message = (
                        f"{type(module).__name__} changed the shape {tuple(shape)} to "
                        f"{tuple(input.shape)} while a mask is carried; keep the spatial "
                        "shape or resample the mask around it"
                    )
                    raise ValueError(message)
        return input, mask, layout

    def __prepare_scriptable__(self):
        """Refuse TorchScript, as the partial layers inside it do."""
        raise NotImplementedError("PartialSequential does not support TorchScript")
