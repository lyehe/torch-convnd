"""Backward-compatible imports; implementations live in focused modules."""

from ._build import _build_environment as _build_environment
from ._build import load_native
from ._geometry import DEFAULT_WORKSPACE_BYTES as DEFAULT_WORKSPACE_BYTES
from .functional import conv_nd
from .modules import Conv1d, Conv2d, Conv3d, Conv4d, ConvNd

__all__ = ["Conv1d", "Conv2d", "Conv3d", "Conv4d", "ConvNd", "conv_nd", "load_native"]
