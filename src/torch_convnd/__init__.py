"""Optional native operations; importing this package does not compile extensions."""

from ._build import load_native
from ._dispatch import ensure_compatible_operator
from ._operators import ensure_native_definitions
from ._version import __version__
from .diagnostics import get_runtime_info
from .functional import conv_nd
from .modules import Conv1d, Conv2d, Conv3d, Conv4d, ConvNd
from .onnx import export_onnx

ensure_native_definitions()
ensure_compatible_operator()

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Conv4d",
    "ConvNd",
    "conv_nd",
    "load_native",
    "export_onnx",
    "get_runtime_info",
    "__version__",
]
