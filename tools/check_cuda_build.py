"""Compile and load the CUDA addon on a machine without a GPU.

This checks that the CUDA sources build against the installed PyTorch and toolkit,
that every kernel registers, and that the CPU kernels of that build are correct.
GPU execution still needs a real device; see the CUDA workflow.
"""

import argparse
import json
import os

import torch

from torch_convnd import _build, conv_nd, load_native
from torch_convnd._compat import has_fp32_output_gemm

KERNELS = [
    ("conv_nd", "CPU"),
    ("conv_nd_backward", "CPU"),
    ("conv_nd", "AutocastCPU"),
    ("conv_nd", "CUDA"),
    ("conv_nd_backward", "CUDA"),
    ("conv_nd", "AutocastCUDA"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="8.0", help="TORCH_CUDA_ARCH_LIST unless already set")
    args = parser.parse_args()
    if torch.version.cuda is None:
        raise SystemExit("A CUDA build of PyTorch is required")
    # Without a device, the arch list cannot be detected and would default to every target.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.arch)
    load_native(with_cuda=True, verbose=True)
    assert _build._WITH_CUDA, "the addon loaded without its CUDA kernels"
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    missing = [
        f"{name}/{key}" for name, key in KERNELS if not has_kernel(f"torch_convnd::{name}", key)
    ]
    if missing:
        raise SystemExit(f"Kernels missing after the CUDA build: {missing}")
    torch.manual_seed(1013)
    input = torch.randn(2, 4, 3, 4, 3, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 2, 2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(6, dtype=torch.float64, requires_grad=True)
    options = {"padding": 1, "stride": (1, 2, 1, 1), "groups": 2}
    actual = conv_nd(input, weight, bias, backend="native", **options)
    expected = conv_nd(input, weight, bias, backend="torch", **options)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (input, weight, bias), cotangent)
    second = torch.autograd.grad(expected, (input, weight, bias), cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
                "fp32_output_gemm": has_fp32_output_gemm(),
                "kernels": [f"{name}/{key}" for name, key in KERNELS],
                "cpu_parity": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
