"""Compare backend policies under matched inputs, precision, gradients and timing."""

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from torch_convnd import conv_nd, load_native
from torch_convnd.benchmark import measure


def benchmark_case(name, shape, output_channels, precision, device, iterations):
    """Reject numerical regressions before collecting backend timing samples."""
    torch.manual_seed(603)
    rank = len(shape) - 2
    input = torch.randn(shape, device=device, requires_grad=True)
    weight = (
        torch.randn(output_channels, shape[1], *((3,) * rank), device=device)
        / math.sqrt(shape[1] * 3**rank)
    ).requires_grad_()
    bias = (torch.randn(output_channels, device=device) * 0.01).requires_grad_()
    tensors = input, weight, bias
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    def operation(backend):
        def run(x, w, b):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                return conv_nd(x, w, b, padding=1, workspace_bytes=64 * 1024**2, backend=backend)

        return run

    reference = operation("native")(*tensors)
    cotangent = torch.randn_like(reference) / math.sqrt(reference.numel())
    expected_grad = torch.autograd.grad(reference, tensors, cotangent)
    results = {}
    for backend in ["native", "torch", "auto"]:
        run = operation(backend)
        observed = run(*tensors)
        gradient = torch.autograd.grad(observed, tensors, cotangent)
        tolerance = 0.03 if precision == "bf16" else 0.005 if precision == "fp16" else 2e-5
        torch.testing.assert_close(observed, reference, atol=tolerance, rtol=tolerance)
        for actual, expected in zip(gradient, expected_grad, strict=True):
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        results[backend] = {
            "forward": measure(
                run, tensors, backward=False, device=device, warmup=5, iterations=iterations
            ),
            "forward_backward": measure(
                run, tensors, backward=True, device=device, warmup=5, iterations=iterations
            ),
            "max_output_error": (observed - reference).abs().max().item(),
        }
    return {
        "name": name,
        "input_shape": shape,
        "out_channels": output_channels,
        "precision": precision,
        "workspace_mib": 64,
        "results": results,
    }


def main():
    """Run a bounded matrix and write raw timing samples plus implementation hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1:
        parser.error("choose a fresh output path and a positive iteration count")
    device = torch.device(args.device)
    load_native(with_cuda=device.type == "cuda")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = (
        [
            ("2d_fp32", [1, 4, 32, 32], 8, "fp32"),
            ("3d_fp32", [1, 4, 8, 8, 8], 8, "fp32"),
            ("4d_small_fp32", [1, 4, 8, 8, 8, 16], 8, "fp32"),
        ]
        if device.type == "cuda"
        else [
            ("2d_fp32", [1, 4, 32, 32], 8, "fp32"),
            ("3d_fp32", [1, 4, 8, 8, 8], 8, "fp32"),
            ("4d_fp32", [1, 2, 4, 4, 4, 8], 4, "fp32"),
        ]
    )
    if device.type == "cuda":
        cases += [
            ("4d_medium_fp32", [1, 8, 16, 16, 16, 32], 16, "fp32"),
            ("4d_small_fp16", [1, 4, 8, 8, 8, 16], 8, "fp16"),
            ("4d_medium_fp16", [1, 8, 16, 16, 16, 32], 16, "fp16"),
            ("4d_medium_bf16", [1, 8, 16, 16, 16, 32], 16, "bf16"),
        ]
    records = []
    for name, shape, channels, precision in cases:
        record = benchmark_case(name, shape, channels, precision, device, args.iterations)
        records.append(record)
        print(
            name,
            {
                key: round(value["forward_backward"]["median_ms"], 3)
                for key, value in record["results"].items()
            },
            flush=True,
        )
    package = Path(__file__).resolve().parents[1] / "src" / "torch_convnd"
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": torch.get_num_threads(),
        "iterations": args.iterations,
        "warmup": 5,
        "tf32": False,
        "cases": records,
        "source_sha256": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.rglob("*")
            if path.suffix in {".py", ".cpp", ".cu", ".h"}
        },
        "scope": "Synchronized operator timing on a shared machine; no optimizer state.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
