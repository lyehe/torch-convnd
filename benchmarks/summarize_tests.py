"""Archive pytest JUnit, numerical measurements and environment/source provenance."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


def package_version(name):
    """Record optional runtime packages without requiring them for CPU checks."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    """Write a fresh report, preserving all historical experiment records."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path")
    document = ET.parse(args.junit)
    cases = []
    modules = defaultdict(Counter)
    features = Counter()
    for case in document.iter("testcase"):
        outcome = next(
            (tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None),
            "passed",
        )
        module, name = case.get("classname"), case.get("name")
        modules[module][outcome] += 1
        features[f"{module}::{name.split('[')[0]}"] += 1
        cases.append(
            {"module": module, "name": name, "outcome": outcome, "seconds": float(case.get("time"))}
        )
    measurement_path = args.measurements or args.junit.with_suffix(".measurements.json")
    if not measurement_path.exists() and args.measurements is None:
        measurement_path = root / ".artifacts" / "parity_measurements.json"
    measurements = json.loads(measurement_path.read_text(encoding="utf-8"))
    errors = defaultdict(list)
    for value in measurements:
        errors[value["dtype"]].append(value["relative_l2_error"])
    sources = [root / "pyproject.toml"]
    for folder in ("src", "tests", "benchmarks"):
        sources.extend(
            path
            for path in (root / folder).rglob("*")
            if path.suffix in {".py", ".pyi", ".cpp", ".cu", ".h"} or path.name == "py.typed"
        )
    counts = Counter(case["outcome"] for case in cases)
    report = {
        "package_version": package_version("torch-convnd"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "optional_packages": {
            name: package_version(name)
            for name in ("triton-windows", "triton", "onnx", "onnxscript", "onnxruntime")
        },
        "counts": {key: counts[key] for key in ("passed", "failure", "error", "skipped")},
        "seconds": sum(float(suite.get("time", 0)) for suite in document.iter("testsuite")),
        "module_counts": modules,
        "feature_case_counts": features,
        "numerical_summary": {
            dtype: {"comparisons": len(values), "max_relative_l2_error": max(values)}
            for dtype, values in sorted(errors.items())
        },
        "numerical_comparisons": measurements,
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
        },
        "junit_sha256": hashlib.sha256(args.junit.read_bytes()).hexdigest(),
        "tests": cases,
        "scope": "Measured coverage only; see docs/PARITY.md for unverified platforms and limits.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
