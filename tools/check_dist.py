"""Validate wheel/sdist metadata and bundled sources without importing PyTorch."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def source_version(root: Path) -> str:
    """Read the version without executing package code or importing dependencies."""
    document = ast.parse((root / "src/torch_convnd/_version.py").read_text(encoding="utf-8"))
    for node in document.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing __version__ in _version.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--install", action="store_true", help="install the checked wheel with pip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = source_version(root)
    tag = os.environ.get("RELEASE_TAG")
    if tag and tag != f"v{version}":
        parser.error(f"release tag {tag!r} does not match v{version}")
    wheel = args.dist_dir / f"torch_convnd-{version}-py3-none-any.whl"
    sdist = args.dist_dir / f"torch_convnd-{version}.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        required = [
            "torch_convnd/py.typed",
            "torch_convnd/__init__.pyi",
            "torch_convnd/csrc/convnd.cpp",
            "torch_convnd/csrc/convnd.h",
            "torch_convnd/csrc/convnd_cuda.cu",
            "torch_convnd/csrc/convnd_mixed.cpp",
        ]
        for name in required:
            if name not in names:
                raise ValueError(f"Wheel is missing {name}")
        for path in (root / "src/torch_convnd").rglob("*"):
            if path.suffix in {".py", ".pyi", ".cpp", ".cu", ".h"} or path.name == "py.typed":
                name = path.relative_to(root / "src").as_posix()
                if archive.read(name) != path.read_bytes():
                    raise ValueError(f"Wheel differs from source: {name}")
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata["Version"] != version:
            raise ValueError("Wheel metadata version disagrees with source")
        requirements = metadata.get_all("Requires-Dist", [])
        mandatory = [value for value in requirements if "extra ==" not in value]
        if len(mandatory) != 1 or not mandatory[0].startswith("torch>=1.11"):
            raise ValueError(f"Unexpected mandatory dependencies: {mandatory}")
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        for suffix in [
            "pyproject.toml",
            "README.md",
            "tools/smoke_test.py",
            "tests/test_convnd.py",
        ]:
            if not any(name.endswith("/" + suffix) for name in names):
                raise ValueError(f"Source archive is missing {suffix}")
    print(
        json.dumps(
            {
                "version": version,
                "artifacts": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (wheel, sdist)
                },
                "sources_match": True,
                "typed": True,
                "mandatory_dependencies": mandatory,
            },
            indent=2,
        )
    )
    if args.install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)],
            check=True,
        )


if __name__ == "__main__":
    main()
