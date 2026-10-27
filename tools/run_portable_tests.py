"""Run pre-push tests in the project environment, rather than the hook's own venv."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("TORCH_CONVND_TEST_PYTHON")
    if configured:
        executable = shutil.which(configured) or configured
    else:
        candidates = [root / ".venv/Scripts/python.exe", root / ".venv/bin/python"]
        executable = next((str(path) for path in candidates if path.is_file()), None)
        if executable is None:
            # pre-commit prepends its isolated environment to PATH. Search the
            # caller's remaining PATH when no conventional project venv exists.
            prefix = Path(sys.prefix).resolve()
            paths = [
                value
                for value in os.environ.get("PATH", "").split(os.pathsep)
                if not Path(value).resolve().is_relative_to(prefix) and "WindowsApps" not in value
            ]
            executable = shutil.which("python", path=os.pathsep.join(paths))
    if executable is None:
        raise SystemExit(
            "Create a project .venv with the dev extra, or set TORCH_CONVND_TEST_PYTHON "
            "to the Python interpreter containing your development dependencies."
        )
    return subprocess.run(
        [executable, "-m", "pytest", "-q", "-m", "portable"], cwd=root, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
