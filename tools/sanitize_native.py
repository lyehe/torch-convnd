"""Run tests against the native addon built with AddressSanitizer and UBSan.

Linux and GCC only. Preload the sanitizer runtimes, so that every allocation, PyTorch's
tensors included, is checked:

    LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libubsan.so)" \
    ASAN_OPTIONS=detect_leaks=0 python tools/sanitize_native.py tests/test_native_compatibility.py

Any out-of-bounds access, use after free or undefined behavior in the addon aborts. The
run fails unless the tests built the sanitized addon with the ASan runtime loaded.
"""

import functools
import sys
from pathlib import Path

import pytest
import torch.utils.cpp_extension as cpp_extension

SANITIZERS = "-fsanitize=address,undefined"
COMPILE = [SANITIZERS, "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g", "-O1"]


def main() -> int:
    build = cpp_extension.load
    built = []

    @functools.wraps(build)
    def sanitized(*args, **kwargs):
        # A separate cache entry, so an unsanitized build is never reused.
        kwargs["name"] = f"{kwargs['name']}_sanitized"
        kwargs["extra_cflags"] = [*kwargs.get("extra_cflags", []), *COMPILE]
        kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), SANITIZERS]
        built.append(kwargs["name"])
        return build(*args, **kwargs)

    # load_native imports load at call time, so it builds through this wrapper.
    cpp_extension.load = sanitized
    status = pytest.main(sys.argv[1:])
    if not built:
        print("No native build ran, so nothing was sanitized", file=sys.stderr)
        return 1
    if "libasan" not in Path("/proc/self/maps").read_text():
        print("The ASan runtime is not loaded; preload libasan", file=sys.stderr)
        return 1
    for name in built:
        # pytest captures the build log, so read the flags from the build file itself.
        rules = Path(cpp_extension._get_build_directory(name, verbose=False)) / "build.ninja"
        flags = [line for line in rules.read_text().splitlines() if line.startswith("cflags")]
        if not flags or SANITIZERS not in flags[0]:
            print(f"{name} was not compiled with {SANITIZERS}", file=sys.stderr)
            return 1
        print(f"{name}: {flags[0]}")
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
