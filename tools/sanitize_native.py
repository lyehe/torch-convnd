"""Run tests against the native addon built with AddressSanitizer and UBSan.

Linux and GCC only. Preload the sanitizer runtimes, so that every allocation, PyTorch's
tensors included, is checked:

    LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libubsan.so)" \
    ASAN_OPTIONS=detect_leaks=0 python tools/sanitize_native.py tests/test_native_compatibility.py

Any out-of-bounds access, use after free or undefined behavior in the addon aborts.
"""

import functools
import sys

import pytest
import torch.utils.cpp_extension as cpp_extension

SANITIZERS = "-fsanitize=address,undefined"
COMPILE = [SANITIZERS, "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-g", "-O1"]


def main() -> int:
    build = cpp_extension.load

    @functools.wraps(build)
    def sanitized(*args, **kwargs):
        # A separate cache entry, so an unsanitized build is never reused.
        kwargs["name"] = f"{kwargs['name']}_sanitized"
        kwargs["extra_cflags"] = [*kwargs.get("extra_cflags", []), *COMPILE]
        kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), SANITIZERS]
        return build(*args, **kwargs)

    # load_native imports load at call time, so it builds through this wrapper.
    cpp_extension.load = sanitized
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
