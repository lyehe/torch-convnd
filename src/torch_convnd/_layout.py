"""Layout handling for native kernels without transform-incompatible script queries."""

import torch


def restore_native_layout(output, input, weight):
    """Preserve 2D/3D channels-last layouts after an ordinary native operation."""
    rank = weight.ndim - 2
    if rank not in {2, 3}:
        return output
    memory_format = torch.channels_last if rank == 2 else torch.channels_last_3d
    if torch.jit.is_tracing():
        channels_last = input.is_contiguous(memory_format=memory_format) and input.stride(1) == 1
        channels_last = channels_last or (
            weight.is_contiguous(memory_format=memory_format) and weight.stride(1) == 1
        )
    else:
        # Legacy portable-only releases do not provide this private helper.
        from torch._prims_common import suggest_memory_format

        channels_last = (
            suggest_memory_format(input) == memory_format
            or suggest_memory_format(weight) == memory_format
        )
    return output.contiguous(memory_format=memory_format) if channels_last else output
