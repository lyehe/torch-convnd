"""Padding semantics shared by every public spatial rank."""

import torch
from torch.nn import functional as F


def padding_pairs(padding, kernel, stride, dilation):
    """Resolve PyTorch's integer or string padding to left/right pairs."""
    if isinstance(padding, str):
        if padding not in {"same", "valid"}:
            raise ValueError(f"Invalid padding string {padding!r}; expected 'same' or 'valid'")
        if padding == "same" and any(step != 1 for step in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
        return tuple(
            (total // 2, total - total // 2)
            for total in (
                (size - 1) * dil if padding == "same" else 0
                for size, dil in zip(kernel, dilation, strict=True)
            )
        )
    return tuple((pad, pad) for pad in padding)


def apply_padding(input, pairs, mode):
    """Apply zero, reflection, replication or circular padding over N axes."""
    if not any(left or right for left, right in pairs):
        return input
    if mode == "zeros":
        return F.pad(input, tuple(v for pair in reversed(pairs) for v in pair))
    if len(pairs) <= 3:
        return F.pad(input, tuple(v for pair in reversed(pairs) for v in pair), mode=mode)
    for axis, (left, right) in enumerate(pairs, start=2):
        if not (left or right):
            continue
        size = input.shape[axis]
        if mode == "reflect" and (left >= size or right >= size):
            raise RuntimeError("Reflection padding must be less than the input dimension")
        if mode == "circular" and (left > size or right > size):
            raise RuntimeError("Padding value causes wrapping around more than once")
        coordinates = torch.arange(-left, size + right, device=input.device)
        if mode == "replicate":
            indices = coordinates.clamp(0, size - 1)
        elif mode == "circular":
            indices = coordinates.remainder(size)
        else:
            indices = torch.where(coordinates < 0, -coordinates, coordinates)
            indices = torch.where(indices >= size, 2 * (size - 1) - indices, indices)
        input = input.index_select(axis, indices)
    return input
