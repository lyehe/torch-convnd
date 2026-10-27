"""Common spatial argument validation."""

from collections.abc import Sequence

DEFAULT_WORKSPACE_BYTES = 64 * 1024 * 1024


def _tuple(value: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * rank
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        message = f"{name} must be an integer or a sequence of integers"
        raise TypeError(message)
    result = tuple(value)
    if len(result) != rank or any(not isinstance(v, int) or isinstance(v, bool) for v in result):
        message = f"{name} must contain exactly {rank} integers"
        raise ValueError(message)
    return result
