"""Common spatial argument validation."""

from collections.abc import Sequence

DEFAULT_WORKSPACE_BYTES = 64 * 1024 * 1024


def any_true(values: list) -> bool:
    """``any`` over a list, in a form that compiles as one graph from PyTorch 2.0.

    Dynamo before PyTorch 2.5 cannot trace ``any``/``all`` over generators, nor over
    symbolic sizes; a loop over a list comprehension traces on every release.
    """
    for value in values:
        if value:
            return True
    return False


def _tuple(value: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * rank
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        message = f"{name} must be an integer or a sequence of integers"
        raise TypeError(message)
    result = tuple(value)
    if len(result) != rank or any_true(
        [not isinstance(v, int) or isinstance(v, bool) for v in result]
    ):
        message = f"{name} must contain exactly {rank} integers"
        raise ValueError(message)
    return result
