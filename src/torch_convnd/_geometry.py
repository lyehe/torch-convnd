"""Common spatial argument validation."""

from collections.abc import Sequence

DEFAULT_WORKSPACE_BYTES = 64 * 1024 * 1024


def any_true(values: list) -> bool:
    """``any`` over a list, in a form every Dynamo release compiles as one graph.

    Dynamo does not trace ``any``/``all`` over generators on every release, nor over
    symbolic sizes before PyTorch 2.5; a loop over a list comprehension always traces.
    """
    for value in values:
        if value:
            return True
    return False


def has_rank(ndim: int, rank: int) -> bool:
    """Whether ``ndim`` dimensions hold a batched or unbatched rank-``rank`` input.

    Plain comparisons: Dynamo can make integer module attributes symbolic, and testing
    membership against them does not trace on every release.
    """
    return ndim == rank + 2 or ndim == rank + 1


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
