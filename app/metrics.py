"""Small, dependency-free inference metrics helpers."""


def output_tokens_per_second(completion_tokens: int, elapsed_seconds: float) -> float:
    """Return completion output speed, guarding against empty durations."""
    if elapsed_seconds <= 0:
        return 0.0
    return completion_tokens / elapsed_seconds
