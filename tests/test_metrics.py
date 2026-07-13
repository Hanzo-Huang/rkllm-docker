from app.metrics import output_tokens_per_second


def test_output_tokens_per_second_uses_elapsed_generation_time():
    assert output_tokens_per_second(40, 2.0) == 20.0


def test_output_tokens_per_second_returns_zero_for_zero_duration():
    assert output_tokens_per_second(40, 0.0) == 0.0
