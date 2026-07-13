# Output Speed Logging Design

## Goal

Make model output speed easy to read from server logs without changing the OpenAI- or Ollama-compatible API responses.

## Behavior

After each successful inference, the server logs output speed in tokens per second. The metric is calculated as the estimated number of completion tokens divided by elapsed inference time. Prompt tokens are excluded.

The completion token count uses the server's existing `estimate_tokens` helper so the new metric remains consistent with existing usage and Ollama statistics. Elapsed time uses a monotonic clock to avoid wall-clock adjustments affecting the result.

The log entry includes the request ID, estimated completion token count, elapsed seconds, and speed. Empty responses and zero-duration measurements produce `0.00 tokens/s` without raising an error.

## Scope

- Apply once in the shared inference completion path so OpenAI and Ollama requests, streaming and non-streaming, get the same log output.
- Do not add fields or headers to API responses.
- Do not change tokenization or inference behavior.

## Verification

Unit tests cover normal speed calculation and safe handling of zero duration. A logging-path test verifies that a completed request emits the expected tokens-per-second metric.
