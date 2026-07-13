# Output Speed Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log estimated completion output speed in tokens per second for every completed RKLLM inference.

**Architecture:** Add a dependency-free metrics helper in `app/metrics.py`. The shared `process_chat_completion` path will calculate and log speed after inference, covering OpenAI and Ollama plus streaming and non-streaming requests without changing API responses.

**Tech Stack:** Python 3.11, pytest, standard-library logging/formatting.

## Global Constraints

- Prompt tokens are excluded from output speed.
- Use existing `estimate_tokens` for completion token estimation.
- Use monotonic elapsed time and safely return zero for empty/zero-duration output.
- Do not change API response schemas.

---

### Task 1: Add tested output-speed helper

**Files:**
- Create: `app/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] Write a failing test for normal and zero-duration calculations.
- [ ] Run `pytest tests/test_metrics.py -q` and verify failure because the helper is missing.
- [ ] Implement `output_tokens_per_second(completion_tokens: int, elapsed_seconds: float) -> float` with a zero guard.
- [ ] Run the test and verify it passes.

### Task 2: Log speed from the shared inference path

**Files:**
- Modify: `app/fastapi_server_llm.py` in `process_chat_completion`

- [ ] Add a testable call using the helper after inference completion, using monotonic elapsed time and `estimate_tokens(req_state.full_response)`.
- [ ] Log request ID, token count, elapsed seconds, and formatted `tokens/s`.
- [ ] Run `pytest -q` and `python -m py_compile app/fastapi_server_llm.py app/metrics.py`.

### Task 3: Verify the final diff

**Files:**
- Modify: `README.md` only if the log format needs user-facing documentation.

- [ ] Run `git diff --check` and inspect the diff for API/schema changes.
- [ ] Confirm the final log line is emitted once per completed inference.
