# Gemma4 E2B IT conversion and board validation

Job date: 2026-10-18. Vision and language only; audio is excluded.
Hugging Face files and published Docker images were not changed.

## Machines and artifacts

- PC: `seeed-ubuntu`, output root `/home/hanzo/workspace/models/gemma-4-E2B-it-RKLLM/v1.3.1/vlm-test-20261018`.
- Conversion server: `Seeed-Dev`, job `/home/ae/hanzos_workspace/models/jobs/gemma4-e2b-it-vlm-1.3.1-20261018`.
- Source checkpoint: `/home/ae/hanzos_workspace/models/origin/google/gemma-4-E2B-it`, previously downloaded through ModelScope. The source revision was not recorded; the job manifest identifies the exact weights/config/tokenizer files by SHA-256.
- Board: `recomputer-rk3576-devkit.local`, test root `/home/hanzo/workspace/models/gemma4-vlm-131-test`.
- Environment: `conda activate rkllm-3.11`; RKLLM Toolkit/runtime 1.3.1, RKNN Toolkit2/runtime 2.3.2, Transformers 5.8.0, Torch 2.6.0. Board NPU driver 0.9.8.

The converter checkout was updated to fork `dev` commit `6aad14d` before applying
local Gemma4 fixes. The previous server branch and existing models/output were
preserved. Generated conversion work ran in the isolated job directory.

Language exports use 4096 maximum context and two RK3576 NPU cores or three
RK3588 cores. RK3576 `w4a16_g128` uses GRQ; both `w8a8` exports use normal
quantization. Calibration captures language-model inputs for the SDK's 20 demo
images/prompts. The held-out astronaut image is not in that calibration set.

Vision preprocessing pads RGB images to a square with gray 127, resizes bilinearly
to 480 x 480, and normalizes pixels by 255. The 16-pixel patches and 3 x 3 pooling
produce 100 image tokens with 1536-dimensional embeddings. Both platform-specific
vision encoders include the projector.

## Compatibility fixes

- Transformers 5.8's mask helper mistakes JIT scalar lengths for cache-position vectors. The fixed-size exporter normalizes scalar shape arguments during tracing. A regression test checks exact agreement with the eager padding mask.
- The fixed image grid contains no padding patches. Replacing the final boolean token selection with an equivalent reshape avoids an RKNN GatherND runtime error and native heap corruption.
- RKNN's slice/concat-to-gather rewrite generates out-of-range RoPE indices; only that rewrite is disabled for Gemma4.
- RKNN lowers normalization to FP16. The original square exceeds FP16 range at vision layer 6: a residual around 505 produces a square around 255043 and becomes infinity. The resulting normalized attention inputs collapse to zero. The export wrapper rescales normalization while preserving its mathematical formula; the upstream model file and original weights are unchanged.
- Serving uses explicit Gemma4 turn formatting because the native parser cannot load the checkpoint's separate complex Jinja template. It preserves system/assistant/user roles and removes the model's end-of-turn token from responses.
- The VLM server loads the vision model by filename rather than keeping duplicate Python buffers. `RKLLM_EMBED_FLASH=1` enables native storage-backed word embeddings on the 8 GB board.
- Shared ctypes layouts match the SDK 1.3.1 header, including the new audio tail required for text and image requests.

## Validation

The repository test suites pass: 17 converter tests in the actual SDK environment,
and 8 runtime/profile/ABI tests. The ARM64 image builds and passes its native
library smoke check. Tests run with `RUN_FREQ_FIX=false`, 2048 runtime context,
`temperature=0`, `top_k=1`, and storage-backed word embeddings. The board test
containers are stopped after each run.

Text, image, and SSE streaming probes use `scripts/test_vlm_endpoint.py` and save
responses, logs, container metadata, and memory snapshots in the board report
directory. The image question asks what the astronaut holds and its color.
Acceptable wording includes a green bottle or can, matching the original model's
answer with float32 PyTorch image embeddings. This is a smoke test, not a model
quality benchmark.

Final artifact checksums, board outcomes, and numerical comparisons are recorded
in the validation files accompanying this report. RK3588 has no supplied target
board and must be treated as export-only; its RKNN compiler also emits REGTASK
bit-width warnings, retained in the conversion logs.

The existing LLM-only Hugging Face artifacts remain unchanged. Publication and
model-image registration should use separately versioned VLM artifacts after
reviewing these results.

## Numerical acceptance limit

| Vision check on the held-out image | Cosine similarity to original float32 PyTorch |
| --- | ---: |
| Final ONNX, before RKNN compilation | 0.99999988 |
| RK3576 before normalization correction | 0.72137904 |
| RK3576 with large-value rescaling | 0.97298056 |
| Final RK3576, with small-value/epsilon refinement | 0.97914374 |

The final encoder has finite output and passes the image question with both
language quantizations. However, the strict numerical checker requires cosine
similarity above 0.99, so **that numerical test still fails**. Mean absolute
embedding error is 0.08323 and maximum absolute error is 2.94509. These test
artifacts are **not approved for publication**. Broader image evaluation and
further RKNN precision investigation remain necessary before treating them as
release-quality VLMs. An additional diagnostic that casts the PyTorch vision
model wholesale to FP16 produces nonfinite output and is not a valid reference.

## Final board results

| Language artifact | Health | Text | Held-out image | SSE stream |
| --- | --- | --- | --- | --- |
| RK3576 W4A16_G128 | Pass | `4` | `The astronaut is holding a green bottle.` | `hello`, complete |
| RK3576 W8A8 | Pass | `4` | `The astronaut is holding a green can.` | `hello`, complete |
| RK3588 W8A8 | Exported | Not tested | Encoder exported with compiler warnings | Not tested |

All final artifact checksums passed on the PC. The three RK3576 files also
passed SHA-256 verification on the board. The final test image is
`rkllm-env:gemma4-test-1.3.1`, image ID
`sha256:cc9eaeb0c19316897a16479462cf4a7b5381ffc8558a187d3025569d425855d5`.
The final converter is commit `5707a36`, available as draft
[converter PR #3](https://github.com/Hanzo-Huang/rknn-llm/pull/3).

See [recorded API responses and numeric results](validation/gemma4-vlm-1.3.1-20261018.json)
and [final artifact SHA-256 manifest](validation/gemma4-vlm-1.3.1-20261018.sha256).
The PC output root also contains the complete conversion logs, board logs,
container/memory snapshots, diagnostic error analysis, source snapshot, and
conversion/source provenance manifest.
