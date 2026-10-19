# Model conversion workflow: PC → Seeed-Dev → PC → Hugging Face → Docker

This runbook coordinates `Hanzo-Huang/rknn-llm` (conversion), the Notion
conversion database (tracking), Hugging Face (publication), and this repository
(runtime images). It describes an operational workflow; an automatic pipeline
has not been implemented.

The PC has Hugging Face access. The conversion server's SSH alias is
`Seeed-Dev`; source checkpoints are downloaded directly there from ModelScope.
Activate `rkllm-3.11` before ModelScope downloads and Python 3.11 conversions.
The PC controls transfers and publication. GitHub Actions builds the images.

## 1. Prepare one conversion job

Check the [Notion conversion database](https://app.notion.com/p/38c3347c89b84fcc8cf6d57a97de71f9)
and the versioned [RKLLM 1.3.1 support list](https://github.com/airockchip/rknn-llm/blob/release-v1.3.1/README.md).
Record the source checkpoint separately from the component being exported:

| Conversion kind | Required deployable outputs | Docker model kind |
| --- | --- | --- |
| LLM | One `.rkllm` per platform/dtype | `llm` |
| VLM | Language `.rkllm` plus matching vision `.rknn` for each platform | `vlm`, after checking the runtime implements that model's input processing |
| VLM language component only | `.rkllm`; images/audio are unavailable | `llm` |

SDK support, an available encoder exporter, successful conversion, and runtime
support are separate facts. For example, SDK 1.3.1 provides Gemma4 vision/audio
exporters. This Docker application accepts Gemma4 audio when a matching audio
encoder is configured with `AUDIO_MODEL_FILE` or `AUDIO_MODEL_PATH`. Validate
audio preprocessing and real speech responses on the board before publishing.
LFM2.5-VL language support does not supply a bundled LFM vision exporter.

Create a job record before downloading. A YAML file or a Notion record plus a
local manifest is sufficient initially. Record:

- Actual ModelScope source repository and pinned revision; canonical HF model link for identification and attribution.
- Original model modality and requested conversion kind.
- Target SoC, dtype, algorithm, NPU core count, context length, and loading device/dtype.
- Converter Git commit, RKLLM/RKNN toolkit versions, and calibration dataset hash.
- For VLMs: vision wrapper, image dimensions, batch size, normalization, and encoder inputs.
- Output HF repository, Docker model ID, and optional target-board SSH alias.

Use a separate output directory for each source revision and conversion plan.
The current converter skips existing filenames; changing settings while reusing
that directory can reuse old artifacts. Use a new job directory or deliberately
rebuild with `--force`.

Recommended initial matrix for this Docker repository:

| Target | Quantizations | Default cores |
| --- | --- | --- |
| RK3576 | `w4a16_g128`, `w8a8` | 2 |
| RK3588 | `w8a8` | 3 |

`rknn-llm` also exposes RK3562/RV1126B and additional quantizations. The Docker
workflow currently accepts RK3576/RK3588; extending conversion coverage does not
automatically extend image/runtime support. Avoid `-p ALL` when only these two
Docker targets are wanted.

## 2. Download from ModelScope on Seeed-Dev

This workflow uses ModelScope on the server for checkpoint acquisition. Hugging
Face is used on the PC for output publication and source-model identification.
Record the actual ModelScope repository/revision; do not infer an HF commit
from a similarly named ModelScope repository. Confirm the intended architecture,
weights, tokenizer, processor, and configuration, then save a source file/checksum
manifest.

On the PC, keep these connection and retrieval variables for the job. Replace
example paths and placeholder values before running; remote paths in these
examples should be absolute and contain no spaces.

```bash
SERVER_HOST=Seeed-Dev
SERVER_ROOT=/home/ae/hanzos_workspace/models
WORK_ROOT=/home/hanzo/workspace/models
JOB_ID="REPLACE_WITH_UNIQUE_JOB_ID"
```

Connect to the server:

```bash
ssh Seeed-Dev
```

In the server shell, activate the required Conda environment before using
[ModelScope](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md):

```bash
conda activate rkllm-3.11

MS_REPO="REPLACE_WITH_VERIFIED_MODELSCOPE_REPOSITORY"
MS_REVISION="REPLACE_WITH_PINNED_MODELSCOPE_REVISION"
MODEL_DIR=/path/to/server/rkllm-work/models/Qwen3-1.7B

modelscope download --model "$MS_REPO" --revision "$MS_REVISION" \
  --local_dir "$MODEL_DIR"
```

Check `modelscope download --help` in that environment for the installed CLI's
syntax. Keep the complete checkpoint: weight shards and indexes, configuration,
tokenizer, processor, chat templates, and model-specific Python files. Verify
the local inventory and hashes after download. Activate `rkllm-3.11` again in
each new SSH or tmux shell before downloading or converting.

## 3. Prepare the server and convert

The server environment is `rkllm-3.11`. Before the first conversion, confirm
its Python/toolkit versions and the server OS, architecture, GPU/VRAM, and available
disk space. The PC is `seeed-ubuntu` (`/home/hanzo/workspace/models`); the
server checkout is `/home/ae/hanzos_workspace/rknn-llm`, with model/job data
under `/home/ae/hanzos_workspace/models`. RKLLM 1.3.1 conversion wheels require x86 Linux and Python
3.10–3.12. RWKV7 is an exception: it requires a separate Python 3.12 environment
and its special requirements, so do not run that conversion in `rkllm-3.11`.

Prepare `rknn-llm`'s merged `dev` checkout on the PC, record its Git commit, and
transfer it to `Seeed-Dev` if the server cannot reach GitHub. Similarly install
or transfer the appropriate toolkit wheels and dependencies before running.
A macOS wheel cache is not a Linux/CUDA dependency environment.

In the activated `rkllm-3.11` server environment:

- Install the RKLLM 1.3.1 wheel matching Python and the pinned requirements.
- Install compatible RKNN Toolkit2 for vision/audio encoder conversion.
- Set the selected CUDA device; GRQ requires a CUDA GPU. W8A8 normally uses `normal`.
- Prepare a representative calibration dataset; record its hash and any multimodal sample files.
- Pass a complete **local** checkpoint directory. Once dependencies and files
  are available, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to expose
  accidental HF dependencies before starting a long job.

Example server commands, assuming the converter checkout is the current directory:

```bash
conda activate rkllm-3.11

SERVER_ROOT=/home/ae/hanzos_workspace/models
JOB_ID="REPLACE_WITH_THE_SAME_JOB_ID_USED_ON_PC"
MODEL_ID=Qwen3-1.7B
MODEL_DIR="$SERVER_ROOT/models/$MODEL_ID"
JOB_DIR="$SERVER_ROOT/jobs/$JOB_ID"
OUTPUT_DIR="$JOB_DIR/output"
DATASET="$JOB_DIR/calibration.json"
mkdir -p "$OUTPUT_DIR"

python tools/custom_export.py -m "$MODEL_DIR" -p RK3576 -q w4a16_g128 \
  --device cuda --dataset "$DATASET" --output-dir "$OUTPUT_DIR"
python tools/custom_export.py -m "$MODEL_DIR" -p RK3576 -q w8a8 \
  --dataset "$DATASET" --output-dir "$OUTPUT_DIR"
python tools/custom_export.py -m "$MODEL_DIR" -p RK3588 -q w8a8 \
  --dataset "$DATASET" --output-dir "$OUTPUT_DIR"
```

For VLM conversion, add `--kind vlm` and the correct vision options to each
command. For a VLM language component only, add `--kind vlm --skip-vision`.
The single entry point automatically prepares missing quantized-language
calibration for supported Qwen and Gemma4 families and reuses its cached manifest.
Pass `--dataset` to use an existing manifest or `--prepare-dataset` to regenerate
automatic VLM calibration. Encoder-only and unquantized conversions do not
require language calibration. There is no separate Gemma4 calibration script. Explicitly record the vision dimensions
instead of accepting unrelated defaults.

Run long conversions in a persistent server session such as `tmux`, with a
separate log per platform/dtype and a recorded exit status. Mark conversion
complete only after every requested artifact exists and its export succeeded.
Do not overlap converter commands in the same checkout: upstream examples share
intermediate files.

Create a result manifest containing artifact relative paths, SHA-256 checksums,
conversion settings, tool versions, source provenance, and log paths. Include
LoRA correction files when the selected algorithm produces them. Do not claim
a board test based on successful export alone.

## 4. Retrieve the result bundle to the PC

From the PC, using the configured `SERVER_HOST`, `SERVER_ROOT`, `WORK_ROOT`, and
`JOB_ID` variables:

```bash
mkdir -p "$WORK_ROOT/results/$JOB_ID"
rsync -aL --partial --progress \
  "$SERVER_HOST:$SERVER_ROOT/jobs/$JOB_ID/" \
  "$WORK_ROOT/results/$JOB_ID/"
```

Verify every artifact against the server's SHA-256 manifest. A failed transfer
or checksum mismatch must be retried before publication. Retain conversion logs
and calibration provenance locally. Keep the private working bundle separate
from the folder intended for public upload.

## 5. Board validation before this release

For this Gemma4 release, both board targets are required: the PC can access
`recomputer-rk3576-devkit.local` and `recomputer-rk3588-devkit.local` directly.
Check SSH host identity before connecting; a changed key requires independent
verification before replacing the saved entry.

Copy only the selected platform's deployable files to a board with the matching
SoC. Initially use the SDK demo/runtime matching the conversion, unless the
Docker runtime's compatibility has already been established.

Minimum smoke test:

- Load the model and generate a text response.
- For VLMs, load the matched vision encoder and test an image request.
- Exercise the intended API if the model will be served through Docker.
- Record runtime/driver versions, SoC, RAM use, speed, and errors.

Record `passed`, `failed`, or `skipped` per platform/configuration. Missing board
access is `skipped`, not `passed`. A failed test stops release of that variant
until investigated; an intentionally skipped test may be published with its
untested status stated in the model card.

## 6. Publish from the PC

Prepare an upload directory containing only the intended public artifacts,
README, license/attribution files, and a sanitized conversion manifest. Exclude
caches, tokens, private calibration samples, and large intermediate ONNX files
unless deliberately publishing them. When publishing ONNX, include all external
tensor files it references.

The agent prepares the README using the manifest and actual test results:

- Exact base checkpoint and source provider/revision.
- LLM, full VLM, or language-only capability; available vision/audio encoders.
- Actual conversion toolkit version, platform/dtype file table, and hashes.
- Calibration method, context/image settings, required companion files, and usage.
- Board test results or a clear untested statement.

The connected Hugging Face MCP currently exposes repository inspection but no
direct README-write tool. Use MCP for inspection and verification; the agent can
edit the local README and publish it with the authenticated PC's `hf` CLI.

```bash
HF_OUTPUT="HanzoHuang/REPLACE_WITH_MODEL_NAME-RKLLM"
UPLOAD_DIR=/path/to/curated-upload-directory

hf upload "$HF_OUTPUT" "$UPLOAD_DIR" .
```

Use the installed CLI's help if its argument syntax differs. Verify the published
file list and README, then record the resulting HF commit. Credentials remain
on the PC; the server does not need an HF upload token.

## 7. Register compatible models in rkllm-docker

This repository bundles RKLLM **1.3.1**, with shared Python ctypes structures
verified against its vendored C++ header. Rebuild the environment image when
updating the runtime, then select that explicit environment version for model
images. Model quality and NPU behavior still require the intended device test;
ABI validation and an image build do not establish model compatibility by themselves.
For later SDK upgrades, update the native library, matching header, shared
bindings, provenance manifest, and version labels together.

For a new VLM family, check preprocessing, encoder tensor shapes, image markers,
and embedding handling in the application. Gemma4 audio additionally requires
application support before it can be offered through an image's API.

Once compatible, create one definition per requested variant:

```text
models/<llm-or-vlm>/<model-id>/<platform>/<quantization>.env
```

Populate the fields documented in [models/README.md](../models/README.md):
`MODEL_URL`, `MODEL_FILE`, `RKLLM_TOOLKIT_VERSION`, and `MODEL_SHA256`. Full VLMs
also require `VISION_MODEL_URL`, `VISION_MODEL_FILE`, `VISION_MODEL_SHA256`, and
the correct `VLM_IMAGE_SIZE`. For Gemma4 audio, also set `VLM_MODEL_FAMILY=gemma4`,
`AUDIO_MODEL_URL`, `AUDIO_MODEL_FILE`, and `AUDIO_MODEL_SHA256`; the model build
downloads, checksums, and packages this separate encoder.

Use immutable HF `/resolve/<commit>/...` URLs from the verified upload. The
artifact filename uses `w4a16_g128`; its definition/image tag uses `w4a16-g128`.
Package a VLM language-only conversion under `models/llm/`. Check companion
artifact requirements before packaging LoRA correction exports: the existing
image definitions do not automatically bundle arbitrary companion files.

## 8. Build, verify, and close the job

1. Commit/push the model definitions and any required runtime changes.
2. If the runtime changed, run **Build environment image** first and wait for
   success. Select that environment's explicit version tag for model builds.
3. Run **Build model images** with `scope=model`, the correct `model_kind`,
   `model_id`, and compatible `base_tag`. Use `scope=configuration` to retry
   one variant; avoid rebuilding the whole catalog unnecessarily.
4. Check workflow results, download/checksum validation, and published image
   tags/digests. A successful build verifies packaging; it does not prove NPU inference.
5. If a board is available, perform a final smoke test of the published image.
6. Update Notion with the HF link, actual toolkit version, per-variant conversion
   status, test result, image link/digest, and build outcome.

A workflow job is **Done** when the requested conversions are verified, artifact
transfer and HF publication are verified, required Docker definitions are
registered, image builds succeed, and board tests are either passed or explicitly
skipped. Keep failed variants visible rather than treating a partially successful
model run as complete. Existing Notion `Done` conversion statuses should not be
interpreted as proof that an image was built or a board test passed.

Resume a failed job at its failed stage using the saved manifest. A changed
checkpoint, calibration dataset, or conversion plan requires a new conversion
result; a failed image build alone does not require reconverting the model.


## Gemma4 1.3.1 procedure and current evidence

Use the updated converter checkout; record its commit before starting. The
validated audio export fixes are in converter commit `9ad8d7a`; runtime audio
serving was tested from `c87e2af`. Pull with `git pull --ff-only` when the checkout
is clean and its branch is correct. If GitHub is inaccessible from Seeed-Dev,
transfer a Git bundle from the PC and fast-forward to the recorded commit.

In a persistent server shell:

```bash
source /home/ae/miniforge3/etc/profile.d/conda.sh
conda activate rkllm-3.11
cd /home/ae/hanzos_workspace/rknn-llm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL=/home/ae/hanzos_workspace/models/origin/google/gemma-4-E2B-it
OUT=/home/ae/hanzos_workspace/models/jobs/NEW_JOB_ID/artifacts
python tools/custom_export.py --kind vlm -m "$MODEL" -p RK3576 -q w4a16_g128 --height 480 --width 480 --audio --output-dir "$OUT"
python tools/custom_export.py --kind vlm -m "$MODEL" -p RK3576 -q w8a8 --height 480 --width 480 --audio --output-dir "$OUT"
python tools/custom_export.py --kind vlm -m "$MODEL" -p RK3588 -q w8a8 --height 480 --width 480 --audio --output-dir "$OUT"
```

Each invocation handles required calibration, language, vision, and audio stages.
The two RK3576 quantizations share their platform's vision/audio encoder files.
Use `--audio-only` to repair only audio. After changing export code, create a new
output directory or use `--force` for the affected stage: an existing filename
would otherwise be reused. Record the source's actual ModelScope revision from
the download metadata; the local directory name alone does not establish it.

The export compatibility layer rescales RMS normalization before squaring to
avoid FP16 overflow, adapts fixed-shape masks/context extraction for ONNX, and
replaces Gemma4 audio's 5D attention-head transpose with equivalent per-head 3D
views and concatenation. The latter fixed RK3576 audio cosine from 0.59504 to
0.99223. Runtime audio inputs use FP16 with RKNN tensor format `UNDEFINED` (3).
These are conversion/native-input fixes; changing FastAPI routing alone cannot
repair an incorrectly exported encoder.

Validate preprocessing against the source processor, ONNX against the original
float32 model, and each native encoder against float32 embeddings on its target
board. Require finite tensors, correct shapes, and cosine at least 0.99; keep
numerical checks separate from successful HTTP responses. Then test text,
image questions, audio transcription, regular and SSE responses, 16 kHz mono,
24 kHz stereo resampling, and invalid-audio HTTP 400. Record actual answers and
latency. Run one model container at a time and stop temporary services afterward.
Use baked application code for final image checks. Keep fixtures and reports
in the job bundle rather than adding test files to the repositories.

Evidence retained on the PC:
`/home/hanzo/workspace/models/gemma-4-E2B-it-RKLLM/v1.3.1/vlm-test-20261018`.
RK3576 W4A16_G128 and W8A8 pass the functional API checks; audio cosine is
0.99222647 and ONNX/reference cosine is 0.99999999917. Two synthetic speech clips
are functional checks, not a broad ASR benchmark. W4 adds one word on the first
clip, also reproduced with reference embeddings. Vision cosine remains
0.97914374, below the 0.99 gate. RK3588 was then checked after SSH identity verification: vision cosine
0.99876202 and the green-can image question pass, but audio cosine0.34071939
and speech transcription fail. Single-core execution, float32 inputs, and the
earlier attention layout yield the same audio result. A text arithmetic check
returns5 for2+2; text SSE passes. Its vision compiler emitted REGTASK bit-width
warnings. These results do not satisfy the all-success publication gate.
Gemma4 remains unpublished; its reports are separate from the passing Spark job.

After all required checks pass, curate a public upload folder on the PC with
only the seven deployable files (three language, two vision, two audio), a model
card, attribution, hashes, and a sanitized manifest. Preserve existing LLM-only
HF paths and add the new multimodal set under a separate versioned directory.
Use `/home/hanzo/.local/bin/hf` if the noninteractive PC shell does not put it on
PATH. Verify authentication without displaying tokens, upload from the PC, read
back the remote file inventory/hashes, and record the resulting HF commit.
Register the three VLM definitions using that immutable commit; retain existing
LLM definitions. Update Notion's published capability/toolkit fields only after
the new files are verified public, with separate legacy LLM-only and new VLM
versions and measured board results. Build images with an environment image
containing the audio runtime and verify the packaged artifacts and serving API.


## Spark-X2.5-1.7B job

Spark-X2.5 is newly listed in RKLLM 1.3.1. The selected checkpoint is the
instruction model `XHToken/Spark-X2.5-1.7B`, a text-only LLM. The ModelScope
`master` download exposes no immutable commit through the installed API;
`source-checksums.json` records the 20 downloaded checkpoint files instead.

The checkpoint's custom model code uses Transformers 4.x tied-weight metadata;
automatic calibration fails under 5.8 with a list/dictionary mismatch. For this
job, activate `rkllm-3.11`, then an isolated venv using its system packages with
Transformers 4.57.1 and huggingface-hub 0.36.2. The base Conda environment retains
5.8 for Gemma4. RKLLM 1.3.1 declares Transformers 5.8 as a dependency, so this
exception is recorded in the job manifest and checked through real exports and
board inference. Do not silently change the shared environment for other jobs.

For the Spark job only, prepare the isolated environment after activating Conda:

```bash
conda activate rkllm-3.11
python -m venv --system-site-packages /path/to/job/venv
source /path/to/job/venv/bin/activate
python -m pip install transformers==4.57.1 huggingface-hub==0.36.2
```

Run the single conversion entry point from this environment. Use `deactivate`
before returning to a model that needs the base environment's Transformers 5.8.

The same `tools/custom_export.py --kind llm` command handles calibration and
conversion: pass the local checkpoint, `--max-context 4096`, the target and dtype,
and one fresh job output directory. Automatic calibration produced 21 samples.
All three requested exports succeeded. RK3576 W4A16_G128 and W8A8 passed
arithmetic, factual response, SSE, and conversation-context checks using the
baked RKLLM 1.3.1 environment image. These are functional smoke checks, not a
broad model-quality benchmark. Temporary model services were stopped afterward.
RK3588 W8A8 subsequently passed the same four checks after the user verified its
new ED25519 host fingerprint. All 12 functional checks across the three variants
passed. Checksum verification succeeded after retrieval through the PC.

The PC bundle is
`/home/hanzo/workspace/models/Spark-X2.5-1.7B-RKLLM/v1.3.1/job-20261019`.
Its server counterpart is
`/home/ae/hanzos_workspace/models/jobs/spark-x2.5-1.7b-1.3.1-20261019`.
Keep artifact checksums, source inventory, calibration, conversion logs, and
per-variant board responses with this bundle. The host key was retrieved by the board IP resolved through `getent` because
`ssh-keyscan` could not resolve the `.local` name directly; its fingerprint was
compared with the user-verified value before replacing the saved hostname entry.
All temporary test containers were stopped after testing.

After verification, the PC curates only the three deployable artifacts, README,
license, hashes, source inventory, and a sanitized manifest for HF upload. The
Docker definitions must pin the resulting HF commit. Branch environment builds
publish `env-<commit>`; only main builds update `env-latest` and runtime version
tags. Use the immutable environment tag as `base_tag` for the model image build.


For this PC, the first Xet upload stalled around 2 GB with repeated timeouts.
The retry used the [HF client's documented Xet-disable setting](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables):

```bash
HF_HUB_DISABLE_XET=1 /home/hanzo/.local/bin/hf upload \
  HanzoHuang/Spark-X2.5-1.7B-RKLLM /path/to/curated-upload .
```

Keep the upload folder unchanged during a transfer. After upload, verify all
remote LFS hashes/sizes and read back the model card and manifest before pinning
the HF commit in Docker definitions. A published repository shell or a partial
upload is not a completed model release.

Spark release references:

- HF artifacts: `HanzoHuang/Spark-X2.5-1.7B-RKLLM`, pinned artifact commit `ce762cbd90333607857da517fd7988fc626feb1c`.
- Environment build: https://github.com/Hanzo-Huang/rkllm-docker/actions/runs/39185013301; tag `env-a1335e3bd00a1ddc9fcf765f5ec2d8d6ea79525d`.
- Model builds: https://github.com/Hanzo-Huang/rkllm-docker/actions/runs/39187189317; source commit `d0321a83a21e190cee437d2af538f88617615b45`.
- Registry prefix: `ghcr.io/hanzo-huang/rkllm-docker/llm/spark-x2.5-1.7b`.
- Immutable tags: `rk3576-w4a16-g128-d0321a83a21e190cee437d2af538f88617615b45`, `rk3576-w8a8-d0321a83a21e190cee437d2af538f88617615b45`, `rk3588-w8a8-d0321a83a21e190cee437d2af538f88617615b45`.

For final image verification, pull ARM64 images on the PC, record their registry
digests and image content IDs, then use `docker save`/`docker load` to transfer
them to the target boards. A saved archive may omit the registry digest, so
compare the board image's content ID with the PC's verified registry image.
Check the packaged model SHA-256, then start the image without a host model
mount and repeat the same API acceptance cases. All three published Spark
images passed this final check; temporary services were stopped. Keep image
inspection data, registry digests, hashes, and final responses in the job bundle.
