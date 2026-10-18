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
exporters, but this Docker application currently has no audio API/backend.
Do not publish an audio-capable Docker image merely because audio conversion
succeeded. LFM2.5-VL language support does not supply a bundled LFM vision
exporter.

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
SERVER_ROOT=/path/to/server/rkllm-work
WORK_ROOT="$HOME/rkllm-work"
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
its Python/toolkit versions and the server OS, architecture, GPU/VRAM, available
disk space, and actual workspace paths; those hardware/path details remain
unconfirmed. RKLLM 1.3.1 conversion wheels require x86 Linux and Python
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

SERVER_ROOT=/path/to/server/rkllm-work
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
Automatic VLM calibration is available only for the supported Qwen helpers;
other families need a prepared manifest. Explicitly record the vision dimensions
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

## 5. Optional board test

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
the correct `VLM_IMAGE_SIZE`.

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
