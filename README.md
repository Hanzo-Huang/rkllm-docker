# RKLLM Docker

A personal Docker test bed for running pre-converted RKLLM LLM and VLM models on Rockchip NPUs through an OpenAI- or Ollama-compatible HTTP API.

[![Build environment image](https://github.com/Hanzo-Huang/rkllm-docker/actions/workflows/build-env-image.yml/badge.svg)](https://github.com/Hanzo-Huang/rkllm-docker/actions/workflows/build-env-image.yml)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64-blue)
![Project status](https://img.shields.io/badge/status-personal%20testing-orange)

RKLLM Docker packages the Rockchip RKLLM runtime, RKNN Toolkit Lite, device setup, and a FastAPI server into `linux/arm64` images. The repository exists primarily to reproduce experiments across RK3576 and RK3588 devices, models, and quantizations.

> [!IMPORTANT]
> This repository **runs and packages pre-converted `.rkllm` models**. Model conversion is performed with Rockchip's RKLLM toolchain before using this project.

> [!WARNING]
> **Personal testing project:** This is not a stable release or a complete RKLLM platform. Bugs, missing functions, API compatibility gaps, and breaking changes are expected. Results reflect the author's test environment; verify everything on your own hardware before relying on it.

**Start here:** [Quick Start](#quick-start) · [Use your own model](#use-your-own-model) · [Development](#development)

## Purpose

Run pre-converted RKLLM models on Rockchip NPUs in reproducible ARM64
containers. Each image bundles the matching runtime, model, platform setup,
and HTTP server so models can be compared with one Docker command.

## What's included

- ARM64 images for RK3576 and RK3588-family devices
- LLM images with OpenAI-compatible and Ollama-compatible APIs
- VLM images with OpenAI-compatible image chat
- Streaming responses, host-mounted models, and SHA-256 checks
- RK3576 benchmark data for the configured LLM models

This is a personal testing project, not a production inference platform.

## Bundled Versions

- RKLLM runtime: `v1.3.0`
- RKNN Toolkit Lite 2 wheel: `v2.3.2`

The RKLLM Toolkit version used to convert a model is stored in that model's
env definition and published as `io.rkllm.toolkit.version` on its image. This
allows models produced by different toolkit versions to coexist.

## Quick Start

### Requirements

- A Linux ARM64 device with a Rockchip RK3576 or RK3588-family SoC
- A working Rockchip NPU driver
- Docker Engine with permission to access `/dev`
- Enough RAM for the selected model, plus approximately 0.5 GB during startup

### 1. Start a model

#### LLM

The following image runs Qwen2.5 1.5B Instruct W4A16 on RK3576:

```bash
sudo docker run --rm -it \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -e INTERACTIVE_CHAT=true \
  -e LOG_LEVEL=warning \
  ghcr.io/hanzo-huang/rkllm-docker/llm/qwen2.5-1.5b-instruct:rk3576-w4a16
```

With this command, wait for `Chat demo ready`, then type at the `You:` prompt.
Use `/reset` to clear the conversation or `/exit` to stop the container. The
HTTP API remains available at `http://localhost:8001` while the terminal chat
is running.

For a background/API-only LLM:

```bash
sudo docker run --rm -d \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  ghcr.io/hanzo-huang/rkllm-docker/llm/qwen2.5-1.5b-instruct:rk3576-w4a16
```

> [!NOTE]
> On RK3588, use an image tagged `rk3588-w8a8`. A model compiled for one target platform must not be used on a different target.

#### VLM

The following image runs Qwen3.5 2B W4A16-g128 on RK3576:

```bash
sudo docker run --rm -it \
  --name rkllm-vlm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  ghcr.io/hanzo-huang/rkllm-docker/vlm/qwen3.5-2b:rk3576-w4a16-g128
```

Both LLM and VLM images use port `8001`.

### 2. Check the server

Model initialization may take a moment. Wait until the health endpoint reports `healthy`:

```bash
curl http://localhost:8001/health
```

Interactive API documentation is available at `http://localhost:8001/docs`.

### 3. Send a chat request

```bash
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Explain edge AI in one sentence."}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

### Stop the Server

```bash
sudo docker stop rkllm
sudo docker stop rkllm-vlm
```

## Configured Model Images

Images follow this naming convention:

```text
ghcr.io/hanzo-huang/rkllm-docker/<llm-or-vlm>/<model>:<platform>-<quantization>
```

| LLM model | RK3576 W4A16 | RK3576 W4A16-g128 | RK3576 W8A8 | RK3588 W8A8 |
| --- | :---: | :---: | :---: | :---: |
| DeepSeek R1 Distill Qwen 1.5B | — | ✅ | ✅ | ✅ |
| Gemma 3 4B IT (LLM-only conversion) | ✅ | — | ✅ | ✅ |
| Gemma 4 E2B IT (LLM-only conversion) | — | ✅ | ✅ | ✅ |
| Llama 3.2 1B Instruct | — | ✅ | ✅ | ✅ |
| Llama 3.2 3B Instruct | — | ✅ | ✅ | ✅ |
| MiniCPM3 4B | — | ✅ | ✅ | ✅ |
| MiniCPM4 0.5B | — | ✅ | ✅ | ✅ |
| Qwen2 0.5B Instruct | — | ✅ | ✅ | ✅ |
| Qwen2.5 1.5B Instruct | ✅ | — | ✅ | ✅ |
| Qwen2.5 3B Instruct | ✅ | — | ✅ | ✅ |
| Qwen3 1.7B | ✅ | ✅ | ✅ | ✅ |
| Qwen3 4B | ✅ | ✅ | ✅ | ✅ |

For future W4A16 conversions, use **W4A16-g128** (`w4a16-g128` in image tags
and definition names) instead of W4A16 (`w4a16`). The older `w4a16` variant
remains listed only for existing model artifacts and compatibility.

Gemma 3 4B IT and Gemma 4 E2B IT are VLM-capable upstream models, but their
RKLLM repositories contain only language-model artifacts. In this project they
are LLM images and do not accept image input.

VLM images use a separate package namespace and include both the RKLLM language
model and RKNN vision encoder:

| VLM model | RK3576 W4A16-g128 | RK3576 W8A8 | RK3588 W8A8 |
| --- | :---: | :---: | :---: |
| Qwen3.5 2B | ✅ | ✅ | ✅ |
| Qwen3.5 4B | ✅ | ✅ | ✅ |

Toolkit versions are recorded in every model definition and model card:

- **RKLLM Toolkit v1.2.3:** Gemma 3 4B IT, Qwen2.5 1.5B, Qwen2.5 3B, and Qwen3 4B.
- **RKLLM Toolkit v1.3.0:** DeepSeek R1 Distill Qwen 1.5B, MiniCPM3 4B,
  MiniCPM4 0.5B, Qwen2 0.5B, Qwen3 1.7B, Llama 3.2 1B/3B,
  Gemma 4 E2B IT, and Qwen3.5 2B/4B.

Example VLM image:

```text
ghcr.io/hanzo-huang/rkllm-docker/vlm/qwen3.5-2b:rk3576-w4a16-g128
```

Model image tags use `<platform>-<quantization>`; runtime environment tags use
the RKLLM runtime version, currently `1.3.0`.

Model licenses and usage restrictions are determined by their original authors. Review them before deployment or redistribution.

## Test Matrix

| Platform | Test status | Model images | Notes |
| --- | --- | --- | --- |
| RK3576 | Measured | W4A16 and W8A8 | Current benchmark data comes from this platform. |
| RK3588 | Configured | W8A8 | Model definitions and platform tuning are included; no benchmark is published here. |

All published images target `linux/arm64`. They are not intended to perform NPU inference on x86-64, macOS, or Windows hosts.

## Use Your Own Model

The model-free runtime supports two separate modes. Use `MODEL_KIND=llm` for a
text-only model, or `MODEL_KIND=vlm` when you have both an RKLLM language model
and an RKNN vision encoder.

### Custom LLM

Mount one platform-compatible `.rkllm` file into the model-free image:

```bash
sudo docker run --rm -it \
  --name rkllm-llm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v /absolute/path/to/models:/app/models:ro \
  -e MODEL_KIND=llm \
  -e MODEL_FILE=my-model.rkllm \
  -e TARGET_PLATFORM=rk3576 \
  -e API_MODEL_NAME=my-llm \
  ghcr.io/hanzo-huang/rkllm-docker:env-latest
```

Replace the host path, file name, and platform with values for your device.
The container exits with a clear error if the selected model file does not
exist.

### Custom VLM

Mount the paired `.rkllm` language model and `.rknn` vision encoder, and select
the VLM backend explicitly:

```bash
sudo docker run --rm -it \
  --name rkllm-vlm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v /absolute/path/to/models:/app/models:ro \
  -e MODEL_KIND=vlm \
  -e MODEL_FILE=my-vlm-language-model.rkllm \
  -e VISION_MODEL_FILE=my-vlm-vision-model.rknn \
  -e TARGET_PLATFORM=rk3576 \
  -e API_MODEL_NAME=my-vlm \
  ghcr.io/hanzo-huang/rkllm-docker:env-latest
```

The `.rkllm` and `.rknn` files must be a matching conversion for the same
platform. The VLM API is available at `http://localhost:8001/v1`.

For interactive text chat with a custom LLM, use `-it`, set
`INTERACTIVE_CHAT=true`, and set `LOG_LEVEL=warning`. Interactive chat is not
available for the VLM server; use its HTTP API for image requests.

## Configuration

| Variable | Default | Allowed values | Purpose |
| --- | --- | --- | --- |
| `MODEL_PATH` | `/app/models/model.rkllm` | Container file path | Select the RKLLM model to load. |
| `MODEL_FILE` | empty | Filename under `/app/models` | Select the RKLLM model by filename; overrides `MODEL_PATH` when set. |
| `MODEL_KIND` | `llm` | `llm`, `vlm` | Select the native model backend. |
| `API_MODEL_NAME` | `rkllm-model` for LLM, `rkllm-vision` for VLM | Model name | Optional stable public name returned by `/v1/models` and API responses. |
| `VISION_MODEL_PATH` | empty | Container file path | RKNN vision encoder path for VLM images. |
| `VISION_MODEL_FILE` | empty | Filename under `/app/models` | Select the VLM vision encoder by filename; overrides `VISION_MODEL_PATH` when set. |
| `TARGET_PLATFORM` | `auto` | `auto`, `rk3576`, `rk3588`, `rk3588s` | Select or detect the target SoC. |
| `RUN_FREQ_FIX` | `true` | `true`, `false` | Apply platform-specific frequency settings at startup. |
| `PORT` | `8001` | TCP port | Set the HTTP server port inside the container. |
| `LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug` | Set Python and Uvicorn logging verbosity. |
| `INTERACTIVE_CHAT` | `false` | `true`, `false` | Open the LLM terminal chat on the container's stdin/stdout. |

Set variables with `docker run -e NAME=value`.

For a clean terminal conversation immediately after starting an LLM container,
attach stdin/TTY, enable interactive chat, and lower routine logging:

```bash
sudo docker run --rm -it \
  --name rkllm-chat \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -e INTERACTIVE_CHAT=true \
  -e LOG_LEVEL=warning \
  ghcr.io/hanzo-huang/rkllm-docker/llm/qwen2.5-1.5b-instruct:rk3576-w4a16
```

Interactive mode disables Uvicorn access logs automatically. Use
`LOG_LEVEL=info` or `LOG_LEVEL=debug` when startup/request diagnostics are
needed; those messages may appear alongside the conversation.

### API Endpoints

| Format | Endpoints | Streaming format |
| --- | --- | --- |
| OpenAI | `GET /v1/models`, `POST /v1/chat/completions` | Server-sent events |
| Ollama | `GET /api/tags`, `POST /api/generate`, `POST /api/chat` | Newline-delimited JSON |
| Shared | `GET /`, `GET /health`, `GET /docs` | — |

LLM images expose both API styles by default. VLM images expose the
OpenAI-compatible endpoints.

### VLM Image Request

VLM requests use the standard OpenAI image content format. The server accepts
one image per request as an HTTPS URL or base64 data URL:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rkllm-vision",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
      {"type": "text", "text": "Describe this image."}
    ]}],
    "max_tokens": 128
  }'
```

### Ollama-Compatible Example

```bash
curl http://localhost:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Hello from RKLLM."}
    ],
    "stream": false
  }'
```

## Running an Experiment

1. Choose a model and convert it to `.rkllm` with Rockchip's tools, or select a configured model image.
2. Use a model built for the exact Rockchip target platform.
3. Mount the model or start its model-specific image.
4. Send the same prompts and settings to each test candidate.
5. Record the device, runtime, model, quantization, memory, and token speed with the result.

Already-converted Qwen, Gemma, Llama, MiniCPM, and DeepSeek R1 Distill Qwen
models can skip the conversion step by using a published model image. Other
model families—such as GLM—require a compatible `.rkllm` conversion and enough
device memory.

## Performance

These experimental RK3576 results use the same long-form prompt. Stable RAM is measured after startup; allow approximately 0.5 GB of additional free memory while loading. They are observations from one setup, not performance guarantees or general hardware benchmarks.

| Model | Quantization | Stable RAM | Output speed |
| --- | --- | ---: | ---: |
| Qwen2.5 1.5B Instruct | W4A16 | **1.6 GB** | **19.55 tok/s** |
| Qwen2.5 1.5B Instruct | W8A8 | 2.2 GB | 13.99 tok/s |
| Qwen3 1.7B | W4A16 | **2.0 GB** | **13.31 tok/s** |
| Qwen3 1.7B | W8A8 | 2.7 GB | 12.57 tok/s |
| Qwen2.5 3B Instruct | W4A16 | **2.5 GB** | **11.45 tok/s** |
| Qwen2.5 3B Instruct | W8A8 | 3.8 GB | 6.21 tok/s |
| Qwen3 4B | W4A16 | **3.4 GB** | 8.30 tok/s |
| Qwen3 4B | W8A8 | 5.2 GB | 5.85 tok/s |
| Gemma 3 4B IT | W4A16 | **4.0 GB** | **8.40 tok/s** |
| Gemma 3 4B IT | W8A8 | 5.5 GB | 5.33 tok/s |

<details>
<summary>Benchmark method and interpretation</summary>

- Platform: RK3576
- Metric: observed output tokens per second during generation
- RAM: stable usage after model startup
- Prompt: one identical, long-form transformer explanation prompt for every model
- Quality: not evaluated; the results compare throughput and memory only

W4A16 was faster and used less memory than W8A8 in every recorded test. Results are comparative, not guaranteed: cooling, clock settings, runtime versions, and background workloads affect performance.

The machine-readable results are in [`benchmarks/rk3576.csv`](benchmarks/rk3576.csv).

</details>

## Repository Structure

```text
.
├── app/                      Separate LLM and VLM FastAPI servers
├── benchmarks/               Measured speed and memory data
├── docker/                   Runtime Dockerfile, model Dockerfile, entrypoint
├── models/<llm-or-vlm>/<model>/<platform>/<quantization>.env
│                             Model download definitions
├── runtime/                  ARM64 RKLLM/RKNN libraries and Python wheel
├── scripts/                  Platform detection and frequency tuning
└── .github/workflows/        Environment and model image builds
```

## Development

### Build the runtime image

The Dockerfile only builds for ARM64. Build on an ARM64 host or use Docker Buildx:

```bash
docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile \
  -t rkllm-env:dev \
  --load .
```

Run the local image with your model:

```bash
sudo docker run --rm -it \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v /absolute/path/to/models:/app/models:ro \
  -e MODEL_PATH=/app/models/my-model.rkllm \
  -e TARGET_PLATFORM=rk3576 \
  -e INTERACTIVE_CHAT=true \
  -e LOG_LEVEL=warning \
  rkllm-env:dev
```

Changes under `app/`, `docker/`, `runtime/`, `scripts/`, or `requirements.txt` require rebuilding the runtime image.

### Add a Model Image

1. Create `models/<llm-or-vlm>/<model-id>/<platform>/<quantization>.env`.
2. Add the direct model URL, file name, and checksum. The platform is
   derived from its directory.
3. Commit the definition and run **Actions → Build model images → Run workflow**.

The workflow's `scope` selects how many definitions to build:

| Scope | Required filters | Builds |
| --- | --- | --- |
| `all` | None | Every LLM and VLM definition under `models/`. |
| `model` | `model_kind`, `model_id` | Every platform and quantization variant of one model. |
| `configuration` | `model_kind`, `model_id`, `quantization`, `platform` | One exact variant; use `platform=all` for that quantization on every platform. |

For `scope=all`, leave `model_id`, `platform`, and `quantization` empty. The
`model_kind` value is ignored in this scope.

Use `image_tag` only when building one exact configuration. Leave it empty for
`platform=all`; the workflow then creates unique `<platform>-<quantization>`
tags and prevents platform images from overwriting each other.

```dotenv
MODEL_URL=https://huggingface.co/<account>/<repo>/resolve/main/path/model.rkllm
MODEL_FILE=model.rkllm
RKLLM_TOOLKIT_VERSION=1.2.3
MODEL_SHA256=<lowercase-sha256>
```

Use a direct `/resolve/` URL, not a `/blob/` page. Never commit access tokens. For a private model, configure the `MODEL_DOWNLOAD_TOKEN` repository secret.

See [`models/README.md`](models/README.md) for build scopes, image tags, and private download options.

## Limitations

- Model conversion is not included; input models must already be in `.rkllm` format.
- The runtime is ARM64-only and requires compatible Rockchip NPU drivers.
- Containers currently need privileged device access and a `/dev` mount.
- Only one model is loaded per container.
- The server defaults to two concurrent requests and serializes access to the model where required by the runtime.
- Authentication, TLS termination, quotas, and multi-tenant isolation are not built in. Put a trusted reverse proxy in front of the API before exposing it outside a private network.
- Available model families and context behavior depend on the RKLLM runtime and the converted model.

## Feedback

This is a personal experiment, so there is no formal support or release schedule. Reproducible issues and tested pull requests are still welcome.

Include the board, SoC, operating system, kernel, NPU driver, image tag, model variant, test command, and relevant logs with compatibility reports. Keep model binaries, credentials, and download tokens out of Git.
