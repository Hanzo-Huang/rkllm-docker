# RKLLM Docker

A personal Docker test bed for running pre-converted RKLLM models on Rockchip NPUs through an OpenAI- or Ollama-compatible HTTP API.

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

Running an LLM on a Rockchip NPU requires matching ARM64 libraries, Python packages, device access, model files, and platform-specific settings. Rebuilding that environment for every experiment makes results harder to compare.

This repository uses Docker to:

- keep the RKLLM test environment repeatable;
- compare models and quantizations on the same device;
- test OpenAI- and Ollama-style API adapters;
- switch model images without rebuilding the host environment;
- record experimental speed and memory results.

Other Rockchip and embedded AI developers may use it as a reference, but the repository does not promise broad hardware compatibility or production support.

## Current Test Capabilities

- [x] Prebuilt ARM64 runtime and model images on GitHub Container Registry
- [x] Rockchip RK3576, RK3588, and RK3588S runtime configuration
- [x] OpenAI-compatible chat completions
- [x] Ollama-compatible generate and chat endpoints
- [x] Streaming responses
- [x] Host-mounted or image-bundled `.rkllm` models
- [x] Automatic platform detection and optional NPU frequency tuning
- [x] Model-image builds with optional SHA-256 verification
- [x] RK3576 performance and memory benchmarks

## Quick Start

### Requirements

- A Linux ARM64 device with a Rockchip RK3576 or RK3588-family SoC
- A working Rockchip NPU driver
- Docker Engine with permission to access `/dev`
- Enough RAM for the selected model, plus approximately 0.5 GB during startup

### 1. Start a model

The following image runs Qwen2.5 1.5B Instruct W4A16 on RK3576:

```bash
sudo docker run --rm -d \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
```

> [!NOTE]
> On RK3588, use an image tagged `w8a8-rk3588`. A model compiled for one target platform must not be used on a different target.

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
```

## Configured Model Images

Images follow this naming convention:

```text
ghcr.io/hanzo-huang/rkllm-docker/<model>:<quantization>-<platform>
```

| Model | RK3576 W4A16 | RK3576 W8A8 | RK3588 W8A8 |
| --- | :---: | :---: | :---: |
| Gemma 3 4B IT | Yes | Yes | Yes |
| Qwen2.5 1.5B Instruct | Yes | Yes | Yes |
| Qwen2.5 3B Instruct | Yes | Yes | Yes |
| Qwen3 1.7B | Yes | Yes | Yes |
| Qwen3 4B | Yes | Yes | Yes |

Example: replace the image in the Quick Start with Qwen3 4B for RK3588:

```text
ghcr.io/hanzo-huang/rkllm-docker/qwen3-4b:w8a8-rk3588
```

The model-free runtime image is:

```text
ghcr.io/hanzo-huang/rkllm-docker:env-latest
```

Model licenses and usage restrictions are determined by their original authors. Review them before deployment or redistribution.

## Test Matrix

| Platform | Test status | Model images | Notes |
| --- | --- | --- | --- |
| RK3576 | Measured | W4A16 and W8A8 | Current benchmark data comes from this platform. |
| RK3588 | Configured | W8A8 | Model definitions and platform tuning are included; no benchmark is published here. |
| RK3588S | Code path only | Bring your own | Accepted by the server and mapped to RK3588 tuning; not validated as a separate test target. |
| RK3568 | Out of scope | None | No runtime configuration, tuning script, model definition, or validation is included. |

All published images target `linux/arm64`. They are not intended to perform NPU inference on x86-64, macOS, or Windows hosts.

## Use Your Own Model

Place a platform-compatible `.rkllm` file in a host directory, then mount it into the model-free image:

```bash
sudo docker run --rm -d \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v /absolute/path/to/models:/app/models:ro \
  -e MODEL_PATH=/app/models/my-model.rkllm \
  -e TARGET_PLATFORM=rk3576 \
  ghcr.io/hanzo-huang/rkllm-docker:env-latest
```

Replace the host path, file name, and platform with values for your device. The container exits with a clear error if `MODEL_PATH` does not exist.

## Docker Compose

Clone the repository and start a published model image:

```bash
git clone https://github.com/Hanzo-Huang/rkllm-docker.git
cd rkllm-docker
sudo env \
  RKLLM_IMAGE=ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576 \
  docker compose up -d
```

To use a host-mounted model, update `MODEL_PATH` and `TARGET_PLATFORM` in `compose.mount-model.yaml`, then run:

```bash
sudo env MODEL_DIR=/absolute/path/to/models \
  docker compose -f compose.mount-model.yaml up -d
```

View startup logs:

```bash
sudo docker compose logs -f
```

## Configuration

| Variable | Default | Allowed values | Purpose |
| --- | --- | --- | --- |
| `MODEL_PATH` | `/app/models/model.rkllm` | Container file path | Select the model to load. |
| `TARGET_PLATFORM` | `auto` | `auto`, `rk3576`, `rk3588`, `rk3588s` | Select or detect the target SoC. |
| `RUN_FREQ_FIX` | `true` | `true`, `false` | Apply platform-specific frequency settings at startup. |
| `PORT` | `8001` | TCP port | Set the HTTP server port inside the container. |
| `API_FORMAT` | `openai` | `openai`, `ollama`, `both` | Enable one or both API formats. |

Set variables with `docker run -e NAME=value` or in the Compose `environment` section.

### API Endpoints

| Format | Endpoints | Streaming format |
| --- | --- | --- |
| OpenAI | `GET /v1/models`, `GET /models`, `POST /v1/chat/completions` | Server-sent events |
| Ollama | `GET /api/version`, `GET /api/tags`, `POST /api/show`, `POST /api/generate`, `POST /api/chat` | Newline-delimited JSON |
| Shared | `GET /`, `GET /health`, `GET /docs` | — |

To enable both API styles:

```bash
sudo docker run --rm -d \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -e API_FORMAT=both \
  ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
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

Already-converted Qwen and Gemma models can skip the conversion step by using a published model image. Other model families—such as DeepSeek, GLM, or Llama—require a compatible `.rkllm` conversion and enough device memory.

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
├── app/                      FastAPI server and generation metrics
├── benchmarks/               Measured speed and memory data
├── docker/                   Runtime Dockerfile, model Dockerfile, entrypoint
├── models/<model>/<variant>/ Model download and platform definitions
├── runtime/                  ARM64 RKLLM/RKNN libraries and Python wheel
├── scripts/                  Platform detection and frequency tuning
├── tests/                    Python tests
├── .github/workflows/        Environment and model image builds
├── compose.yaml              Run an image with a bundled model
└── compose.mount-model.yaml  Run the environment with a mounted model
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
  rkllm-env:dev
```

Changes under `app/`, `docker/`, `runtime/`, `scripts/`, or `requirements.txt` require rebuilding the runtime image.

### Run Tests

```bash
python -m pip install pytest
python -m pytest -q
```

### Add a Model Image

1. Create `models/<model-id>/<quantization>-<platform>/model.env`.
2. Add the direct model URL, file name, target platform, and optional checksum.
3. Commit the definition and run **Actions → Build model images → Run workflow**.

```dotenv
MODEL_URL=https://huggingface.co/<account>/<repo>/resolve/main/path/model.rkllm
MODEL_FILE=model.rkllm
TARGET_PLATFORM=rk3576
MODEL_SHA256=
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
