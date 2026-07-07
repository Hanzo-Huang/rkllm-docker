# RKLLM Docker

Docker images for serving Rockchip RKLLM models on ARM64 devices through an
OpenAI-compatible HTTP API.

## Highlights

- Run RKLLM models on Rockchip RK3576 and RK3588 devices with Docker.
- Use an OpenAI-compatible `/v1/chat/completions` endpoint on port `8001`.
- Optionally expose Ollama-compatible `/api/generate` and `/api/chat`
  endpoints.
- Stream chat completions with server-sent events.
- Choose a ready-to-run model image or mount your own `.rkllm` file.
- Publish model-specific images to GHCR with GitHub Actions.
- Apply platform-specific RK3576/RK3588 frequency settings at startup.

## Overview

RKLLM Docker packages the Rockchip RKLLM/RKNN runtime, Python 3.11, RKNN
Toolkit Lite, device setup scripts, and a FastAPI server into `linux/arm64`
container images.

The project is meant for edge AI deployments where a Rockchip board should run
a local LLM behind a familiar OpenAI-style API. Model binaries are not stored
in this repository. They are either mounted from the host at runtime or
downloaded during a GitHub Actions build and baked into a model image.

## Quick Start

Run the Qwen2.5 1.5B Instruct W4A16 image for RK3576:

```bash
sudo docker run --rm -it \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
```

Check the server:

```bash
curl http://localhost:8001/health
```

Send a chat request:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Write a one-sentence hello from RKLLM."}
    ],
    "temperature": 0.8,
    "max_tokens": 128
  }'
```

Expose the Ollama-compatible API instead:

```bash
sudo docker run --rm -it \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -e API_FORMAT=ollama \
  ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
```

```bash
curl http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Write a one-sentence hello from RKLLM."}
    ],
    "stream": false
  }'
```

Set `API_FORMAT=both` to expose the OpenAI-compatible and
Ollama-compatible endpoints at the same time.

Use `w8a8-rk3576` instead of `w4a16-rk3576` to run the W8A8 variant.
Use `qwen2.5-3b-instruct` instead of `qwen2.5-1.5b-instruct` to run the
3B model.

## Images

```text
ghcr.io/hanzo-huang/rkllm-docker:env-latest
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w8a8-rk3576
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-3b-instruct:w4a16-rk3576
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-3b-instruct:w8a8-rk3576
```

The `env-latest` image contains the runtime and API server, but no model. Use
it when you want to mount a model from the host.

The model images include one `.rkllm` file under `/app/models` and start the
server with the correct `MODEL_PATH` and `TARGET_PLATFORM`.

## Requirements

- A Rockchip device supported by RKLLM, such as RK3576 or RK3588.
- Docker running on the target device.
- Access to Rockchip device nodes through `--privileged` and `/dev` mounting.
- An `.rkllm` model built for the selected target platform.

## Run Your Own Model

Mount a host directory containing your `.rkllm` file and point `MODEL_PATH` at
the model inside the container:

```bash
sudo docker run --rm -it \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v /home/hanzo/llm/models:/app/models:ro \
  -e MODEL_PATH=/app/models/my-model.rkllm \
  -e TARGET_PLATFORM=rk3576 \
  ghcr.io/hanzo-huang/rkllm-docker:env-latest
```

Set `RUN_FREQ_FIX=false` to skip frequency tuning at startup.

## Docker Compose

Run an image that already contains a model:

```bash
docker compose up -d
```

Run the environment image with a host-mounted model:

```bash
docker compose -f compose.mount-model.yaml up -d
```

Override the default image or model directory with environment variables:

```bash
RKLLM_IMAGE=ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576 docker compose up -d
MODEL_DIR=/home/hanzo/llm/models docker compose -f compose.mount-model.yaml up -d
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `/app/models/model.rkllm` | Path to the RKLLM model inside the container. |
| `TARGET_PLATFORM` | `auto` | Target platform, such as `rk3576`, `rk3588`, or `rk3588s`. |
| `RUN_FREQ_FIX` | `true` | Apply platform-specific frequency settings before serving. |
| `PORT` | `8001` | HTTP server port. |
| `API_FORMAT` | `openai` | API format to expose: `openai`, `ollama`, or `both`. |

### API Formats

`API_FORMAT=openai` exposes:

- `GET /health`
- `GET /v1/models`
- `GET /models`
- `POST /v1/chat/completions`

`API_FORMAT=ollama` exposes:

- `GET /health`
- `GET /api/version`
- `GET /api/tags`
- `POST /api/show`
- `POST /api/generate`
- `POST /api/chat`

Ollama streaming responses use newline-delimited JSON, matching Ollama's API.
OpenAI streaming responses use server-sent events.

## Add A Model Image

Create a model definition:

```text
models/<model-id>/<quantization>-<platform>/model.env
```

Example:

```text
models/qwen2.5-1.5b-instruct/w4a16-rk3576/model.env
```

```dotenv
MODEL_URL=https://huggingface.co/<account>/<repo>/resolve/main/path/model.rkllm
MODEL_FILE=model.rkllm
TARGET_PLATFORM=rk3576
MODEL_SHA256=
```

Then run the model build workflow:

- Commit and push the new `model.env`.
- Open `Actions > Build model image > Run workflow`.
- Enter `model_id` and `variant` using the directory names.
- Keep `base_tag=env-latest` unless you need a different runtime image.

The published image will be:

```text
ghcr.io/hanzo-huang/rkllm-docker/<model-id>:<variant>
```

If you copy a Hugging Face URL from the web UI, change `/blob/main/` to
`/resolve/main/` before saving it in `MODEL_URL`; Docker builds need the raw
file download URL, not the HTML file page. For private models, add a repository
secret named `MODEL_DOWNLOAD_TOKEN`.

## Repository Layout

```text
app/                     FastAPI RKLLM server
docker/                  Runtime and model Dockerfiles, plus entrypoint
models/<model>/<variant> Model download and platform definitions
runtime/lib/             RKLLM/RKNN ARM64 shared libraries
runtime/wheels/          RKNN Toolkit Lite ARM64 wheel
scripts/                 RK3576/RK3588 frequency scripts
.github/workflows/       Environment and model image builds
compose.yaml             Run an image containing a model
compose.mount-model.yaml Run the environment with a host model
```

## Feedback And Contributing

Issues and pull requests are welcome. Useful contributions include new model
definitions, additional Rockchip platform notes, runtime fixes, and clearer
deployment examples.
