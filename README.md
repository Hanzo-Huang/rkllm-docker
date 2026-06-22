# RKLLM Docker images

## 1. Summary

This repository builds `linux/arm64` Docker images for running RKLLM models on
Rockchip devices. Images include the RKLLM/RKNN runtime, Python 3.11, and an
OpenAI-compatible FastAPI server on port `8001`.

Published images:

```text
ghcr.io/hanzo-huang/rkllm-docker:env-latest
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w8a8-rk3576
```

## 2. Repository structure

```text
app/                     FastAPI RKLLM server
docker/                  Environment/model Dockerfiles and entrypoint
models/<model>/<variant> Model download and platform definitions
runtime/lib/             RKLLM/RKNN ARM64 shared libraries
runtime/wheels/          RKNN Toolkit Lite ARM64 wheel
scripts/                 RK3576/RK3588 frequency scripts
.github/workflows/       Environment and model image builds
compose.yaml             Run an image containing a model
compose.mount-model.yaml Run the environment with a host model
```

Model binaries are downloaded during GitHub Actions builds and are not stored
in Git.

## 3. Run a model

### A. Model included in the image

The model path, target platform, and startup command are already configured:

```bash
sudo docker run --rm -it \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
```

Replace `w4a16-rk3576` with `w8a8-rk3576` to use the W8A8 model.

### B. Use your own model

Mount the model directory and specify the file and target platform:

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

The model must be built for the selected `TARGET_PLATFORM`. Set
`RUN_FREQ_FIX=false` to disable frequency tuning. With `--privileged` and
`-v /dev:/dev`, Rockchip device nodes are available without separate
`--device` flags.

## 4. Add a model image to GitHub

Create a definition using this layout:

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

Then:

1. Commit and push the new `model.env`.
2. Open **Actions > Build model image > Run workflow**.
3. Enter `model_id` and `variant` using the directory names.
4. Keep `base_tag=env-latest` and run the workflow.

The published image will be:

```text
ghcr.io/hanzo-huang/rkllm-docker/<model-id>:<variant>
```

Use a Hugging Face `/resolve/main/` download URL instead of `/blob/main/`.
For private models, add a repository secret named `MODEL_DOWNLOAD_TOKEN`.
