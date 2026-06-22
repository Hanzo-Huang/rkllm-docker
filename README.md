# RKLLM Docker images

ARM64 Docker images for running RKLLM models on Rockchip devices. The shared
environment contains Python 3.11, RKLLM/RKNN libraries, FastAPI, and automatic
RK3576/RK3588 frequency-script selection.

## Images

```text
ghcr.io/<owner>/<repo>:env-latest
ghcr.io/<owner>/<repo>/qwen2.5-1.5b-instruct:w4a16-rk3576
ghcr.io/<owner>/<repo>/qwen2.5-1.5b-instruct:w8a8-rk3576
```

Use **Build environment image** to publish the base environment. Then run
**Build model image** for each model variant:

```text
model_id: qwen2.5-1.5b-instruct
variant:  w4a16-rk3576
```

```text
model_id: qwen2.5-1.5b-instruct
variant:  w8a8-rk3576
```

## Run

An embedded model image already contains its startup command:

```bash
RKLLM_IMAGE=ghcr.io/<owner>/<repo>/qwen2.5-1.5b-instruct:w4a16-rk3576 \
  docker compose up -d
```

To use the environment image with a model mounted from the host:

```bash
RKLLM_ENV_IMAGE=ghcr.io/<owner>/<repo>:env-latest \
  docker compose -f compose.mount-model.yaml up -d
```

## Add a model

Create one definition for each model variant:

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

Run **Build model image** with the directory names as `model_id` and `variant`.
The result is:

```text
ghcr.io/<owner>/<repo>/<model-id>:<variant>
```

Use Hugging Face `/resolve/main/` URLs, not `/blob/main/`. Model binaries are
downloaded by GitHub Actions and should not be committed.

## Notes

- Images are `linux/arm64` only.
- Models must match `TARGET_PLATFORM`.
- Frequency tuning requires a privileged container with `/dev` mounted. Set
  `RUN_FREQ_FIX=false` to disable it.
