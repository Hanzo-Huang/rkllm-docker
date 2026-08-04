# Model definitions

Definitions are organized by model, then target platform. Each quantization is
an independent env file:

```text
models/
  gemma-3-4b-it/
    rk3576/
      w4a16.env
      w8a8.env
    rk3588/
      w8a8.env
  qwen2.5-1.5b-instruct/
    rk3576/
      w4a16.env
      w8a8.env
    rk3588/
      w8a8.env
  qwen2.5-3b-instruct/
    rk3576/
      w4a16.env
      w8a8.env
    rk3588/
      w8a8.env
  qwen3-1.7b/
    rk3576/
      w4a16.env
      w8a8.env
    rk3588/
      w8a8.env
  qwen3-4b/
    rk3576/
      w4a16.env
      w8a8.env
    rk3588/
      w8a8.env
```

This layout makes the hardware boundary visible before the quantization choice.
It also keeps every downloadable artifact independently configurable without
repeating the platform inside the env file.

Each env file contains:

```dotenv
MODEL_URL=https://example.com/model.rkllm
MODEL_FILE=model.rkllm
RKLLM_TOOLKIT_VERSION=1.2.3
MODEL_SHA256=optional-lowercase-sha256
```

`RKLLM_TOOLKIT_VERSION` is the RKLLM Toolkit version that produced that exact
`.rkllm` file. Keep it in each quantization env because models built with
different toolkit versions may require different runtime compatibility.

`MODEL_URL` must be a direct download URL. For Hugging Face, use
`/resolve/main/...`, not a `/blob/main/...` browser page. Model binaries are
ignored by Git and downloaded only while an image is built.

## Add a model configuration

1. Add `models/<model-id>/<platform>/<quantization>.env`.
2. Run **Build model images** with `configuration` scope and select the model,
   platform, and quantization.
3. For private or temporary URLs, leave `MODEL_URL` empty and pass `model_url`
   to that workflow run. A `MODEL_DOWNLOAD_TOKEN` Actions secret is sent as a
   Bearer token when configured.

Use `model` scope to build every platform and quantization for one model. Use
`all` after publishing a new environment image.

Published image tags remain `<quantization>-<platform>`:

```text
ghcr.io/<owner>/<repo>/gemma-3-4b-it:w4a16-rk3576
ghcr.io/<owner>/<repo>/gemma-3-4b-it:w8a8-rk3576
ghcr.io/<owner>/<repo>/gemma-3-4b-it:w8a8-rk3588
```
