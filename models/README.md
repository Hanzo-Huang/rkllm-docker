# Model definitions

Each model has one directory, with a subdirectory per quantization/platform
variant. Model binaries are ignored by Git and downloaded only while an image
is built.

```text
models/
  qwen2.5-1.5b-instruct/
    w4a16-rk3576/model.env
    w8a8-rk3576/model.env
```

Required values:

```dotenv
MODEL_URL=https://example.com/model.rkllm
MODEL_FILE=model.rkllm
TARGET_PLATFORM=rk3576
MODEL_SHA256=optional-lowercase-sha256
```

To add a model:

1. Add `<model-id>/<variant>/model.env` or copy an existing variant.
2. Run **Build model image** with both `model_id` and `variant`.
3. For private or temporary URLs, leave `MODEL_URL` empty and provide
   `model_url` to the workflow. A `MODEL_DOWNLOAD_TOKEN` Actions secret is sent
   as a Bearer token when configured.

Images use the model as the repository and the variant as the tag:

```text
ghcr.io/<owner>/<repo>/qwen2.5-1.5b-instruct:w4a16-rk3576
ghcr.io/<owner>/<repo>/qwen2.5-1.5b-instruct:w8a8-rk3576
```
