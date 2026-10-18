# Model definitions

Model metadata is organized as:

```text
models/<llm-or-vlm>/<model-id>/<platform>/<quantization>.env
```

LLM definitions download one `.rkllm` file. VLM definitions download a paired
`.rkllm` language model and `.rknn` vision encoder for the same platform.
Gemma4 can additionally bundle a separate audio `.rknn` encoder.

An LLM definition contains:

```dotenv
MODEL_URL=https://example.com/Qwen3-4B_RK3576_w8a8.rkllm
MODEL_FILE=Qwen3-4B_RK3576_w8a8.rkllm
RKLLM_TOOLKIT_VERSION=1.2.3
MODEL_SHA256=lowercase-sha256
```

A complete VLM definition contains both model artifacts and the vision runtime
settings:

```dotenv
MODEL_KIND=vlm
MODEL_URL=https://example.com/Qwen3.5-4B_RK3576_w8a8.rkllm
MODEL_FILE=Qwen3.5-4B_RK3576_w8a8.rkllm
VISION_MODEL_URL=https://example.com/Qwen3.5-4B_vision_RK3576.rknn
VISION_MODEL_FILE=Qwen3.5-4B_vision_RK3576.rknn
RKLLM_TOOLKIT_VERSION=1.3.0
MODEL_SHA256=lowercase-sha256
VISION_MODEL_SHA256=lowercase-sha256
VLM_IMAGE_SIZE=392
```

For optional Gemma4 audio, add all three audio artifact fields and select the
model family. The build downloads and verifies the encoder before packaging it:

```dotenv
VLM_MODEL_FAMILY=gemma4
AUDIO_MODEL_URL=https://example.com/gemma-4-E2B-it_audio_RK3576.rknn
AUDIO_MODEL_FILE=gemma-4-E2B-it_audio_RK3576.rknn
AUDIO_MODEL_SHA256=lowercase-sha256
```

`MODEL_FILE`, `VISION_MODEL_FILE`, and optional `AUDIO_MODEL_FILE` must exactly match the final filename in
their corresponding URL. The converted artifacts in this repository put the
platform before the quantization, for example:

```text
Qwen3-4B_RK3576_w8a8.rkllm
Qwen3.5-4B_RK3576_w8a8.rkllm
Qwen3.5-4B_vision_RK3576.rknn
```

Do not use the older form where the platform appears after the quantization,
such as `Qwen3-4B_w8a8_RK3576.rkllm`.

For all future group-size-128 W4A16 conversions, use the `w4a16-g128`
definition name and image-tag suffix; its corresponding filename contains
`w4a16_g128`. Do not create new `w4a16` definitions. The older `w4a16` name is
retained only for existing artifacts and compatibility. Every committed
artifact definition includes a SHA256 checksum so the build workflow verifies
the download before creating an image.

The runtime selects 2 RKNN cores for RK3576 and 3 for RK3588/RK3588S from
`TARGET_PLATFORM`. The `.rkllm` and `.rknn` files must target the same platform.

The application runtime in this repository is RKLLM v1.3.1. Its ABI is the
v1.3.1 `rkllm.h` layout: `ignore_eos_token` is present in `RKLLMParam`, image
marker strings are supplied in `RKLLMMultiModalInput.image`, and per-request
sampling/max-token overrides are supplied through `RKLLMInferParam`. Do not
copy the older v1.2.x `RKLLMParam` layout into the server; the field offsets
are different. SDK 1.3.1 additionally appends an audio block to the multimodal
input. This changes the input union size even for text-only requests; both
servers use the shared definitions in `app/rkllm_types.py`. Gemma4 accepts OpenAI `input_audio` content through chat completions when the
audio encoder is configured. It supports regular and SSE text responses from
one PCM16 WAV clip (mono/stereo, 8–48 kHz, longer than 10 ms, up to 7.56 s).
The vision and audio encoders are separate files; the language model is shared.

**Toolkit version:** `RKLLM_TOOLKIT_VERSION` records the toolkit used to create
the `.rkllm` file. It is intentionally kept as a separate, prominent field so
the conversion version is easy to identify before building an image.
`MODEL_URL`, `VISION_MODEL_URL`, and optional `AUDIO_MODEL_URL` must be direct download URLs, such as
Hugging Face `/resolve/main/...` URLs. Model binaries are downloaded during
the image build and are not committed to Git.

## Build model images

Use **Actions → Build model images**:

- `scope: configuration` builds one selected quantization and platform;
- `scope: model` builds every definition for one model;
- `scope: all` scans and builds every definition under `models/`;
- `platform: all` builds the selected quantization on every available platform.

For private or temporary artifacts, provide the URL override in the workflow
and configure `MODEL_DOWNLOAD_TOKEN` as a repository secret when needed.

Published images use separate namespaces and `<platform>-<quantization>` tags:

```text
ghcr.io/<owner>/<repo>/llm/<model-id>:rk3576-w4a16
ghcr.io/<owner>/<repo>/vlm/<model-id>:rk3576-w4a16-g128
```
