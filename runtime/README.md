# Bundled runtime and SDK updates

The application uses Rockchip **RKLLM runtime 1.3.1**, built for Linux AArch64.
`rkllm-manifest.json` records the exact upstream release commit and SHA-256
hashes of the native library, matching header, and upstream license. The files
come from [release-v1.3.1](https://github.com/airockchip/rknn-llm/releases/tag/release-v1.3.1),
commit `f7390530443bf84f0394255a449d7cbe81e69d1c`:

- `lib/librkllmrt.so`: upstream `rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so`
- `include/rkllm.h`: upstream `rkllm-runtime/Linux/librkllm_api/include/rkllm.h`
- `LICENSE.rkllm`: upstream SDK license

The image includes the SDK license and provenance manifest under
`/usr/share/doc/rkllm/`.

The RKNN runtime and Toolkit Lite wheel are separate dependencies and retain
their existing versions. Model definitions record the **conversion toolkit**
version used to produce each artifact; upgrading the runtime does not change
those historical toolkit versions.

## Python versus C++

The current Python/FastAPI HTTP servers call the same native `librkllmrt.so`
used by Rockchip's C++ demos. Model inference already executes in the native
runtime; Python handles HTTP, request preparation, callbacks, and streaming.

Keep Python for the current OpenAI/Ollama APIs unless measurements show that
application overhead matters. C++ may reduce interpreter memory and some
preprocessing/callback overhead, and provides compile-time checking against
the SDK header. It also requires replacing or maintaining the HTTP, protocol,
streaming, and image-processing implementations. No token-speed improvement
has been measured for this repository, so a rewrite would be premature.

To evaluate a C++ implementation, compare the same model, quantization,
platform, runtime, CPU/NPU frequencies, sampling settings, context, prompt,
and output length. Measure time to first token, prefill/decode throughput,
RSS, CPU use, and streaming latency. Treat the upstream command-line demo as
a native inference baseline; it does not include the same HTTP workload.
Keep conversion tools in Python regardless of the serving implementation.

## ABI validation

Both servers import `app/rkllm_types.py`. SDK 1.3.1 adds the audio block to
`RKLLMMultiModalInput`, increasing its LP64 size from 144 to 192 bytes and
`RKLLMInput` from 160 to 208 bytes. The new allocation is required for text
and image requests too. Zero-initialized audio fields do not expose audio
inference through the API; an audio encoder/backend and request handling are
separate work.

Run from the repository root with Python 3.11 and a C++ compiler:

```bash
python -m unittest discover -s tests -v
```

The tests compile a probe against the exact vendored header, compare all
bound structure sizes/alignments/member offsets, verify that both servers
use the shared types, and check the bundled binary architecture/checksums and
image version labels. They require no ML packages or NPU. The Dockerfile additionally runs `scripts/check_runtime.py` inside the ARM64
image: it imports both HTTP servers, loads RKLLM/RKNN, and calls the native
`rkllm_createDefaultParam` through the bindings without loading a model. You can
repeat this check with:

```bash
docker run --rm --entrypoint python rkllm-env:dev-1.3.1 /app/check_runtime.py
```

On-board model inference still requires a separate validation step.

## Gemma4 vision serving

Use `MODEL_KIND=vlm` and a matching language model and vision encoder. Set
`VLM_MODEL_FAMILY=gemma4` when the language-model filename does not identify
the family. The server uses Gemma4 image markers and renders explicit canonical
chat turns because the native parser cannot read some checkpoints' separate
`chat_template.jinja` files. System messages and assistant history retain their
roles. Audio and tool calling are not implemented by this profile.

`RKLLM_EMBED_FLASH=1` selects the native storage-backed word embeddings for
memory-constrained VLM boards. Vision models load directly from their file path
to avoid holding duplicate model buffers in Python. The encoder receives packed
RGB uint8 data; RKNN applies the normalization recorded during conversion.

The endpoint smoke-test client saves actual text, image, and streaming responses:

```bash
python3 scripts/test_vlm_endpoint.py --base-url http://127.0.0.1:18001 \
  --model gemma4-test --image /path/to/demo.jpg \
  --question 'What is the astronaut holding, and what color is it?' \
  --expect green --expect-any bottle can --report /path/to/test-report.json
```

This is a smoke test with explicit expected content, not a quality benchmark.

Environment-image builds run these checks before packaging. Pull requests
also run them through the runtime-test workflow. When updating the SDK,
replace the library/header/license together, update the shared bindings and
manifest, adjust Docker/CI versions, and run the checks before publishing.
