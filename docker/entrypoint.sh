#!/bin/sh
set -eu

TARGET_PLATFORM="$(/app/fix_freq.sh --detect "${TARGET_PLATFORM:-auto}")"
export TARGET_PLATFORM

if [ "${RUN_FREQ_FIX:-true}" = "true" ]; then
    /app/fix_freq.sh "$TARGET_PLATFORM"
fi

if [ "${1:-serve}" != "serve" ]; then
    exec "$@"
fi

# A filename is convenient for custom model mounts. Full paths remain
# supported through MODEL_PATH and VISION_MODEL_PATH.
if [ -n "${MODEL_FILE:-}" ]; then
    case "$MODEL_FILE" in
        /*) MODEL_PATH="$MODEL_FILE" ;;
        *) MODEL_PATH="/app/models/$MODEL_FILE" ;;
    esac
fi
if [ -n "${VISION_MODEL_FILE:-}" ]; then
    case "$VISION_MODEL_FILE" in
        /*) VISION_MODEL_PATH="$VISION_MODEL_FILE" ;;
        *) VISION_MODEL_PATH="/app/models/$VISION_MODEL_FILE" ;;
    esac
fi
export MODEL_PATH VISION_MODEL_PATH

if [ ! -f "${MODEL_PATH}" ]; then
    echo "RKLLM model not found: ${MODEL_PATH}" >&2
    echo "Mount a model there or use an image built by build-model-image.yml." >&2
    exit 1
fi

shift || true

MODEL_KIND="${MODEL_KIND:-llm}"
case "$MODEL_KIND" in
llm)
    ;;
vlm)
    if [ ! -f "${VISION_MODEL_PATH:-}" ]; then
        echo "VLM vision model not found: ${VISION_MODEL_PATH:-}" >&2
        exit 1
    fi

    case "$TARGET_PLATFORM" in
        rk3576) vlm_core_num=2 ;;
        rk3588|rk3588s) vlm_core_num=3 ;;
        *) vlm_core_num=3 ;;
    esac

    exec python3 /app/fastapi_server_vlm.py \
        --encoder_model "${VISION_MODEL_PATH}" \
        --llm_model "${MODEL_PATH}" \
        --model_name "${API_MODEL_NAME:-rkllm-vision}" \
        --target_platform "${TARGET_PLATFORM}" \
        --port "${PORT:-8001}" \
        --rknn_core_num "$vlm_core_num" \
        --img_start "${VLM_IMAGE_START:-<|vision_start|>}" \
        --img_end "${VLM_IMAGE_END:-<|vision_end|>}" \
        --img_content "${VLM_IMAGE_CONTENT:-<|image_pad|>}" \
        "$@"
    ;;
*)
    echo "Unsupported MODEL_KIND: $MODEL_KIND (expected llm or vlm)" >&2
    exit 1
    ;;
esac

if [ "${INTERACTIVE_CHAT:-false}" != "true" ]; then
    set -- --no_chat "$@"
fi

exec python3 /app/fastapi_server_llm.py \
    --rkllm_model_path "${MODEL_PATH}" \
    --target_platform "${TARGET_PLATFORM}" \
    --port "${PORT:-8001}" \
    --model_name "${API_MODEL_NAME:-rkllm-model}" \
    --api_format "${API_FORMAT:-openai}" \
    "$@"
