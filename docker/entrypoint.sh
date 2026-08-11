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

if [ ! -f "${MODEL_PATH}" ]; then
    echo "RKLLM model not found: ${MODEL_PATH}" >&2
    echo "Mount a model there or use an image built by build-model-image.yml." >&2
    exit 1
fi

shift || true

if [ "${MODEL_KIND:-llm}" = "vlm" ]; then
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
        --model_name "${MODEL_ID:-rkllm-vision}" \
        --target_platform "${TARGET_PLATFORM}" \
        --port "${PORT:-8001}" \
        --rknn_core_num "$vlm_core_num" \
        --img_start "${VLM_IMAGE_START:-<|vision_start|>}" \
        --img_end "${VLM_IMAGE_END:-<|vision_end|>}" \
        --img_content "${VLM_IMAGE_CONTENT:-<|image_pad|>}" \
        "$@"
fi

exec python3 /app/fastapi_server_llm.py \
    --rkllm_model_path "${MODEL_PATH}" \
    --target_platform "${TARGET_PLATFORM}" \
    --port "${PORT:-8001}" \
    --model_name "${MODEL_ID:-rkllm-model}" \
    --no_chat \
    --api_format "${API_FORMAT:-openai}" \
    "$@"
