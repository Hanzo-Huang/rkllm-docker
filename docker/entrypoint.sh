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
exec python3 /app/fastapi_server_llm.py \
    --rkllm_model_path "${MODEL_PATH}" \
    --target_platform "${TARGET_PLATFORM}" \
    --port "${PORT:-8001}" \
    "$@"
