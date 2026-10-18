# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG TARGETARCH
ARG VERSION=dev
ARG RKLLM_RUNTIME_VERSION=1.3.1

LABEL org.opencontainers.image.title="RKLLM runtime" \
      org.opencontainers.image.description="ARM64 RKLLM OpenAI-compatible API runtime" \
      org.opencontainers.image.version="${VERSION}" \
      io.rkllm.runtime.version="${RKLLM_RUNTIME_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/lib \
    MODEL_PATH=/app/models/model.rkllm \
    MODEL_FILE= \
    MODEL_KIND=llm \
    VISION_MODEL_PATH= \
    VISION_MODEL_FILE= \
    AUDIO_MODEL_PATH= \
    AUDIO_MODEL_FILE= \
    API_MODEL_NAME= \
    LOG_LEVEL=info \
    INTERACTIVE_CHAT=false \
    RKLLM_RUNTIME_VERSION=${RKLLM_RUNTIME_VERSION} \
    TARGET_PLATFORM=auto \
    RUN_FREQ_FIX=true \
    PORT=8001

RUN test "${TARGETARCH}" = "arm64" \
    && apt-get update \
    && apt-get install -y --no-install-recommends bash libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY runtime/lib/*.so /usr/lib/
COPY runtime/wheels/*.whl /tmp/wheels/
COPY requirements.txt /tmp/requirements.txt

RUN ldconfig \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /tmp/requirements.txt

WORKDIR /app
COPY runtime/LICENSE.rkllm runtime/rkllm-manifest.json /usr/share/doc/rkllm/
COPY app/rkllm_types.py /app/rkllm_types.py
COPY app/gemma4_audio.py /app/gemma4_audio.py
COPY app/vlm_profiles.py /app/vlm_profiles.py
COPY app/fastapi_server_llm.py /app/fastapi_server_llm.py
COPY app/fastapi_server_vlm.py /app/fastapi_server_vlm.py
COPY scripts/fix_freq*.sh /app/
COPY docker/entrypoint.sh /app/docker-entrypoint.sh

RUN chmod 0755 /app/docker-entrypoint.sh /app/fix_freq*.sh \
    && mkdir -p /app/models

EXPOSE 8001
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve"]
