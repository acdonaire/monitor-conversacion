FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir \
    soxr \
    librosa \
    soundfile \
    "mistral-common>=1.9.0"

ENV VLLM_DISABLE_COMPILE_CACHE=1
ENV HF_HOME=/root/.cache/huggingface

ARG HF_TOKEN
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
    fi && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Voxtral-Mini-4B-Realtime-2602')" || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["vllm", "serve", \
     "mistralai/Voxtral-Mini-4B-Realtime-2602", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--compilation_config", "{\"cudagraph_mode\": \"PIECEWISE\"}", \
     "--max-model-len", "32768"]
