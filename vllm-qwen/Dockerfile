#FROM vllm/vllm-openai:latest
FROM dustynv/vllm:0.6.6.post1-r36.4.0

ENV HF_HOME=/root/.cache/huggingface

ARG HF_TOKEN
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
    fi && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')" || true

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["vllm", "serve", \
     "Qwen/Qwen2.5-7B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--max-model-len", "32768"]
