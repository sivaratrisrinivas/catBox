FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    HF_HOME=/home/ubuntu/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface/transformers \
    DIFFUSERS_CACHE=/home/ubuntu/.cache/huggingface/diffusers

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /uvx /usr/local/bin/

USER ubuntu
WORKDIR /home/ubuntu/app

COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock README.md ./
COPY --chown=ubuntu:ubuntu catbox ./catbox
COPY --chown=ubuntu:ubuntu experiments ./experiments

RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["uv", "run", "--no-sync", "python", "-m", "catbox.browser_ui", "--host", "0.0.0.0", "--port", "7860"]
