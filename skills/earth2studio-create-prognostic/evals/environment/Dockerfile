FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Earth2Studio dev sandbox for Harbor skill evals (Python 3.13 + uv).
# Clones the repo scaffold and pre-installs dependencies to avoid timeout.
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    jq \
    make \
    ripgrep

ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.13
# NV-ACES default repo location
ENV EARTH2STUDIO_ROOT=/workspace/repo

RUN mkdir -p /workspace/skills /workspace/input /workspace/output \
    /logs/verifier /logs/agent

# Clone earth2studio repo (depth 1, no history) and pre-install dependencies
RUN git clone --depth 1 https://github.com/NVIDIA/earth2studio.git /workspace/repo && \
    cd /workspace/repo && \
    uv venv --python 3.13 && \
    uv sync --group dev --extra data

WORKDIR /workspace/repo

COPY setup/bootstrap.sh /usr/local/bin/e2s-eval-bootstrap
RUN chmod +x /usr/local/bin/e2s-eval-bootstrap
