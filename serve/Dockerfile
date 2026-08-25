FROM nvcr.io/nvidia/pytorch:26.01-py3

RUN apt-get update && apt-get install -y git make && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --upgrade pip

ENV NGC_CLI_ORG=no-org
ENV NGC_CLI_TEAM=no-team
ENV EARTH2STUDIO_CACHE=/workspace/data/earth2cache
ENV EARTH2STUDIO_PACKAGE_TIMEOUT=1200
# Following is important for accelerated FCN3 inference
ENV FORCE_CUDA_EXTENSION=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARG E2G_HASH=11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3
ARG TH_HASH=a632ca748a12bd9f74dbc1e00653317810991f74

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/
RUN uv pip install --system --break-system-packages --no-build-isolation \
    "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@${TH_HASH}"
RUN uv pip install --system --break-system-packages --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@${E2G_HASH}"
RUN uv pip install --system --break-system-packages natten==0.21.5+torch2100cu130 -f https://whl.natten.org
RUN uv pip install --system --break-system-packages cartopy xskillscore matplotlib

# Copy the serve collateral (scripts, config, etc.) and setup env.
COPY serve /workspace/earth2studio-project/serve
COPY examples /workspace/earth2studio-project/examples
RUN ln -s /outputs /workspace/earth2studio-project/examples/outputs
# Config and scripts live under serve/server/ after COPY serve
ENV SCRIPT_DIR=/workspace/earth2studio-project/serve/server/scripts
ENV CONFIG_DIR=/workspace/earth2studio-project/serve/server/conf
ENV WORKFLOW_DIR=/workspace/earth2studio-project/serve/server/example_workflows

# Install local earth2studio from repo, then various extras and serve requirements.
# Copy project root files needed for pip install (pyproject.toml is at repo root, not inside earth2studio).
COPY pyproject.toml README.md /workspace/earth2studio-project/
COPY earth2studio /workspace/earth2studio-project/earth2studio
RUN apt-get update && apt-get install -y redis-server && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env && \
    cd /workspace/earth2studio-project && \
    uv pip install --system --break-system-packages ".[dlwp,dlesym,pangu,sfno,stormcast,fcn3,serve,cbottle,corrdiff,cyclone,data,perturbation,statistics]" && \
    uv pip install --system --break-system-packages -r serve/server/requirements.txt && \
    rm -rf /workspace/earth2studio-project/earth2studio /workspace/earth2studio-project/pyproject.toml /workspace/earth2studio-project/README.md

# Critical CVEs
# Libtasn1: CVEs
RUN apt-get update && apt-get install -y --only-upgrade libtasn1-6 && rm -rf /var/lib/apt/lists/*

# Go stdlib: CVEs
RUN rm -f /usr/local/cuda-13.1/NsightSystems-cli-2025.6.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && find /usr /root -name "wandb-core" -type f -delete \
    && rm -rf /root/.cache/uv \
    && rm -rf /var/lib/apt/lists/*

# aiohttp: CVEs
RUN uv pip install --system --break-system-packages "aiohttp>=3.13.4"

# onnx: CVEs
RUN uv pip install --system --break-system-packages "onnx>=1.21.0" \
    && rm -rf /opt/pytorch/pytorch/third_party/onnx/onnx.egg-info

# HIGH/CRITICAL CVE fixes — Ubuntu system libraries.
RUN apt-get update && apt-get install -y \
        curl \
        libcurl4 \
        libcurl3t64-gnutls \
        libexpat1 \
        libc6 \
        libc-bin \
        libgnutls30t64 \
        libpng16-16 \
        openssl \
        libssl3t64 \
        libssh-4 \
        python3-minimal \
        protobuf-compiler \
        libprotoc32t64 \
        libprotobuf32t64 \
    && (apt-get remove -y --purge vim vim-common vim-tiny vim-runtime linux-libc-dev 2>/dev/null || true) \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL CVE fixes — Slurm runtime libs (CVE-2023-41914, -49933, -49937, -49938, CVE-2025-43904).
# No patched Ubuntu package available; packages are unused by the serve stack.
RUN apt-get update \
    && apt-get remove -y --purge libslurm40t64 libpmi2-0t64 libpmi2-0-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# HIGH/CRITICAL CVE fixes — Python packages.
RUN uv pip install --system --break-system-packages \
    "pip>=25.0" \
    "setuptools>=78.1.1" \
    "wheel>=0.46.2" \
    "aiohttp>=3.14.3" \
    "jupyter_server>=2.20.0" \
    "jupyterlab>=4.5.10" \
    "mistune>=3.3.0" \
    "msgpack>=1.2.1" \
    "nbconvert>=7.17.0" \
    "notebook>=7.5.6" \
    "pillow>=12.3.0" \
    "protobuf>=6.33.5" \
    "pyarrow>=23.0.1" \
    "soupsieve>=2.8.4" \
    "tornado>=6.5.6" \
    "urllib3>=2.7.0" \
    "black>=26.3.1" \
    && find /usr/lib /usr/local/lib -name "wheel-0.45*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

# CVE-2024-7254: the PyTorch base image vendors protobuf 3.13.0 source at
# /opt/pytorch/pytorch/third_party/protobuf/.
RUN rm -rf /opt/pytorch/pytorch/third_party/protobuf

CMD ["/workspace/earth2studio-project/serve/server/scripts/startup.sh"]
