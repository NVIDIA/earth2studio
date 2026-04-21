FROM nvcr.io/nvidia/pytorch:26.03-py3

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

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/
RUN uv pip install --system --break-system-packages --no-build-isolation torch-harmonics==0.8.0
RUN uv pip install --system --break-system-packages --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@${E2G_HASH}"
RUN uv pip install --system --break-system-packages natten==0.21.5+torch2100cu130 -f https://whl.natten.org
RUN uv pip install --system --break-system-packages cartopy xskillscore matplotlib

# Install the requirements for stormcast.
RUN pip uninstall -y torch-harmonics && \
    FORCE_CUDA_EXTENSION=1 uv pip install --system --break-system-packages --no-build-isolation torch-harmonics==0.8.0

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

# Security patch: pin packages to versions that fix known critical/high CVEs.
RUN uv pip install --system --break-system-packages \
    "aiohttp>=3.13.4" \
    "black>=26.3.1" \
    "nbconvert>=7.17.0" \
    "protobuf>=6.33.5" \
    "tornado>=6.5.5" \
    "pillow>=12.2.0" \
    "setuptools>=78.1.1" \
    "wheel>=0.46.2"

# nvidia-physicsnemo 2.0.0 (latest on PyPI) imports torch.distributed.tensor._ops.registration
# which was removed in PyTorch 2.7 (NGC 26.03). The fix landed in main at a56b5e8 (PR #1574)
# but has not been released yet; install from that commit directly.
RUN uv pip install --system --break-system-packages \
    "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@a56b5e8"

# Force-upgrade onnx past the CVE-fixed version (1.21.0).
RUN printf 'onnx>=1.21.0\n' > /tmp/onnx-override.txt && \
    uv pip install --system --break-system-packages \
        --override /tmp/onnx-override.txt \
        "onnx>=1.21.0" && \
    rm /tmp/onnx-override.txt

# Remove vulnerable binaries and packages not required by the application.
RUN rm -f /usr/local/cuda-13.2/NsightSystems-cli-2026.1.2/target-linux-x64/plugins/efa_metrics/nic_sampler && \
    find /usr /root -name "wandb-core" -type f -delete && \
    rm -rf /root/.cache/uv && \
    apt-get remove -y --purge linux-libc-dev python3-setuptools python3-protobuf python3-wheel && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["/workspace/earth2studio-project/serve/server/scripts/startup.sh"]
