FROM nvcr.io/nvidia/pytorch:26.01-py3

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --upgrade pip

ENV NGC_CLI_ORG=no-org
ENV NGC_CLI_TEAM=no-team
ENV EARTH2STUDIO_CACHE=/workspace/data/earth2cache
ENV EARTH2STUDIO_PACKAGE_TIMEOUT=1200
# Following is important for accelerated FCN3 inference
ENV FORCE_CUDA_EXTENSION=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARG MAKANI_HASH=28f38e3e929ed1303476518552c64673bbd6f722
ARG E2G_HASH=11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3
ARG E2S_HASH=f2b02f6a18962c5b689b9f4ac8ef999e1578ff70

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/
RUN uv pip install --system --break-system-packages --no-build-isolation torch-harmonics==0.8.0
RUN uv pip install --system --break-system-packages "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_HASH}"
RUN uv pip install --system --break-system-packages --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@${E2G_HASH}"
RUN uv pip install --system --break-system-packages natten==0.21.5+torch2100cu130 -f https://whl.natten.org
RUN uv pip install --system --break-system-packages "earth2studio[dlwp,dlesym,pangu,sfno,stormcast,fcn3] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
RUN uv pip install --system --break-system-packages "earth2studio[serve] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
RUN uv pip install --system --break-system-packages "earth2studio[cbottle,corrdiff,cyclone] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
RUN uv pip install --system --break-system-packages "earth2studio[data,perturbation,statistics] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
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

# Install local earth2studio from repo, then serve requirements.
# Copy project root files needed for pip install (pyproject.toml is at repo root, not inside earth2studio).
COPY pyproject.toml README.md /workspace/earth2studio-project/
COPY earth2studio /workspace/earth2studio-project/earth2studio
RUN apt-get update && apt-get install -y redis-server && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env && \
    cd /workspace/earth2studio-project && \
    uv pip install --system --break-system-packages . && \
    uv pip install --system --break-system-packages -r serve/server/requirements.txt && \
    rm -rf /workspace/earth2studio-project/earth2studio /workspace/earth2studio-project/pyproject.toml /workspace/earth2studio-project/README.md

CMD ["/workspace/earth2studio-project/serve/server/scripts/startup.sh"]
