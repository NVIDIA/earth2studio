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

CMD ["/workspace/earth2studio-project/serve/server/scripts/startup.sh"]
