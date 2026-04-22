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

# Critical CVE fixes (all marked Patchable: true in the security scan).
# Libtasn1: CVE-2021-46848, CVE-2025-13151 (USN-7954-1)
RUN apt-get update && apt-get install -y --only-upgrade libtasn1-6 && rm -rf /var/lib/apt/lists/*

# Go stdlib: CVE-2025-22871, CVE-2025-68121 (CRITICAL) + 12 HIGH Go stdlib CVEs.
# Sole Go binary is the NsightSystems EFA nic_sampler plugin (Go 1.23.4), unused
# by the serve application. wandb-core is also a Go binary flagged for the same
# stdlib CVEs; the wandb Python package remains importable without it.
# uv cache is purged so scanners don't flag archive copies of installed packages.
RUN rm -f /usr/local/cuda-13.1/NsightSystems-cli-2025.6.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && find /usr /root -name "wandb-core" -type f -delete \
    && rm -rf /root/.cache/uv \
    && rm -rf /var/lib/apt/lists/*

# aiohttp: CVE-2026-34520 (GHSA-63hf-3vf5-4wqf) — also covers HIGH aiohttp CVEs below
RUN uv pip install --system --break-system-packages "aiohttp>=3.13.4"

# onnx: CVE-2026-27489, CVE-2026-28500, CVE-2026-34445, GHSA-q56x-g2fj-4rj6
# The pytorch base image vendors onnx 1.18.0 as a C++ third-party build dep;
# its stale egg-info is removed so scanners don't report it as an installation.
RUN uv pip install --system --break-system-packages "onnx>=1.21.0" \
    && rm -rf /opt/pytorch/pytorch/third_party/onnx/onnx.egg-info

# HIGH CVE fixes — Ubuntu system libraries.
# curl (USN-8062-1, USN-8084-1), expat (USN-8022-2), glibc (USN-8005-1),
# libpng (USN-7963-1, USN-8039-1), openssl (USN-7980-1, USN-8155-1),
# libssh (USN-8093-1), python3 (USN-7951-1).
# linux-libc-dev (CVE-2025-37849, CVE-2025-37899, CVE-2025-38561, CVE-2025-39698,
#                 CVE-2026-23111, CVE-2026-23268, CVE-2026-23410, CVE-2026-23411) —
#                 kernel headers not needed at runtime; removed to eliminate CVEs.
# vim (USN-8101-1) is removed since it is not needed in a production serve image.
RUN apt-get update && apt-get install -y \
        curl \
        libcurl4 \
        libexpat1 \
        libc6 \
        libc-bin \
        libpng16-16 \
        openssl \
        libssl3 \
        libssh-4 \
        python3-minimal \
    && (apt-get remove -y --purge vim vim-common vim-tiny vim-runtime linux-libc-dev 2>/dev/null || true) \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# HIGH CVE fixes — Python packages.
# pip (CVE-2025-8869, CVE-2026-1703), pillow (CVE-2026-25990),
# nbconvert (CVE-2025-53000), protobuf (CVE-2026-0994),
# tornado (CVE-2026-31958), black (CVE-2026-32274).
# aiohttp and onnx are already pinned in the CRITICAL section above.
RUN uv pip install --system --break-system-packages \
    "pip>=25.0" \
    "setuptools>=78.1.1" \
    "wheel>=0.46.2" \
    "pillow>=12.2.0" \
    "nbconvert>=7.17.0" \
    "protobuf>=6.33.5" \
    "tornado>=6.5.5" \
    "black>=26.3.1" \
    && find /usr/lib /usr/local/lib -name "wheel-0.45*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

CMD ["/workspace/earth2studio-project/serve/server/scripts/startup.sh"]
