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
ARG E2S_HASH=b1f856388f7665e2cd1942e90d6460dcebffb7ba

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/
RUN uv pip install --system --break-system-packages --no-build-isolation torch-harmonics==0.8.0
RUN uv pip install --system --break-system-packages "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_HASH}"
RUN uv pip install --system --break-system-packages --no-build-isolation "earth2grid @ git+https://github.com/NVlabs/earth2grid@${E2G_HASH}"
RUN uv pip install --system --break-system-packages natten==0.21.5+torch2100cu130 -f https://whl.natten.org
RUN uv pip install --system --break-system-packages "earth2studio[dlwp,dlesym,pangu,sfno,stormcast,fcn3] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
RUN uv pip install --system --break-system-packages "earth2studio[cbottle,corrdiff,cyclone] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
RUN uv pip install --system --break-system-packages "earth2studio[data,perturbation,statistics] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_HASH}"
RUN uv pip install --system --break-system-packages cartopy xskillscore matplotlib

# Install the requirements for stormcast.
RUN pip uninstall -y torch-harmonics && \
    FORCE_CUDA_EXTENSION=1 uv pip install --system --break-system-packages --no-build-isolation torch-harmonics==0.8.0

# Install the requirements for the inference server.
COPY serve /workspace/earth2studio-project/serve
RUN cd / && mkdir /workspace/earth2studio-project/examples && ln -s /outputs /workspace/earth2studio-project/examples/outputs
RUN apt-get update && apt-get install -y redis-server && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env && \
    cd /workspace/earth2studio-project && \
    uv pip install --system --break-system-packages -r serve/inferenceserver/requirements.txt

CMD ["/workspace/earth2studio-project/serve/inferenceserver/scripts/startup.sh"]

#RUN python -c "import earth2studio; print(f'Earth2Studio version: {earth2studio.__version__}')"
