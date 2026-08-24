FROM nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04

ARG PYTHON_VERSION=3.13.13

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV FLASH_ATTN_CUDA_ARCHS=90 \
    NATTEN_CUDA_ARCH=8.6;9.0;10.0 \
    NATTEN_N_WORKERS=8 \
    UV_CACHE_DIR=/opt/uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    ca-certificates \
    gnupg \
    git \
    make \
    curl \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install eccodes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-tools \
    libeccodes-dev \
 && rm -rf /var/lib/apt/lists/*

# Installing newer version of cmake (for dm-tree and natten)
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
  && sh -c 'echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" > /etc/apt/sources.list.d/kitware.list' \
  && apt-get update \
  && apt-get install -y --no-install-recommends cmake \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Disable contraint files in the container
# PyTorch container started constraining packages which we don't want
ENV PIP_CONSTRAINT=
RUN for PIP_CONF in "/etc/xdg/pip/pip.conf" "/root/.pip/pip.conf" "/usr/pip.conf" "/root/.config/pip/pip.conf" "/etc/pip.conf"; do \
        if [ -f "$PIP_CONF" ]; then \
          sed -i 's/^no-cache-dir = true/#no-cache-dir = true/' "$PIP_CONF"; \
        fi; \
    done

# Install tox for venv managment
RUN uv tool install tox --with tox-uv

RUN /usr/bin/uv python install "${PYTHON_VERSION}"

# Build this image from the repository root so these lock files are in context:
#   docker build -f test/Dockerfile .
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p earth2studio
COPY earth2studio/__init__.py earth2studio/__init__.py

# Warm uv's package/build cache into the image. CI runs use this same cache
# directory, avoiding a large GitHub Actions uv cache archive.
RUN uv venv --python="${PYTHON_VERSION}" \
    && uv sync --locked --no-install-project \
    && for EXTRA in all aifs aifs2 aifs2ens aifsens stormcast-conus; do \
        uv sync --locked --no-install-project --extra "${EXTRA}"; \
    done \
    && TORCH_VERSION="$(.venv/bin/python -c 'from importlib.metadata import version; print(version("torch"))')" \
    && printf 'export TORCH_VERSION=%s\n' "${TORCH_VERSION}" > /opt/earth2studio-ci-env \
    && chmod -R a+rwX "${UV_CACHE_DIR}"

RUN rm -rf .venv earth2studio pyproject.toml uv.lock README.md

# GitHub Actions runs container steps with bash, which automatically sources
# BASH_ENV before each step. This exposes the version captured above without
# inspecting or activating the warmed uv cache during CI.
ENV BASH_ENV=/opt/earth2studio-ci-env
