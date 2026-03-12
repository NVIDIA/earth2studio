FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
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

# Use the build argument
ARG PYTHON_VERSION=3.12.9
RUN /usr/bin/uv python install "${PYTHON_VERSION}"
