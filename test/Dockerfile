FROM nvcr.io/nvidia/pytorch:25.03-py3
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Disable contraint files in the container
RUN unset PIP_CONSTRAINT
RUN for PIP_CONF in "/etc/xdg/pip/pip.conf" "/root/.pip/pip.conf" "/usr/pip.conf" "/root/.config/pip/pip.conf" "/etc/pip.conf"; do \
        if [ -f "$PIP_CONF" ]; then \
          sed -i 's/^no-cache-dir = true/#no-cache-dir = true/' "$PIP_CONF"; \
        fi; \
    done

ENTRYPOINT ["bash"]
