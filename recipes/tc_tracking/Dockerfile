# Image for the TC tracking recipe: PhysicsNeMo base, TempestExtremes and the
# recipe's Python dependencies (installed with uv). The recipe code itself is
# not baked in, mount it at runtime (see launch_tc_tracking.sh).
#
#   docker build -t tc_tracking:1.0 .                                  # AIFS-ENS v1
#   docker build --build-arg AIFS_VERSION=2 -t tc_tracking:1.0-aifs2 . # AIFS-ENS v2
#
# Only the last install layer depends on AIFS_VERSION, so switching between
# v1 and v2 reuses everything else from the build cache.
FROM nvcr.io/nvidia/physicsnemo/physicsnemo:26.08

# Compile parallelism: MAX_JOBS source files at once, each compiling
# NVCC_THREADS GPU architectures at once. Every architecture pass of a CUDA
# file needs up to 3 GB of RAM, so by default MAX_JOBS is derived from the
# available memory (at most one job per core). If the build still runs out of
# memory, use e.g. --build-arg MAX_JOBS=1 --build-arg NVCC_THREADS=1
ARG MAX_JOBS=""
ARG NVCC_THREADS=5

ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    FORCE_CUDA_EXTENSION=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 12.0+PTX" \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_CACHE=1 \
    UV_CONSTRAINT=/etc/uv/constraint.txt \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=vmm

# TempestExtremes, providing the DetectNodes and StitchNodes executables
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libibmad5 unixodbc git-lfs netcdf-bin libnetcdf-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install && \
    git clone https://github.com/ClimateGlobalChange/tempestextremes.git /tmp/tempestextremes && \
    cd /tmp/tempestextremes && \
    git checkout 80d2b54f60117bd386f8786c8c35e739e63bfc97 && \
    cmake -S . -B build && \
    cmake --build build -j "${MAX_JOBS:-$(nproc)}" && \
    cp build/bin/DetectNodes build/bin/StitchNodes /usr/local/bin && \
    rm -rf /tmp/tempestextremes

# Pin the container's CUDA builds of torch and friends, so that uv never
# replaces them with generic wheels from PyPI. torch-geometric is pure Python
# and not pinned, AIFS-ENS v1 needs an older version than the container's
RUN mkdir -p /etc/uv && \
    pip list --format=freeze 2>/dev/null | grep -i -E \
        '^(torch|torchvision|triton|flash[-_]attn|natten|nvidia-physicsnemo|transformer[-_]engine|apex|cupy-cuda13x|torch[-_](scatter|sparse|cluster))==' \
        > /etc/uv/constraint.txt && \
    cat /etc/uv/constraint.txt

# Model-independent dependencies plus all models except AIFS-ENS. CUDA
# extensions (torch-harmonics) consist of few source files, so besides parallel
# jobs nvcc compiles the architectures of TORCH_CUDA_ARCH_LIST in parallel
COPY pyproject.toml /tc_tracking/pyproject.toml
WORKDIR /tc_tracking
RUN mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) && \
    jobs=$(( mem_gb / (3 * NVCC_THREADS) )) && \
    jobs=$(( jobs < 1 ? 1 : (jobs > $(nproc) ? $(nproc) : jobs) )) && \
    export MAX_JOBS="${MAX_JOBS:-${jobs}}" NVCC_APPEND_FLAGS="--threads ${NVCC_THREADS}" && \
    echo "compiling with MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS} (${mem_gb} GB RAM available)" && \
    uv pip install -r pyproject.toml \
        --extra fcn3 --extra sfno --extra interp --extra weathernext --extra plot --extra dev || \
    { echo "ERROR: installing the dependencies failed. If the log above shows 'Killed'" \
           "or 'signal 9', compiling ran out of memory: rebuild with fewer parallel" \
           "jobs, e.g. --build-arg MAX_JOBS=1 --build-arg NVCC_THREADS=1" >&2; exit 1; }

# AIFS-ENS, v1 and v2 depend on incompatible anemoi-models versions
ARG AIFS_VERSION=1
RUN case "${AIFS_VERSION}" in \
        1) extra=aifs ;; \
        2) extra=aifs-v2 ;; \
        *) echo "AIFS_VERSION must be 1 or 2, got ${AIFS_VERSION}" >&2; exit 1 ;; \
    esac && \
    uv pip install -r pyproject.toml --extra "${extra}"

# Fail the build early if a model stack or TempestExtremes is not importable
RUN python -c "import earth2studio, anemoi.models, makani, torch_harmonics, natten, physicsnemo, jax, weathernext, tropycal, hydra" && \
    command -v DetectNodes StitchNodes

# Writable home for running the container as the host user (--user uid:gid)
RUN mkdir -m 1777 /home/tc_hunt
WORKDIR /workspace
