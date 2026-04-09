FROM nvcr.io/nvidia/physicsnemo/physicsnemo:26.03

ENV PATH="/root/.local/bin:$PATH" \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    FORCE_CUDA_EXTENSION=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 12.0+PTX"

RUN apt update -y && \
    apt install -y libibmad5 unixodbc git-lfs netcdf-bin libnetcdf-dev && \
    git lfs install && \
    apt remove cmake -y && \
    pip install cmake --upgrade && \
    wget -qO- https://astral.sh/uv/install.sh | sh && \
    cd / && \
    git clone https://github.com/ClimateGlobalChange/tempestextremes.git && \
    cd /tempestextremes && \
    git checkout 80d2b54f60117bd386f8786c8c35e739e63bfc97 && \
    mkdir -p /tempestextremes/build && \
    cd /tempestextremes/build && \
    cmake .. && \
    make -j && \
    cp ./bin/DetectNodes /usr/local/bin && \
    cp ./bin/StitchNodes /usr/local/bin

COPY . /tc_tracking_src

RUN cd /tc_tracking_src && \
    uv pip install --system --break-system-packages --no-cache-dir ".[all]"
