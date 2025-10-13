FROM nvcr.io/nvidia/pytorch:25.08-py3

# update repo info
RUN apt update -y && \
    apt install -y libibmad5 unixodbc && \
    apt install -y netcdf-bin libnetcdf-dev

# upgrade cmake
RUN apt remove cmake -y && \
    pip install cmake --upgrade

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++

# install TempestExtremes
WORKDIR /
RUN git clone https://github.com/ClimateGlobalChange/tempestextremes.git && \
    mkdir -p /tempestextremes/build
WORKDIR /tempestextremes/build
RUN cmake .. && \
    make -j && \
    cp ./bin/DetectNodes /usr/local/bin && \
    cp ./bin/StitchNodes /usr/local/bin


# copy source into the container.
RUN mkdir -p /fcn3_tcs
COPY . /fcn3_tcs/

# fcn3_tc
WORKDIR /fcn3_tcs

ENV FORCE_CUDA_EXTENSION 1
ENV TORCH_CUDA_ARCH_LIST "8.0 8.6 9.0 10.0 12.0+PTX"
RUN uv pip install --system --break-system-packages --no-cache-dir .

