# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/physicsnemo/physicsnemo:26.03

# update repo info
RUN apt update -y && \
    apt install -y libibmad5 unixodbc git-lfs && \
    apt install -y netcdf-bin libnetcdf-dev && \
    git lfs install

# upgrade cmake
RUN apt remove cmake -y && \
    pip install cmake --upgrade

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

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
RUN mkdir -p /tc_tracking_src
COPY . /tc_tracking_src
WORKDIR /tc_tracking_src

ENV FORCE_CUDA_EXTENSION=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 12.0, 13.0+PTX"
RUN uv pip install --system --break-system-packages --no-cache-dir .
