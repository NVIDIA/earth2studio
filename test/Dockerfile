FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./src

# Install earth2studio
RUN cd src && make setup-ci
RUN cd src && make install
RUN pip uninstall -y earth2studio

ENTRYPOINT ["bash"]
