FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./src

# Disable contraint file in /etc/pip/constraint.txt
RUN unset PIP_CONSTRAINT

# Install earth2studio
RUN cd src && ls && git init
RUN cd src && ls && unset PIP_CONSTRAINT && make setup-ci
RUN cd src && unset PIP_CONSTRAINT && make install
RUN pip uninstall -y earth2studio

ENTRYPOINT ["bash"]
