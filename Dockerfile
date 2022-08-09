# Triton

FROM nvcr.io/nvidia/cuda:9.0-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    lsb-release \
    gpg \
    && \
    rm -rf /var/lib/apt/lists/*

# Triton Version
ARG TRITON_VERSION=1.1.0

# Download Triton
RUN TRITON_INSTALLER=nvidia-triton-inference-server_${TRITON_VERSION}-1+%7Eubuntu16.04+%7Ex86_64.tar.gz && \
    curl -O -L https://api.github.com/repos/triton-inference-server/server/releases/latest \
        | grep browser_download_url \
        | grep ${TRITON_INSTALLER} \
        | cut -d '"' -f 4 \
        | xargs curl -O -L \
        && tar -zxvf ${TRITON_INSTALLER} \
        && cd nvidia-triton-inference-server-${TRITON_VERSION} \
        && ./install.sh --skip-manager

# Download the Triton client
RUN pip install tritonclient==${TRITON_VERSION}

# Download the Tensor2Tensor library
RUN pip install tensor2tensor==1.14.0

# Create a user
RUN useradd -ms /bin/bash triton

# Create a triton data directory
RUN mkdir -p /tmp/triton-t2t/data
RUN chown triton /tmp/triton-t2t/data

# Create a triton model repository
RUN mkdir -p /tmp/triton-t2t/triton_model_dir
RUN chown triton /tmp/triton-t2t/triton_model_dir

# Setup the triton env vars
ENV TRITON_IMAGE_NAME "triton-dev:${GIT_COMMIT_HASH}"
ENV TRITON_CONTAINER_NAME "triton-t2t"
ENV TRITON_DATA_DIR "/tmp/triton-t2t/data"

# Setup the triton model dir
ENV TRITON_MODEL_REPO "/tmp/triton-t2t/triton_model_dir"

# We will be using the "triton" user inside the container
USER triton

# Setup the default command to launch the triton server
CMD ["/bin/bash"]
