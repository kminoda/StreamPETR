# Use an alternative CUDA 11.2 base image
# FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Set the working directory to the root of StreamPETR
WORKDIR /workspace

# Set environment variable to prevent time zone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && ~/miniconda/bin/conda init bash

ENV PATH="/root/miniconda/bin:$PATH"

# Create and activate the conda environment
RUN /bin/bash -c "conda create -n streampetr python=3.8 -y && \
    echo 'source activate streampetr' > ~/.bashrc"

# # Install PyTorch and torchvision
# RUN /bin/bash -c "source ~/.bashrc && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"

# # Install optional flash-attn
# RUN /bin/bash -c "source ~/.bashrc && pip install flash-attn==0.2.8"

# Install PyTorch 1.13.0 and torchvision
RUN /bin/bash -c "source ~/.bashrc && pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html"

# Install optional flash-attn
RUN /bin/bash -c "source ~/.bashrc && pip install flash-attn==0.2.8"

# Copy the mmdetection3d directory into the Docker image
COPY mmdetection3d /workspace/mmdetection3d

# Install mmdet3d and dependencies
RUN /bin/bash -c "source ~/.bashrc && cd /workspace/mmdetection3d && \
    git checkout v1.0.0rc6 && \
    pip install -e ."

# Install mmdet3d and dependencies
RUN /bin/bash -c "source ~/.bashrc && pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html && \
    pip install mmdet==2.28.2 && \
    pip install mmsegmentation==0.30.0"

RUN /bin/bash -c "source ~/.bashrc && pip install onnx onnxruntime onnxsim"

# Set the entry point to bash
ENTRYPOINT ["/bin/bash"]
