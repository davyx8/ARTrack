FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    g++ \
    bash \
    vim \
    libgl1-mesa-glx \
    git \
    cuda-toolkit-11-3 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Clone ARTrack repository and checkout ARTrackV2 branch
RUN git clone https://github.com/iMaTzzz/ARTrack.git /workspace/ARTrack && \
    cd /workspace/ARTrack \

# Copy required files
COPY hand.mp4 /workspace/ARTrack/
COPY artrackv2_seq_256_full.pth.tar /workspace/ARTrack/

# Set working directory
WORKDIR /workspace/ARTrack

# Create the conda environment
COPY requirements.txt /workspace/ARTrack/
RUN conda env create -f /workspace/ARTrack/ARTrack_env_cuda113.yaml

# Set the default command
CMD ["/bin/bash", "-c", "source /opt/conda/bin/activate ARTrack_env_cuda113 && tail -f /dev/null"]
