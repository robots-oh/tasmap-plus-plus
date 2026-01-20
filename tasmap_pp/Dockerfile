# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/anaconda3/bin:$PATH"
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7


WORKDIR /workspace
COPY ./requirements.txt .

# Update system and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    libopenexr-dev \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /root/anaconda3 && \
    rm /tmp/anaconda.sh && \
    /root/anaconda3/bin/conda init

# Change the default shell to bash and enable conda
SHELL ["/bin/bash", "-c"]

# Create a new conda environment with Python 3.8
RUN conda create -n tasmap_pp python=3.8 -y

# Activate the conda environment and install required packages
RUN echo "source activate tasmap_pp" >> ~/.bashrc && \
    /root/anaconda3/bin/conda run -n tasmap_pp pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    /root/anaconda3/bin/conda run -n tasmap_pp pip install "pytorch3d==0.7.3" -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html && \
    /root/anaconda3/bin/conda run -n tasmap_pp pip install -r requirements.txt && \
    /root/anaconda3/bin/conda run -n tasmap_pp pip install imageio

ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=0

# Clone and build detectron2 + CropFormer
RUN source activate tasmap_pp && \
    cd /workspace && \
    mkdir -p third_party && cd third_party && \
    git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/qqlu/Entity.git && \
    cp -r Entity/Entityv2/CropFormer detectron2/projects && \
    cd detectron2/projects/CropFormer/entity_api/PythonAPI && \
    make && \
    cd ../.. && \
    cd mask2former/modeling/pixel_decoder/ops && \
    sh make.sh && \
    pip install -U openmim && \
    mim install mmcv && \
    pip install open_clip_torch

# Set the default command to use bash
CMD ["/bin/bash"]
