# 1. Start from a standard NVIDIA runtime that extracts cleanly rootless
FROM docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 2. Install Python and standard system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Combine and install PyTorch + ML dependencies using explicit pip3
RUN pip3 install --no-cache-dir \
    torch==2.2.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.19.0 \
    trl==0.8.6 \
    accelerate==0.29.3 \
    fsspec==2024.2.0

# 4. Use a path guaranteed to be writable by rootless containers
WORKDIR /tmp

# Copy the python script into the container
COPY train.py /tmp/train.py

ENTRYPOINT ["python3", "/tmp/train.py"]