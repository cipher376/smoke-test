FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install minimal deep learning dependencies
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.19.0 \
    trl==0.8.6 \
    accelerate==0.29.3 \
    fsspec==2024.2.0 \
    rich  

# Create cache directory with proper permissions
RUN mkdir -p /.cache/huggingface && \
    chmod 777 /.cache

# Set environment variables for cache directories
ENV HF_HOME=/.cache/huggingface \
    TRANSFORMERS_CACHE=/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/.cache/huggingface/hub

WORKDIR /workspace

# Copy the python script into the container
COPY train.py /workspace/train.py

ENTRYPOINT ["python", "/workspace/train.py"]