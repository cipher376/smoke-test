FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime


    
# Install minimal deep learning dependencies
RUN pip install --no-cache-dir --break-system-packages  \
    transformers==5.9.0 \
    datasets==4.8.5 \
    trl==1.4.0 \
    accelerate==1.13.0\
    fsspec==2026.2.0 \
    rich==15.0.0

# Create a user with the same UID as in the securityContext
# RUN useradd -m -u 1000 -s /bin/bash ubuntu

# Set environment variables for cache directories (using user's home)
ENV HF_HOME=/home/ubuntu/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub

WORKDIR /workspace

# Copy the python script
COPY train.py /workspace/train.py

# Set ownership to ubuntu
RUN chown -R ubuntu:ubuntu /workspace && \
    mkdir -p /home/ubuntu/.cache && \
    chown -R ubuntu:ubuntu /home/ubuntu

# Switch to non-root ubuntu
USER ubuntu

CMD ["python", "/workspace/train.py"] 