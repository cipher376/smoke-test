FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install minimal deep learning dependencies
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.19.0 \
    trl==0.8.6 \
    accelerate==0.29.3 \
    fsspec==2024.2.0 \
    rich

# Create a user with the same UID as in the securityContext
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set environment variables for cache directories (using user's home)
ENV HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/home/appuser/.cache/huggingface/hub

WORKDIR /workspace

# Copy the python script
COPY train.py /workspace/train.py

# Set ownership to appuser
RUN chown -R appuser:appuser /workspace && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

CMD ["python", "/workspace/train.py"]