# -------------------------------------------------------------
# RunPod Serverless GPU Base Image (CUDA 12.1, Ubuntu 22.04)
# -------------------------------------------------------------
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

# -------------------------------------------------------------
# System Dependencies (minimal + stable)
# -------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        wget && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Python Dependencies
# Torch 2.1.2 + TorchVision 0.16.1 are the correct CUDA 12.1 wheels
# -------------------------------------------------------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------
# Application Code
# -------------------------------------------------------------
COPY app.py .
COPY remove_sam_lama_fast.py .
COPY best.pt .

# Download SAM ViT-B checkpoint (fast, stable, GPU-compatible)
RUN wget -O sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# -------------------------------------------------------------
# RunPod Serverless: port is dynamic via $PORT
# DO NOT hardcode port 80
# -------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# -------------------------------------------------------------
# Start FastAPI via uvicorn on the correct port
# -------------------------------------------------------------
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
