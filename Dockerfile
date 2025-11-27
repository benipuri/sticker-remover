FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code + models
COPY remove_sam_lama_fast.py .
COPY app.py .
COPY best.pt .
COPY sam_vit_b_01ec64.pth .

# Expose RunPod's load balancer port (80)
EXPOSE 80

# Start Uvicorn on port 80 (RunPod proxy default for health checks)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
