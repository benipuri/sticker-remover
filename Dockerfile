# Base image with PyTorch 2.4.0 and CUDA (RunPod recommended)
FROM runpod/pytorch:2.4.0

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# No buffering for logs
ENV PYTHONUNBUFFERED=1

# FastAPI server will be started by app.py (which reads PORT env var)
CMD ["python3", "app.py"]
