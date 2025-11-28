# Valid, existing RunPod PyTorch GPU image
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "app.py"]
