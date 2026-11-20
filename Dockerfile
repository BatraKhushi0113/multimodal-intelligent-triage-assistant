# Multi-stage optimized Python 3.10 image for Render Cloud Deployment
FROM python:3.10-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    MPLCONFIGDIR=/tmp/mpl \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Install essential system libraries for audio (Whisper), vision (OpenCV), and health checks (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight PyTorch CPU wheels (avoids 3GB+ CUDA wheels on Render CPU instances)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements-deployment.txt /app/
RUN pip install --no-cache-dir -r requirements-deployment.txt

# Copy project files
COPY . /app/

# Ensure runtime directories exist and have proper permissions
RUN mkdir -p /app/logs/.mpl /tmp/mpl && chmod -R 777 /app/logs /tmp/mpl

# Expose default ports
EXPOSE 8501 8000 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || curl -f http://localhost:${PORT}/health || exit 1

# Launch application
CMD ["python", "run_render.py"]
