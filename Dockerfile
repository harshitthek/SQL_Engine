# ============================================================
# Text-to-SQL Engine — Multi-Stage Dockerfile
# ============================================================
# Build:  docker build -t sql-engine .
# Run:    docker run -p 8000:8000 -p 7860:7860 \
#           -v ./models:/app/models sql-engine
# ============================================================

# ---------- Stage 1: Base with system dependencies ----------
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Stage 2: Install Python dependencies ----------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 3: Production image ----------
FROM deps AS production

# Copy deployment code
COPY deployment/ ./deployment/

# Copy src package (inference engine used by prediction.py fallback)
COPY src/ ./src/
COPY data/__init__.py ./data/__init__.py
COPY data/processing/ ./data/processing/

# Copy configuration files
COPY pyproject.toml .
COPY .env_example .env

# Expose ports: FastAPI (8000) + Gradio (7860)
EXPOSE 8000 7860

# Model weights are mounted as a volume, not baked into the image
VOLUME ["/app/models"]

# Health check against FastAPI
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health', timeout=3); exit(0 if r.ok else 1)" \
    || exit 1

# Default: launch the orchestrator (starts FastAPI + Gradio)
CMD ["python", "deployment/app.py"]
