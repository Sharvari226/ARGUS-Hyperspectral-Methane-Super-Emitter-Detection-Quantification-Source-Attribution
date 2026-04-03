# ═══════════════════════════════════════════════════════════════════
# ARGUS — Production Dockerfile
# Multi-stage build for minimal image size
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libgdal-dev gdal-bin \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .

# Install PyTorch CPU version first (smaller for deployment)
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install rest of requirements
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Try Modulus (optional — don't fail build if unavailable)
RUN pip install --no-cache-dir --prefix=/install nvidia-modulus==0.6.0 || \
    echo "Modulus unavailable — PyTorch PINN fallback will be used"

# ── Stage 2: runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p \
    data/raw/tropomi \
    data/raw/ecmwf \
    data/raw/emit \
    data/processed \
    data/facilities \
    data/active_learning \
    data/runs \
    models/checkpoints \
    logs

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GEE auth directory
RUN mkdir -p /root/.config/earthengine

EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: run API
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]