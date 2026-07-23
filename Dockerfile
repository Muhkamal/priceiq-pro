# PriceIQ Pro V5 — Dockerfile
# Optimised for Render.com deployment (free/starter tier)
# Multi-stage build: keeps final image slim

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directory for persistence files
RUN mkdir -p /app/data && chmod 777 /app/data

# Non-root user for security
RUN useradd -m -u 1000 priceiq && chown -R priceiq:priceiq /app
USER priceiq

# Environment defaults (override in Render dashboard)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    REGIME_MODEL_PATH=/app/data/regime_model.pkl \
    LEARNING_STATE_PATH=/app/data/learning_state.json \
    REGIME_WEIGHTS_PATH=/app/data/regime_weights.json \
    WIN_PROB_PATH=/app/data/win_prob_calibrator.json \
    TRANSITION_PATH=/app/data/regime_transitions.json \
    JOURNAL_PATH=/app/data/trade_journal.json \
    EQUITY_CURVE_PATH=/app/data/equity_curve.json \
    POSITIONS_SNAPSHOT=/app/data/open_positions_snapshot.json

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
