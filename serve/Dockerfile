# syntax=docker/dockerfile:1
FROM python:3.9-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git curl libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY serve/requirement.txt .

# Upgrade pip first (with timeout)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1200 --upgrade pip setuptools wheel

# Install PyTorch with aggressive caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=3000 --retries 5 \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Install remaining deps (filter torch to avoid conflicts)
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" requirement.txt > requirements_no_torch.txt && \
    pip install --default-timeout=3000 --retries 5 \
    -r requirements_no_torch.txt \
    --no-cache-dir

FROM python:3.9-slim

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 postgresql-client curl netcat-openbsd libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY serve/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

COPY serve/ /app/serve/
COPY verification_service/ /app/verification_service/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
