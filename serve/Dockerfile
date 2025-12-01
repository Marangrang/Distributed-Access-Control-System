# syntax=docker/dockerfile:1
FROM python:3.14-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirement.txt /app/requirement.txt

# Upgrade pip with longer timeout
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1200 --upgrade pip setuptools wheel

# Install PyTorch CPU version with retry logic and cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1200 --retries 10 \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other requirements (excluding torch)
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" requirement.txt > requirements_no_torch.txt || cp requirement.txt requirements_no_torch.txt && \
    pip install --default-timeout=1200 -r requirements_no_torch.txt

# Stage: Final image
FROM python:3.14-slim

WORKDIR /app

# Install runtime system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 postgresql-client curl netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY serve/ /app/
COPY verification_service/ /app/verification_service/

# Make entrypoint executable
RUN test -f /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]