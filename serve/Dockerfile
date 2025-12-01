# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 postgresql-client curl build-essential netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
# If your file is actually requirements.txt, rename or change this line
COPY requirement.txt /app/requirement.txt

# Upgrade pip (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install PyTorch CPU (cached layer) with increased timeout
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=600 \
    torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements (cached) with increased timeout
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" /app/requirement.txt > /app/requirements_no_torch.txt || cp /app/requirement.txt /app/requirements_no_torch.txt && \
    pip install --default-timeout=600 -r /app/requirements_no_torch.txt

# Copy app code and entrypoint
COPY serve/ /app/
# Make sure entrypoint is present and executable
RUN test -f /app/entrypoint.sh && chmod +x /app/entrypoint.sh
# Default command (or entrypoint) to run the script
ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8000