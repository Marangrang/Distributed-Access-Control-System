FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirement.txt /app/requirement.txt

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch without hash verification
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 --trusted-host pypi.org --trusted-host files.pythonhosted.org torch

# Install other requirements
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirement.txt

# Copy application code
COPY verification_service /app/verification_service
COPY serve/entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]