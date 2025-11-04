FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirement.txt /app/requirement.txt

# Install torch separately (uncomment / adjust version for CUDA images if needed)
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 torch

# Install remaining requirements
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r /app/requirement.txt

# copy app code into image
COPY . /app

# Ensure runtime dirs
RUN mkdir -p /app/verification_service/faiss_index/thumbs /app/models

# copy entrypoint and make executable
COPY serve/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV UVICORN_WORKERS=2

# Expose port used by Gunicorn/Uvicorn
EXPOSE 8080

# Use Gunicorn + Uvicorn worker, read gunicorn config from config/gunicorn.conf.py
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["gunicorn", "-c", "/app/config/gunicorn.conf.py", "serve.wsgi:app"]