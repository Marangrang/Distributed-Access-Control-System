# Use official Python image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirement.txt ./
# psycopg2-binary is included for PostgreSQL support
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8080

# Optionally install torch with CUDA for GPU support (uncomment if using GPU base image)
# RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Run the FastAPI app with configurable workers (default 2)
ENV UVICORN_WORKERS=2
CMD ["sh", "-c", "uvicorn verification_service.main:app --host 0.0.0.0 --port 8080 --workers ${UVICORN_WORKERS}"] 