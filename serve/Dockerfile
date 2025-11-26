FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 postgresql-client curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt /app/requirement.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install smaller CPU-only PyTorch
RUN pip install --no-cache-dir --default-timeout=300 torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (skip torch line if present)
RUN grep -v "^torch" requirement.txt > requirements_no_torch.txt || cp requirement.txt requirements_no_torch.txt \
 && pip install --no-cache-dir --default-timeout=300 -r requirements_no_torch.txt

COPY verification_service /app/verification_service
COPY serve/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]