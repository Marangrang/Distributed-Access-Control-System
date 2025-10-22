# Face Verification System

A modular, distributed face verification system for edge-cloud environments, designed for secure, high-speed access control across multiple sites with intermittent connectivity.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture & File Structure](#architecture--file-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Deployment notes](#deployment-notes)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Adding or Updating Drivers](#adding-or-updating-drivers)
- [Storage Layout](#storage-layout)
- [Access Control](#access-control)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing & Contact](#contributing--contact)

---

## Overview
This system enables fast, local face verification for up to 80,000 drivers, each with three reference photos. The cloud stores originals and embeddings; each site keeps the minimal artifacts needed for fast matching and guard review.

## Features
- Cloud ingestion of images and embeddings
- Delta sync to edge sites
- Fast local verification API (sub-200ms)
- Dockerized with Prometheus metrics and MinIO for local S3 compatibility

## Architecture & File Structure
```
app/
├── config/                   # runtime config and gunicorn/logging
│   ├── config.yaml
│   ├── logging.conf
│   └── gunicorn.conf.py
├── serve/                    # deployment / runtime files (canonical)
│   ├── main.py                # FastAPI app (entrypoint moved if needed)
│   ├── wsgi.py                # for gunicorn
│   ├── nginx.conf             # reverse proxy (used by nginx service)
│   ├── entrypoint.sh          # waits for DB/MinIO on startup
│   ├── Dockerfile             # canonical image builder for app
│   └── .env                   # local env vars (do NOT commit secrets)
├── train/                    # tools to build embeddings/index
│   ├── build_index.py
│   ├── dataset_prep.py
│   ├── model_export.py
│   └── requirements.txt
├── verification_service/     # service code and runtime index
│   ├── main.py
│   ├── build_index.py
│   ├── test_verify.py
│   ├── faiss_index/
│   │   ├── driver_vectors.index   # large — exclude from image, mount at runtime
│   │   ├── metadata.json
│   │   └── thumbs/
│   └── metrics.py
├── .dockerignore
├── .gitignore
├── requirement.txt
├── docker-compose.yaml
└── README.md
```

## Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- Docker Compose (for multi-service orchestration)
- pip (for local development)
- PostgreSQL (for cloud embedding storage; see below)
- MinIO (for local S3-compatible storage; see below)
- (Optional) AWS CLI or compatible S3 credentials for cloud storage

## PostgreSQL Setup
- You must have a running PostgreSQL instance for cloud embedding storage (pgvector extension recommended).
- Create a database and user, and ensure the `face_vectors` and `driver_metadata` tables exist:
  ```sql
  CREATE TABLE face_vectors (
      id SERIAL PRIMARY KEY,
      driver_id TEXT NOT NULL,
      embedding vector(512)
  );

  CREATE TABLE IF NOT EXISTS driver_metadata (
      driver_id VARCHAR PRIMARY KEY,
      thumb_path VARCHAR NOT NULL
  );
  ```
- Update your connection details in `cloud_ingestion/ingest.py` and `verification_service/build_index.py` as needed.
- Note: the repo uses the ankane/pgvector image in docker-compose for local PG+pgvector support. I avoided an external PostgreSQL docs hyperlink in earlier edits to keep the README self-contained — if you want a link back to official docs add:
  - PostgreSQL docs: https://www.postgresql.org/docs/
  - pgvector: https://github.com/ankane/pgvector
- To access PostgreSQL from your host:
  ```bash
  psql -h localhost -p 5432 -U myuser -d mydb
  # Password: mypassword
  ```
- Or, from inside the container:
  ```bash
  docker compose exec db bash
  psql -U myuser -d mydb
  ```

## MinIO Setup (Recommended for Local/Dev)
- MinIO is included as a service in `docker-compose.yaml`.
- Default credentials: `minioadmin` / `minioadmin`.
- S3 API: `http://localhost:9000`
- Web Console: `http://localhost:9001`
- The app and ingestion scripts use MinIO by default via environment variables.

## MinIO, Edge Sync and Index Management (NEW)

These notes collect the new upload/download and sync behavior added to the repo.

Important environment variables
- Ensure MINIO_ENDPOINT includes the scheme (http:// or https://) — e.g.:
  - MINIO_ENDPOINT=http://minio:9000
- Add these to serve/.env or provide via Docker secrets:
  - MINIO_ENDPOINT
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
  - MINIO_BUCKET (default used in scripts: faiss-index)
- Ensure docker-compose exposes MinIO ports to the host:
  - ports: "9000:9000" and "9001:9001"

Upload index & assets to MinIO
- Script: train/upload_index_to_minio.py
- What it uploads:
  - driver_vectors.index -> key: driver_vectors.index (binary, ContentType application/octet-stream)
  - metadata.json -> key: metadata.json (application/json)
  - thumbs/* -> keys under thumbs/ (image/jpeg)
- Usage (from repo root, ensure MINIO_* env set):
  ```bash
  python train/upload_index_to_minio.py
  ```

Edge sync: download directly from MinIO (recommended)
- Script: edge_sync/sync.py now supports downloading index/metadata/thumbs directly from MinIO using boto3.download_file.

## Setup & Installation
### Local
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Ensure you have the required FAISS index and metadata in `verification_service/faiss_index/`.

### Docker Compose (Recommended)
This project includes `docker-compose.yaml` and a canonical app builder in `serve/Dockerfile`.

1. Build and start services:
   ```bash
   docker compose up --build
   ```
2. FastAPI: http://localhost:8080 (docs at /docs)
3. MinIO console: http://localhost:9001 (minioadmin / minioadmin)

---

## Deployment notes
- Canonical runtime files live under `serve/` and `config/`. Use `serve/Dockerfile` to build the app image.
- Do NOT bake large model/index artifacts into the image. Add a `.dockerignore` and `.gitignore` to exclude:
  - `verification_service/faiss_index/driver_vectors.index`
  - `verification_service/faiss_index/thumbs/`
  - `models/` and large model files (`*.pt`, `*.pth`)
  Prefer mounting `verification_service/faiss_index/` and `./models` at runtime or downloading them during container startup (edge_sync).
- Use `serve/.env` for local development env vars. For production, prefer Docker secrets or an external secrets manager; do not commit real secrets.
- The app exposes `/health` used by docker-compose healthchecks.
- A lightweight entrypoint script (`serve/entrypoint.sh`) waits for DB and MinIO before launching the app — improves first-boot reliability.
- Logging: Gunicorn/uvicorn reads `config/gunicorn.conf.py` and `config/logging.conf`. Ensure logging config is loaded in app startup if you customize logging.
- CI: A basic GitHub Actions workflow is provided at `.github/workflows/ci.yml` (lint, build, run tests if present).

Recommended mounts (docker-compose):
- `./verification_service/faiss_index:/app/verification_service/faiss_index:ro`
- `./models:/app/models:ro`
- `./config:/app/config:ro`

If you prefer the container to fetch models and index files itself, ensure the relevant scripts (e.g., edge_sync) are triggered on startup, and consider using a startup script or command in your docker-compose.yml to handle this.

## Usage
### Local Development
Start the API server:
```bash
python3 -m verification_service.main
```

### Docker Compose
Start all services:
```bash
docker compose up --build
```

## API Documentation
### POST /verify
- **Request JSON:**
  ```json
  {
    "driver_id": "12345",
    "image_base64": "..."
  }
  ```
- **Response JSON:**
  ```json
  {
    "match": true,
    "score": 0.87,
    "suggestedDriverId": "12345",
    "suggestedImagePath": "/thumbs/12345.jpg"
  }
  ```
- **Description:**
  - Accepts a base64-encoded image and a driver ID.
  - Compares the image to the three reference vectors for the driver.
  - Returns match status, similarity score, and the best-matching reference.

## Testing
Run the included test script:
```bash
python3 verification_service/test_verify.py
```
- Prints the API response and latency in milliseconds.

## Adding or Updating Drivers
1. Add new images to `data/originals/` (use filenames like `driverid_1.jpg`, `driverid_2.jpg`, `driverid_3.jpg`).
2. Run `build_index.py` to update the FAISS index and metadata.
3. Use `cloud_ingestion/ingest.py` to upload new originals, thumbnails, and embeddings to the cloud (MinIO by default).
4. Edge sites will receive updates on the next sync cycle.

## Storage Layout
- **Originals:** `originals/{driver_id}.jpg` in S3-compatible object storage (audit/retraining only).
- **Thumbnails:** `thumbnails/{driver_id}.jpg` in S3-compatible object storage (synced to edge for guard review).
- **Embeddings:** 512-D vectors in cloud vector store (e.g., PostgreSQL+pgvector), keyed by driver_id.

## Access Control
- Only authorized edge sites can sync embeddings and thumbnails.
- Full-resolution originals are accessible only in the cloud for compliance and retraining.
- API and storage access is restricted by site and user role.

## Troubleshooting & FAQ
- **Service not starting?** Ensure all dependencies are installed and index files are present.
- **Docker build fails?** Check that all files are in the correct locations and Docker has access to them.
- **MinIO connection refused?** Make sure MinIO is running (`docker-compose up`) and the endpoint is correct for your environment.
- **Latency too high?** Make sure you are running on a machine with sufficient CPU resources.
- **Need to reset sync?** Delete the `sync_checkpoint.txt` file on the edge device.

---

## Embedding model & retrieval improvements

We updated the repo to use an image-first, production-friendly retrieval pipeline. Summary of what changed and why:

- Image embedding model
  - Use facenet-pytorch (MTCNN + InceptionResnetV1 pretrained on vggface2) for face embeddings.
  - Rationale: text embedding models (BAAI/bge-*, sentence-transformers) are not appropriate for image face verification — dedicated face models give far better performance for this task.

- Embedding normalization
  - All image embeddings are L2-normalized (unit length). Normalization makes cosine similarity stable and comparable across vectors.

- FAISS configuration
  - Index built as IndexFlatIP over normalized vectors. Inner product on unit vectors == cosine similarity.
  - This keeps retrieval semantics intuitive (scores in [-1, 1]) and efficient for our dataset sizes. For very large corpora you may switch to IndexHNSWFlat or IVF+PQ (profiling required).

- Cross-encoder / text reranker
  - Removed / disabled by default — this service is image-only. If you later add textual metadata and want reranking, a CrossEncoder can be added as a separate reranker stage.

- Code locations changed
  - verification_service/build_index.py — encodes images with facenet-pytorch, L2-normalizes embeddings, builds IndexFlatIP and writes index + metadata JSON.
  - verification_service/main.py — image-only pipeline: decodes base64 probe images, runs MTCNN->InceptionResnetV1 -> L2-normalize, queries FAISS and returns candidates.
  - verification_service/test_verify.py — test harness uses the same image pipeline.

- Requirements
  - requirement.txt updated to include facenet-pytorch, faiss-cpu, and supporting libs. Rebuild environment or container after pulling changes:
    - pip install -r requirement.txt
    - or docker compose up --build

- Docker / deployment notes
  - Do NOT copy large artifacts (driver_vectors.index, thumbs, model checkpoints) into images — mount them or download at startup.
  - serve/entrypoint.sh added to wait for DB/MinIO before starting gunicorn.
  - serve/Dockerfile updated to include facenet dependencies and entrypoint.

- How to rebuild and test locally
  1. Ensure images and metadata exist at verification_service/faiss_index/ (driver_vectors.index, metadata.json, thumbs/).
  2. Build index locally (example usage — adapt to your dataset):
     ```bash
     python verification_service/build_index.py  # or run the helper that collects image_paths + metadata then calls build_index()
     ```
  3. Start services:
     ```bash
     docker compose up --build
     ```
  4. Verify with the test script:
     ```bash
     python verification_service/test_verify.py
     ```

- Scoring / thresholds
  - Returned scores are cosine similarity (inner-product on normalized vectors). Values near 1.0 indicate very close matches, 0 indicates orthogonal, -1 opposite.
  - Select thresholds empirically on held-out pairs (e.g., TPR/FPR tradeoff). Typical face-verification thresholds vary by model and data quality (run ROC/PR evaluation to pick).

- Notes & next steps
  - If you want to support multi-modal retrieval (text + image), we can add a separate text-embedding pipeline (e.g., BGE or sentence-transformers) and an optional reranker.
  - For large-scale deployment consider HNSW/IVF+PQ plus incremental update tooling (or remote indexing with periodic full index rebuilds).



