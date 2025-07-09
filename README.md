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
  - [Local Development](#local-development)
  - [Docker Compose (Recommended)](#docker-compose-recommended)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Adding or Updating Drivers](#adding-or-updating-drivers)
- [Storage Layout](#storage-layout)
- [Access Control](#access-control)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing & Contact](#contributing--contact)

---

## Overview
This system enables fast, local face verification for up to 80,000 drivers, each with three reference photos. The cloud stores all originals and embeddings, while each site keeps only what is needed for rapid matching and guard review. Designed for environments with unreliable connectivity.

## Features
- Cloud ingestion of images and embeddings
- Efficient delta sync to edge sites
- Fast local verification API (sub-200ms)
- Modular, scalable, and Dockerized
- Prometheus metrics for monitoring
- S3-compatible storage with MinIO (default for local/dev)

## Architecture & File Structure
```
face-verification-system/
├── cloud_ingestion/         # Scripts for cloud upload
│   ├── ingest.py
│   ├── layout.md
│   └── vector_index.sql
├── edge_sync/              # Delta sync for edge sites
│   ├── sync.py
│   └── SYNC.md
├── verification_service/   # FastAPI verification API
│   ├── main.py
│   ├── test_verify.py
│   ├── build_index.py
│   ├── faiss_index/
│   │   ├── driver_vectors.index
│   │   ├── metadata.json
│   │   └── thumbs/
│   └── metrics.py
├── requirement.txt
├── Dockerfile
├── docker-compose.yaml
├── README.md
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
- You must have a running PostgreSQL instance for cloud embedding storage.
- Create a database and user, and ensure the `face_vectors` table exists:
  ```sql
  CREATE TABLE face_vectors (
      id SERIAL PRIMARY KEY,
      driver_id TEXT NOT NULL,
      embedding vector(512)
  );
  ```
- Update your connection details in `cloud_ingestion/ingest.py` as needed.

## MinIO Setup (Recommended for Local/Dev)
- MinIO is included as a service in `docker-compose.yaml`.
- Default credentials: `minioadmin` / `minioadmin`.
- S3 API: `http://localhost:9000`
- Web Console: `http://localhost:9001`
- The app and ingestion scripts use MinIO by default via environment variables.

## Setup & Installation
### Local
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Ensure you have the required FAISS index and metadata in `verification_service/faiss_index/`.

### Docker Compose (Recommended)
This project includes a `docker-compose.yaml` that runs the verification service, PostgreSQL, and MinIO together.

1. Build and start all services:
   ```bash
   docker compose up --build
   ```
2. The FastAPI app will be available at [http://localhost:8080](http://localhost:8080)
3. MinIO web console will be at [http://localhost:9001](http://localhost:9001) (login: minioadmin / minioadmin)
4. PostgreSQL will be available at `localhost:5432` (user: myuser, password: mypassword, db: mydb)

### Environment Variables for MinIO (used by app and scripts)
- `MINIO_ENDPOINT` (e.g., `http://minio:9000` in Docker Compose, `http://localhost:9000` on host)
- `MINIO_ACCESS_KEY` (default: `minioadmin`)
- `MINIO_SECRET_KEY` (default: `minioadmin`)
- `MINIO_REGION` (default: `us-east-1`)

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

## Contributing & Contact
- Contributions are welcome! Please open issues or pull requests.
- For help, contact the maintainer or open a GitHub issue.
