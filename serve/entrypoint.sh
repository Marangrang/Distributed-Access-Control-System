#!/bin/bash
set -e

echo "Starting Face Verification Service..."

# Load env file if present (so container gets any values from serve/.env)
set -a
[ -f /app/serve/.env ] && . /app/serve/.env || true
set +a


# Defaults to prevent uvicorn arg errors
UVICORN_HOST=${UVICORN_HOST:-0.0.0.0}
UVICORN_PORT=${UVICORN_PORT:-8000}
UVICORN_WORKERS=${UVICORN_WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}
FAISS_INDEX_PATH=${FAISS_INDEX_PATH:-/app/verification_service/faiss_index}
IMAGES_DIR=${IMAGES_DIR:-/app/verification_service/faiss_index/thumbs}


# Wait for database to be ready
echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
while ! pg_isready -h "${DB_HOST:-db}" -p "${DB_PORT:-5432}" -U "${DB_USER:-appuser}"; do
  sleep 2
done
echo "PostgreSQL is ready!"

# Wait for MinIO to be ready
echo "Waiting for MinIO at ${MINIO_ENDPOINT:-http://minio:9000}..."
while ! curl -sf "${MINIO_ENDPOINT:-http://minio:9000}/minio/health/live" > /dev/null 2>&1; do
  sleep 2
done
echo "MinIO is ready!"

# Initialize FAISS index if it doesn't exist
if [[ ! -f "$FAISS_INDEX_PATH/driver_vectors.index" || ! -f "$FAISS_INDEX_PATH/metadata.json" ]]; then
  echo "Initializing FAISS index at $FAISS_INDEX_PATH..."
  python - <<'PY'
import os, glob
from pathlib import Path

try:
    from verification_service.build_index import build_index
except Exception as e:
    print(f"build_index import failed: {e}")
else:
    images_dir = os.environ.get("IMAGES_DIR", "/app/verification_service/faiss_index/thumbs")
    index_dir  = os.environ.get("FAISS_INDEX_PATH", "/app/verification_service/faiss_index")

    exts = ("*.jpg","*.jpeg","*.png")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_paths:
        print(f"No images found in {images_dir}. Skipping index build.")
    else:
        # Minimal metadata: derive ID from filename
        metadata_list = [
            {"driver_id": Path(p).stem, "thumb_path": os.path.relpath(p, start=index_dir)}
            for p in image_paths
        ]
        try:
            build_index(image_paths, metadata_list)
        except TypeError as te:
            print(f"build_index signature mismatch: {te}")
        except Exception as e:
            print(f"Index build failed: {e}")
PY
  echo "FAISS index initialization completed or skipped"
else
  echo "FAISS index already present. Skipping initialization."
fi

# Start the application
echo "Starting Uvicorn server on $UVICORN_HOST:$UVICORN_PORT..."
exec uvicorn serve.main:app --host "$UVICORN_HOST" --port "$UVICORN_PORT"
