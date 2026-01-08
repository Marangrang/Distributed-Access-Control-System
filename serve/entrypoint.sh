#!/bin/bash
set -e

echo "Starting Face Verification Service..."

# Load env file if present
set -a
if [ -f /app/serve/.env ]; then
  . /app/serve/.env
else
  echo "[INFO] /app/serve/.env not found, continuing with defaults"
fi
set +a

# Defaults to prevent uvicorn arg errors
UVICORN_HOST=${UVICORN_HOST:-0.0.0.0}
UVICORN_PORT=${UVICORN_PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-info}
FAISS_INDEX_PATH=${FAISS_INDEX_PATH:-/app/verification_service/faiss_index}
IMAGES_DIR=${IMAGES_DIR:-/app/verification_service/faiss_index/thumbs}


mkdir -p "$FAISS_INDEX_PATH" "$IMAGES_DIR"

# Optionally wait for database if enabled
if [[ "${WAIT_FOR_POSTGRES:-false}" == "true" ]]; then
  echo "Waiting for PostgreSQL at ${DB_HOST:-db}:${DB_PORT:-5432}..."
  while ! pg_isready -h "${DB_HOST:-db}" -p "${DB_PORT:-5432}" -U "${DB_USER:-appuser}" >/dev/null 2>&1; do
    sleep 2
  done
  echo "PostgreSQL is ready!"
fi

# Wait for MinIO
echo "Waiting for MinIO at minio:9000..."
MINIO_PROTO="http"
if [[ "${MINIO_SECURE:-false}" == "true" ]]; then
  MINIO_PROTO="https"
fi
while ! curl -sf "${MINIO_PROTO}://minio:9000/minio/health/live" >/dev/null 2>&1; do
  sleep 2
done
echo "MinIO is ready!"

# Download trained artifacts from MinIO (preferred) and fallback to local build
echo "Syncing trained model artifacts from MinIO into $FAISS_INDEX_PATH..."
python - <<'PY'
import os, glob
from pathlib import Path

endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
access_key = os.environ.get('MINIO_ACCESS_KEY')
secret_key = os.environ.get('MINIO_SECRET_KEY')
bucket = os.environ.get('MODEL_BUCKET') or os.environ.get('MINIO_BUCKET', 'trained-models')
secure = os.environ.get('MINIO_SECURE', 'false').lower() in ('1','true','yes')

local_dir = os.environ.get('FAISS_INDEX_PATH', '/app/verification_service/faiss_index')
Path(local_dir).mkdir(parents=True, exist_ok=True)

download_ok = False
try:
  from edge_sync.sync import ModelSyncManager
  mgr = ModelSyncManager(endpoint=endpoint, access_key=access_key, secret_key=secret_key,
               bucket=bucket, local_model_dir=local_dir, secure=secure)
  download_ok = mgr.download_model_files()
  print(f"MinIO download success: {download_ok}")
except Exception as e:
  print(f"MinIO download failed: {e}")

index_path = os.path.join(local_dir, 'driver_vectors.index')
metadata_path = os.path.join(local_dir, 'metadata.json')
if not (os.path.isfile(index_path) and os.path.isfile(metadata_path)):
  print("Artifacts missing after MinIO sync; attempting local index build from thumbs...")
  try:
    from verification_service.build_index import build_index
    images_dir = os.environ.get('IMAGES_DIR', '/app/verification_service/faiss_index/thumbs')
    exts = ("*.jpg","*.jpeg","*.png")
    image_paths = []
    for ext in exts:
      image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    if not image_paths:
      print(f"No images found in {images_dir}. Skipping index build.")
    else:
      metadata_list = [{"driver_id": Path(p).stem, "thumb_path": os.path.relpath(p, start=local_dir)} for p in image_paths]
      try:
        build_index(image_paths, metadata_list)
      except Exception as e:
        print(f"Index build failed: {e}")
  except Exception as e:
    print(f"Fallback build_index import failed: {e}")
else:
  print("Artifacts present; skipping local build.")
PY

# Start the application
echo "Starting Uvicorn server on $UVICORN_HOST:$UVICORN_PORT (single worker)..."
exec uvicorn serve.main:app --host "$UVICORN_HOST" --port "$UVICORN_PORT" --log-level "$LOG_LEVEL"
