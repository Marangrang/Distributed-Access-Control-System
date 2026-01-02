from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import faiss
import base64
import io
import logging
import os
from pathlib import Path
from datetime import datetime

from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from minio import Minio
from minio.error import S3Error
import json
from prometheus_client import make_asgi_app
import time

# Import metrics
from metrics import (
    record_request,
    observe_latency,
    set_sync_lag,
    record_similarity,
    record_verification_result,
    increment_active_connections,
    decrement_active_connections
)

# Import config utilities
from config.config_loader import get_config, get_config_value
from config.logging_config import setup_logging

# ✅ FIX 1: Setup logging ONCE at the top
setup_logging()  # Configures the entire logging system
logger = logging.getLogger(__name__)  # Get module-specific logger

# ✅ FIX 2: Load config ONCE
config = get_config()

# ✅ FIX 3: Use config values consistently
MINIO_ENDPOINT = get_config_value('minio.endpoint')
MINIO_ACCESS_KEY = get_config_value('minio.access_key')
MINIO_SECRET_KEY = get_config_value('minio.secret_key')
MINIO_SECURE = get_config_value('minio.secure', False)
MODEL_BUCKET = get_config_value('minio.buckets.models', 'trained-models')
SIMILARITY_THRESHOLD = get_config_value('model.similarity_threshold', 0.6)
LOCAL_MODEL_DIR = Path(get_config_value('model.faiss.local_cache_dir', 'faiss_index'))

# Device configuration
device_config = get_config_value('model.device', 'auto')
if device_config == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(device_config)

logger.info(f"Starting Face Verification Service")
logger.info(f"MinIO Endpoint: {MINIO_ENDPOINT}")
logger.info(f"Model Bucket: {MODEL_BUCKET}")
logger.info(f"Device: {device}")
logger.info(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")

# Global state
last_model_update = None

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)


def download_model_from_minio():
    """Download latest model files from MinIO"""
    global last_model_update

    logger.info(f"Downloading model from MinIO bucket: {MODEL_BUCKET}")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        'driver_vectors.npy',
        'driver_vectors.index',
        'metadata.json'
    ]

    for filename in files_to_download:
        object_name = f"models/latest/{filename}"
        local_file = LOCAL_MODEL_DIR / filename

        try:
            minio_client.fget_object(
                MODEL_BUCKET,
                object_name,
                str(local_file)
            )
            logger.info(f"✓ Downloaded {filename}")
        except S3Error as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    last_model_update = datetime.now()
    set_sync_lag(0)
    logger.info(f"✓ Model downloaded successfully at {last_model_update}")


# Download model on startup
try:
    download_model_from_minio()
except Exception as e:
    logger.warning(f"Could not download model from MinIO: {e}")
    logger.info("Using local model files if available")

# Load FAISS index
INDEX_PATH = LOCAL_MODEL_DIR / "driver_vectors.index"
METADATA_PATH = LOCAL_MODEL_DIR / "metadata.json"

index = None
metadata = []

try:
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"✓ Loaded FAISS index with {index.ntotal} vectors")
    else:
        logger.error(f"Index file not found: {INDEX_PATH}")
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
except Exception as e:
    logger.error(f"Error loading index: {e}")
    raise

try:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        logger.info(f"✓ Loaded metadata for {len(metadata)} drivers")
    else:
        logger.warning("Metadata file not found")
        metadata = []
except Exception:
    logger.exception("Error loading metadata")
    metadata = []

# Initialize FastAPI
app = FastAPI(
    title="Face Verification Service",
    version="1.0.0",
    description="Face recognition verification API using FaceNet and FAISS"
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize face detection and recognition models
mtcnn_config = get_config_value('model.mtcnn', {})
mtcnn = MTCNN(
    image_size=mtcnn_config.get('image_size', 160),
    margin=mtcnn_config.get('margin', 0),
    min_face_size=mtcnn_config.get('min_face_size', 20),
    thresholds=mtcnn_config.get('thresholds', [0.6, 0.7, 0.7]),
    factor=mtcnn_config.get('factor', 0.709),
    keep_all=mtcnn_config.get('keep_all', False),
    device=device
)

model_pretrained = get_config_value('model.pretrained', 'vggface2')
model = InceptionResnetV1(pretrained=model_pretrained).eval().to(device)

logger.info(f"✓ Models initialized (FaceNet: {model_pretrained})")


class VerifyPayload(BaseModel):
    driver_id: str = None
    image_base64: str = None


def embed_image_from_base64(b64: str):
    """Extract face embedding from base64 image"""
    try:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        face = mtcnn(img)
        if face is None:
            return None

        with torch.no_grad():
            face = face.unsqueeze(0).to(device)
            embedding = model(face).cpu().numpy()

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        return embedding.flatten().astype('float32')

    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None


@app.middleware("http")
async def track_connections(request: Request, call_next):
    """Middleware to track active connections"""
    increment_active_connections()
    try:
        response = await call_next(request)
        return response
    finally:
        decrement_active_connections()


@app.get("/health")
def health():
    """Health check endpoint"""
    # Update sync lag
    if last_model_update:
        lag_seconds = (datetime.now() - last_model_update).total_seconds()
        set_sync_lag(lag_seconds)

    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "num_vectors": index.ntotal if index else 0,
        "num_metadata": len(metadata),
        "last_model_update": last_model_update.isoformat() if last_model_update else None,
        "device": str(device),
        "model": model_pretrained,
        "threshold": SIMILARITY_THRESHOLD
    }


@app.post("/verify")
def verify(payload: VerifyPayload):
    """Verify a face image against stored embeddings"""
    start_time = time.time()

    try:
        if not payload.image_base64:
            record_request('invalid_request')
            raise HTTPException(status_code=400, detail="image_base64 is required")

        # Extract embedding from input image
        query_emb = embed_image_from_base64(payload.image_base64)
        if query_emb is None:
            record_request('no_face_detected')
            latency_ms = (time.time() - start_time) * 1000
            observe_latency(latency_ms)
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Search in FAISS index
        query_emb = query_emb.reshape(1, -1)
        distances, indices = index.search(query_emb, k=1)

        distance = float(distances[0][0])
        idx = int(indices[0][0])

        # Get metadata for matched driver
        if idx < len(metadata):
            matched_driver = metadata[idx]
            driver_id = matched_driver.get('driver_id', 'unknown')
            name = matched_driver.get('name', driver_id)
        else:
            driver_id = 'unknown'
            name = 'Unknown'

        # Calculate similarity score (convert distance to similarity)
        similarity = 1.0 / (1.0 + distance)

        # Use threshold from config
        verified = similarity > SIMILARITY_THRESHOLD

        # Record metrics
        record_request('success')
        record_similarity(similarity)
        record_verification_result(verified, similarity)

        latency_ms = (time.time() - start_time) * 1000
        observe_latency(latency_ms)

        result = {
            "verified": verified,
            "driver_id": driver_id,
            "name": name,
            "similarity": float(similarity),
            "distance": float(distance),
            "threshold": SIMILARITY_THRESHOLD,
            "latency_ms": float(latency_ms)
        }

        logger.info(
            f"Verification: {driver_id} - "
            f"Verified={verified}, "
            f"Similarity={similarity:.4f}, "
            f"Latency={latency_ms:.2f}ms"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        record_request('error')
        latency_ms = (time.time() - start_time) * 1000
        observe_latency(latency_ms)
        logger.error(f"Verification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Face Verification Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "verify": "/verify (POST)",
            "metrics": "/metrics",
            "stats": "/stats"
        }
    }


@app.get("/stats")
def stats():
    """Service statistics"""
    return {
        "total_vectors": index.ntotal if index else 0,
        "total_people": len(metadata),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_bucket": MODEL_BUCKET,
        "minio_endpoint": MINIO_ENDPOINT,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "model_pretrained": model_pretrained,
        "last_update": last_model_update.isoformat() if last_model_update else None
    }


if __name__ == "__main__":
    import uvicorn

    host = get_config_value('api.host', '0.0.0.0')
    port = get_config_value('api.port', 8080)

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None  # Use our logging config instead of uvicorn's
    )
