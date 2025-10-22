from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import base64
from io import BytesIO
from PIL import Image
import torch
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
from verification_service import metrics
import os
import logging
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification_service")

# Load runtime config (optional - falls back to env/defaults)
CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
try:
    import yaml
    cfg_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    if os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
            SIMILARITY_THRESHOLD = float(cfg.get("model", {}).get("similarity_threshold", SIMILARITY_THRESHOLD))
except Exception:
    pass  # yaml may not be installed in minimal env; env var still works

# Single device + image model init (image-only pipeline)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "verification_service/faiss_index/driver_vectors.index")
METADATA_PATH = os.getenv("METADATA_PATH", "verification_service/faiss_index/metadata.json")

# Removed text-embedding / cross-encoder code because this service is image-only.
# Load FAISS index + metadata robustly
index = None
metadata = []
try:
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        logger.info("Loaded FAISS index from %s (ntotal=%d)", INDEX_PATH, index.ntotal)
    else:
        logger.warning("FAISS index not found at %s — creating empty IndexFlatIP(512).", INDEX_PATH)
        index = faiss.IndexFlatIP(int(os.getenv("EMBED_DIM", "512")))
except Exception as e:
    logger.exception("Error loading FAISS index: %s", e)
    index = faiss.IndexFlatIP(int(os.getenv("EMBED_DIM", "512")))

try:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    else:
        metadata = []
except Exception as e:
    logger.exception("Error loading metadata: %s", e)
    metadata = []

app = FastAPI()

class VerifyPayload(BaseModel):
    driver_id: str = None
    image_base64: str = None

def embed_image_from_base64(b64: str):
    try:
        data = base64.b64decode(b64)
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="invalid base64 image")

    face = mtcnn(img)
    if face is None:
        # fallback: resize full image to expected size and convert to tensor
        img_resized = img.resize((160,160))
        face = torch.tensor(np.array(img_resized).transpose(2,0,1)).float() / 255.0
    face = face.to(device).unsqueeze(0) if face.ndim == 3 else face.unsqueeze(0)
    with torch.no_grad():
        emb = resnet(face).cpu().numpy()[0].astype('float32')
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.reshape(1, -1)

@app.post("/verify")
def verify(payload: VerifyPayload):
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 required")

    q_emb = embed_image_from_base64(payload.image_base64)

    k = int(os.getenv("K_NEIGHBORS", "5"))
    D, I = index.search(q_emb, k)
    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        candidates.append({
            "index": int(idx),
            "score": float(score),  # cosine on normalized vectors
            "meta": meta
        })

    # Simple thresholding example (can be tuned)
    match = len(candidates) > 0 and candidates[0]["score"] >= SIMILARITY_THRESHOLD
    result = {"match": bool(match), "candidates": candidates}
    return result

@app.get("/metrics")
def metrics_endpoint():
    return Response(metrics.get_metrics(), media_type="text/plain")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": (index is not None and getattr(index, "ntotal", 0) >= 0),
        "vectors": int(getattr(index, "ntotal", 0)),
        "metadata_count": len(metadata)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on port 8080...")
    uvicorn.run("verification_service.main:app", host="0.0.0.0", port=8080, reload=False)

# NOTE: All models, index, and metadata are preloaded at startup for low latency. No per-request disk I/O.

