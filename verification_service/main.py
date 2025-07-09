from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import base64
from io import BytesIO
from PIL import Image
import torch
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
import time
import metrics
import os
from fastapi.responses import Response

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Use FAISS IndexHNSWFlat for faster approximate search if not already used
INDEX_PATH = "verification_service/faiss_index/driver_vectors.index"
METADATA_PATH = "verification_service/faiss_index/metadata.json"

# Load or create HNSW index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    # If not HNSW, convert
    if not isinstance(index, faiss.IndexHNSWFlat):
        xb = index.reconstruct_n(0, index.ntotal)
        hnsw_index = faiss.IndexHNSWFlat(index.d, 32)
        hnsw_index.add(xb)
        index = hnsw_index
else:
    index = faiss.IndexHNSWFlat(512, 32)

with open(METADATA_PATH) as f:
    metadata = json.load(f)

app = FastAPI()

class VerifyRequest(BaseModel):
    image_base64: str
    driver_id: str

@app.post("/verify")
async def verify(req: VerifyRequest):
    start = time.time()
    img = Image.open(BytesIO(base64.b64decode(req.image_base64))).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected"}
    emb = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()
    emb = normalize(emb)

    # Find all indices for this driver_id (simulate: assume 3 per driver, consecutive in metadata)
    driver_indices = [i for i, m in enumerate(metadata) if m["driver_id"] == req.driver_id]
    if len(driver_indices) == 0:
        return {"error": "DriverId not found"}
    # If less than 3, just use what we have
    ref_indices = driver_indices[:3]
    # Get reference vectors
    ref_vecs = np.vstack([index.reconstruct(i) for i in ref_indices])
    # Compute cosine similarity
    emb_norm = emb / np.linalg.norm(emb)
    ref_norms = ref_vecs / np.linalg.norm(ref_vecs, axis=1, keepdims=True)
    scores = np.dot(ref_norms, emb_norm[0])
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    suggested_idx = ref_indices[best_idx]
    suggested_driver = metadata[suggested_idx]["driver_id"]
    suggested_image = metadata[suggested_idx]["thumb_path"]
    is_match = bool((req.driver_id == suggested_driver) and (best_score > 0.7))  # ensure Python bool
    latency_ms = (time.time() - start) * 1000
    metrics.record_request()
    metrics.observe_latency(latency_ms)
    return {
        "match": is_match,
        "score": float(best_score),
        "suggestedDriverId": suggested_driver,
        "suggestedImagePath": suggested_image
    }

@app.get("/metrics")
def metrics_endpoint():
    return Response(metrics.get_metrics(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on port 8080...")
    uvicorn.run("verification_service.main:app", host="0.0.0.0", port=8080, reload=False)

# NOTE: All models, index, and metadata are preloaded at startup for low latency. No per-request disk I/O.

