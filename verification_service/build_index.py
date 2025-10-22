# verification_service/build_index.py
import os
import json
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

# use facenet-pytorch for face embeddings (image-based)
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

INDEX_PATH = "verification_service/faiss_index/driver_vectors.index"
METADATA_PATH = "verification_service/faiss_index/metadata.json"
IMAGE_SIZE = 160

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_image(path):
    return Image.open(path).convert("RGB")

def encode_images(image_paths, batch_size=32):
    embeddings = []
    for p in tqdm(image_paths, desc="Embedding images"):
        img = load_image(p)
        # detect & crop face; mtcnn returns a Tensor [3,H,W] or None
        face = mtcnn(img)
        if face is None:
            # fallback: resize full image to expected size
            face = torch.nn.functional.interpolate(
                torch.tensor(np.array(img).transpose(2,0,1))[None].float()/255.0,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode='bilinear'
            )[0]
        with torch.no_grad():
            face = face.to(device).unsqueeze(0) if face.ndim == 3 else face.unsqueeze(0)
            emb = resnet(face).cpu().numpy()[0]
        embeddings.append(emb)
    if len(embeddings) == 0:
        return np.zeros((0, resnet.embedding_size), dtype='float32')
    return np.vstack(embeddings).astype('float32')

def build_index(image_paths, metadata_list):
    """
    image_paths: list of image file paths, one vector per entry
    metadata_list: list of dicts (same length) with driver_id, thumb_path, etc.
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    if len(image_paths) == 0:
        print("No images provided.")
        return

    embeddings = encode_images(image_paths)

    # L2-normalize embeddings so inner-product = cosine on normalized vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors -> cosine
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata_list, f)

    print(f"Saved FAISS index ({index.ntotal} vectors) to {INDEX_PATH} and metadata to {METADATA_PATH}")
