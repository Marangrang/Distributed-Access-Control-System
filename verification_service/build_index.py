# verification_service/build_index.py
import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from sklearn.preprocessing import normalize
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

os.makedirs("verification_service/faiss_index/thumbs", exist_ok=True)
embed_index = faiss.IndexFlatIP(512)
metadata = []

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

driver_dir = "data/originals"

def get_driver_id(fname):
    # Assume filenames like 12345_1.jpg, 12345_2.jpg, 12345_3.jpg
    return fname.split('_')[0]

driver_files = sorted([f for f in os.listdir(driver_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
drivers = defaultdict(list)
for fname in driver_files:
    driver_id = get_driver_id(fname)
    drivers[driver_id].append(fname)

idx = 0
for driver_id, files in drivers.items():
    if len(files) != 3:
        print(f"WARNING: Driver {driver_id} has {len(files)} reference photos (expected 3). Skipping.")
        continue
    for fname in sorted(files):
        path = os.path.join(driver_dir, fname)
    img = Image.open(path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        print(f"No face detected in {fname}")
        continue
    emb = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()
    emb = normalize(emb, axis=1)
    embed_index.add(emb)
    # Save thumbnail
    thumb = transform(img)
    thumb_img = transforms.ToPILImage()(thumb)
        thumb_path = f"verification_service/faiss_index/thumbs/{idx}.jpg"
    thumb_img.save(thumb_path)
    metadata.append({
            "driver_id": driver_id,
        "thumb_path": thumb_path
    })
        idx += 1

faiss.write_index(embed_index, "verification_service/faiss_index/driver_vectors.index")
with open("verification_service/faiss_index/metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"Indexed {len(metadata)} driver images.")
