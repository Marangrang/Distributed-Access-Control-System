# Upload local FAISS index + metadata + thumbs/ to a MinIO bucket
import os
import sys
from botocore.client import Config
import boto3

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "faiss-index")

INDEX_PATH = "verification_service/faiss_index/driver_vectors.index"
METADATA_PATH = "verification_service/faiss_index/metadata.json"
THUMBS_DIR = "verification_service/faiss_index/thumbs"

if not os.path.exists(INDEX_PATH):
    print(f"Index not found at {INDEX_PATH}")
    sys.exit(1)

s3 = boto3.resource(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

# create bucket if it doesn't exist (MinIO allows create)
client = s3.meta.client
try:
    client.head_bucket(Bucket=BUCKET)
    print(f"Bucket {BUCKET} exists")
except Exception:
    print(f"Creating bucket {BUCKET}")
    s3.create_bucket(Bucket=BUCKET)

print("Uploading index and metadata...")
s3.Bucket(BUCKET).upload_file(INDEX_PATH, "driver_vectors.index", ExtraArgs={"ContentType": "application/octet-stream"})
s3.Bucket(BUCKET).upload_file(METADATA_PATH, "metadata.json", ExtraArgs={"ContentType": "application/json"})

if os.path.isdir(THUMBS_DIR):
    for root, _, files in os.walk(THUMBS_DIR):
        for fn in files:
            local = os.path.join(root, fn)
            # key under thumbs/
            key = os.path.join("thumbs", fn)
            print(f"Uploading {local} -> {key}")
            s3.Bucket(BUCKET).upload_file(local, key, ExtraArgs={"ContentType": "image/jpeg"})
else:
    print("No thumbs directory to upload.")

print("Upload complete.")