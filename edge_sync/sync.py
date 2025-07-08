# sync.py: Delta syncs updated embeddings and thumbnails from cloud to edge device
import requests
import json
import os
import time
import sys
sys.path.append("..")
import metrics
import boto3

def load_checkpoint(path="sync_checkpoint.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return "0"

def save_checkpoint(ts, path="sync_checkpoint.txt"):
    with open(path, "w") as f:
        f.write(str(ts))

def get_minio_client():
    endpoint_url = os.environ.get('MINIO_ENDPOINT', 'http://localhost:9000')
    access_key = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
    region = os.environ.get('MINIO_REGION', 'us-east-1')
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

def sync_vectors(api_url="https://your-api.com/vectors", thumb_url="https://your-api.com/thumbs"):
    last_sync = load_checkpoint()
    # Simulate delta fetch: only get items updated since last_sync
    response = requests.get(api_url, params={"since": last_sync})
    if response.status_code == 200:
        data = response.json()
        # Upsert embeddings
        with open("verification_service/faiss_index/driver_vectors.index", "wb") as f:
            f.write(bytes(data["binary_index"]))  # Simulated upsert
        # Upsert thumbnails
        for thumb in data.get("thumbnails", []):
            thumb_resp = requests.get(f"{thumb_url}/{thumb['driver_id']}.jpg")
            if thumb_resp.status_code == 200:
                thumb_path = f"verification_service/faiss_index/thumbs/{thumb['driver_id']}.jpg"
                with open(thumb_path, "wb") as tf:
                    tf.write(thumb_resp.content)
        # Save new checkpoint
        save_checkpoint(data["latest_ts"])
        # Set Prometheus sync lag (assume latest_ts is a unix timestamp)
        now = int(time.time())
        lag = now - int(data["latest_ts"])
        metrics.set_sync_lag(lag)
        print("Delta sync complete. Upserts applied.")
    else:
        print("Failed to sync:", response.text)

if __name__ == "__main__":
    sync_vectors()
