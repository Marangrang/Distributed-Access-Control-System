# sync.py: Delta syncs updated embeddings and thumbnails from cloud to
# edge device
import requests
import os
import time
import logging
from botocore.client import Config
import boto3
from verification_service import metrics

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "faiss-index")
LOCAL_INDEX_DIR = os.getenv(
    "LOCAL_INDEX_DIR",
    "verification_service/faiss_index")
LOCAL_INDEX_PATH = os.path.join(LOCAL_INDEX_DIR, "driver_vectors.index")
LOCAL_METADATA_PATH = os.path.join(LOCAL_INDEX_DIR, "metadata.json")
THUMBS_LOCAL_DIR = os.path.join(LOCAL_INDEX_DIR, "thumbs")


def minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def load_checkpoint(path="sync_checkpoint.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return "0"


def save_checkpoint(ts, path="sync_checkpoint.txt"):
    with open(path, "w") as f:
        f.write(str(ts))


def download_index_from_minio(bucket=MINIO_BUCKET, prefix=""):
    client = minio_client()
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
    # download index file
    try:
        logger.info(
            "Downloading %s/driver_vectors.index -> %s",
            bucket,
            LOCAL_INDEX_PATH)
        client.download_file(bucket, "driver_vectors.index", LOCAL_INDEX_PATH)
    except Exception as e:
        logger.exception("Failed to download driver_vectors.index: %s", e)
        raise

    # download metadata.json
    try:
        logger.info(
            "Downloading %s/metadata.json -> %s",
            bucket,
            LOCAL_METADATA_PATH)
        client.download_file(bucket, "metadata.json", LOCAL_METADATA_PATH)
    except Exception as e:
        logger.warning("metadata.json not found in bucket: %s", e)

    # download thumbs/ prefix
    os.makedirs(THUMBS_LOCAL_DIR, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="thumbs/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # skip folder keys
            if key.endswith("/"):
                continue
            local_path = os.path.join(LOCAL_INDEX_DIR, key)
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)
            logger.info("Downloading %s -> %s", key, local_path)
            client.download_file(bucket, key, local_path)


# Example sync loop: call download_index_from_minio when a new version is
# available
def sync_loop(poll_interval_seconds=60 * 10):
    while True:
        try:
            download_index_from_minio()
            # update your checkpoint / notify service (e.g., touch a reload
            # file)
            logger.info("Index sync complete")
        except Exception as e:
            logger.exception("Index sync failed: %s", e)
        time.sleep(poll_interval_seconds)


def sync_vectors(api_url="https://your-api.com/vectors",
                 thumb_url="https://your-api.com/thumbs"):
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
                # FIX: Break long string across multiple lines
                thumb_path = (
                    f"verification_service/faiss_index/thumbs/"
                    f"{thumb['driver_id']}.jpg"
                )
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
