"""
Model sync module - Download updated models from MinIO
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from minio import Minio
from minio.error import S3Error
from metrics import set_sync_lag

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() in ('1','true','yes')
MODEL_BUCKET = os.getenv('MODEL_BUCKET', 'trained-models')
LOCAL_MODEL_DIR = Path(os.getenv('LOCAL_MODEL_DIR', '/app/verification_service/faiss_index'))


class ModelSyncManager:
    """Manage model synchronization from MinIO"""

    def __init__(self,
                 endpoint: str = MINIO_ENDPOINT,
                 access_key: str = MINIO_ACCESS_KEY,
                 secret_key: str = MINIO_SECRET_KEY,
                 bucket: str = MODEL_BUCKET,
                 local_model_dir: str | Path = LOCAL_MODEL_DIR,
                 secure: bool = MINIO_SECURE):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        self.local_model_dir = Path(local_model_dir)
        self.last_sync = None
        self.local_model_dir.mkdir(parents=True, exist_ok=True)

    def get_remote_model_timestamp(self) -> datetime:
        """Get last modified time of remote model"""
        try:
            stat = self.client.stat_object(
                self.bucket,
                'models/latest/training_info.json'
            )
            return stat.last_modified
        except S3Error as e:
            logger.error(f"Failed to get remote timestamp: {e}")
            return None

    def get_local_model_timestamp(self) -> datetime | None:
        """Get last modified time of local training_info.json if present"""
        info_path = self.local_model_dir / 'training_info.json'
        try:
            stat = info_path.stat()
            return datetime.fromtimestamp(stat.st_mtime)
        except FileNotFoundError:
            return None

    def download_model_files(self) -> bool:
        """Download all model files from MinIO"""
        files_to_download = [
            'driver_vectors.npy',
            'driver_vectors.index',
            'metadata.json',
            'training_info.json'
        ]

        success_count = 0
        for filename in files_to_download:
            object_name = f"models/latest/{filename}"
            local_file = self.local_model_dir / filename
            tmp_file = local_file.with_suffix(local_file.suffix + '.tmp')

            try:
                # Download to a temporary file, then atomically replace
                self.client.fget_object(self.bucket, object_name, str(tmp_file))
                os.replace(tmp_file, local_file)
                logger.info(f"✓ Downloaded {filename}")
                success_count += 1
            except S3Error as e:
                logger.error(f"Failed to download {filename}: {e}")
            except Exception as e:
                logger.error(f"Failed to finalize {filename}: {e}")

        if success_count == len(files_to_download):
            self.last_sync = datetime.now()
            set_sync_lag(0)
            return True
        return False

    def check_for_updates(self) -> bool:
        """Check if remote model is newer than local"""
        # Compare remote timestamp to local file timestamp
        remote_time = self.get_remote_model_timestamp()
        if not remote_time:
            return False
        local_time = self.get_local_model_timestamp() or self.last_sync
        if not local_time:
            return True
        return remote_time > local_time

    def sync_if_needed(self) -> bool:
        """Sync model if updates available"""
        if self.check_for_updates():
            logger.info("New model version detected, syncing...")
            return self.download_model_files()
        else:
            logger.info("Model is up to date")
            # Update sync lag
            if self.last_sync:
                lag = (datetime.now() - self.last_sync).total_seconds()
                set_sync_lag(lag)
            return True


# For backward compatibility
def download_index_from_minio():
    """Legacy function - downloads model from MinIO"""
    sync_manager = ModelSyncManager()
    return sync_manager.download_model_files()
