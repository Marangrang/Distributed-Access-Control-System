"""
Train face recognition model using data from MinIO storage
Pulls data from face-images/train and face-images/test buckets
Saves trained model back to MinIO
"""
import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from minio import Minio
from minio.error import S3Error
import json
from tqdm import tqdm
from datetime import datetime
import faiss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinIODataLoader:
    """Load training and test data from MinIO"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False
    ):
        """
        Initialize MinIO client

        Args:
            endpoint: MinIO endpoint (e.g., 'localhost:9000')
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Use HTTPS if True
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        logger.info(f"Connected to MinIO at {endpoint}")

    def download_dataset(
        self,
        bucket_name: str,
        prefix: str,
        local_path: Path
    ) -> Dict[str, List[Path]]:
        """
        Download dataset from MinIO to local directory

        Args:
            bucket_name: Name of MinIO bucket (e.g., 'face-images')
            prefix: Prefix path in bucket (e.g., 'train/' or 'test/')
            local_path: Local directory to download to

        Returns:
            Dictionary mapping person names to list of image paths
        """
        logger.info(f"Downloading dataset from {bucket_name}/{prefix}")

        # Create local directory
        local_path.mkdir(parents=True, exist_ok=True)

        dataset = {}
        downloaded_count = 0

        try:
            # List all objects in bucket with prefix
            objects = self.client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=True
            )

            # Group by person name
            for obj in objects:
                if obj.size == 0:  # Skip directories
                    continue

                # Extract person name from path
                # Format: train/PersonName/image.jpg or test/PersonName/image.jpg
                parts = obj.object_name.split('/')
                if len(parts) < 3:
                    continue

                person_name = parts[1]
                image_name = parts[2]

                # Skip non-image files
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    logger.debug(f"Skipping non-image file: {image_name}")
                    continue

                # Create person directory
                person_dir = local_path / person_name
                person_dir.mkdir(exist_ok=True)

                # Download file
                local_file = person_dir / image_name
                try:
                    self.client.fget_object(
                        bucket_name,
                        obj.object_name,
                        str(local_file)
                    )

                    # Add to dataset
                    if person_name not in dataset:
                        dataset[person_name] = []
                    dataset[person_name].append(local_file)
                    downloaded_count += 1

                    if downloaded_count % 100 == 0:
                        logger.info(f"Downloaded {downloaded_count} images...")

                except S3Error as e:
                    logger.error(f"Failed to download {obj.object_name}: {e}")
                    continue

            logger.info(f"✓ Downloaded {downloaded_count} images for {len(dataset)} people")
            return dataset

        except S3Error as e:
            logger.error(f"Error accessing bucket {bucket_name}: {e}")
            raise

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path,
        content_type: str = 'application/octet-stream'
    ):
        """
        Upload file to MinIO

        Args:
            bucket_name: Target bucket name
            object_name: Object name in bucket
            file_path: Local file path to upload
            content_type: MIME type of file
        """
        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                str(file_path),
                content_type=content_type
            )
            logger.info(f"✓ Uploaded {object_name} to {bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to upload {object_name}: {e}")
            raise

    def create_bucket_if_not_exists(self, bucket_name: str):
        """Create bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"✓ Created bucket: {bucket_name}")
            else:
                logger.info(f"✓ Bucket exists: {bucket_name}")
        except S3Error as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            raise


class FaceRecognitionTrainer:
    """Train face recognition model using FaceNet"""

    def __init__(self, device: str = None):
        """
        Initialize trainer

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False
        )

        # Initialize FaceNet model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        logger.info("✓ Loaded pretrained FaceNet model (VGGFace2)")

    def extract_face(self, image_path: Path) -> torch.Tensor:
        """
        Extract face from image and return embedding

        Args:
            image_path: Path to image file

        Returns:
            Face embedding tensor or None if no face detected
        """
        try:
            img = Image.open(image_path).convert('RGB')

            # Detect face
            face = self.mtcnn(img)

            if face is None:
                logger.warning(f"No face detected in {image_path.name}")
                return None

            # Generate embedding
            with torch.no_grad():
                face = face.unsqueeze(0).to(self.device)
                embedding = self.model(face)

            return embedding.cpu()

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def build_embeddings(
        self,
        dataset: Dict[str, List[Path]],
        dataset_type: str = "train"
    ) -> Tuple[np.ndarray, List[int], Dict[str, int]]:
        """
        Build embeddings for all images in dataset

        Args:
            dataset: Dictionary mapping person names to image paths
            dataset_type: 'train' or 'test' for logging

        Returns:
            Tuple of (embeddings array, labels list, label_map dict)
        """
        logger.info(f"Building {dataset_type} embeddings...")

        embeddings_list = []
        labels_list = []
        label_map = {}
        current_label = 0

        total_images = sum(len(images) for images in dataset.values())
        processed = 0
        failed = 0

        with tqdm(total=total_images, desc=f"Processing {dataset_type} images") as pbar:
            for person_name, image_paths in dataset.items():
                # Assign label to person
                if person_name not in label_map:
                    label_map[person_name] = current_label
                    current_label += 1

                label = label_map[person_name]

                for image_path in image_paths:
                    embedding = self.extract_face(image_path)

                    if embedding is not None:
                        embeddings_list.append(embedding.numpy().flatten())
                        labels_list.append(label)
                        processed += 1
                    else:
                        failed += 1

                    pbar.update(1)

        embeddings = np.array(embeddings_list)

        logger.info(f"✓ Built {len(embeddings)} embeddings for {len(label_map)} people")
        logger.info(f"  Success: {processed}, Failed: {failed}")

        return embeddings, labels_list, label_map

    def calculate_accuracy(
        self,
        train_embeddings: np.ndarray,
        train_labels: List[int],
        test_embeddings: np.ndarray,
        test_labels: List[int],
        threshold: float = 0.6
    ) -> Dict[str, float]:
        """
        Calculate verification accuracy and metrics

        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            test_embeddings: Test embeddings
            test_labels: Test labels
            threshold: Distance threshold for verification

        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Calculating accuracy...")

        correct = 0
        total = len(test_embeddings)

        # Track per-person accuracy
        person_correct = {}
        person_total = {}

        for i, test_emb in enumerate(test_embeddings):
            test_label = test_labels[i]

            # Find closest match in training set
            distances = np.linalg.norm(train_embeddings - test_emb, axis=1)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]

            predicted_label = train_labels[min_idx]

            # Update per-person stats
            if test_label not in person_total:
                person_total[test_label] = 0
                person_correct[test_label] = 0
            person_total[test_label] += 1

            # Check if prediction is correct and within threshold
            if min_distance < threshold and predicted_label == test_label:
                correct += 1
                person_correct[test_label] += 1

        accuracy = correct / total if total > 0 else 0

        logger.info(f"✓ Overall Accuracy: {accuracy:.2%} ({correct}/{total})")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'threshold': threshold
        }


def main():
    """Main training pipeline"""

    # Configuration from environment variables
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    SOURCE_BUCKET = os.getenv('SOURCE_BUCKET', 'face-images')
    MODEL_BUCKET = os.getenv('MODEL_BUCKET', 'trained-models')
    TRAIN_PREFIX = 'train/'
    TEST_PREFIX = 'test/'
    THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))

    logger.info("=" * 80)
    logger.info("Face Recognition Training Pipeline (MinIO)")
    logger.info("=" * 80)
    logger.info(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    logger.info(f"Source Bucket: {SOURCE_BUCKET}")
    logger.info(f"Model Bucket: {MODEL_BUCKET}")
    logger.info(f"Similarity Threshold: {THRESHOLD}")
    logger.info("=" * 80)

    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp(prefix='face_training_'))
    train_dir = temp_dir / 'train'
    test_dir = temp_dir / 'test'
    output_dir = temp_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Download data from MinIO
        logger.info("\n[STEP 1/7] Downloading data from MinIO...")
        minio_loader = MinIODataLoader(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=os.getenv('MINIO_SECURE', 'false').lower() in ('1','true','yes')
        )

        train_dataset = minio_loader.download_dataset(
            SOURCE_BUCKET,
            TRAIN_PREFIX,
            train_dir
        )

        test_dataset = minio_loader.download_dataset(
            SOURCE_BUCKET,
            TEST_PREFIX,
            test_dir
        )

        logger.info(f"Train: {len(train_dataset)} people")
        logger.info(f"Test: {len(test_dataset)} people")

        if len(train_dataset) == 0:
            logger.error("No training data found in MinIO!")
            sys.exit(1)

        # Step 2: Initialize trainer
        logger.info("\n[STEP 2/7] Initializing trainer...")
        trainer = FaceRecognitionTrainer()

        # Step 3: Build training embeddings
        logger.info("\n[STEP 3/7] Building training embeddings...")
        train_embeddings, train_labels, label_map = trainer.build_embeddings(
            train_dataset, "train"
        )

        if len(train_embeddings) == 0:
            logger.error("No valid training embeddings extracted!")
            sys.exit(1)

        # Step 4: Build test embeddings
        logger.info("\n[STEP 4/7] Building test embeddings...")
        test_embeddings, test_labels, _ = trainer.build_embeddings(
            test_dataset, "test"
        )

        # Step 5: Calculate accuracy
        logger.info("\n[STEP 5/7] Evaluating model...")
        metrics = trainer.calculate_accuracy(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
            threshold=THRESHOLD
        )

        # Step 6: Save files locally (temporary)
        logger.info("\n[STEP 6/7] Saving model files...")
        env_model_version = os.getenv('MODEL_VERSION')
        timestamp = env_model_version if env_model_version else datetime.now().strftime('%Y%m%d_%H%M%S')

        # Normalize embeddings for cosine similarity
        train_embeddings_norm = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-12)

        # Save embeddings
        embeddings_file = output_dir / 'driver_vectors.npy'
        np.save(embeddings_file, train_embeddings_norm)
        logger.info(f"✓ Saved embeddings: {embeddings_file}")

        # Build and save FAISS index
        logger.info("Building FAISS index...")
        dimension = train_embeddings_norm.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(train_embeddings_norm.astype('float32'))

        index_file = output_dir / 'driver_vectors.index'
        faiss.write_index(index, str(index_file))
        logger.info(f"✓ Saved FAISS index: {index_file}")

        # Save metadata
        metadata = []
        inverse_label_map = {v: k for k, v in label_map.items()}
        for i, label in enumerate(train_labels):
            person_name = inverse_label_map.get(label, f"person_{label}")
            metadata.append({
                "id": i,
                "driver_id": person_name,
                "name": person_name,
                "label": int(label)
            })

        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_file}")

        # Save label map
        label_map_file = output_dir / 'label_map.json'
        with open(label_map_file, 'w') as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"✓ Saved label map: {label_map_file}")

        # Save training info
        training_info = {
            'timestamp': timestamp,
            'minio_endpoint': MINIO_ENDPOINT,
            'source_bucket': SOURCE_BUCKET,
            'model_bucket': MODEL_BUCKET,
            'num_people': len(label_map),
            'num_train_samples': len(train_embeddings),
            'num_test_samples': len(test_embeddings),
            'embedding_dim': train_embeddings.shape[1],
            'accuracy': float(metrics['accuracy']),
            'threshold': THRESHOLD,
            'model': 'InceptionResnetV1 (VGGFace2)',
            'device': str(trainer.device)
        }

        training_info_file = output_dir / 'training_info.json'
        with open(training_info_file, 'w') as f:
            json.dump(training_info, f, indent=2)
        logger.info(f"✓ Saved training info: {training_info_file}")

        # Step 7: Upload to MinIO
        logger.info("\n[STEP 7/7] Uploading model to MinIO...")

        # Create model bucket if it doesn't exist
        minio_loader.create_bucket_if_not_exists(MODEL_BUCKET)

        # Upload all files
        files_to_upload = [
            ('driver_vectors.npy', 'application/octet-stream'),
            ('driver_vectors.index', 'application/octet-stream'),
            ('metadata.json', 'application/json'),
            ('label_map.json', 'application/json'),
            ('training_info.json', 'application/json')
        ]

        for filename, content_type in files_to_upload:
            local_file = output_dir / filename
            object_name = f"models/{timestamp}/{filename}"
            minio_loader.upload_file(
                MODEL_BUCKET,
                object_name,
                local_file,
                content_type
            )

        # Also upload latest version (without timestamp for easy access)
        for filename, content_type in files_to_upload:
            local_file = output_dir / filename
            object_name = f"models/latest/{filename}"
            minio_loader.upload_file(
                MODEL_BUCKET,
                object_name,
                local_file,
                content_type
            )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"Training Samples: {len(train_embeddings)}")
        logger.info(f"Test Samples: {len(test_embeddings)}")
        logger.info(f"Number of People: {len(label_map)}")
        logger.info(f"Embedding Dimension: {train_embeddings.shape[1]}")
        logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"Threshold: {THRESHOLD}")
        logger.info(f"MinIO Location: {MODEL_BUCKET}/models/{timestamp}/")
        logger.info(f"Latest Version: {MODEL_BUCKET}/models/latest/")
        logger.info("=" * 80)
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"✓ Model saved to MinIO bucket: {MODEL_BUCKET}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Cleanup temporary directories
        logger.info("\nCleaning up temporary files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info("✓ Cleanup complete")


if __name__ == '__main__':
    main()
