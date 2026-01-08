"""Test MinIO storage operations."""
import pytest
from minio import Minio
from minio.error import S3Error
import os
import io


@pytest.fixture
def minio_client(request):
    """Create MinIO client for testing."""
    # Skip if not running integration tests
    if "integration" not in request.config.option.markexpr:
        pytest.skip("MinIO tests require --run-integration flag or -m integration")

    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() in ('1','true','yes')
    )
    return client

@pytest.mark.integration
def test_minio_connection(minio_client):
    """Test MinIO connectivity."""
    try:
        buckets = minio_client.list_buckets()
        assert buckets is not None
    except S3Error as e:
        pytest.fail(f"MinIO connection failed: {e}")


@pytest.mark.integration
def test_bucket_exists(minio_client):
    """Test bucket existence check."""
    bucket_name = "test-bucket"
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        assert minio_client.bucket_exists(bucket_name)
    except S3Error as e:
        pytest.fail(f"Bucket operation failed: {e}")


@pytest.mark.integration
def test_upload_download_file(minio_client):
    """Test file upload and download."""
    bucket_name = "test-bucket"
    object_name = "test-file.txt"
    content = b"test content"

    # Ensure bucket exists
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    # Upload
    minio_client.put_object(
        bucket_name,
        object_name,
        io.BytesIO(content),
        len(content)
    )

    # Download
    response = minio_client.get_object(bucket_name, object_name)
    downloaded = response.read()
    assert downloaded == content

    # Cleanup
    minio_client.remove_object(bucket_name, object_name)
