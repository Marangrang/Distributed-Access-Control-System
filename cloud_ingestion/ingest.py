# ingest.py: Upload embeddings to cloud (PostgreSQL with pgvector) and
# images to S3
import psycopg2
import os
from PIL import Image
import boto3
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from sklearn.preprocessing import normalize
import io


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


def upload_to_s3(local_path, bucket, key):
    s3 = get_minio_client()
    s3.upload_file(local_path, bucket, key)


def upload_bytes_to_s3(img_bytes, bucket, key):
    s3 = get_minio_client()
    s3.upload_fileobj(io.BytesIO(img_bytes), bucket, key)


def insert_embedding(conn, driver_id, embedding):
    embedding_1d = embedding.squeeze().tolist()  # Ensure 1D
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO face_vectors (driver_id, embedding) VALUES (%s, %s)",
            (driver_id,
             embedding_1d))
        conn.commit()


if __name__ == "__main__":
    # Config
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="mypassword",
        host="localhost",
        port="5432")
    bucket = "face-verification-bucket"
    originals_prefix = "originals/"
    thumbs_prefix = "thumbnails/"
    driver_dir = "data/originals"
    device = 'cpu'
    mtcnn = MTCNN(image_size=160, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    s3 = get_minio_client()
    try:
        s3.create_bucket(Bucket='face-verification-bucket')
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass  # Ignore if the bucket already exists
    for fname in sorted(os.listdir(driver_dir)):
        path = os.path.join(driver_dir, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            print(f"No face detected in {fname}")
            continue
        emb = model(face.unsqueeze(0)).detach().cpu().numpy()
        emb = normalize(emb)
        driver_id = os.path.splitext(fname)[0]
        insert_embedding(conn, driver_id, emb)
        # Upload original
        s3_key_orig = originals_prefix + fname
        upload_to_s3(path, bucket, s3_key_orig)
        # Create and upload thumbnail
        thumb = transform(img)
        thumb_img = transforms.ToPILImage()(thumb)
        thumb_bytes = io.BytesIO()
        thumb_img.save(thumb_bytes, format='JPEG')
        thumb_bytes = thumb_bytes.getvalue()
        s3_key_thumb = thumbs_prefix + driver_id + ".jpg"
        upload_bytes_to_s3(thumb_bytes, bucket, s3_key_thumb)
        print(f"Processed and uploaded {fname}")
