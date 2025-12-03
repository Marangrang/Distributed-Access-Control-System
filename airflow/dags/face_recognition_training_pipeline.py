"""
DAG for training and deploying face recognition model.

This DAG orchestrates the complete ML pipeline:
1. Check for new training data
2. Train face recognition model
3. Build FAISS index
4. Upload artifacts to MinIO
5. Restart application service
6. Validate deployment

Owner: Data Engineering Team
Tags: ml, face-recognition, training, production
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.utils.db import provide_session
from airflow.models import XCom
from airflow.hooks.base import BaseHook

# Configure logging
logger = logging.getLogger(__name__)

# Default arguments following best practices
default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


@provide_session
def cleanup_xcom(context, session=None):
    """Clean up XCom data after task completion."""
    dag_id = context["ti"]["dag_id"]
    session.query(XCom).filter(XCom.dag_id == dag_id).delete()
    logger.info(f"Cleaned up XComs for DAG: {dag_id}")


def get_db_connection() -> Dict[str, str]:
    """
    Retrieve database connection from Airflow connections.

    Returns:
        Dict containing database connection parameters
    """
    try:
        conn = BaseHook.get_connection('face_verification_db')
        return {
            'host': conn.host,
            'port': conn.port,
            'database': conn.schema,
            'user': conn.login,
            'password': conn.password,
        }
    except Exception as e:
        logger.error(f"Failed to get database connection: {str(e)}")
        raise


def get_minio_connection() -> Dict[str, str]:
    """
    Retrieve MinIO connection from Airflow connections.

    Returns:
        Dict containing MinIO connection parameters
    """
    try:
        conn = BaseHook.get_connection('minio_default')
        return {
            'endpoint': conn.host,
            'access_key': conn.login,
            'secret_key': conn.password,
        }
    except Exception as e:
        logger.error(f"Failed to get MinIO connection: {str(e)}")
        raise


with DAG(
    dag_id='face_recognition_training_pipeline',
    default_args=default_args,
    description='Train, build index, upload to MinIO, and deploy face recognition model',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'face-recognition', 'training', 'production'],
    doc_md=__doc__,
) as dag:

    @task()
    def check_new_training_data(**context) -> bool:
        """
        Check if new training data is available in MinIO.

        Returns:
            bool: True if new data exists, False otherwise
        """
        try:
            from minio import Minio

            # Get connection from Airflow (not hardcoded)
            minio_conn = get_minio_connection()

            client = Minio(
                minio_conn['endpoint'],
                access_key=minio_conn['access_key'],
                secret_key=minio_conn['secret_key'],
                secure=False
            )

            # Check for new images in training bucket
            bucket_name = "{{ var.value.get('training_bucket', 'training-data') }}"
            objects = list(client.list_objects(bucket_name, recursive=True))

            if len(objects) > 0:
                logger.info(f"✓ Found {len(objects)} training images")
                # Push count to XCom for downstream tasks
                context['ti'].xcom_push(key='training_data_count', value=len(objects))
                return True
            else:
                logger.info("No new training data found")
                return False

        except Exception as e:
            logger.error(f"Error checking training data: {str(e)}")
            raise

    @task()
    def train_face_recognition_model(**context) -> Dict[str, Any]:
        """
        Train the face recognition model using new data.

        Returns:
            Dict containing training metrics
        """
        try:
            import sys
            sys.path.append('/opt/airflow/train')
            from train import train_face_recognition_model

            # Get training data count from previous task
            data_count = context['ti'].xcom_pull(
                task_ids='check_new_training_data',
                key='training_data_count'
            )

            logger.info(f"Starting model training with {data_count} images...")

            # Train model (function should return metrics)
            metrics = train_face_recognition_model()

            logger.info(f"✓ Model training completed: {metrics}")

            return {
                'status': 'success',
                'images_processed': data_count,
                'accuracy': metrics.get('accuracy', 'N/A'),
                'training_time': metrics.get('training_time', 'N/A'),
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    @task()
    def build_faiss_index(**context) -> str:
        """
        Build FAISS index from trained embeddings.

        Returns:
            str: Path to the built index
        """
        try:
            import sys
            sys.path.append('/opt/airflow/verification_service')
            from build_index import build_faiss_index as build_index

            logger.info("Building FAISS index...")

            index_path = build_index()

            logger.info(f"✓ FAISS index built: {index_path}")

            return index_path

        except Exception as e:
            logger.error(f"FAISS index build failed: {str(e)}")
            raise

    @task()
    def upload_artifacts_to_minio(index_path: str, **context) -> bool:
        """
        Upload trained model and FAISS index to MinIO.

        Args:
            index_path: Path to the FAISS index file

        Returns:
            bool: True if upload successful
        """
        try:
            import sys
            sys.path.append('/opt/airflow/train')
            from upload_index_to_minio import upload_index_to_minio

            logger.info(f"Uploading artifacts from {index_path} to MinIO...")

            # Upload using connection (not hardcoded credentials)
            upload_index_to_minio(index_path)

            logger.info("✓ Artifacts uploaded successfully")

            return True

        except Exception as e:
            logger.error(f"Artifact upload failed: {str(e)}")
            raise

    @task()
    def log_training_metrics(**context) -> None:
        """Log training metrics to database for monitoring."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            # Get metrics from previous task
            training_metrics = context['ti'].xcom_pull(
                task_ids='train_face_recognition_model'
            )

            # Get DB connection
            db_conn = get_db_connection()

            conn = psycopg2.connect(**db_conn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Insert metrics
            insert_query = """
                INSERT INTO training_metrics
                (training_date, images_processed, accuracy, training_time, status)
                VALUES (NOW(), %s, %s, %s, %s)
            """

            cursor.execute(
                insert_query,
                (
                    training_metrics['images_processed'],
                    training_metrics['accuracy'],
                    training_metrics['training_time'],
                    training_metrics['status']
                )
            )

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✓ Training metrics logged to database")

        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            # Don't fail the pipeline if logging fails
            pass

    # Task: Restart application service
    restart_service = BashOperator(
        task_id='restart_app_service',
        bash_command='docker compose restart app',
        on_success_callback=cleanup_xcom,
    )

    # Task: Wait for service health check
    wait_for_health = HttpSensor(
        task_id='wait_for_service_health',
        http_conn_id='app_service',
        endpoint='/health',
        timeout=300,
        poke_interval=10,
        mode='poke',
    )

    @task()
    def validate_deployment(**context) -> Dict[str, str]:
        """
        Validate the deployed service is working correctly.

        Returns:
            Dict with validation results
        """
        try:
            import requests

            base_url = "{{ var.value.get('app_base_url', 'http://localhost:8000') }}"

            # Test health endpoint
            health_response = requests.get(f'{base_url}/health', timeout=10)
            health_status = health_response.status_code == 200

            # Test metrics endpoint
            metrics_response = requests.get(f'{base_url}/metrics', timeout=10)
            metrics_status = metrics_response.status_code == 200

            if health_status and metrics_status:
                logger.info("✓ All validation checks passed")
                return {
                    'status': 'success',
                    'health_check': 'passed',
                    'metrics_check': 'passed',
                }
            else:
                raise Exception("Validation checks failed")

        except Exception as e:
            logger.error(f"Deployment validation failed: {str(e)}")
            raise

    @task()
    def send_notification(validation_result: Dict[str, str], **context) -> None:
        """Send notification about pipeline completion."""
        try:
            logger.info(f"Pipeline completed successfully: {validation_result}")
            # Add your notification logic here (Slack, email, etc.)

        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")

    # Define task dependencies using TaskFlow API
    data_available = check_new_training_data()
    training_result = train_face_recognition_model()
    index_path = build_faiss_index()
    upload_result = upload_artifacts_to_minio(index_path)

    # Traditional task dependencies
    data_available >> training_result >> index_path >> upload_result
    upload_result >> log_training_metrics()
    upload_result >> restart_service >> wait_for_health

    validation_result = validate_deployment()
    wait_for_health >> validation_result >> send_notification(validation_result)
