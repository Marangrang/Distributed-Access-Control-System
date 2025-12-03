"""
DAG for ingesting face images from cloud storage to MinIO.

This DAG handles hourly data ingestion from various cloud sources.

Owner: Data Engineering Team
Tags: data, ingestion, cloud-sync
"""
from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}

with DAG(
    dag_id='face_data_ingestion_pipeline',
    default_args=default_args,
    description='Ingest face images from cloud to MinIO',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    tags=['data', 'ingestion', 'cloud-sync'],
    doc_md=__doc__,
) as dag:

    @task()
    def ingest_from_cloud_storage(**context) -> int:
        """
        Ingest data from cloud storage (S3, Azure Blob, etc.).

        Returns:
            int: Number of files ingested
        """
        try:
            import sys
            sys.path.append('/opt/airflow/cloud_ingestion')
            from ingest import ingest_face_data

            logger.info("Starting cloud data ingestion...")

            # Get source from Airflow Variable (not hardcoded)
            source = "{{ var.value.get('ingestion_source', 's3') }}"

            files_ingested = ingest_face_data(source=source)

            logger.info(f"✓ Ingested {files_ingested} files")

            return files_ingested

        except Exception as e:
            logger.error(f"Cloud ingestion failed: {str(e)}")
            raise

    # Create ingestion log table if not exists
    create_log_table = PostgresOperator(
        task_id='create_ingestion_log_table',
        postgres_conn_id='face_verification_db',
        sql="""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id SERIAL PRIMARY KEY,
                ingestion_date TIMESTAMP NOT NULL,
                status VARCHAR(50) NOT NULL,
                record_count INTEGER NOT NULL,
                source VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """,
    )

    @task()
    def log_ingestion_result(files_count: int, **context) -> None:
        """Log ingestion results to database."""
        try:
            postgres_hook = PostgresHook(postgres_conn_id='face_verification_db')

            insert_query = """
                INSERT INTO ingestion_log (ingestion_date, status, record_count, source)
                VALUES (NOW(), %s, %s, %s);
            """

            postgres_hook.run(
                insert_query,
                parameters=('completed', files_count, 'cloud_storage')
            )

            logger.info(f"✓ Logged {files_count} ingested files to database")

        except Exception as e:
            logger.error(f"Failed to log ingestion: {str(e)}")
            raise

    # Define task dependencies
    files_ingested = ingest_from_cloud_storage()
    create_log_table >> files_ingested >> log_ingestion_result(files_ingested)

    create_table = SQLExecuteQueryOperator(
        task_id='create_face_data_table',
        conn_id='postgres_default',
        sql=""",
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255),
            embedding_vector FLOAT8[],
            image_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        dag=dag,
    )
