"""Helper functions for managing Airflow connections."""
import logging
import os
from typing import Dict, Optional
from airflow.hooks.base import BaseHook
import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)


def get_db_connection(conn_id: str = 'face_verification_db') -> Dict[str, str]:
    """
    Retrieve database connection from Airflow connections.
    Falls back to environment variables if connection not found.

    Args:
        conn_id: Airflow connection ID

    Returns:
        Dict containing database connection parameters
    """
    try:
        conn = BaseHook.get_connection(conn_id)
        return {
            'host': conn.host,
            'port': conn.port or 5432,
            'database': conn.schema,
            'user': conn.login,
            'password': conn.password,
        }
    except Exception as e:
        logger.warning(f"Airflow connection '{conn_id}' not found, using environment variables: {str(e)}")
        # Fallback to environment variables
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'face_verification_db'),
            'user': os.getenv('DB_USER', 'NWUUSER'),
            'password': os.getenv('DB_PASSWORD', 'mypassword'),
        }


def get_minio_connection(conn_id: str = 'minio_default') -> Dict[str, str]:
    """
    Retrieve MinIO connection from Airflow connections.
    Falls back to environment variables if connection not found.

    Args:
        conn_id: Airflow connection ID

    Returns:
        Dict containing MinIO connection parameters
    """
    try:
        conn = BaseHook.get_connection(conn_id)

        # Handle different endpoint formats
        endpoint = conn.host
        port = conn.port or 9000

        # Check if schema (http/https) is in extra
        extra = conn.extra_dejson if hasattr(conn, 'extra_dejson') else {}
        secure = extra.get('secure', False)

        return {
            'endpoint': f"{endpoint}:{port}",
            'access_key': conn.login,
            'secret_key': conn.password,
            'secure': secure,
        }
    except Exception as e:
        logger.warning(f"Airflow connection '{conn_id}' not found, using environment variables: {str(e)}")
        # Fallback to environment variables
        endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        secure = os.getenv('MINIO_SECURE', 'false').lower() in ('1','true','yes')
        return {
            'endpoint': endpoint,
            'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            'secret_key': os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            'secure': secure,
        }


def get_minio_client(conn_id: str = 'minio_default'):
    """
    Create boto3 S3 client for MinIO using Airflow connection.

    Args:
        conn_id: Airflow connection ID

    Returns:
        boto3.client configured for MinIO
    """
    minio_conn = get_minio_connection(conn_id)

    protocol = 'https' if minio_conn['secure'] else 'http'
    endpoint_url = f"{protocol}://{minio_conn['endpoint']}"

    logger.info(f"Connecting to MinIO at {endpoint_url}")

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=minio_conn['access_key'],
        aws_secret_access_key=minio_conn['secret_key'],
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )


def get_minio_resource(conn_id: str = 'minio_default'):
    """
    Create boto3 S3 resource for MinIO using Airflow connection.

    Args:
        conn_id: Airflow connection ID

    Returns:
        boto3.resource configured for MinIO
    """
    minio_conn = get_minio_connection(conn_id)

    protocol = 'https' if minio_conn['secure'] else 'http'
    endpoint_url = f"{protocol}://{minio_conn['endpoint']}"

    return boto3.resource(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=minio_conn['access_key'],
        aws_secret_access_key=minio_conn['secret_key'],
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
