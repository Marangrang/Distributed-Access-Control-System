"""Test database operations."""
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os


@pytest.fixture
def db_connection():
    """Create database connection for testing."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER", "NWUUSER"),
        password=os.getenv("DB_PASSWORD", "mypassword"),
        database=os.getenv("DB_NAME", "face_verification_db")
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    yield conn
    conn.close()


def test_database_connection(db_connection):
    """Test database connectivity."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
    cursor.close()


def test_pgvector_extension(db_connection):
    """Test pgvector extension is available."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM pg_extension WHERE extname='vector'")
    result = cursor.fetchone()
    assert result is not None
    cursor.close()


def test_embeddings_table_exists(db_connection):
    """Test that embeddings table exists."""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'embeddings'
        )
    """)
    result = cursor.fetchone()
    assert result[0] is True
    cursor.close()
