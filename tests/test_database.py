"""Test database operations."""
import os
import pytest
import psycopg2

# Treat this as an integration test (CI excludes with -m "not integration")
pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def db_connection():
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    name = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=name, user=user, password=password
        )
    except psycopg2.OperationalError as e:
        pytest.skip(f"PostgreSQL not available at {host}:{port}: {e}")

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
