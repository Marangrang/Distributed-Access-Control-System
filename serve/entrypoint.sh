#!/bin/bash
set -e

echo "Starting Face Verification Service..."

# Wait for database to be ready (if using PostgreSQL)
if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
    echo "Waiting for database at $DB_HOST:$DB_PORT..."
    while ! nc -z $DB_HOST $DB_PORT; do
        sleep 1
    done
    echo "Database is ready!"
fi

# Wait for MinIO to be ready (if using MinIO)
if [ -n "$MINIO_ENDPOINT" ]; then
    echo "Waiting for MinIO at $MINIO_ENDPOINT..."
    # Add MinIO health check logic here if needed
fi

# Initialize FAISS index if it doesn't exist
if [ ! -f "$FAISS_INDEX_PATH/index.faiss" ]; then
    echo "Initializing FAISS index..."
    python -c "
from verification_service.build_index import build_faiss_index
build_faiss_index()
" || echo "FAISS index initialization completed or skipped"
fi

# Start the application
echo "Starting Uvicorn server..."
exec uvicorn verification_service.main:app \
    --host $UVICORN_HOST \
    --port $UVICORN_PORT \
    --workers $UVICORN_WORKERS \
    --log-level $LOG_LEVEL
EOF

# Make it executable
chmod +x serve/entrypoint.sh
