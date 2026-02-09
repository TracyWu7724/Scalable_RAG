#!/bin/bash
set -e

# Wait for PostgreSQL to be ready (when running inside docker-compose)
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for PostgreSQL..."
    # Extract host and port from DATABASE_URL
    # Format: postgresql+asyncpg://user:pass@host:port/db
    DB_HOST=$(echo "$DATABASE_URL" | sed -n 's|.*@\(.*\):\([0-9]*\)/.*|\1|p')
    DB_PORT=$(echo "$DATABASE_URL" | sed -n 's|.*@\(.*\):\([0-9]*\)/.*|\2|p')

    for i in $(seq 1 30); do
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" > /dev/null 2>&1; then
            echo "PostgreSQL is ready!"
            break
        fi
        echo "Waiting for PostgreSQL ($i/30)..."
        sleep 2
    done
fi

echo "Starting FastAPI server on port 8000..."
uvicorn server:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit UI on port 8501..."
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
