FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer independently of code changes.
# requirements-backend.txt pins torch to the CPU-only build (~800 MB installed),
# keeping the total image well under Railway's 4 GB limit.
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy only what the backend needs at runtime.
COPY app/ ./app/
COPY data/processed/ ./data/processed/
COPY data/catalog_thumbnails/ ./data/catalog_thumbnails/

EXPOSE 8000

# PORT is injected by Railway at runtime; fallback to 8000 for local Docker testing.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
