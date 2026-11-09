# Results-auditor API + viewer (FastAPI, serves static/viewer.html at "/").
# Build:  docker build -t us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/results-auditor:dev .
# Run locally:  docker run --rm -p 8080:8080 -e PEMAD_DB_DSN="postgresql://..." <image>
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# Cloud Run supplies $PORT (8080 by default) and routes to 0.0.0.0.
# The DB connection comes from PEMAD_DB_DSN at runtime -- never baked into the image.
RUN useradd --create-home appuser
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "uvicorn pemad_esb_inference.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
