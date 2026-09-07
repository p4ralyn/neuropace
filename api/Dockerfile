FROM python:3.11-slim

WORKDIR /srv

# Install the package first so its layer caches independently of the service.
COPY pyproject.toml README.md ./
COPY neuropace ./neuropace
RUN pip install --no-cache-dir ".[api,train]"

COPY api ./api

# Cloud Run and friends inject PORT; default to 8080 locally.
ENV PORT=8080
ENV NEUROPACE_MODEL=/srv/artifacts/model.keras
EXPOSE 8080

# No model is baked into the image: mount or copy artifacts/ at deploy time.
# Without one the service still starts and /health reports model_loaded false.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
