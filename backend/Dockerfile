# Sprint 6.3 — Multi-stage, sin dev deps, sin modelo .joblib en imagen (D10.3)
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias de sistema mínimas para psycopg/asyncpg y LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Etapa final ligera — solo código y deps de producción
FROM python:3.12-slim AS final

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copia deps instaladas del builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia estrictamente app/ y archivos necesarios (no data/models/final/*.joblib, no tests)
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

# Modelo LightGBM incluído para que el backend arranque sin volumen externo
# (disco solo copia de la DB de Render en el plan free, por eso incluímoslo aquí)
COPY data/models/final/lightgbm_production.joblib ./data/models/final/lightgbm_production.joblib

EXPOSE 8000

# FastAPI con lifespan (carga única), escucha en el puerto que Render asigna (PORT)
CMD sh -c "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
