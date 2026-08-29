# Multi-stage production Dockerfile for ModelForge AI
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY sdk/ ./sdk/
RUN pip install -e ./sdk

COPY run.py ./
COPY Makefile ./
COPY package.json ./

EXPOSE 8000 3000

ENTRYPOINT ["python", "run.py"]
CMD ["--mode", "all"]
