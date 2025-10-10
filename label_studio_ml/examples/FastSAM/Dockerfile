FROM python:3.10-slim
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    FASTSAM_CHECKPOINT=/app/models/FastSAM-x.pt

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y wget git libopencv-dev python3-opencv ffmpeg; \
    apt-get autoremove -y

# Base requirements (gunicorn and LS-ML)
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --upgrade pip && pip install -r requirements-base.txt

# App requirements (FastSAM and middleware deps)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Download FastSAM weights into the image to avoid runtime downloads
RUN mkdir -p /app/models \
    && wget -O /app/models/FastSAM-x.pt \
       https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt

# Test requirements if needed
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

# Copy app
COPY . .

EXPOSE 9090

CMD gunicorn --bind :${PORT:-9090} --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app