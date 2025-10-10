<!--
---
title: SAM2 with Images
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM2 with Label Studio for Image Annotation
categories:
    - Computer Vision
    - Image Annotation
    - Object Detection
    - Segment Anything Model
image: "/tutorials/sam2-images.png"
---
-->

# FastSAM Dedicated ML Backend

A dedicated FastSAM server for Label Studio, with organization middleware (V3) support and Docker deployment.

## Quickstart (Docker)

```bash
docker-compose up -d
```

Connect in Label Studio → Project → Settings → Machine Learning → Add Model: `http://fastsam-backend:9090`

## Environment (.env)

```bash
# Middleware (optional but recommended)
USE_ORG_MIDDLEWARE=true

# PostgreSQL (docker-compose)
LABEL_STUDIO_DB_TYPE=postgres
LABEL_STUDIO_DB_HOST=label-studio-db
LABEL_STUDIO_DB_PORT=5432
LABEL_STUDIO_DB_NAME=postgres
LABEL_STUDIO_DB_USER=postgres
LABEL_STUDIO_DB_PASSWORD=postgres

# Label Studio URL (internal)
LABEL_STUDIO_HOST=http://label-studio-app:8000

# FastSAM
RESPONSE_TYPE=both
POLYGON_DETAIL_LEVEL=0.002
MAX_RESULTS=10
FASTSAM_CHECKPOINT=models/FastSAM-x.pt
```

## Labeling Config

Use `BrushLabels` for masks, optionally `PolygonLabels` for contours, plus `KeyPointLabels` or `RectangleLabels` for prompts.

## Health

```bash
curl http://localhost:9090/health
```
