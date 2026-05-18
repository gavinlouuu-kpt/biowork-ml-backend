# Dagster Integration

The `biowork-ml-backend` prediction server is called by the Dagster
`yolo_trained_model` asset during the active-learning loop.  No code
changes are required on the ml-backend side; this document records the
contract so both sides stay in sync.

## Endpoint used by Dagster

### `POST /train`

**Purpose:** trigger a full YOLO retraining run for one Biowork project.

**Request body (JSON)**

```json
{
  "project": "<project_id>",
  "label_config": "<Label Studio XML label config>",
  "annotations": [
    {
      "id": 1,
      "data": {"image": "..."},
      "annotations": [
        {
          "result": [ ... ],
          "was_cancelled": false
        }
      ]
    }
  ]
}
```

`annotations` is a list of Label Studio task objects (the same format
returned by `GET /api/projects/{id}/export?exportType=JSON`).
The ml-backend ignores unannotated tasks automatically.

**Response body (JSON, HTTP 201)**

```json
{
  "status": "ok",
  "result": {
    "status": "trained",
    "model_version": "yolo-auto-segment-20250518_120000",
    "mlflow_run_id": "abc123",
    "mlflow_tracking_uri": "http://mlflow.tailnet:5000",
    "mlflow_experiment": "biowork-yolo-training",
    "num_samples": 42,
    "num_train": 33,
    "num_val": 9
  }
}
```

Possible `result.status` values:

| Value | Meaning |
|-------|---------|
| `trained` | Training completed; `model_version` and `mlflow_run_id` are set. |
| `skipped` | No annotated tasks found or label config has no supported controls. |

If `result.mlflow_run_id` is set, the Dagster `inference_run` asset
constructs `runs:/<mlflow_run_id>/model` as the model URI for the
subsequent inference step.

## How the annotation data flows

```
Dagster active_learning_job
  │
  ├── biowork_project      ← GET /api/projects/{id}/
  ├── biowork_annotations  ← GET /api/projects/{id}/export
  ├── yolo_trained_model   → POST /train  (ml-backend)
  └── inference_run        → MLflow + RustFS (direct, no ml-backend call)
```

## Environment variables required by the ml-backend

| Variable | Purpose |
|----------|---------|
| `LABEL_STUDIO_HOST` | Internal Biowork URL for downloading task images |
| `LABEL_STUDIO_API_KEY` | Biowork API token |
| `MLFLOW_TRACKING_URI` | Where to register trained models |
| `YOLO_TRAIN_EPOCHS` | (optional) training epoch count, default 25 |
| `YOLO_TRAIN_MLFLOW_ENABLED` | (optional) `true`/`false`, default `true` |
