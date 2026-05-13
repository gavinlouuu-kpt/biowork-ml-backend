# Biowork Product Workflow

Biowork is managed as separate repos with one shared integration discipline:

- `biowork`: product app, annotation UI, project data flow, and user workflows.
- `biowork-ml-backend`: ML backend services, active training, MLflow tracking, and model APIs.
- `rustfs_yolo_sam2_inference`: Kedro full-dataset inference using RustFS data and a selected MLflow model.

## ML Backend Responsibilities

- Serve prediction APIs for SAM2, FastSAM, and YOLO.
- Receive Label Studio training events and run long training work asynchronously where possible.
- Log training lifecycle to MLflow, including progress, metrics, failures, artifacts, best weights, and model selection metadata.
- Keep model API behavior explicit enough for Biowork and batch inference to depend on it intentionally.

## Coordination

- Use the GitHub Project `Biowork Product Integration` for planning.
- Use development branches for each issue or feature.
- Link PRs across repos when frontend, backend, and inference behavior depend on each other.
- Record MLflow run IDs, failure causes, fixes, commit hashes, and verification results on the related GitHub issue/project.

## Service Boundaries

- MLflow is shared product infrastructure, not a dataset-specific service.
- RustFS can hold datasets and MLflow artifacts, but paths should be scoped by purpose, project, and dataset.
- Keep MLflow tailnet-only unless the user explicitly changes that.
- Keep dev and prod Docker networks separate. Verify connectivity from the caller container or network.
