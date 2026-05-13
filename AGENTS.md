# AGENTS.md

## Biowork Product Workflow

This repo is the ML service layer for the Biowork product system:

- `biowork`: product app and annotation workflows.
- `biowork-ml-backend`: SAM2, FastSAM, YOLO, active training, MLflow logging, and model APIs.
- `rustfs_yolo_sam2_inference`: Kedro batch inference over Biowork datasets with a selected MLflow model.

Keep the repos separate and coordinate with explicit contracts, linked PRs/issues, and the GitHub Project `Biowork Product Integration`.

## Default Habit

- Check branch and dirty worktree before editing.
- Create a purpose-named development branch from `origin/master` for new work.
- Do not mix unrelated issue work into an existing branch.
- Link paired `biowork` or inference PRs when model behavior changes product workflows.
- Update the related GitHub issue/project with investigation notes, MLflow run IDs, commit hashes, test results, and remaining risks.

## ML Backend Rules

- Keep dev and prod Docker networks separate. When changing Docker networking or Label Studio ML backend connectivity, use the local `biowork-ml-topology` skill.
- Treat Label Studio prediction/training payloads and response formats as contracts.
- Training should be asynchronous when it can take user-visible time.
- MLflow runs should log params, metrics, dataset/class metadata, progress, failure reasons, best weights, and model promotion state.
- MLflow is shared product infrastructure and should remain tailnet-only unless explicitly changed.

See `docs/biowork_product_workflow.md` for the cross-repo workflow notes.
