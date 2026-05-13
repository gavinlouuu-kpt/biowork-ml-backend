import importlib
import sys
import types
from types import SimpleNamespace


def load_training(monkeypatch):
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    yaml = types.ModuleType("yaml")
    yaml.safe_dump = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "yaml", yaml)

    control_models = types.ModuleType("control_models")
    control_models_base = types.ModuleType("control_models.base")
    control_models_base.MODEL_ROOT = "/tmp/model-root"
    control_models_base._model_cache = {}
    monkeypatch.setitem(sys.modules, "control_models", control_models)
    monkeypatch.setitem(sys.modules, "control_models.base", control_models_base)

    import training

    return importlib.reload(training)


def test_build_sample_uses_resolved_label_studio_credentials(monkeypatch, tmp_path):
    training = load_training(monkeypatch)
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"test")

    calls = []

    class Backend:
        def get(self, key):
            return None

        def get_local_path(self, image_url, **kwargs):
            calls.append((image_url, kwargs))
            return str(image_path)

    trainer = object.__new__(training.YoloAutoTrainer)
    trainer.backend = Backend()

    controls = {
        "box": training.ControlSpec(
            name="box",
            control_type="RectangleLabels",
            labels=["Cell"],
            data_key="image",
        )
    }
    task = {
        "id": 472,
        "data": {"image": "https://example.test/tasks/472/resolve/?fileuri=image"},
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "box",
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 40,
                            "rectanglelabels": ["Cell"],
                        },
                    }
                ]
            }
        ],
    }

    lines, resolved_image_path, has_polygon = trainer._build_sample(
        task,
        controls,
        {"Cell": 0},
        ls_host="http://label-studio-app-dev:8000",
        ls_access_token="resolved-token",
    )

    assert lines == ["0 0.250000 0.400000 0.300000 0.400000"]
    assert resolved_image_path == image_path
    assert has_polygon is False
    assert calls == [
        (
            "http://label-studio-app-dev:8000/tasks/472/resolve/?fileuri=image",
            {
                "task_id": 472,
                "ls_host": "http://label-studio-app-dev:8000",
                "ls_access_token": "resolved-token",
            },
        )
    ]


def test_normalize_label_studio_media_url_only_rewrites_resolve_urls(monkeypatch):
    training = load_training(monkeypatch)

    assert training.YoloAutoTrainer._normalize_label_studio_media_url(
        "https://external.example/tasks/472/resolve/?fileuri=image",
        "http://label-studio-app-dev:8000",
    ) == "http://label-studio-app-dev:8000/tasks/472/resolve/?fileuri=image"

    assert training.YoloAutoTrainer._normalize_label_studio_media_url(
        "https://external.example/data/image.png",
        "http://label-studio-app-dev:8000",
    ) == "https://external.example/data/image.png"


def test_build_dataset_adds_image_extension_for_label_studio_cache_files(monkeypatch, tmp_path):
    training = load_training(monkeypatch)
    cached_image = tmp_path / "d4749655__"
    cached_image.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")

    trainer = object.__new__(training.YoloAutoTrainer)
    trainer.project_root = tmp_path / "project_231"

    controls = {
        "box": training.ControlSpec(
            name="box",
            control_type="RectangleLabels",
            labels=["Cell"],
            data_key="image",
        )
    }

    def build_sample(task, controls, class_map, ls_host=None, ls_access_token=None):
        return ["0 0.250000 0.400000 0.300000 0.400000"], cached_image, False

    trainer._build_sample = build_sample

    dataset = trainer._build_dataset([{"id": 472}], controls)

    train_images = list((dataset["dataset_dir"] / "images" / "train").iterdir())
    val_images = list((dataset["dataset_dir"] / "images" / "val").iterdir())
    assert len(train_images) == 1
    assert len(val_images) == 1
    assert train_images[0].suffix == ".png"
    assert val_images[0].suffix == ".png"


def test_train_records_mlflow_run_with_rustfs_artifact_root(monkeypatch, tmp_path):
    tracking = {
        "tracking_uri": None,
        "created_experiment": None,
        "started_run": None,
        "params": {},
        "metrics": {},
        "metric_steps": [],
        "artifacts": [],
        "artifact_dirs": [],
        "tags": {},
        "ended_status": None,
    }

    class MlflowClient:
        def get_experiment_by_name(self, name):
            return None

        def create_experiment(self, name, artifact_location=None):
            tracking["created_experiment"] = {
                "name": name,
                "artifact_location": artifact_location,
            }
            return "experiment-1"

    mlflow = types.ModuleType("mlflow")
    mlflow.tracking = SimpleNamespace(MlflowClient=MlflowClient)
    mlflow.set_tracking_uri = lambda uri: tracking.update({"tracking_uri": uri})
    mlflow.start_run = lambda **kwargs: tracking.update({"started_run": kwargs}) or SimpleNamespace(
        info=SimpleNamespace(run_id="run-1")
    )
    mlflow.log_params = lambda params: tracking["params"].update(params)
    mlflow.log_param = lambda key, value: tracking["params"].update({key: value})
    def log_metric(key, value, step=None):
        tracking["metrics"].update({key: value})
        tracking["metric_steps"].append((key, value, step))

    mlflow.log_metric = log_metric
    mlflow.log_artifact = lambda path, artifact_path=None: tracking["artifacts"].append(
        (path, artifact_path)
    )
    mlflow.log_artifacts = lambda path, artifact_path=None: tracking["artifact_dirs"].append(
        (path, artifact_path)
    )
    mlflow.set_tags = lambda tags: tracking["tags"].update(tags)
    mlflow.end_run = lambda status=None: tracking.update({"ended_status": status})
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)

    training = load_training(monkeypatch)
    monkeypatch.setattr(training, "MODEL_ROOT", str(tmp_path / "models"))

    class FakeYOLO:
        def __init__(self, source_model):
            self.source_model = source_model
            self.trainer = None
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)

        def train(self, **kwargs):
            save_dir = tmp_path / "runs" / kwargs["name"]
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True)
            best_path = weights_dir / "best.pt"
            best_path.write_bytes(b"model")
            (save_dir / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")
            (save_dir / "labels.jpg").write_bytes(b"labels")
            (save_dir / "train_batch0.jpg").write_bytes(b"train")
            (save_dir / "results.csv").write_text(
                "epoch,metrics/mAP50(B),train/box_loss\n1,0.50,0.2\n2,0.75,0.1\n",
                encoding="utf-8",
            )
            self.trainer = SimpleNamespace(best=str(best_path), save_dir=str(save_dir), epoch=1)
            for callback in self.callbacks.get("on_fit_epoch_end", []):
                callback(self.trainer)

    monkeypatch.setattr(training, "YOLO", FakeYOLO)

    class Backend:
        def get(self, key):
            return None

    trainer = object.__new__(training.YoloAutoTrainer)
    trainer.backend = Backend()
    trainer.project_id = "231"
    trainer.project_root = tmp_path / "project_231"
    trainer.train_root = tmp_path
    trainer.epochs = 3
    trainer.imgsz = 640
    trainer.batch = 2
    trainer.workers = 1
    trainer.patience = 5
    trainer.base_detect_model = "yolov8m.pt"
    trainer.base_segment_model = "yolov8n-seg.pt"
    trainer.mlflow_enabled = True
    trainer.mlflow_tracking_uri = f"file://{tmp_path / 'mlflow'}"
    trainer.mlflow_artifact_root = "mlflow-artifacts:/biowork"
    trainer.mlflow_experiment_name = "biowork-yolo-training"

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: dataset\n", encoding="utf-8")
    dataset = {
        "dataset_dir": tmp_path / "dataset",
        "data_yaml": data_yaml,
        "num_samples": 4,
        "num_train": 3,
        "num_val": 1,
        "task": "detect",
    }

    result = trainer._train(dataset)

    assert result["mlflow_run_id"] == "run-1"
    assert result["mlflow_artifact_root"] == "mlflow-artifacts:/biowork"
    assert tracking["tracking_uri"] == trainer.mlflow_tracking_uri
    assert tracking["created_experiment"] == {
        "name": "biowork-yolo-training",
        "artifact_location": "mlflow-artifacts:/biowork",
    }
    assert tracking["started_run"]["experiment_id"] == "experiment-1"
    assert tracking["params"]["project_id"] == "231"
    assert tracking["params"]["mlflow_artifact_root"] == "mlflow-artifacts:/biowork"
    assert tracking["metrics"]["dataset_num_samples"] == 4
    assert tracking["metrics"]["metrics/mAP50B"] == 0.75
    assert ("metrics/mAP50B", 0.75, 2) in tracking["metric_steps"]
    assert tracking["tags"]["status"] == "succeeded"
    assert tracking["tags"]["last_logged_epoch"] == "2"
    assert tracking["tags"]["biowork.artifact_store"] == "mlflow-artifacts:/biowork"
    assert any(artifact_path == "progress/epoch_0002" for _, artifact_path in tracking["artifacts"])
    assert any(artifact_path == "config" for _, artifact_path in tracking["artifacts"])
    assert any(artifact_path == "dataset/plots" for _, artifact_path in tracking["artifacts"])
    assert any(artifact_path == "progress/samples" for _, artifact_path in tracking["artifacts"])
    assert any(artifact_path == "model" for _, artifact_path in tracking["artifacts"])
    assert any(artifact_path == "ultralytics_run" for _, artifact_path in tracking["artifact_dirs"])
    assert tracking["ended_status"] == "FINISHED"
    assert (tmp_path / "models" / result["active_model_path"]).exists()
