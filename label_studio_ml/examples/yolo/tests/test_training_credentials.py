import importlib
import sys
import types


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
