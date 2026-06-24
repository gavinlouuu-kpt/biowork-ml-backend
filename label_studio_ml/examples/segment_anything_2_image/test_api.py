import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (EXAMPLE_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="tag1" toName="image">
    <Label value="Object" background="#ff0000"/>
  </BrushLabels>
  <KeyPointLabels name="tag2" toName="image" smart="true">
    <Label value="Object" background="#0000ff"/>
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="Object" background="#00ff00"/>
  </RectangleLabels>
</View>
"""


def install_fake_sam2(monkeypatch):
    class FakeSAM2ImagePredictor:
        instances = []

        def __init__(self, _model):
            self.last_predict = None
            self.image_shape = None
            FakeSAM2ImagePredictor.instances.append(self)

        def set_image(self, image):
            self.image_shape = image.shape[:2]

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            self.last_predict = {
                "point_coords": point_coords.copy() if point_coords is not None else None,
                "point_labels": point_labels.copy() if point_labels is not None else None,
                "box": box.copy() if box is not None else None,
                "multimask_output": multimask_output,
            }
            mask = np.zeros((80, 100), dtype=np.uint8)
            mask[10:40, 20:70] = 1
            return np.stack([mask]), np.array([0.95]), np.zeros((1, 1), dtype=np.float32)

    class FakeSAM2AutomaticMaskGenerator:
        generate_calls = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, _image):
            type(self).generate_calls += 1
            return []

    sam2_pkg = types.ModuleType("sam2")
    build_mod = types.ModuleType("sam2.build_sam")
    predictor_mod = types.ModuleType("sam2.sam2_image_predictor")
    generator_mod = types.ModuleType("sam2.automatic_mask_generator")

    build_mod.build_sam2 = lambda *_args, **_kwargs: object()
    predictor_mod.SAM2ImagePredictor = FakeSAM2ImagePredictor
    generator_mod.SAM2AutomaticMaskGenerator = FakeSAM2AutomaticMaskGenerator

    monkeypatch.setitem(sys.modules, "sam2", sam2_pkg)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_mod)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_mod)
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", generator_mod)
    return FakeSAM2ImagePredictor, FakeSAM2AutomaticMaskGenerator


@pytest.fixture()
def sam2_model(monkeypatch, tmp_path):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("RESPONSE_TYPE", "brush")
    monkeypatch.setenv("SAM_PREANNOTATE", "1")
    _fake_predictor_cls, fake_generator_cls = install_fake_sam2(monkeypatch)

    sys.modules.pop("model", None)
    module = importlib.import_module("model")
    module.fake_mask_generator = fake_generator_cls

    image_path = tmp_path / "sanitized-sam2-test.png"
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[10:40, 20:70] = [180, 60, 30]
    Image.fromarray(image).save(image_path)

    def fake_get_local_path(self, _url, **_kwargs):
        return str(image_path)

    monkeypatch.setattr(module.NewModel, "get_local_path", fake_get_local_path)
    module.predictor.last_predict = None
    module.predictor._last_image_url = None
    return module


@pytest.fixture()
def client(sam2_model):
    from label_studio_ml.api import init_app

    app = init_app(model_class=sam2_model.NewModel)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def predict_request(context):
    return {
        "tasks": [{"id": 1, "project": 1, "data": {"image": "local://sanitized-sam2-test.png"}}],
        "project": "1.0",
        "label_config": LABEL_CONFIG,
        "params": {"context": context},
    }


def post_predict(client, context):
    return client.post("/predict", data=json.dumps(predict_request(context)), content_type="application/json")


def test_structured_prompts_are_passed_to_sam2(client, sam2_model):
    response = post_predict(
        client,
        {
            "prompts": {
                "original_width": 100,
                "original_height": 80,
                "coordinate_system": "pixel",
                "label": "Object",
                "points": [
                    {"x": 10, "y": 20, "is_positive": True},
                    {"x": 30, "y": 40, "is_positive": False},
                ],
                "boxes": [
                    {"x": 5, "y": 6, "width": 10, "height": 12},
                    {"box": [20, 10, 40, 30]},
                ],
            }
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["results"][0]["result"][0]["value"]["brushlabels"] == ["Object"]

    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"].tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert predict_call["point_labels"].tolist() == [1, 0]
    assert predict_call["box"].tolist() == [5.0, 6.0, 40.0, 30.0]
    assert predict_call["multimask_output"] is True


def test_singular_point_prompt_object_is_passed_to_sam2(client, sam2_model):
    response = post_predict(
        client,
        {
            "prompt": {
                "original_width": 100,
                "original_height": 80,
                "coordinate_system": "pixel",
                "type": "point",
                "x": 10,
                "y": 20,
                "label": 1,
            }
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"].tolist() == [[10.0, 20.0]]
    assert predict_call["point_labels"].tolist() == [1]
    assert predict_call["box"] is None


def test_singular_point_prompt_accepts_generic_image_dimensions(client, sam2_model):
    response = post_predict(
        client,
        {
            "prompt": {
                "width": 100,
                "height": 80,
                "coordinate_system": "pixel",
                "type": "point",
                "x": 10,
                "y": 20,
                "label": 1,
            }
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"].tolist() == [[10.0, 20.0]]
    assert predict_call["point_labels"].tolist() == [1]
    assert predict_call["box"] is None


def test_singular_box_prompt_object_is_passed_to_sam2(client, sam2_model):
    response = post_predict(
        client,
        {
            "prompt": {
                "original_width": 100,
                "original_height": 80,
                "coordinate_system": "pixel",
                "type": "box",
                "x": 5,
                "y": 6,
                "width": 10,
                "height": 12,
            }
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"] is None
    assert predict_call["point_labels"] is None
    assert predict_call["box"].tolist() == [5.0, 6.0, 15.0, 18.0]


def test_singular_box_prompt_uses_outer_context_dimensions(client, sam2_model):
    response = post_predict(
        client,
        {
            "original_width": 100,
            "original_height": 80,
            "coordinate_system": "pixel",
            "prompt": {
                "type": "box",
                "x": 5,
                "y": 6,
                "width": 10,
                "height": 12,
            },
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"] is None
    assert predict_call["point_labels"] is None
    assert predict_call["box"].tolist() == [5.0, 6.0, 15.0, 18.0]


def test_flat_mixed_prompt_list_is_split_by_type(client, sam2_model):
    response = post_predict(
        client,
        {
            "original_width": 100,
            "original_height": 80,
            "coordinate_system": "pixel",
            "prompts": [
                {"type": "point", "x": 10, "y": 20, "label": 1},
                {"type": "point", "x": 30, "y": 40, "label": 0},
                {"type": "box", "x": 5, "y": 6, "width": 10, "height": 12},
            ],
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"].tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert predict_call["point_labels"].tolist() == [1, 0]
    assert predict_call["box"].tolist() == [5.0, 6.0, 15.0, 18.0]


def test_box_only_prompt_is_passed_to_sam2(client, sam2_model):
    response = post_predict(
        client,
        {
            "prompts": {
                "original_width": 100,
                "original_height": 80,
                "coordinate_system": "pixel",
                "boxes": [{"x": 20, "y": 10, "width": 30, "height": 40}],
            }
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"] is None
    assert predict_call["point_labels"] is None
    assert predict_call["box"].tolist() == [20.0, 10.0, 50.0, 50.0]


def test_normalized_prompt_coordinates_are_scaled(client, sam2_model):
    response = post_predict(
        client,
        {
            "prompts": {
                "original_width": 100,
                "original_height": 80,
                "coordinate_system": "normalized",
                "points": [{"x": 0.5, "y": 0.25, "is_positive": True}],
                "boxes": [{"box": [0.1, 0.2, 0.4, 0.6]}],
            }
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"].tolist() == [[50.0, 20.0]]
    assert predict_call["point_labels"].tolist() == [1]
    assert predict_call["box"].tolist() == [10.0, 16.0, 40.0, 48.0]


def test_legacy_label_studio_context_still_works(client, sam2_model):
    response = post_predict(
        client,
        {
            "result": [
                {
                    "original_width": 100,
                    "original_height": 80,
                    "value": {"x": 50, "y": 25, "width": 1, "keypointlabels": ["Object"]},
                    "is_positive": True,
                    "type": "keypointlabels",
                },
                {
                    "original_width": 100,
                    "original_height": 80,
                    "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rectanglelabels": ["Object"]},
                    "type": "rectanglelabels",
                },
            ]
        },
    )

    assert response.status_code == 200
    predict_call = sam2_model.predictor.last_predict
    assert predict_call["point_coords"].tolist() == [[50.0, 20.0]]
    assert predict_call["point_labels"].tolist() == [1]
    assert predict_call["box"].tolist() == [10.0, 16.0, 40.0, 48.0]


def assert_empty_prediction_response(response):
    assert response.status_code == 200
    assert response.get_json() == {"results": [{"model_version": "0.0.1", "result": [], "score": 0.0}]}


def test_empty_structured_prompt_returns_empty_prediction(client, sam2_model):
    response = post_predict(
        client,
        {"prompts": {"original_width": 100, "original_height": 80, "points": [], "boxes": []}},
    )

    assert_empty_prediction_response(response)
    assert sam2_model.predictor.last_predict is None
    assert sam2_model.fake_mask_generator.generate_calls == 0


def test_empty_structured_prompt_list_returns_empty_prediction(client, sam2_model):
    response = post_predict(client, {"prompts": []})

    assert_empty_prediction_response(response)
    assert sam2_model.predictor.last_predict is None
    assert sam2_model.fake_mask_generator.generate_calls == 0


def test_empty_legacy_context_returns_empty_prediction(client, sam2_model):
    response = post_predict(client, {"result": []})

    assert_empty_prediction_response(response)
    assert sam2_model.predictor.last_predict is None
    assert sam2_model.fake_mask_generator.generate_calls == 0


def test_only_negative_prompts_return_validation_error(client):
    response = post_predict(
        client,
        {
            "prompts": {
                "original_width": 100,
                "original_height": 80,
                "points": [{"x": 10, "y": 20, "is_positive": False}],
            }
        },
    )

    assert response.status_code == 400
    assert "at least one positive point or box prompt is required" in response.get_json()["detail"]


def test_null_structured_prompt_returns_clear_validation_error(client):
    response = post_predict(client, {"prompts": None})

    assert response.status_code == 400
    assert "prompts must be an object" in response.get_json()["detail"]


def test_invalid_structured_prompt_returns_clear_validation_error(client):
    response = post_predict(
        client,
        {"prompts": {"original_width": 100, "original_height": 80, "points": [{"x": "bad", "y": 20}]}},
    )

    assert response.status_code == 400
    assert "prompts.points[0].x must be a number" in response.get_json()["detail"]
