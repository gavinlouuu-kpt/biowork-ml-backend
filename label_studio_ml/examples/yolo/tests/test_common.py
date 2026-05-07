"""
This file contains tests for the API
"""

import os
import pickle
import pytest
import json

from unittest.mock import MagicMock, patch
from model import YOLO
from control_models.keypoint_labels import KeypointLabelsModel
from control_models.rectangle_labels import RectangleLabelsModel


def load_file(path):
    # json
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    # pickle
    if path.endswith(".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)


@pytest.fixture
def client():
    from _wsgi import init_app

    app = init_app(model_class=YOLO)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


TEST_DIR = os.path.dirname(__file__)


label_configs = [
    # test 1: wrong key in task data
    """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
    </View>
    """,
    # test 2: model_skip=true
    """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" model_skip="true">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
    </View>
    """,
]

tasks = [
    # test 1: wrong key in task data
    {"data": {"wrong_key": "https://some/path"}},
    # test 2: model_skip
    {"data": {"image": "https://some/path"}},
]

expected = [
    # test 1: wrong key in task data
    "Can't load path using key",
    # test 2: model skip
    "No suitable control tags",
]


@pytest.mark.parametrize(
    "label_config, task, expect", zip(label_configs, tasks, expected)
)
def test_label_configs(client, label_config, task, expect, capsys):
    data = {"schema": label_config, "project": "42"}
    response = client.post(
        "/setup", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )
    assert (
        response.status_code == 500
    ), "Error was expected, but another status code found"

    # Capture stdout and stderr
    captured = capsys.readouterr()

    # Check for specific words in the output
    assert expect in captured.out, "Text not found in error string"


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch("model.logger")


@pytest.fixture
def mock_ml_backend():
    return MagicMock()


@pytest.fixture
def mock_label_interface(mock_ml_backend):
    mock_label_interface = MagicMock()
    mock_ml_backend.label_interface = mock_label_interface
    return mock_label_interface


@pytest.fixture
def mock_model_class():
    mock_model_class = MagicMock()
    mock_model_class.type = "RectangleLabels"
    mock_model_class.is_control_matched = MagicMock(return_value=True)
    mock_model_class.create = MagicMock()
    return mock_model_class


@pytest.fixture
def available_model_classes(mock_model_class):
    return [mock_model_class, MagicMock(), MagicMock()]


@pytest.fixture
def yolo_instance(mock_label_interface):
    yolo = YOLO()
    yolo.label_interface = mock_label_interface
    return yolo


def test_control_no_to_name(yolo_instance, mock_label_interface, mock_logger):
    mock_label_interface.controls = [
        MagicMock(to_name=None, tag="RectangleLabels", name="test_label")
    ]

    with pytest.raises(ValueError):
        yolo_instance.detect_control_models()

    mock_logger.warning.assert_called_once()


def test_control_no_label_map(
    yolo_instance,
    mock_label_interface,
    mock_model_class,
    available_model_classes,
    mock_logger,
):
    mock_instance = MagicMock()
    mock_instance.label_map = None
    mock_model_class.create.return_value = mock_instance
    mock_label_interface.controls = [
        MagicMock(to_name=["image"], tag="RectangleLabels", name="test_label")
    ]

    with patch("model.available_model_classes", available_model_classes):
        result = yolo_instance.detect_control_models()

    # Updated expectation to handle control models that should be skipped if label_map is None
    assert len(result) == 1
    mock_logger.error.assert_called_once()


def test_control_with_valid_label_map(
    yolo_instance,
    mock_label_interface,
    mock_model_class,
    available_model_classes,
    mock_logger,
):
    mock_instance = MagicMock()
    mock_instance.label_map = {"car": "Car"}
    mock_model_class.create.return_value = mock_instance
    mock_label_interface.controls = [
        MagicMock(to_name=["image"], tag="RectangleLabels", name="test_label")
    ]

    with patch("model.available_model_classes", available_model_classes):
        result = yolo_instance.detect_control_models()

    assert len(result) == 1
    assert result[0] == mock_instance
    mock_logger.debug.assert_called_once()


def test_setup_sets_lowercase_model_version():
    yolo = YOLO()
    yolo.setup()
    assert yolo.get("model_version") == "yolo"


def test_setup_keeps_existing_trained_model_version():
    yolo = YOLO()
    yolo.set("model_version", "yolo-auto-detect-20260101_000000")
    yolo.setup()
    assert yolo.get("model_version") == "yolo-auto-detect-20260101_000000"


def test_detect_checkpoint_skips_keypoint_control(mock_ml_backend):
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.names = {0: "Object"}
    fake_object = MagicMock(tag="Image", value_name="image")
    keypoint_control = MagicMock(
        tag="KeyPointLabels",
        name="tag3",
        to_name=["image"],
        objects=[fake_object],
        attr={"model_path": "autotrain/project_226/best.pt"},
    )
    rectangle_control = MagicMock(
        tag="RectangleLabels",
        name="tag4",
        to_name=["image"],
        objects=[fake_object],
        attr={"model_path": "autotrain/project_226/best.pt"},
    )
    mock_ml_backend.build_label_map.return_value = {"Object": "Object"}

    with patch.object(KeypointLabelsModel, "get_cached_model", return_value=fake_model):
        assert KeypointLabelsModel.create(mock_ml_backend, keypoint_control) is None

    with patch.object(RectangleLabelsModel, "get_cached_model", return_value=fake_model):
        rectangle_model = RectangleLabelsModel.create(mock_ml_backend, rectangle_control)

    assert rectangle_model is not None
    assert rectangle_model.from_name == "tag4"


def test_train_endpoint_calls_fit(client, mocker):
    fit_mock = mocker.patch("model.YOLO.fit", return_value={"status": "trained"})
    request = {
        "project": "42.111",
        "label_config": "<View><Image name='image' value='$image'/></View>",
        "annotations": [],
    }
    response = client.post("/train", data=json.dumps(request), content_type="application/json")
    assert response.status_code == 201
    fit_mock.assert_called_once()
