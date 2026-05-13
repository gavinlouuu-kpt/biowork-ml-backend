import copy
import pytest
from unittest.mock import patch, mock_open
from label_studio_ml.model import LabelStudioMLBase


@pytest.fixture
def model():
    return LabelStudioMLBase(label_config="""<View>
        <Text name="Text" value="$text"/>
        <Labels name="Labels" toName="Text">
            <Label value="label1"/>
            <Label value="label2"/>
        </Labels>
    </View>
    """)


@pytest.mark.parametrize("url, expected_result", [
    ("s3://bucket/test/1.jpg", "data"),
    ("gs://bucket/test/1.jpg", "data"),
    ("azure-blob://bucket/test/1.jpg", "data"),
    ("/data/local-files?d=test", "data"),
    ("/upload/123.jpg", "data"),
    ("c:\\test.jpg", "data"),
    ("/home/user/123.jpg", "data"),
    ("this is text", "this is text"),
])
@patch("builtins.open", new_callable=mock_open, read_data="body")
@patch("label_studio_ml.model.LabelStudioMLBase.get_local_path", return_value="path")
@patch("os.path.exists", return_value=True)  # Mock os.path.exists to return True
def test_preload_task_data(mock_exists, mock_get_local_path, mock_file, model, url, expected_result):
    task = {"id": 1, "data": {"url": url}}
    result = model.preload_task_data(task, value=copy.deepcopy(task['data']))
    assert result == {"url": "body"}
    mock_get_local_path.assert_called_once_with(url=task["data"]["url"], task_id=task["id"])
    mock_file.assert_called_once_with("path", "r")
    print(result)


@patch("builtins.open", new_callable=mock_open, read_data="body")
@patch("label_studio_ml.model.LabelStudioMLBase.get_local_path", return_value="path")
def test_preload_task_data_complex_structure(mock_get_local_path, mock_file, model):
    def generate_task(url):
        return {
            "id": 1,
            "data": {
                "root": {
                    "url": url,
                    "urls": [url, url, url],
                    "text": "this is text",
                },
                "int": 123,
                "float": 42.42,
                "bool": False,
                "null": None,
            }
        }

    url = "s3://bucket/test/1.jpg"
    task = generate_task(url)
    expected = generate_task("body")
    result = model.preload_task_data(task, value=copy.deepcopy(task['data']))
    assert result == expected['data']
    mock_get_local_path.assert_called_with(url=url, task_id=task["id"])
    mock_file.assert_called_with("path", "r")
    print(result)


@patch("label_studio_ml.model.get_local_path", return_value="path")
def test_get_local_path_uses_setup_credentials(mock_get_local_path, model):
    model.set("ls_host", "http://label-studio-app:8000")
    model.set("ls_access_token", "setup-token")

    assert model.get_local_path("s3://bucket/test/1.jpg", task_id=123) == "path"

    _, kwargs = mock_get_local_path.call_args
    assert kwargs["hostname"] == "http://label-studio-app:8000"
    assert kwargs["access_token"] == "setup-token"
    assert kwargs["task_id"] == 123


@patch.dict(
    "os.environ",
    {
        "LABEL_STUDIO_HOST": "http://label-studio-app-dev:8000",
        "LABEL_STUDIO_API_KEY": "env-token",
    },
)
@patch("label_studio_ml.model.get_local_path", return_value="path")
def test_get_local_path_uses_env_credentials(mock_get_local_path):
    model = LabelStudioMLBase(
        project_id="env-fallback-test",
        label_config="""<View><Text name="Text" value="$text"/></View>""",
    )

    assert model.get_local_path("s3://bucket/test/1.jpg", task_id=456) == "path"

    _, kwargs = mock_get_local_path.call_args
    assert kwargs["hostname"] == "http://label-studio-app-dev:8000"
    assert kwargs["access_token"] == "env-token"
    assert kwargs["task_id"] == 456


@patch("label_studio_ml.model.get_local_path", return_value="path")
def test_get_local_path_accepts_sdk_credential_aliases(mock_get_local_path, model):
    assert model.get_local_path(
        "s3://bucket/test/1.jpg",
        hostname="http://label-studio-app:8000",
        access_token="sdk-token",
        task_id=789,
    ) == "path"

    _, kwargs = mock_get_local_path.call_args
    assert kwargs["hostname"] == "http://label-studio-app:8000"
    assert kwargs["access_token"] == "sdk-token"
    assert kwargs["task_id"] == 789
