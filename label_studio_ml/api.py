import hmac
import logging
import os
import uuid
import threading

from flask import Flask, request, jsonify, Response

from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import exception_handler

logger = logging.getLogger(__name__)

_server = Flask(__name__)
MODEL_CLASS = LabelStudioMLBase
BASIC_AUTH = None

# In-memory store for async prediction jobs
_prediction_jobs = {}
_prediction_jobs_lock = threading.Lock()


def init_app(model_class, basic_auth_user=None, basic_auth_pass=None):
    global MODEL_CLASS
    global BASIC_AUTH

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    return _server


def _run_async_predict(job_id, data):
    """Background worker: runs prediction and writes result into _prediction_jobs."""
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    with _prediction_jobs_lock:
        _prediction_jobs[job_id]['status'] = 'started'

    try:
        model = MODEL_CLASS(project_id=project_id, label_config=label_config)
        response = model.predict(tasks, context=context, **params)

        if isinstance(response, ModelResponse):
            if not response.has_model_version():
                mv = model.model_version
                if mv:
                    response.set_version(str(mv))
            else:
                response.update_predictions_version()
            response = response.model_dump()

        results = response
        if results is None:
            results = []
        if isinstance(results, dict):
            results = results.get('predictions', results)

        with _prediction_jobs_lock:
            _prediction_jobs[job_id]['status'] = 'done'
            _prediction_jobs[job_id]['results'] = results

    except Exception as e:
        logger.error(f'Async prediction job {job_id} failed: {e}', exc_info=True)
        with _prediction_jobs_lock:
            _prediction_jobs[job_id]['status'] = 'error'
            _prediction_jobs[job_id]['error'] = str(e)


@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    """
    Predict tasks (synchronous).

    Example request:
    request = {
            'tasks': tasks,
            'model_version': model_version,
            'project': '{project.id}.{int(project.created_at.timestamp())}',
            'label_config': project.label_config,
            'params': {
                'login': project.task_data_login,
                'password': project.task_data_password,
                'context': context,
            },
        }

    @return:
    Predictions in LS format
    """
    data = request.json
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    # model.use_label_config(label_config)

    response = model.predict(tasks, context=context, **params)

    # if there is no model version we will take the default
    if isinstance(response, ModelResponse):
        if not response.has_model_version():
            mv = model.model_version
            if mv:
                response.set_version(str(mv))
        else:
            response.update_predictions_version()

        response = response.model_dump()

    res = response
    if res is None:
        res = []

    if isinstance(res, dict):
        res = response.get("predictions", response)

    return jsonify({'results': res})


@_server.route('/predict_async', methods=['POST'])
@exception_handler
def _predict_async():
    """
    Submit a batch prediction job asynchronously.

    Accepts the same payload as /predict but returns immediately with a
    job_id. Poll GET /predictions/<job_id> to check status and retrieve
    results when done.

    @return: {"job_id": "<uuid>", "status": "pending"}
    """
    data = request.json
    job_id = str(uuid.uuid4())

    with _prediction_jobs_lock:
        _prediction_jobs[job_id] = {
            'status': 'pending',
            'results': None,
            'error': None,
        }

    thread = threading.Thread(
        target=_run_async_predict,
        args=(job_id, data),
        daemon=True,
    )
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'pending'})


@_server.route('/predictions/<job_id>', methods=['GET'])
@exception_handler
def _prediction_status(job_id):
    """
    Get the status and results of an async prediction job.

    @return:
    - {"job_id": "...", "status": "pending"|"started"} while processing
    - {"job_id": "...", "status": "done", "results": [...]} on completion
    - {"job_id": "...", "status": "error", "error": "..."} on failure
    """
    with _prediction_jobs_lock:
        job = _prediction_jobs.get(job_id)

    if job is None:
        return jsonify({'error': f'Job {job_id} not found'}), 404

    response = {'job_id': job_id, 'status': job['status']}
    if job['status'] == 'done':
        response['results'] = job['results']
    elif job['status'] == 'error':
        response['error'] = job['error']

    return jsonify(response)


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('schema')
    extra_params = data.get('extra_params')
    hostname = data.get('hostname')
    access_token = data.get('access_token')
    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    if extra_params:
        model.set_extra_params(extra_params)
    if hostname:
        model.set("ls_host", hostname)
    if access_token:
        model.set("ls_access_token", access_token)

    model_version = model.get('model_version')
    return jsonify({'model_version': model_version})


@_server.route('/train', methods=['POST'])
@exception_handler
def _train():
    """Legacy train endpoint used by Label Studio ML API connector."""
    data = request.json or {}
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    label_config = data.get('label_config')

    model = MODEL_CLASS(project_id=project_id, label_config=label_config)
    result = model.fit("START_TRAINING", data)

    try:
        response = jsonify({'result': result, 'status': 'ok'})
    except Exception as e:
        response = jsonify({'error': str(e), 'status': 'error'})
    return response, 201


TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'START_TRAINING'
)


@_server.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event = data.pop('action')
    if event not in TRAIN_EVENTS:
        return jsonify({'status': 'Unknown event'}), 200
    project_id = str(data['project']['id'])
    label_config = data['project']['label_config']
    model = MODEL_CLASS(project_id, label_config=label_config)
    result = model.fit(event, data)

    try:
        response = jsonify({'result': result, 'status': 'ok'})
    except Exception as e:
        response = jsonify({'error': str(e), 'status': 'error'})

    return response, 201


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    return jsonify({
        'status': 'UP',
        'model_class': MODEL_CLASS.__name__
    })


@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    return jsonify({})


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


@_server.before_request
def check_auth():
    if BASIC_AUTH is not None:

        auth = request.authorization
        if not auth or not (safe_str_cmp(auth.username, BASIC_AUTH[0]) and safe_str_cmp(auth.password, BASIC_AUTH[1])):
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response
