"""Exercise rollout failure/recovery without Docker or remote registry writes."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def deployment(monkeypatch):
    log = ModuleType('airflow.sdk.log')
    log.mask_secret = lambda value: None
    monkeypatch.setitem(sys.modules, 'airflow', ModuleType('airflow'))
    monkeypatch.setitem(sys.modules, 'airflow.sdk', ModuleType('airflow.sdk'))
    monkeypatch.setitem(sys.modules, 'airflow.sdk.log', log)
    path = Path(__file__).resolve().parents[2] / 'airflow/dags/prediction_deployment.py'
    spec = importlib.util.spec_from_file_location('deployment_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in {
        'AIRFLOW_DOCKER_PROJECT_DIR': '/not-a-real-checkout',
        'PREDICT_MODEL_VERSION': '@live',
        'PREDICT_SMOKE_USERNAME': 'test-user',
        'PREDICT_SMOKE_PASSWORD': 'disposable-test-password',
        'AUTH_SECRET_KEY': 'disposable-test-key',
        'MLFLOW_TRACKING_PASSWORD': 'disposable-test-token',
        'MLFLOW_TRACKING_URI': 'http://unused.invalid',
        'MLFLOW_TRACKING_USERNAME': 'test-owner',
    }.items():
        monkeypatch.setenv(key, value)
    return module


def arrange_existing(module, monkeypatch, legacy=False):
    baseline = {'model_uri': 'models:/movie-genre-knn/1', 'model_type': 'knn'}
    if not legacy:
        baseline.update(data_fingerprint='a' * 64, sample_user_id=11)
    replicas = []
    for i in range(3):
        replica = MagicMock()
        replica.attrs = {
            'State': {'Health': {'Status': 'healthy'}},
            'Image': 'sha256:previous-serving-image',
            'Config': {'Env': ['PREDICT_MODEL_VERSION=@live']},
            'NetworkSettings': {
                'Networks': {'personal_default': {'IPAddress': f'172.0.0.{i + 1}'}}
            },
        }
        replicas.append(replica)
    client = MagicMock()
    monkeypatch.setattr(module.docker, 'from_env', lambda: client)
    monkeypatch.setattr(module, '_replicas', lambda client: replicas)
    monkeypatch.setattr(
        module, '_url', lambda container, network: 'http://disposable-replica'
    )
    monkeypatch.setattr(
        module, '_login', lambda *args: {'Authorization': 'disposable-test-jwt'}
    )
    response = MagicMock()
    response.json.return_value = baseline
    monkeypatch.setattr(module.requests, 'get', lambda *args, **kwargs: response)
    monkeypatch.setattr(module, '_reload_nginx', lambda client: None)
    state = {'alias': '1', 'writes': []}

    def registry(method, name, alias, version=None, **kwargs):
        if method == 'GET':
            return state['alias']
        state['alias'] = version
        state['writes'].append(version)

    monkeypatch.setattr(module, '_registry_alias', registry)
    return client, replicas, state


def training_result():
    return {
        'model_name': 'movie-genre-knn',
        'model_version': '2',
        'run_id': 'test-run',
        'data_fingerprint': 'b' * 64,
        'sample_user_id': 22,
    }


def test_failed_rollout_probes_rollback_with_previous_snapshots_user(
    deployment, monkeypatch
):
    module = deployment
    client, replicas, state = arrange_existing(module, monkeypatch)
    probes = []

    def smoke(url, headers, expected, user_id, **kwargs):
        probes.append((expected['model_uri'], user_id))
        if expected['model_uri'].endswith('/2') and state['alias'] == '2':
            raise RuntimeError('injected post-promotion failure')

    monkeypatch.setattr(module, '_smoke', smoke)
    with pytest.raises(RuntimeError, match='restored and verified'):
        module.deploy_prediction(training_result())
    assert state['writes'] == ['2', '1']
    assert state['alias'] == '1'
    assert ('models:/movie-genre-knn/2', 22) in probes
    assert probes[-4:] == [('models:/movie-genre-knn/1', 11)] * 4
    assert all(replica.restart.call_count == 2 for replica in replicas)
    client.containers.run.return_value.remove.assert_called_once_with(force=True)


def test_legacy_baseline_is_rejected_before_any_promotion(deployment, monkeypatch):
    client, _, state = arrange_existing(deployment, monkeypatch, legacy=True)
    with pytest.raises(RuntimeError, match='migrate the legacy deployment'):
        deployment.deploy_prediction(training_result())
    assert state['writes'] == []
    client.containers.run.assert_not_called()


def test_unversioned_training_result_fails_before_docker_access(
    deployment, monkeypatch
):
    result = training_result()
    result.pop('data_fingerprint')
    connect = MagicMock()
    monkeypatch.setattr(deployment.docker, 'from_env', connect)
    with pytest.raises(ValueError, match='versioned serving-data snapshot'):
        deployment.deploy_prediction(result)
    connect.assert_not_called()
