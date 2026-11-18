"""Real artifact/registry checks with tiny data and a temporary SQLite registry."""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from predict_service.model_loader import load_registered_model
from src.serving_data import attach_snapshot, snapshot_processed_data, validate_snapshot
from src.train_service.models import knn

SOURCE = Path(__file__).resolve().parents[2] / 'src'


def write_inputs(directory, user_id=1):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            'movieId': np.arange(100, 130),
            'Action': np.arange(30) % 2,
            'Drama': np.arange(30) % 3,
        }
    ).to_csv(directory / 'movie_matrix.csv', index=False)
    pd.DataFrame({'userId': [user_id], 'Action': [1.0], 'Drama': [0.0]}).to_csv(
        directory / 'user_matrix.csv', index=False
    )


def test_snapshot_rejects_tampered_or_incomplete_profiles(tmp_path):
    source, target = tmp_path / 'inputs', tmp_path / 'snapshot'
    write_inputs(source)
    manifest = snapshot_processed_data(source, target)
    assert validate_snapshot(target, manifest['data_fingerprint']) == manifest
    # A different sample profile also changes the content identity.
    write_inputs(source, user_id=9)
    other = snapshot_processed_data(source, tmp_path / 'other')
    assert other['data_fingerprint'] != manifest['data_fingerprint']
    with (target / 'user_matrix.csv').open('a') as out:
        out.write('99,0,1\n')
    with pytest.raises(ValueError, match='checksum'):
        validate_snapshot(target, manifest['data_fingerprint'])
    with pytest.raises(FileNotFoundError):
        snapshot_processed_data(tmp_path / 'missing', tmp_path / 'incomplete')


def test_training_entrypoint_packages_exact_snapshot(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from src import config
    from src.train_service.train_model import main

    inputs = tmp_path / 'inputs'
    write_inputs(inputs, user_id=7)
    monkeypatch.setattr(config, 'PROCESSED_DATA_DIR', inputs)
    monkeypatch.setattr(
        config, 'MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path}/training.db'
    )
    monkeypatch.setattr(config, 'MLFLOW_EXPERIMENT_NAME', 'snapshot-training')
    monkeypatch.setattr(config, 'MLFLOW_REGISTERED_MODEL_NAME', 'snapshot-knn')
    main()
    result = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('TRAINING_RESULT_JSON=')
    )
    result = json.loads(result.split('=', 1)[1])
    artifact = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f'models:/snapshot-knn/{result["model_version"]}'
        )
    )
    recorded = validate_snapshot(artifact / 'serving_data', result['data_fingerprint'])
    assert recorded['sample_user_id'] == result['sample_user_id'] == 7
    assert (artifact / 'serving_data/user_matrix.csv').read_bytes() == (
        inputs / 'user_matrix.csv'
    ).read_bytes()
    schema = json.loads((artifact / 'training_schema.json').read_text())
    assert schema['serving_data_fingerprint'] == result['data_fingerprint']


def test_alias_rollback_restores_profiles_despite_host_file_changes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    uri = f'sqlite:///{tmp_path}/registry.db'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    monkeypatch.setenv('PREDICT_MODEL_TYPE', 'knn')
    monkeypatch.setenv('PREDICT_MODEL_NAME', 'bundled-knn')
    monkeypatch.setenv('PREDICT_MODEL_VERSION', '@live')
    monkeypatch.delenv('USER_MATRIX_PATH', raising=False)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment('snapshot-rollback')
    inputs = tmp_path / 'inputs'
    versions = []
    for user_id in [11, 22]:
        write_inputs(inputs, user_id)
        snapshot = tmp_path / f'snapshot-{user_id}'
        manifest = snapshot_processed_data(inputs, snapshot)
        data = knn.load_data(snapshot)
        settings = {'n_neighbors': 20, 'algorithm': 'ball_tree', 'metric': 'euclidean'}
        with mlflow.start_run() as run:
            folder = tmp_path / f'model-{user_id}'
            knn.export(
                knn.train(data, settings),
                data,
                knn.parameters(settings, data),
                folder,
                source_root=SOURCE,
                run_id=run.info.run_id,
            )
            attach_snapshot(folder, snapshot)
            mlflow.log_artifacts(str(folder), artifact_path='model')
            version = mlflow.register_model(
                f'runs:/{run.info.run_id}/model', 'bundled-knn'
            )
            versions.append((version.version, manifest))
    client = mlflow.MlflowClient()
    # The shared host CSV now contains neither historical user profile.
    write_inputs(inputs, user_id=999)
    for selected, expected_user in [(0, 11), (1, 22), (0, 11)]:
        version, manifest = versions[selected]
        client.set_registered_model_alias('bundled-knn', 'live', version)
        model = load_registered_model()
        assert model.data_fingerprint == manifest['data_fingerprint']
        assert pd.read_csv(model.user_matrix_path).userId.tolist() == [expected_user]
        with patch(
            'predict_service.model_loader.load_registered_model', return_value=model
        ):
            sys.modules.pop('predict_service.predict', None)
            module = importlib.import_module('predict_service.predict')
        try:
            assert module.user_matrix.index.tolist() == [expected_user]
            info = module.model_info()
            assert info['sample_user_id'] == expected_user
            assert info['data_fingerprint'] == model.data_fingerprint
            assert (
                len(
                    module.predict(
                        module.PredictRequest(user_ids=[expected_user], seed=42)
                    )['recommendations'][0]['movie_ids']
                )
                == 10
            )
        finally:
            for collector in (
                module.PREDICTION_REQUESTS,
                module.PREDICTION_HANDLER_REQUESTS,
                module.PREDICTION_REQUEST_DURATION,
            ):
                REGISTRY.unregister(collector)
            sys.modules.pop('predict_service.predict', None)


def test_declared_bundle_cannot_silently_fall_back_to_host_csv(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path}/unused.db')
    monkeypatch.setenv('PREDICT_MODEL_VERSION', '1')
    monkeypatch.setenv('PREDICT_MODEL_TYPE', 'knn')
    monkeypatch.setattr(
        mlflow.artifacts, 'download_artifacts', lambda **kwargs: str(tmp_path)
    )
    schema = tmp_path / 'training_schema.json'
    schema.write_text(json.dumps({'serving_data_fingerprint': 'a' * 64}))
    with pytest.raises(FileNotFoundError):
        load_registered_model()
    schema.write_text(json.dumps({'serving_data_fingerprint': None}))
    with pytest.raises(ValueError, match='Invalid serving-data fingerprint'):
        load_registered_model()
