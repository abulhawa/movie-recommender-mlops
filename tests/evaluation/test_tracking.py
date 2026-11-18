"""Exercise upload and readback against an isolated local MLflow store."""

import json

import mlflow
import pytest

from src.evaluation import tracking


def test_upload_metrics_artifacts_and_provenance(tmp_path, monkeypatch):
    uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    monkeypatch.setattr(tracking, 'PERSONAL_URI', uri)
    report = {
        'protocol': {'train_through': 6, 'test_part': 7, 'k': 10},
        'metrics': {'genre_knn_serving': {'recall': 0.25, 'ndcg': None}},
        'coverage': {'evaluated_users': 2},
        'status': 'evaluated',
        'source_commit': 'abc123',
        'evaluator_sha256': 'checksum',
        'source_working_tree_dirty': True,
    }
    (tmp_path / 'report.json').write_text(json.dumps(report))
    (tmp_path / 'summary.md').write_text('Aggregate summary')
    (tmp_path / 'per_user.csv').write_text('private user data')
    mlflow.MlflowClient(tracking_uri=uri).create_experiment(
        tracking.EXPERIMENT, artifact_location=str(tmp_path / 'artifacts')
    )
    try:
        run_id = tracking.upload_report(tmp_path, uri)
        client = mlflow.MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)
        assert run.info.status == 'FINISHED'
        assert run.data.metrics == {
            'genre_knn_serving.recall': 0.25,
            'coverage.evaluated_users': 2.0,
        }
        assert run.data.tags['evaluation.scope'] == 'historical_reconstruction'
        assert run.data.params['k'] == '10'
        assert {x.path for x in client.list_artifacts(run_id, 'evaluation')} == {
            'evaluation/report.json',
            'evaluation/summary.md',
        }
        assert (
            json.loads((tmp_path / 'mlflow_upload.json').read_text())['run_id']
            == run_id
        )
    finally:
        mlflow.set_tracking_uri(None)


def test_rejects_nonpersonal_endpoint(tmp_path):
    (tmp_path / 'report.json').write_text('{}')
    with pytest.raises(ValueError, match='personal DagsHub'):
        tracking.upload_report(tmp_path, 'https://dagshub.com/other/repo.mlflow')
