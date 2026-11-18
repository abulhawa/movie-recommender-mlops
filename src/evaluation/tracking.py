"""Upload a saved evaluation to a separate personal MLflow experiment."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

PERSONAL_URI = 'https://dagshub.com/abulhawa/movie-recommender-mlops.mlflow'
EXPERIMENT = 'movie-recommendation-evaluation'


def upload_report(directory, tracking_uri=PERSONAL_URI):
    import mlflow

    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    # Keep all network writes in the owner's personal repository.
    if tracking_uri.rstrip('/') != PERSONAL_URI:
        raise ValueError('Evaluation uploads require the personal DagsHub endpoint')
    metrics = {
        f'{method}.{name}': float(value)
        for method, values in report['metrics'].items()
        for name, value in values.items()
        if value is not None and math.isfinite(value)
    }
    metrics.update(
        {f'coverage.{key}': value for key, value in report['coverage'].items()}
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(
        run_name=f'historical-parts-1-{report["protocol"]["train_through"]}-vs-{report["protocol"]["test_part"]}'
    ) as run:
        mlflow.log_params(report['protocol'])
        mlflow.set_tags(
            {
                'evaluation.scope': 'historical_reconstruction',
                'evaluation.status': report['status'],
                'evaluation.report_sha256': hashlib.sha256(
                    (directory / 'report.json').read_bytes()
                ).hexdigest(),
                'evaluation.source_commit': report['source_commit'] or 'unknown',
                'evaluation.evaluator_sha256': report['evaluator_sha256'],
                'evaluation.source_working_tree_dirty': str(
                    report['source_working_tree_dirty']
                ),
            }
        )
        mlflow.log_metrics(metrics)
        # Only aggregate evidence: never upload per-user histories or recommendations.
        mlflow.log_artifact(str(directory / 'report.json'), artifact_path='evaluation')
        mlflow.log_artifact(str(directory / 'summary.md'), artifact_path='evaluation')
        run_id = run.info.run_id
    client = mlflow.MlflowClient()
    saved = client.get_run(run_id)
    if saved.info.status != 'FINISHED' or any(
        saved.data.metrics.get(key) != value for key, value in metrics.items()
    ):
        raise RuntimeError(f'MLflow readback failed for run {run_id}')
    artifacts = {item.path for item in client.list_artifacts(run_id, 'evaluation')}
    if not {'evaluation/report.json', 'evaluation/summary.md'} <= artifacts:
        raise RuntimeError(f'MLflow artifact readback failed for run {run_id}')
    (directory / 'mlflow_upload.json').write_text(
        json.dumps(
            {
                'run_id': run_id,
                'experiment_id': saved.info.experiment_id,
                'tracking_uri': tracking_uri,
                'metrics_verified': len(metrics),
            },
            indent=2,
        )
        + '\n'
    )
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
    if not os.getenv('MLFLOW_TRACKING_PASSWORD') and os.getenv('DAGSHUB_TOKEN'):
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['DAGSHUB_TOKEN']
    run_id = upload_report(
        args.report_dir, os.getenv('MLFLOW_TRACKING_URI', PERSONAL_URI)
    )
    print(f'Verified MLflow evaluation run: {run_id}')


if __name__ == '__main__':
    main()
