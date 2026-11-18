"""Train, track and register a model together with its exact serving data."""

import json
import tempfile
from pathlib import Path

import mlflow

from src import config
from src.serving_data import attach_snapshot, snapshot_processed_data
from src.train_service.models import implementation


def main():
    """Register one immutable model/data bundle; never train from changing files."""
    for name, value in {
        'MLFLOW_TRACKING_URI': config.MLFLOW_TRACKING_URI,
        'MLFLOW_EXPERIMENT_NAME': config.MLFLOW_EXPERIMENT_NAME,
        'MLFLOW_REGISTERED_MODEL_NAME': config.MLFLOW_REGISTERED_MODEL_NAME,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} cannot be empty')
    settings = implementation.resolve_settings(config)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / 'processed'
        manifest = snapshot_processed_data(config.PROCESSED_DATA_DIR, snapshot)
        with mlflow.start_run(run_name=config.MLFLOW_RUN_NAME) as run:
            # Both the fitted estimator and the profiles later served come from
            # this snapshot, even if the shared processed directory changes later.
            data = implementation.load_data(snapshot)
            parameters = implementation.parameters(settings, data)
            mlflow.log_params(parameters)
            mlflow.set_tags(
                {
                    **implementation.run_tags(),
                    'serving_data_fingerprint': manifest['data_fingerprint'],
                }
            )
            mlflow.log_dict(data['metadata'], 'data/metadata.json')
            result = implementation.train(
                data,
                settings,
                on_epoch=lambda epoch, metrics: mlflow.log_metrics(metrics, step=epoch),
            )
            mlflow.log_metrics(result['metrics'])
            model_directory = Path(temporary) / 'model'
            implementation.export(
                result,
                data,
                parameters,
                model_directory,
                source_root=config.PROJECT_ROOT / 'src',
                run_id=run.info.run_id,
            )
            attach_snapshot(model_directory, snapshot)
            mlflow.log_artifacts(str(model_directory), artifact_path='model')
            version = mlflow.register_model(
                f'runs:/{run.info.run_id}/model', config.MLFLOW_REGISTERED_MODEL_NAME
            )
            client = mlflow.MlflowClient()
            # A failed tagging request can still leave a registered version;
            # do not automatically retry training after an ambiguous response.
            for key, value in {
                **implementation.registry_tags(data),
                'source_training_run_id': run.info.run_id,
                'serving_data_fingerprint': manifest['data_fingerprint'],
            }.items():
                client.set_model_version_tag(
                    config.MLFLOW_REGISTERED_MODEL_NAME,
                    str(version.version),
                    key,
                    str(value),
                )
            print(f'MLflow run ID: {run.info.run_id}')
            print(
                f'Registry URI: models:/{config.MLFLOW_REGISTERED_MODEL_NAME}/{version.version}'
            )

    # Emit only after upload, registration and run closure all succeeded.
    print(
        'TRAINING_RESULT_JSON='
        + json.dumps(
            {
                'model_name': config.MLFLOW_REGISTERED_MODEL_NAME,
                'model_version': str(version.version),
                'run_id': run.info.run_id,
                'data_fingerprint': manifest['data_fingerprint'],
                'sample_user_id': manifest['sample_user_id'],
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
