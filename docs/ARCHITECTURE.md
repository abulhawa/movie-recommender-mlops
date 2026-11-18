# Architecture and code map

The project demonstrates an MLOps lifecycle around genre-based movie retrieval.
It is a local training/demo stack, not a public managed service. Start with the
[root overview](../README.md); use the [restore guide](RESTORE_AND_RUN.md) to run it.

## Two paths into the same feature schema

### DVC: reproducible local pipeline

`dvc.yaml` describes `import_raw_data → split_data → ingest/build_features`.
`params.yaml` selects the chronological rating parts that have “arrived.” Raw
inputs and source splits use `cache: false`; ingested subsets and processed
outputs are cached/versioned. The normal pipeline's data policy differs from the
additional objects retained during personal preservation.

| File | Responsibility |
|---|---|
| `src/initial_data_import/import_raw_data.py` | Obtain the original input files |
| `src/initial_data_import/split_data.py` | Sort/split rating data into chronological parts |
| `src/features/build_features.py` | Reusable genre feature construction |
| `dvc.yaml`, `dvc.lock`, `params.yaml` | Pipeline definition, recorded hashes, selected parameters |

Changing `split.current_part` changes the local DVC pipeline's inputs. It does
not select the parts for the bucket-based Airflow preparation task, which reads
the highest contiguous part stored in the bucket.

### Bucket: simulated arriving data and Airflow

The collection API reads local raw/split files, checks which parts already exist
in the DagsHub repository bucket, and uploads missing files. The preparation
container downloads a cumulative prefix of rating parts, creates matrices, and
validates them before replacing the processed files.

| File | Responsibility |
|---|---|
| `src/data_collection_service/main.py` | Authenticated ingestion HTTP routes |
| `src/data_collection_service/ingest.py` | Validate requested progression and plan uploads |
| `src/data_collection_service/bucket_storage_client.py` | S3-compatible bucket access and part listing |
| `src/data_preparation_service/main.py` | Download inputs, build/check features, replace output files |

Existence checks skip files already uploaded; they are not a distributed lock.
A retry after an ambiguous successful upload can advance the simulation again,
so the ingestion DAG deliberately does not retry automatically. Do not run
multiple writers/preparation jobs concurrently.

A user's feature row is the average genre vector of movies they rated. Numeric
rating values are ignored in this representation. Movie feature rows are binary
genre vectors. Both matrices must agree on genre columns; IDs are identifiers,
not zero-based row positions. Each output file replacement is atomic, but the two
CSV replacements are not a single transaction.

## Storage boundaries

| State | Where it lives | How it is restored |
|---|---|---|
| Code, Docker configuration, public examples | GitHub and personal DagsHub Git repository | Git clone/pull |
| DVC cached outputs | DagsHub DVC remote (`…movie-recommender-mlops.s3`, bucket `dvc`) | DVC token in `.dvc/config.local`, then `dvc pull` |
| Upload simulation inputs | DagsHub repository bucket (`movie-recommender-mlops`) | Bucket client / preparation task using `.env` credentials |
| Runs, registered versions, model artifacts, aliases | DagsHub MLflow endpoint (`…movie-recommender-mlops.mlflow`) | MLflow-authenticated clients |
| Local credentials and machine paths | Ignored `.env`, `.dvc/config.local`, optional custom user records | Private backup and machine-specific review |
| Airflow metadata/logs and Grafana/Prometheus history | Docker named volumes | Preserve or back up volumes separately |

The three DagsHub interfaces have different URLs and purposes. Changing a Git
remote does not automatically update bucket/DVC/MLflow URLs. A DVC token entry
also does not automatically supply the credentials inside running containers.

## Model training and loading

`src/config.py` supplies path conventions, model settings, and model names.
`MODEL_TYPE` selects standalone training; `PREDICT_MODEL_TYPE` independently
selects serving. The Airflow training DAG explicitly chooses KNN.

| File | Responsibility |
|---|---|
| `src/train_service/train_model.py` | Select model module, create an MLflow run, export/register its bundle |
| `src/train_service/prepare_data.py` | Read/check matrices, align IDs/genres, prepare optional two-tower examples |
| `src/train_service/models/knn.py` | Fit nearest-neighbor retrieval over movie genre rows and export catalog/schema |
| `src/train_service/models/two_tower.py` | Optional user/movie embedding model, trained on cosine pseudo-labels |
| `src/train_service/evaluate.py` | Optional two-tower approximation metrics |
| `src/train_service/check_data.py` | Input/feature validation and diagnostics |
| `src/predict_service/model_loader.py` | Resolve a registry selector, download the bundle, validate metadata, expose inference |

A registered bundle includes the estimator and the catalog/schema needed to map
its returned row positions back to actual movie IDs. Reordering features or
using mismatched IDs can silently change recommendations, so loading checks the
bundle's expected schema. Prediction resolves a version/alias once at startup
and retains it in memory. Moving an alias does not hot-swap a running process;
replicas must restart or be recreated.

KNN has no held-out recommendation-quality score here. Two-tower metrics measure
approximation of genre-similarity pseudo-labels; splitting by users does not turn
those labels into future-preference ground truth. There is no Recall@K/NDCG
quality gate in the deployment path. See the [training limitations](../src/train_service/readme.md#evaluation-limitations).

## Serving and authentication

Nginx publishes host port 8000 and balances requests across three `predict`
containers. The collection API publishes port 8001. APIs share the authentication
module in `src/user_authentication_service/auth.py`.

- `POST /token` exchanges an API account password for an expiring signed JWT.
- Prediction access requires a permitted role; ingestion requires an admin.
- `/model` reports the loaded numeric registry URI and model type.
- `/predict` accepts MovieLens profile IDs and an optional seed. It samples ten
  distinct movies from the nearest candidate pool; already-rated movies are not
  explicitly excluded by this implementation.
- Prediction metrics are available at `/metrics` for Prometheus.

API accounts are distinct from numeric profile IDs, DagsHub accounts, and the
Airflow/Grafana dashboard accounts. Replicas share `AUTH_SECRET_KEY`, allowing a
token issued by one replica to work on another. If no private user file exists,
the committed demo accounts are used. They are documented demo values, not
production access controls.

## Orchestration and deployment

Airflow uses LocalExecutor: the scheduler runs task processes and contacts the
host Docker socket to start sibling containers. There is no separate Celery
worker/Redis stack. The DAG processor imports DAG files independently of task
execution. Avoid network calls or credential resolution while parsing DAGs.

| File | Responsibility |
|---|---|
| `airflow/setup_local.py` | Fill missing local Airflow keys/passwords and a daemon-visible project path |
| `airflow/dags/data_ingestion.py` | Weekly/manual upload and successful `rating_parts` event |
| `airflow/dags/rating_assets.py` | Shared event identity linking the ingestion producer and training consumer |
| `airflow/dags/training_pipeline.py` | Preparation → KNN training → deployment, with one active run/task |
| `airflow/dags/prediction_deployment.py` | Validate candidate, manage alias, restart/verify serving, attempt rollback |
| `airflow/dags/infrastructure_smoke_test.py` | Manual scheduler/infrastructure check without model or bucket access |

A successful ingestion emits an event; skipped/failed tasks do not. The training
consumer must be unpaused to react. New DAGs are paused by default, while existing
metadata can retain unpaused schedules across ordinary restarts.

The training task returns its exact registered model name, numeric version, and
run ID via XCom. The deployment helper does not simply deploy whichever model
happens to be “latest.” It validates a temporary candidate against the specified
version, promotes a named alias, restarts serving replicas, reloads Nginx, and
checks each replica plus the public routing path.

Rollback is best-effort for handled errors. For an existing deployment it uses
the version actually served before rollout, which may differ from a manually
moved alias. It does not roll back historical processed CSVs and cannot guarantee
recovery after host loss or forced task termination. The replicas can be briefly
unavailable during restart. See the [detailed recovery flow](../airflow/README.md#exact-version-prediction-deployment).

## Docker names and lifecycle

- Compose project: `movie-recommender-mlops`. Generated serving-container names
  follow `<project>-<service>-<replica>`.
- Local application image tags use `movie-recommender-mlops-…`; training/prediction
  use a model-type tag such as `knn`, while Airflow/preparation use `local`.
- Airflow preparation/training jobs get explicit names containing project, task,
  run ID, and attempt. The temporary prediction preflight container is created
  separately by the Docker SDK and may still receive a generated name.
- Third-party image names (`postgres`, `nginx`, `grafana/grafana`, etc.) stay intact.
- Preparation/training image tags may show “unused” between task executions.
  Retain them unless you intend to rebuild before the next run.

A Compose restart retains its original container environment. Use recreation to
apply changed `.env` values; use rebuild plus recreation for code copied into an
image. Bind-mounted DAG source is reparsed without rebuilding the Airflow image.

## Monitoring and tests

Prometheus discovers/scrapes prediction replicas directly rather than through
Nginx. Request status totals include authentication failures; the handler counter
counts only requests that reach prediction logic. Summing both would double count
successful traffic. Grafana is provisioned from repository dashboard files.
Counters reset per process; named volumes retain the monitoring database history.

Operational monitoring here does not implement drift detection or automatic
retraining. The offline test suite checks auth, ingestion logic, registry loading,
and prediction behavior using mocks and a temporary local SQLite registry. CI's
Compose workflow adds a separate disposable server-backed integration scenario.
Neither suite establishes recommendation ranking quality, and current CI results
must be checked on the actual Actions run rather than inferred from the files.
