# Restore and operate the personal project

Run commands from the repository root in a Linux/WSL shell unless marked
otherwise. Docker Engine must see the same bind-mount paths used by Compose.
Docker Compose must support `include` and `--wait`. On Windows, use Docker
Desktop's WSL integration or follow the daemon-path notes in the Airflow guide.

This guide describes the current configuration. A complete restore on a separate
fresh machine has not yet been performed as part of this documentation update.

## 1. Clone the personal repository

```bash
git clone https://github.com/abulhawa/movie-recommender-mlops.git
cd movie-recommender-mlops
uv sync --locked
```

GitHub access is required if the personal repository is private. A checkout of an
old historical branch can contain old endpoints and Docker names; use current
`main` for the personal configuration. Do not push or log experiments to the
original `rosneru` repositories.

## 2. Restore credentials before starting Docker

If you saved the real file, retrieve **movie-recommender-mlops — .env backup**
from your password manager and restore it as `.env` here. The reminder in
`.env.example` identifies the suggested entry; it does not mean the backup has
already been created or verified.

If no saved file is available, copy `.env.example` to `.env` only when `.env`
does not already exist. Use an editor to fill the required values. Never paste
tokens into Git, issue comments, screenshots, or terminal commands that will be
saved in shell history. Both `.env` and `.dvc/config.local` must remain ignored.

| Settings | How to obtain them |
|---|---|
| `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_USERNAME` | Your current personal DagsHub token and username |
| `DAGSHUB_S3_ENDPOINT_URL`, `DAGSHUB_BUCKET`, `MLFLOW_TRACKING_URI` | Personal defaults in `.env.example`; keep them pointing to `abulhawa/movie-recommender-mlops` |
| `GRAFANA_ADMIN_*` | Restore existing values or let the helper generate a missing password |
| `AIRFLOW_FERNET_KEY`, `AIRFLOW_JWT_SECRET`, `AIRFLOW_POSTGRES_PASSWORD`, `AIRFLOW_ADMIN_*` | Restore existing values, or generate missing values using the helper below |
| `AUTH_SECRET_KEY` | Restore the shared API signing key; the helper generates it only when missing |
| `DATA_COLLECTION_*` | An existing API admin account; needed for the ingestion DAG |
| `PREDICT_SMOKE_*` | An existing prediction account; needed for the deployment task |
| `MODEL_TYPE`, `PREDICT_MODEL_TYPE` | Keep `knn` for the automated training/deployment demo |
| `PREDICT_MODEL_NAME` | May be empty; defaults to the model name for the selected type |
| `PREDICT_MODEL_VERSION` | `latest` or a numeric version for standalone serving; an alias such as `@live` for Airflow deployment |

Fill missing public defaults/local secrets and validate the configuration:

```bash
uv run python airflow/setup_local.py
```

The helper preserves nonempty secrets and selectors, fills public example defaults,
and generates missing Airflow settings, the API signing key and Grafana password.
It repairs the Docker group when the socket is accessible. It exits nonzero with
variable names/guidance if private account settings or machine paths still need
attention. It never prints values, contacts APIs, starts containers, or changes an
initialized database password. API account passwords must match existing users.

Use `uv run python airflow/setup_local.py --check` for read-only validation. Add
`--deployment` to require KNN and an explicit serving alias. Ordinary setup accepts
standalone `latest`/numeric selectors and either supported model type.

### Machine-specific settings

Check `AIRFLOW_DOCKER_PROJECT_DIR` after moving/restoring the project: a saved
nonempty path will not be corrected by the setup helper. It must be the repository
path visible to Docker Engine, not `/opt/airflow` inside the scheduler. The helper
uses the checkout path on Linux and the Docker Desktop host-mount form on Windows.

On Linux/WSL, read the socket group with:

```bash
stat -c '%g' /var/run/docker.sock
```

Setup detects this number and updates a stale value when possible. If detection
is unavailable, set `AIRFLOW_DOCKER_GID` manually. Do not assume `0` or reuse a
number from a different machine. Group access applies when the scheduler is created.

If reusing existing volumes, preserve their PostgreSQL credentials and Airflow
Fernet key. Editing `.env` does not change an initialized database's account
password or re-encrypt stored values. Grafana and Airflow dashboard accounts also
persist independently of later edits to first-start settings.

## 3. Restore DVC access and processed data

The DVC remote is defined in `.dvc/config`; the token is stored separately in
`.dvc/config.local`. Saving `.env` alone does not automatically configure DVC.
In an editor, create/update `.dvc/config.local` with this structure, replacing
both placeholders privately with the current DagsHub token:

```ini
[remote "origin"]
    access_key_id = YOUR_DAGSHUB_TOKEN
    secret_access_key = YOUR_DAGSHUB_TOKEN
```

Verify the files are ignored, test storage access, and restore cached outputs:

```bash
git check-ignore .env .dvc/config.local
uv run dvc status -c
uv run dvc pull
```

A status listing differences can still mean authentication succeeded; 401/403
means the credentials or permissions need checking. `dvc pull` uses the selected
Git revision's tracked outputs. Training and legacy serving need restored processed matrices. New bundled models
download their user profiles with the model instead of using the host CSV. DVC
access alone does not prove registry access. See [versioned bundles](VERSIONED_SERVING_DATA.md).

The normal DVC pipeline marks raw data and chronological source splits
`cache: false`, so do not expect an ordinary pull to recreate those local source
folders. The personal preservation also retained additional data objects; see
[PRESERVATION.md](../PRESERVATION.md) for that snapshot and its historical gaps.

If demonstrating new uploads from the collection API, reconstruct its local
raw inputs and rating parts separately:

```bash
uv run dvc repro split_data
```

This can download the original input dataset and use substantial disk/RAM.
It is not necessary just to serve an already registered model using restored
processed matrices.

## 4. Choose the demo and start

### Show an existing model

Ensure the selected registry version/alias exists. New bundles validate and use
their own profiles; legacy models require compatible host matrices. Startup resolves that selector once; it does not keep polling the registry.
Keep ingestion and training DAGs paused if no new versions should be created.

```bash
docker compose config --quiet
make start-project
docker compose ps
```

`make start-project` is `docker compose up -d --build`. It assumes the preceding
settings/data steps are complete. It builds the ordinary services, not the
profiled preparation/training task images. An unpaused or queued Airflow DAG may
still execute after the scheduler starts.

Initialization order is PostgreSQL healthy → `airflow-init` exits 0 → Airflow
runtime services start. The init container is a successful one-off job, not a
failed long-running service. Prediction replicas become healthy before Nginx
starts. Allow startup time for the model download and health checks.

### Demonstrate preparation, training, and deployment

Read the [Airflow run and recovery guide](../airflow/README.md) first. Use KNN
and a serving alias such as `@live`; if serving already exists, all replicas and
the scheduler must be configured for the same alias and API signing key.

```bash
docker compose build data-preparation train predict
```

The bucket must contain `movies.csv` and contiguous rating parts starting at 1.
Triggering `training_pipeline` creates a registered model version and can move
the serving alias/restart prediction. Uploading through an unpaused
`data_ingestion` DAG can trigger it automatically. Do not run DVC/preparation
commands concurrently with a training/deployment run; they share processed CSVs.

## 5. Verify the demo

- `docker compose ps`: long-running services should be up, with configured health
  checks healthy; check `ps -a airflow-init` separately for exit 0.
- Airflow: http://localhost:8081; check task logs and the actual run state.
- Prediction: http://localhost:8000/docs; authenticate, call `/model`, and verify
  the resolved numeric model version. Choose a user profile from the current
  `user_matrix.csv`, then call `/predict` with a seed for a repeatable example.
- Prometheus: http://localhost:9090/targets; expect each prediction replica to be
  scraped directly.
- Grafana: http://localhost:3000; generate traffic using the password-prompting
  [demo script](../scripts/demo_prediction.py), then allow a scrape interval.

A healthy API or successful rollout is a functional check. It does not establish
recommendation ranking quality. Keep that distinction clear when presenting.

## Configuration updates and token rotation

After changing the DagsHub token, update both `.env` and `.dvc/config.local`.
Recreate affected running containers to reload Compose environment values:

```bash
docker compose up -d --force-recreate --no-deps airflow-scheduler data-collection predict
```

This can briefly interrupt prediction. Changing only a source file under the
bind-mounted `airflow/dags/` is different: allow the DAG processor to reparse it.
Code copied into application images requires a rebuild and container recreation.
`docker compose restart` does not reload new `.env` values.

For a renamed project, keep the Compose project, image tags, DAG image references,
deployment project constant, and bucket/registry URLs in agreement. Docker
Desktop may hide the common project prefix in its grouped container view. A
Docker image tag and a running container's name are separate identifiers.

## Troubleshooting

| Symptom | Check |
|---|---|
| Docker socket permission denied in an Airflow task | Compare socket group with `AIRFLOW_DOCKER_GID`, then recreate the scheduler |
| DVC status/pull returns 401 or 403 | Token in both DVC credential fields and access to the personal repository |
| Training cannot reach/register with MLflow | Scheduler URI, username, token; registry permissions and network access |
| Airflow cannot connect to PostgreSQL after `.env` changes | Password retained by the initialized database volume |
| Deployment rejects `latest` | Use the serving alias required by the deployment task, consistently across replicas/scheduler |
| API login succeeds but profile returns 404 | MovieLens profile ID must exist in the loaded user matrix |
| Old image name still appears in Docker Desktop | Existing container records its creation-time image reference; recreate after checking tags/config |
| Preparation/training image says “unused” | Expected while no one-off task container exists; retain it for the next DAG run |

## Stop, archive, and clean up

A normal stop preserves data:

```bash
make stop-project
```

Do not use `docker compose down --volumes` for a routine stop. It removes this
project's named volumes: Airflow metadata/logs, Grafana settings, and Prometheus
history. It does not remove Git files or remote DagsHub records. Avoid global
Docker prune commands when other projects share the daemon.

### Before deleting the local copy

1. Review `git status`, commit intended source changes, and push `main` to both
   personal remotes. Preserve original authorship/license and attribution.
2. Save and verify the real `.env` in your password manager. The reminder in
   `.env.example` is only a reminder. Treat `.env.*.bak` files as sensitive too.
3. Preserve any custom `users.json` separately; it contains the user records
   needed to reproduce non-demo API logins. Regenerate/revoke service tokens as
   appropriate rather than committing them.
4. Ensure important processed data is versioned/uploaded with DVC. Preparation
   overwrites local CSVs but does not itself update `dvc.lock` or push DVC data.
   Do not overwrite a useful DVC revision just because a newer bucket exists.
5. Keep the independent preservation bundle/snapshot if historical branches or
   older artifacts matter. Some historical outputs were already unavailable.
6. If keeping local Airflow/Grafana history matters, back up their Docker volumes
   separately. Git and DagsHub do not contain those volumes.
7. Ideally rehearse restoration in a different folder/machine before deleting
   the working installation. This guide does not replace that acceptance test.

GitHub issues, Actions logs, Docker volumes, private configuration, and Git history
are different kinds of state. A Git clone alone does not reproduce all of them.
