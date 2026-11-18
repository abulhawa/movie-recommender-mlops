# Model versions own their serving data

New training runs register a self-contained bundle:

```text
model/
  MLmodel, estimator files, training_schema.json
  serving_data/
    movie_matrix.csv
    user_matrix.csv
    manifest.json
```

Before fitting, training copies both processed CSVs into a private temporary
snapshot. The implementation loads that snapshot, and the same files are uploaded
with its exported model before registration. The manifest records file sizes,
SHA-256 hashes, a combined data fingerprint, and a sample user ID. The model schema
records the expected fingerprint. Loading verifies those hashes before serving.

Prediction uses the bundle's user matrix rather than the mutable host CSV. Its
`/model` response includes `data_fingerprint` and `sample_user_id` for a new bundle.
A host `USER_MATRIX_PATH` override is rejected for these bundles, avoiding silent
mixing. The override remains available for legacy models and isolated tests.

This does not change KNN or two-tower learning/recommendation algorithms. It changes
which input snapshot is recorded and loaded. It adds copies of the two processed
CSVs per registered version, not the full raw ratings dataset. Retention/storage
cost must be considered before retaining many versions indefinitely.

## Deployment and rollback

The training result passed to Airflow now includes the exact model version,
fingerprint, and a profile ID from that snapshot. Candidate and final serving
checks verify all of these, not just whether an API responds.

For an existing deployment, the task records the previous model's fingerprint
and its own sample user ID. After a handled post-promotion failure, it restores
the previous alias/version and restarts replicas. Those replicas load the previous
version's profiles from its artifact. Rollback probes use the previous snapshot's
user ID; a new user may not exist in the old snapshot.

The shared host CSVs are not rewritten by rollback. They can retain the newest
prepared data, while the restored serving bundle remains internally consistent.
Preparation still must not run concurrently with snapshot capture: the two source
CSV files are not committed atomically as a pair. Checksums establish the exact
captured inputs, not the absence of upstream races or recommendation quality.

## Existing models need a one-time transition

Old registered versions have no saved historical user matrix. They remain usable
for standalone serving with the existing host matrix. They do not acquire a
historical snapshot simply because the application code is upgraded.

Managed deployment refuses a legacy serving baseline before moving an alias or
creating a candidate. It cannot safely promise restoration of data that was never
saved. A training job built from an older image also lacks the new handoff fields
and is rejected; rebuild the images first.

### Transition the personal demo deliberately

This is a manual one-time transition, not a guarantee of automatic rollback to a
legacy model. Keep a backup of useful current data and record the current model
selection. Use only the personal registry and do not run another DAG/preparation
job concurrently.

1. Build the updated images:

   ```bash
   docker compose build train predict
   ```

2. With valid processed matrices, register a new bundle:

   ```bash
   docker compose run --rm train
   ```

   This writes a real model version to your configured registry. Note the printed
   version and fingerprint; it does not replace currently running replicas.
3. In the personal DagsHub model registry, select an owned serving alias
   for this new version (for example `live`). Set
   `PREDICT_MODEL_VERSION=@live` in `.env`. Preserve the API signing key and
   login settings; all replicas and the scheduler must use the same alias/key.
4. Recreate prediction and the scheduler; then refresh Nginx's upstream addresses:

   ```bash
   docker compose up -d --force-recreate --no-deps --wait predict airflow-scheduler
   docker compose exec nginx nginx -t
   docker compose exec nginx nginx -s reload
   ```

   These commands can briefly interrupt serving. They retain volumes and do not
   change API passwords. Nginx must already be running for the `exec` commands.
5. Authenticate and check `/model`: all three replicas should report the new
   numeric version and matching fingerprint. Confirm a request with the returned
   `sample_user_id` succeeds. Keep legacy versions as historical records, rather
   than treating them as data-safe automatic rollback targets.
6. Subsequent Airflow training/deployment runs can roll between bundled versions.
   Before triggering them, run:

   ```bash
   uv run python airflow/setup_local.py --check --deployment
   ```

## Personal transition completed — 18 November 2026

The personal registry transition was completed with `movie-genre-knn` version 72
and alias `live`. The temporary candidate served predictions without a
host processed-data mount. All three live replicas and Nginx then reported the
same model/data fingerprint and passed a seeded prediction request. Scheduler
Docker access was verified and the previous DAG pause settings were restored.
The local `.env` now selects `PREDICT_MODEL_VERSION=@live`; update any
private configuration backup accordingly. This is a dated acceptance checkpoint,
not a promise that the alias will always refer to version 72.

## Verification scope

Offline tests register tiny bundles in SQLite, change the host inputs, promote
and move an alias back, and verify the older profiles are loaded. Simulated
Docker/HTTP failures check rollback uses the previous snapshot's valid profile,
and that legacy baselines are rejected before promotion. These do not replace
an acceptance run against your personal DagsHub and Docker deployment.
