# Offline recommendation evaluation

This backtests the existing genre KNN against later ratings. It does not change
the learning algorithm, API, serving alias, prepared production files, or model registry
records. The live model already consumed later data, so it is **not** used to
claim a leakage-free historical score. A fresh local KNN index is reconstructed
from the historical inputs using the production `train_model` function.

## Fixed development protocol

- History: chronological parts 1–6. Holdout: part 7. Part 8 remains the final test.
- User activation: at least 20 pre-cutoff rating events matching known movie
  metadata. This is configurable; it is an evaluation cohort rule, not a new API
  activation gate. Rating events, including duplicates if present, contribute to
  the same arithmetic genre mean as production; star values do not weight profiles.
- Cutoff: minimum timestamp in the held-out part. Training events at that exact
  timestamp are removed. Later training timestamps or overlapping/out-of-order
  parts are rejected. Eight ingestion parts are not inherently train/test splits;
  the evaluator makes this boundary explicit.
- Catalog: only movies observed in pre-cutoff interactions. Metadata genres are
  assumed stable. No release-time availability field exists, so this is a
  conservative historically-observed-item evaluation, not new-item generalization.
- Relevant targets: distinct future movies rated at least 4, present in that
  historical catalog, and not rated by that user before the cutoff. Repeated
  future ratings count once.
- Users without enough history are outside personalized evaluation. No cold-user
  fallback is added. Users without eligible future positives have no defined
  recall denominator and are counted separately, rather than assigned fake scores.
- The default sample is 1,000 history-qualified returning users, selected with
  seed 42 **before looking at positive labels**. Use `--max-users 0` for the full
  qualified-returning-user cohort. Coverage and exclusions accompany every score.

Chronological input ranges and read stability are checked. Inputs are read in
chunks; only selected users' histories/profiles are materialized. Source sizes,
SHA-256 hashes, evaluator checksum, protocol, library versions, and Git state are
recorded. Historical CSVs must remain unchanged while evaluation runs.

## Four explicitly named methods

| Method | Policy |
|---|---|
| `genre_knn_serving` | Raw mean genres; production Euclidean ball-tree KNN; sample ten from top twenty with the API's per-user seed. Already-rated movies remain eligible. |
| `popularity` | Top ten by number of positive training ratings, ties by movie ID. Already-rated movies remain eligible. |
| `genre_knn_unseen` | Diagnostic: remove previously rated movies before constructing the nearest candidate pool, then use the same seeded sampling. |
| `popularity_unseen` | Diagnostic: training-only popularity after removing previously rated movies. |

All use the same historical catalog, scored users and relevance definition.
The unseen variants are counterfactual policy changes, not claims about current
API behavior. Popularity is a standard ranked baseline; it does not imitate
KNN's random top-20 sampling. The API returns sampled movies in seeded-random
order, so KNN NDCG evaluates that returned order, not a learned ranking.

## Metrics and denominators

Metrics are macro-averaged over scored users with eligible future positives:

- Recall@K: hits divided by the user's number of relevant held-out movies.
- Precision@K: hits divided by K; missing recommendation slots get zero credit.
- Hit rate: fraction of scored users with at least one hit.
- NDCG@K: binary hits discounted by `1/log2(rank+1)`, divided by the ideal gain
  for `min(K, relevant_count)`. See the [scikit-learn metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).
- Seen fraction and catalog coverage describe repeated recommendations and how
  much of the available catalog is used.

No-positive cohorts produce an explicit `no_evaluable_users` report with null
quality scores. These are conditional retrospective scores. Missing exposure,
self-selection into rating, one seed/sample, and one time window limit any claim
about user satisfaction or online performance. This is not a deployment quality
gate and does not retrofit quality approval onto an existing registered model.

## Run

Use original ratings with `userId,movieId,rating,timestamp` and raw movies with
`movieId,title,genres`. Processed genre matrices alone have lost the information
needed for temporal evaluation. Obtain the original split files through the
[restore guide](RESTORE_AND_RUN.md); the evaluator never downloads or rewrites them.

```bash
uv sync --locked
uv run --offline --frozen python -m src.evaluation.recommendations \
  --ratings-dir data/interim/ratings_parts \
  --movies-csv data/raw/movies.csv \
  --train-through 6 --test-part 7 --min-history 20 \
  --positive-rating 4 --max-users 1000 --seed 42 \
  --output reports/recommendation_evaluation/development
```

Alternatively, `make evaluate-recommendations` uses these development defaults.
Override `EVALUATION_RATINGS_DIR`, `EVALUATION_MOVIES_CSV`, and `EVALUATION_OUTPUT`
for another local source location. No `.env` or DagsHub token is required by the
evaluator. Dependencies must already be installed for the offline command.

Output: `report.json` contains the full protocol/provenance/coverage, `per_user.csv`
contains auditable recommendation IDs and scores, and `summary.md` is readable.
Generated evidence is ignored by Git; preserve or explicitly review/share a
summary. Output directories must be new/empty to avoid replacing previous runs.

For final evaluation, freeze the model and protocol using development results,
then use `--train-through 7 --test-part 8 --final`. The CLI requires explicit
acknowledgement for part 8; it cannot prevent a human from tuning on that result.
Do not call part 8 untouched after running it, or repeatedly optimize against it.
The checked-in [development result](EVALUATION_DEVELOPMENT_RESULT.md) uses part 7.

## Code and tests

`src/evaluation/recommendations.py` is a separate streaming CLI and metric module.
The tests check hand-calculated/Scikit-learn NDCG agreement, future-label isolation,
chronological rejection and boundary ties, cold/insufficient-history counts,
deduplicated relevance, future catalog exclusion, chunk-size/seed repeatability,
input mutation detection, and protection of the final holdout.

```bash
uv run --offline --frozen pytest -q tests/evaluation
```

A small score is a useful finding, not a reason to silently alter the protocol.
Any later changes to profiles, exclusions, or the model should be compared on the
same development split and documented as a new experiment.

## Upload metrics to MLflow

After evaluation, explicitly upload the saved report (no recomputation required):

```bash
uv run --offline --frozen python -m src.evaluation.tracking \
  --report-dir reports/recommendation_evaluation/dev-1-6-v7-verified
```

This loads the project `.env` for MLflow credentials and writes exclusively to
personal DagsHub, experiment `movie-recommendation-evaluation`. Each invocation
creates a new run; inspect `mlflow_upload.json` before retrying an interrupted upload.
Metrics use `<method>.<metric>` names (for example `genre_knn_serving.recall`)
and `coverage.<count>` names; quality values are fractions, not percentages.
Null quality scores are omitted. The run includes protocol parameters, source
commit/checksum/dirty-state tags, aggregate `report.json` and `summary.md` artifacts.
Per-user CSV data is not uploaded. The uploader verifies metrics, finished status,
and artifact listings remotely before saving the local run receipt. Upload errors
are surfaced; the original local evaluation remains available. No model is
registered, promoted or deployed, and these scores are not attributed to version 72.
