# Interaction-trained retrieval: ALS and hybrid two-tower

The main research candidate is a hybrid collaborative/content-aware two-tower.
Implicit ALS is the competitive collaborative baseline. The deployed genre KNN
and its existing Airflow, registry names and rollback workflow remain unchanged.
The previous `models/two_tower.py` is retained as the historical genre-cosine
pseudo-label experiment; it is **not** this interaction-trained model.

## Architecture and supervision

User tower inputs:

- Learned user-ID embedding.
- Signed, rating-weighted mean of learned movie-ID embeddings for up to 32
  strictly earlier rated movies. Padding is zero; the target movie is excluded.
- The same history weights applied to movie genres as optional-content preferences
  (included by default). Rating weight is `(rating - 3) / 2`; neutral ratings have
  zero feature weight, low ratings contribute negatively, and rated items always
  enter the seen-item exclusion set.

Movie tower inputs:

- Learned movie-ID embedding and binary genres.
- Optional release year parsed from the title, standardized using historical
  catalog metadata, with a missing-year indicator (`--use-year`).

Separate MLPs project these inputs into normalized 32-dimensional vectors.
Training compares positive and negative movie scores using BPR:

`softplus(-(user · positive - user · negative) / temperature)`.

Temperature is 0.1. Dense tower/movie parameters use AdamW at learning rate
0.001 with weight decay 0.0001. User IDs use SparseAdam at learning rate 0.001
with sampled-user L2 penalty 0.0001, avoiding full-table updates per batch,
and the hidden layer has 64 units. Ratings >=4 supply positives; their loss
weight is `(rating - 3) / 2`. No synthetic genre-cosine targets are used.

For each positive, sample an explicitly low-rated movie (rating <=2) with 50%
probability when one exists; otherwise sample uniformly from historically
unobserved movies. Unobserved comparisons receive weight 0.25, explicit dislikes
weight 1. All historical rated movies are excluded from the unobserved pool;
future labels are never consulted. This is still an exposure assumption:
unobserved does not mean disliked. BPR loss values are not calibrated probabilities.
Users with neither an unobserved movie nor an explicit dislike cannot supply a
valid comparison and are omitted from BPR training.

At inference, encode the most recent historical context once, score the entire
historical catalog by dot product, exclude **all** previously rated movies, then
return the highest scoring ten. Ties use movie ID; no random top-20 sampling.

## ALS baseline

`models/als.py` uses the CPU implementation from `implicit`. A CSR matrix has
users as rows and movies as columns. Ratings >=4 become positive confidence
`1 + 20 * (rating - 4 + 1)`. Ratings <=2 become negative confidence
`-(1 + 20 * (2 - rating + 1))`. Intermediate ratings have no preference target,
but remain history and seen exclusions. Missing interactions carry the library's
low-confidence zero preference. Defaults: 32 factors, 10 iterations,
regularization 0.1, seeded CPU training.

This preserves the distinction between observed dislike and missing feedback.
It does not claim that implicit ALS models the complete ordinal star-rating scale.

## Temporal protocol and comparison

- Default training: parts 1–6; development: part 7. Part 8 is guarded by `--final`.
- Cutoff is the minimum development timestamp. Training ties at the cutoff are
  removed; later training events and out-of-order parts are rejected.
- Duplicate user/movie events use their latest historical rating. Equal-timestamp
  history events cannot reveal a target; training contexts contain earlier events.
- IDs, catalog, genres, confidence values, and histories come from historical data.
  Genre metadata is assumed stable. External current popularity/tag-genome values
  are not imported into this historical benchmark.
- Evaluate established returning users with at least 20 raw historical rating
  events. Sample up to 1,000 before filtering for eligible positive targets.
- Score distinct future ratings >=4 for unseen, historically available movies.
  Report cold users, insufficient histories, out-of-sample users, excluded future
  positives, and users without evaluable positives separately.
- ALS, two-tower, popularity and genre KNN share the same cohort, catalog, target
  definition and all-rated-item exclusion. The KNN comparator retains seeded
  sampling from its top-20 unseen pool; it uses the deduplicated histories here.
  Consequently, these figures should not be directly substituted into the older
  event-mean KNN report.
- Primary metric: NDCG@10; supporting: Recall@10, Precision@10, hit rate and catalog
  coverage. Catalog coverage measures exposure breadth, not semantic diversity.
- Latency p95 measures sequential local warm scoring/filtering/sorting; it excludes
  HTTP, authentication, tower encoding and concurrent load. It is **not** an API
  load-test result. Compressed bundle bytes are reported, not a RAM measurement.

The CLI uses fixed epochs and never selects a checkpoint on part 7. Any tuning
should use an earlier inner chronological window (e.g. parts 1–5 versus 6), then
refit on 1–6 before reporting part 7. Keep part 8 untouched until model/protocol
selection is frozen. Report capped pilot runs as pilots, not full-data benchmarks.

## Run locally

```bash
uv sync --locked --group retrieval
OPENBLAS_NUM_THREADS=1 uv run --offline --frozen --group retrieval \
  python -m src.train_service.train_retrieval \
  --ratings-dir data/interim/ratings_parts --movies-csv data/raw/movies.csv \
  --output reports/recommendation_evaluation/interaction-development
```

All inputs are read-only. Output must be new/empty. Defaults train on all historical
users and all positives. A bounded pilot adds `--max-train-users 10000 --max-pairs
200000`; user sampling is seeded before looking at future labels, and the positive
cap applies per epoch. Candidate catalog/cohort shrink with a user cap, so compare
models within that experiment. No learning-quality claim follows from a smoke test.

Use the `retrieval` dependency group explicitly; ordinary tests/builds remain
offline and the API image does not acquire Torch/ALS dependencies. CPU Torch uses
an explicit package index scoped to Torch. Implicit 0.7.3 has a compatible wheel
for this Python 3.12 environment; 0.7.2 attempted a source build without a compiler.

## MLflow and frozen retrieval artifacts

Add `--log-mlflow` to log/register candidates in the owner's personal DagsHub only.
Experiment: `movie-interaction-retrieval`; registry names:
`movie-interaction-als` and `movie-interaction-two-tower`.
The command logs protocol, library versions, input/source hashes, metrics, training
loss history and aggregate reports. Per-user recommendation CSVs stay local.
The two-tower bundle also retains neural weights and architecture settings.
Only the minimal serving-wrapper source is packaged, excluding unrelated app files.
A local receipt records run IDs before upload and verification state afterward;
inspect it before retrying an interrupted operation to avoid duplicate training.

Each pyfunc model bundles original ID mappings, frozen user/movie vectors and
seen-item exclusions. Its input is a DataFrame with `user_id` and optional `k`;
output has `user_id` and JSON-encoded `recommendations`. Unknown users produce an
explicit error. Registry readback compares loaded recommendations with the local
bundle. Frozen embeddings require re-encoding/retraining after new ratings; this
is an experimental serving contract, not integration into the existing API.

To upload a completed local experiment without recomputing training:

```bash
uv run --offline --frozen --group retrieval python -m src.train_service.train_retrieval \
  --output reports/recommendation_evaluation/interaction-development --upload-only
```

The upload refuses to start if a receipt already exists; inspect the recorded runs
instead of blindly retrying an ambiguous registration.

No serving alias is changed and no model is automatically promoted. Extending
Airflow/API selection requires measured eligibility and support for this new
artifact contract; a better offline score alone is insufficient.

## Verification

```bash
uv run --offline --frozen --group retrieval pytest -q tests/retrieval
```

Tests check temporal isolation, future-label isolation, target exclusion, signed
rating history, valid negative pools, actual preference learning, repeatability,
ALS matrix orientation, unseen top-K ranking, cold-user handling, stable ties,
final-holdout protection and serving-bundle round trips.

References: [BPR paper](https://arxiv.org/abs/1205.2618),
[implicit ALS confidence semantics](https://benfred.github.io/implicit/api/models/cpu/als.html),
[TensorFlow retrieval architecture](https://www.tensorflow.org/recommenders/examples/basic_retrieval).
