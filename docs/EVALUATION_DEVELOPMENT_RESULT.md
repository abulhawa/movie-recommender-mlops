# Development evaluation — 18 November 2026

This is a retrospective development-window result, not a score for live model version 72.
The algorithm and serving policy were reconstructed at the historical cutoff; no model was registered or deployed.

## Protocol and coverage

- History: parts 1–6; holdout: part 7; cutoff: `2008-11-02T16:12:04+00:00`.
- Minimum history: 20 earlier rating events. Positive target: rating ≥4, unseen and in the historical catalog.
- Held-out users: **20,587**. Without earlier history: **15,321**; below minimum: **172**.
- Qualified returning users: **5,094**; uniformly sampled **1,000** with seed 42 before positive-label filtering.
- Scored users: **937**; sampled users without eligible positives: **63**.
- Available catalog: **10,361** movies. Sampled positive rows outside this historical catalog: **12,574** of **40,779**.
- Part 8 was not read or used.

## Results

| Method | Recall@10 | NDCG@10 | Precision@10 | Hit rate |
|---|---:|---:|---:|---:|
| genre_knn_serving | 0.025% | 0.169% | 0.160% | 1.174% |
| popularity | 2.445% | 5.758% | 4.792% | 26.574% |
| genre_knn_unseen | 0.157% | 0.315% | 0.245% | 2.241% |
| popularity_unseen | 4.716% | 12.146% | 10.171% | 43.436% |

Popularity performs substantially better on this fixed development protocol. The genre-only KNN should not be described as accurate merely because its deployment checks pass. Excluding already-rated movies improves the measured result but is not currently implemented in the API.

Scores are macro averages on the 937 scored users, not estimates for all 20,587 held-out users. Sampling, rating/exposure bias and the historical-catalog restriction matter. These results do not establish online satisfaction or a final test score.

## Reproduction evidence

- Evaluator SHA-256: `0001e37416cc730cb481ab38ae8b5e679d019e158264626dd6ca8ff0767d0873`.
- Library versions: `{"numpy": "2.5.4", "pandas": "2.3.3", "python": "3.12.14", "scikit-learn": "1.4.2"}`.
- The evaluator was uncommitted when run; the source checksum records the executed implementation. Input files and protocol are identified below.

| Input | SHA-256 |
|---|---|
| movies.csv | `eed80dd12706c10362fea572a248f625cee3cfd3dba05086158314ac69d78fac` |
| ratings_01.csv | `af618f6221185dc2fdc827eec7c1ff97f801ea852ff774219e29439c428d0e8f` |
| ratings_02.csv | `218222cd560cfa8a18fa8c58632cde4c029afde55ab52406b39e43209f6e5584` |
| ratings_03.csv | `a95ff5fce0044338a6c966a64d609c16e271cf53663545f6687c352b5a1d32bd` |
| ratings_04.csv | `4e832893332a0e7b1d2991ad48f3e658b2308183e582eb4311ba2397442b8a8d` |
| ratings_05.csv | `8e8d09ce31bbbe360e4790e6aa658de0ed69decb6226fa11dab4a89188846374` |
| ratings_06.csv | `69c9f09bc142cb282fa0c7c8462d5def8bb592e1d9511b4ec13f44b0e65cf45a` |
| ratings_07.csv | `a59f1df6ce3d9b042beb11f917831b6e32534a6181d30d9f52604b4041f483eb` |

Generated full reports are under `reports/recommendation_evaluation/dev-1-6-v7-verified/` in the working installation and are ignored by Git. The pilot and final input-stability-checked run agreed exactly on scores and coverage. See [protocol and commands](RECOMMENDATION_EVALUATION.md).

## Verified MLflow upload

The saved development report was uploaded on 18 November 2026 to the personal
`movie-recommendation-evaluation` experiment, run
[`b80493aabcdc4a0eb53b86b245a0f55b`](https://dagshub.com/abulhawa/movie-recommender-mlops.mlflow/#/experiments/1/runs/b80493aabcdc4a0eb53b86b245a0f55b).
Metrics, finished status, and aggregate report artifact listings were read back
successfully. This is a historical evaluation run, not a model-registration run.
The original evaluator checksum above remains the provenance of these saved scores.
