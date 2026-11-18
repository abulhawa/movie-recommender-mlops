# Interaction retrieval development result - 18 November 2026

This compares fixed initial configurations on parts 1-6 versus development part 7. Part 8 was not read. No candidate was deployed.

## Data and protocol

- Historical training: 15,000,197 deduplicated user/movie pairs, 106,573 users, 10,361 movies.
- Positive ratings: >=4; explicit dislikes: <=2. Two-tower BPR trained on 7,526,296 positive pairs per epoch for three epochs; ALS used 10 iterations. Both used 32 factors and seed 42.
- User tower: learned ID, signed rating-weighted previous-movie embeddings and genre preferences; movie tower: learned ID and genres. Optional year features were disabled.
- Two-tower histories: up to 32 strictly earlier rated movies during training; target and same-timestamp events excluded. Sparse user-ID updates; dense tower/movie updates.
- 20,587 development users; 15,321 had no history, 172 had insufficient history. Of 5,094 qualified returning users, sample 1,000 before positive-label filtering; 937 scored and 63 had no eligible future positive.
- 12,574 sampled positive rows were outside the historical catalog and excluded. No training-user or per-epoch pair cap was applied.
- Every method ranks the same available catalog and excludes all historical rated movies. KNN retains its seeded random ordering from the top-20 unseen pool.
- Metadata genres are assumed stable. No current external popularity/tag-genome information was introduced.

## Ranking results

| Method | NDCG@10 | Recall@10 | Precision@10 | Hit rate | Catalog coverage |
|---|---:|---:|---:|---:|---:|
| interaction_als | 0.169451 | 0.071765 | 0.150907 | 0.574173 | 0.118618 |
| interaction_two_tower | 0.127539 | 0.055713 | 0.104803 | 0.494130 | 0.076827 |
| popularity_unseen | 0.121460 | 0.047155 | 0.101708 | 0.434365 | 0.016890 |
| genre_knn_unseen | 0.003456 | 0.001763 | 0.002882 | 0.026681 | 0.243606 |

All values are fractions; quality metrics are macro averages over the 937 scored users. Catalog coverage measures exposure breadth, not semantic diversity.

ALS leads this development comparison. The neural model learns from actual interactions and scores above popularity, but its small observed improvement is uncertain.

Paired user bootstrap of NDCG differences versus popularity, 2,000 resamples with seed 42:

| Candidate | Mean difference | Percentile 95% interval |
|---|---:|---:|
| ALS | +0.04799 | [+0.03587, +0.05991] |
| Two-tower | +0.00608 | [-0.00646, +0.01952] |

The neural interval includes zero. These intervals describe user-sampling uncertainty conditional on this one trained model, cohort and window; they do not measure training-seed or future-window variation. Resample the same aligned user rows with replacement from `per_user.csv` and average paired NDCG differences.

## Resources and limitations

| Candidate | Training seconds | Compressed bundle bytes | Warm retrieval p95 ms |
|---|---:|---:|---:|
| interaction_als | 14.57 | 36585378 | 0.905 |
| interaction_two_tower | 518.84 | 52445421 | 0.932 |

Latency covers sequential local scoring, seen filtering and sorting, excluding HTTP, concurrent load and tower encoding. Bundle size is not a RAM measurement. Neither candidate is a production latency or memory benchmark.

Fixed configurations were used; no checkpoint or hyperparameter was selected using part 7. The earlier capped dense-optimizer pilot is not directly interchangeable with this full-data sparse-optimizer experiment. An interrupted dense full-data run produced no evaluation result.

Next research: tune and run ablations on parts 1-5 versus 6 (ID-only, history, genres, history length and negative-sampling strength), refit on 1-6, then compare on 7. Reserve 8 for the frozen final protocol. ALS is the stronger current candidate; neural complexity alone is not a selection criterion.

## Reproduction and MLflow

- Source commit: `5b54a677679e65613ddaf3354e5d48d8fee67441`; working tree was clean at report generation.
- Full input/source hashes, library versions, parameters and losses are in the aggregate report artifact. Local evidence: `reports/recommendation_evaluation/interaction-development-sparse-20261118/`.
- Both registry candidates use a separate frozen known-user retrieval contract. New ratings require re-encoding/retraining; API/Airflow promotion remains a separate integration step.

See [architecture, commands and tests](INTERACTION_RETRIEVAL.md).

Verified personal MLflow candidates (metrics read back, registered artifacts downloaded, and recommendations matched to local bundles):

- [interaction_als: run `b07c2cc572f947909327604641ea6f37`](https://dagshub.com/abulhawa/movie-recommender-mlops.mlflow/#/experiments/2/runs/b07c2cc572f947909327604641ea6f37); registry `movie-interaction-als`, version 1.
- [interaction_two_tower: run `2c56657790b24065a5b847f0623264ab`](https://dagshub.com/abulhawa/movie-recommender-mlops.mlflow/#/experiments/2/runs/2c56657790b24065a5b847f0623264ab); registry `movie-interaction-two-tower`, version 1.
