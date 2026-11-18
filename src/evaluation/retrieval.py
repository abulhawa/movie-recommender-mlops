"""Same chronological cohort/catalog for learned retrieval and popularity."""

import json
import time

import numpy as np

from src.evaluation.recommendations import chunks, ranking_metrics
from src.train_service.retrieval import rank_unseen


def evaluate_retrieval(
    data, test_path, methods, *, k=10, min_history=20, max_users=1000, seed=42
):
    user_map = {int(uid): u for u, uid in enumerate(data.user_ids)}
    movie_map = {int(mid): i for i, mid in enumerate(data.movie_ids)}
    test_users = set()
    for frame in chunks(test_path, 200_000):
        test_users.update(map(int, frame.userId.unique()))
    qualified = sorted(
        uid
        for uid in test_users
        if uid in user_map and data.counts.get(uid, 0) >= min_history
    )
    chosen = qualified
    if max_users and len(chosen) > max_users:
        chosen = sorted(
            map(
                int,
                np.random.default_rng(seed).choice(chosen, max_users, replace=False),
            )
        )
    truth = {uid: set() for uid in chosen}
    outside_catalog = 0
    for frame in chunks(test_path, 200_000):
        frame = frame[frame.userId.isin(chosen) & (frame.rating >= 4)]
        for uid, mid in frame[['userId', 'movieId']].itertuples(index=False, name=None):
            if mid not in movie_map:
                outside_catalog += 1
            elif movie_map[mid] not in data.seen[user_map[uid]]:
                truth[uid].add(mid)
    eligible = [uid for uid in chosen if truth[uid]]
    rows, summaries = [], {}
    for name, score_user in methods.items():
        catalog, cases, timings = set(), [], []
        for uid in eligible:
            u = user_map[uid]
            started = time.perf_counter()
            positions = rank_unseen(score_user(u), data.seen[u], data.movie_ids, k)
            timings.append((time.perf_counter() - started) * 1000)
            recs = data.movie_ids[positions].tolist()
            catalog.update(recs)
            scores = ranking_metrics(recs, truth[uid], k)
            cases.append(scores)
            rows.append(
                {
                    'user_id': uid,
                    'method': name,
                    'recommendations': json.dumps(recs),
                    'seen_fraction': 0.0,
                    **scores,
                }
            )
        summaries[name] = {
            key: float(np.mean([case[key] for case in cases])) if cases else None
            for key in ['recall', 'precision', 'ndcg', 'hit_rate']
        }
        summaries[name].update(
            seen_fraction=0.0,
            catalog_coverage=len(catalog) / len(data.movie_ids),
            latency_p95_ms=float(np.percentile(timings, 95)) if timings else None,
        )
    return (
        summaries,
        rows,
        {
            'test_users': len(test_users),
            'users_without_history': sum(uid not in data.counts for uid in test_users),
            'users_below_minimum_history': sum(
                0 < data.counts.get(uid, 0) < min_history for uid in test_users
            ),
            'users_outside_training_sample': sum(
                uid in data.counts and uid not in user_map for uid in test_users
            ),
            'history_qualified_returning_users': len(qualified),
            'sampled_returning_users': len(chosen),
            'evaluated_users': len(eligible),
            'sampled_without_relevant_movies': len(chosen) - len(eligible),
            'catalog_movies': len(data.movie_ids),
            'sampled_positive_rows_outside_catalog': outside_catalog,
        },
    )
