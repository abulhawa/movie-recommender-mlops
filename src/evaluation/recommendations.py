"""Backtest the existing genre KNN using earlier histories and later interactions.

This reconstructs a historical model, not a live version trained on later data.
Only CSV inputs are read; no serving files, DagsHub records or aliases are changed.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_service.models.knn import train_model


@dataclass(frozen=True)
class Protocol:
    train_through: int = 6
    test_part: int = 7
    min_history: int = 20
    positive_rating: float = 4.0
    k: int = 10
    candidate_count: int = 20
    max_users: int = 1000  # 0 means every eligible returning user.
    seed: int = 42
    chunk_size: int = 200_000

    def validate(self):
        if not 1 <= self.train_through < 8 or self.test_part != self.train_through + 1:
            raise ValueError(
                'Use cumulative parts 1..N and the next part N+1 (up to 8)'
            )
        if not 1 <= self.k <= self.candidate_count or self.min_history < 1:
            raise ValueError('Require 1 <= k <= candidate_count and min_history >= 1')
        if (
            not 0 < self.positive_rating <= 5
            or self.max_users < 0
            or self.seed < 0
            or self.chunk_size < 1
        ):
            raise ValueError(
                'Invalid rating threshold, sample size, seed or chunk size'
            )


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def chunks(path, size):
    for frame in pd.read_csv(
        path,
        usecols=['userId', 'movieId', 'rating', 'timestamp'],
        dtype={
            'userId': 'int64',
            'movieId': 'int64',
            'rating': 'float32',
            'timestamp': 'int64',
        },
        chunksize=size,
    ):
        if (
            frame.isna().any().any()
            or not np.isfinite(frame.to_numpy()).all()
            or (frame[['userId', 'movieId']] <= 0).any().any()
            or not frame.rating.between(0, 5, inclusive='right').all()
            or (frame.timestamp < 0).any()
        ):
            raise ValueError(f'Invalid rating rows in {path.name}')
        yield frame


def statistics(path, size):
    users, count, low, high = set(), 0, None, None
    for frame in chunks(path, size):
        if frame.empty:
            continue
        users.update(map(int, frame.userId.unique()))
        count += len(frame)
        low = (
            int(frame.timestamp.min())
            if low is None
            else min(low, int(frame.timestamp.min()))
        )
        high = (
            int(frame.timestamp.max())
            if high is None
            else max(high, int(frame.timestamp.max()))
        )
    if not count:
        raise ValueError(f'Empty ratings file: {path.name}')
    return {'rows': count, 'min_timestamp': low, 'max_timestamp': high}, users


def ranking_metrics(recommendations, relevant, k):
    """Binary relevance; macro-average only users with a nonempty ground truth.

    Missing recommendation slots count as zero precision. Repeated future
    ratings are one relevant item, not multiple opportunities to claim a hit.
    """
    if not relevant:
        raise ValueError('A scored user must have at least one relevant movie')
    recommendations = list(recommendations)[:k]
    if len(set(recommendations)) != len(recommendations):
        raise ValueError('Recommendations must be unique')
    gains = [movie in relevant for movie in recommendations]
    hits = sum(gains)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {
        'recall': hits / len(relevant),
        'precision': hits / k,
        'ndcg': dcg / ideal,
        'hit_rate': float(hits > 0),
    }


def sample_candidates(ids, k, seed, user_id):
    # Same per-user generator and sampling policy as predict.py. This does not
    # exclude rated movies unless the separately labelled diagnostic asks to.
    return (
        np.random.default_rng([seed, user_id])
        .choice(np.asarray(ids), size=min(k, len(ids)), replace=False)
        .tolist()
    )


def evaluate(ratings_dir, movies_csv, protocol):
    protocol.validate()
    ratings_dir, movies_csv = Path(ratings_dir), Path(movies_csv)
    train_paths = [
        ratings_dir / f'ratings_{n:02d}.csv'
        for n in range(1, protocol.train_through + 1)
    ]
    test_path = ratings_dir / f'ratings_{protocol.test_part:02d}.csv'
    for path in [movies_csv, *train_paths, test_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    starting_stats = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [movies_csv, *train_paths, test_path]
    }
    metadata = pd.read_csv(movies_csv)
    if (
        not {'movieId', 'title', 'genres'} <= set(metadata)
        or metadata[['movieId', 'genres']].isna().any().any()
    ):
        raise ValueError('Movie metadata requires movieId, title and genres')
    if (
        not pd.api.types.is_integer_dtype(metadata.movieId)
        or (metadata.movieId <= 0).any()
    ):
        raise ValueError('Movie IDs must be positive integers')
    # Same binary genre encoding as features/build_features.py, without importing
    # its CLI/path defaults (the evaluator accepts explicit, read-only inputs).
    movies = pd.concat(
        [metadata[['movieId', 'title']], metadata.genres.str.get_dummies(sep='|')],
        axis=1,
    )
    if movies.movieId.duplicated().any() or movies.empty:
        raise ValueError('Movie catalog must be nonempty with unique IDs')
    genres = [c for c in movies if c not in {'movieId', 'title'}]
    if not genres:
        raise ValueError('Movie catalog has no genre features')
    metadata_ids = set(map(int, movies.movieId))
    test_stats, test_users = statistics(test_path, protocol.chunk_size)
    cutoff = test_stats['min_timestamp']
    history_counts, movie_counts, popular = Counter(), Counter(), Counter()
    stats, boundary_rows, unknown_train, previous_max = [], 0, 0, None
    for path in train_paths:
        part_rows, low, high = 0, None, None
        for frame in chunks(path, protocol.chunk_size):
            if frame.empty:
                continue
            part_rows += len(frame)
            low = (
                int(frame.timestamp.min())
                if low is None
                else min(low, int(frame.timestamp.min()))
            )
            high = (
                int(frame.timestamp.max())
                if high is None
                else max(high, int(frame.timestamp.max()))
            )
            if int(frame.timestamp.max()) > cutoff:
                raise ValueError(
                    'Chronological leakage: training timestamps overlap the held-out period'
                )
            boundary_rows += int((frame.timestamp == cutoff).sum())
            frame = frame[frame.timestamp < cutoff]
            unknown_train += int((~frame.movieId.isin(metadata_ids)).sum())
            frame = frame[frame.movieId.isin(metadata_ids)]
            history_counts.update(frame.userId.value_counts().to_dict())
            movie_counts.update(frame.movieId.value_counts().to_dict())
            popular.update(
                frame[frame.rating >= protocol.positive_rating]
                .movieId.value_counts()
                .to_dict()
            )
        if not part_rows or previous_max is not None and previous_max > low:
            raise ValueError(
                'Training parts are empty, overlapping or out of chronological order'
            )
        previous_max = high
        stats.append(
            {
                'name': path.name,
                'rows': part_rows,
                'min_timestamp': low,
                'max_timestamp': high,
            }
        )
    # Metadata does not provide availability timestamps. Requiring an observed
    # pre-cutoff interaction avoids recommending catalog items introduced later.
    catalog = (
        movies[movies.movieId.isin(movie_counts)]
        .drop(columns='title')
        .reset_index(drop=True)
    )
    if len(catalog) < protocol.k:
        raise ValueError('Historically available catalog is smaller than k')
    catalog_ids = set(map(int, catalog.movieId))
    returning = sorted(
        int(u) for u in test_users if history_counts[u] >= protocol.min_history
    )
    chosen = returning
    if protocol.max_users and len(chosen) > protocol.max_users:
        chosen = sorted(
            map(
                int,
                np.random.default_rng(protocol.seed).choice(
                    chosen, protocol.max_users, replace=False
                ),
            )
        )
    selected = set(chosen)
    seen = {u: set() for u in chosen}
    sums = pd.DataFrame(0.0, index=chosen, columns=genres)
    counts = pd.Series(0, index=chosen, dtype='int64')
    for path in train_paths:
        for frame in chunks(path, protocol.chunk_size):
            frame = frame[(frame.timestamp < cutoff) & frame.userId.isin(selected)]
            merged = frame[['userId', 'movieId']].merge(
                catalog, on='movieId', how='inner'
            )
            if merged.empty:
                continue
            grouped = merged.groupby('userId')
            partial = grouped[genres].sum()
            sums.loc[partial.index] += partial
            partial_count = grouped.size()
            counts.loc[partial_count.index] += partial_count
            for uid, group in grouped:
                seen[int(uid)].update(map(int, group.movieId))
    profiles = sums.div(counts.replace(0, np.nan), axis=0)
    if len(chosen) and not np.isfinite(profiles.to_numpy()).all():
        raise ValueError('Failed to construct pre-cutoff profiles')
    truth = {u: set() for u in chosen}
    selected_positive_rows = unknown_positive_rows = already_seen_positive_rows = 0
    for frame in chunks(test_path, protocol.chunk_size):
        frame = frame[
            frame.userId.isin(selected) & (frame.rating >= protocol.positive_rating)
        ]
        selected_positive_rows += len(frame)
        for uid, mid in frame[['userId', 'movieId']].itertuples(index=False, name=None):
            uid, mid = int(uid), int(mid)
            if mid not in catalog_ids:
                unknown_positive_rows += 1
            elif mid in seen[uid]:
                already_seen_positive_rows += 1
            else:
                truth[uid].add(mid)
    eligible = [u for u in chosen if truth[u]]
    model = train_model(catalog, n_neighbors=protocol.candidate_count)
    movie_ids = catalog.movieId.to_numpy()
    popularity = sorted(catalog_ids, key=lambda mid: (-popular[mid], mid))
    names = ['genre_knn_serving', 'popularity', 'genre_knn_unseen', 'popularity_unseen']
    rows, recommended = [], {name: set() for name in names}
    for uid in eligible:
        user = profiles.loc[[uid], genres]
        pool_size = min(len(catalog), protocol.candidate_count)
        positions = model.kneighbors(
            user, n_neighbors=pool_size, return_distance=False
        )[0]
        base_pool = movie_ids[positions].tolist()
        # Search enough neighbors to obtain the unseen pool, rather than filtering
        # only top-20 and unfairly returning a short list when further items exist.
        search_size = min(len(catalog), protocol.candidate_count + len(seen[uid]))
        positions = model.kneighbors(
            user, n_neighbors=search_size, return_distance=False
        )[0]
        unseen_pool = [
            int(movie_ids[i]) for i in positions if int(movie_ids[i]) not in seen[uid]
        ][: protocol.candidate_count]
        lists = {
            'genre_knn_serving': sample_candidates(
                base_pool, protocol.k, protocol.seed, uid
            ),
            'popularity': popularity[: protocol.k],
            'genre_knn_unseen': sample_candidates(
                unseen_pool, protocol.k, protocol.seed, uid
            ),
            'popularity_unseen': [mid for mid in popularity if mid not in seen[uid]][
                : protocol.k
            ],
        }
        for name, recs in lists.items():
            recommended[name].update(recs)
            rows.append(
                {
                    'user_id': uid,
                    'method': name,
                    'history_events': int(counts[uid]),
                    'relevant_movies': len(truth[uid]),
                    'returned': len(recs),
                    'recommendations': json.dumps([int(mid) for mid in recs]),
                    'seen_fraction': sum(mid in seen[uid] for mid in recs) / len(recs)
                    if recs
                    else 0.0,
                    **ranking_metrics(recs, truth[uid], protocol.k),
                }
            )
    summaries = {}
    for name in names:
        cases = [row for row in rows if row['method'] == name]
        summaries[name] = {
            metric: float(np.mean([row[metric] for row in cases])) if cases else None
            for metric in ['recall', 'precision', 'ndcg', 'hit_rate', 'seen_fraction']
        }
        summaries[name]['catalog_coverage'] = len(recommended[name]) / len(catalog)
    inputs = [
        {'name': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in [movies_csv, *train_paths, test_path]
    ]
    for path, before in starting_stats.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != before:
            raise ValueError(
                'Input files changed during evaluation; no report was written'
            )
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        dirty = bool(
            subprocess.check_output(
                ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    report = {
        'source_working_tree_dirty': dirty,
        'status': 'evaluated' if eligible else 'no_evaluable_users',
        'protocol': asdict(protocol),
        'cutoff_timestamp': cutoff,
        'history_rule': 'minimum pre-cutoff rating events matching the known catalog',
        'relevance_rule': 'unique future rating>=threshold, historically available, not rated in history',
        'user_sampling': 'seeded sample of history-qualified returning users, before checking positive labels',
        'catalog_rule': 'movies observed in pre-cutoff training interactions; same catalog for all methods',
        'model_scope': 'historical reconstruction of existing KNN; not evaluation of the live registered model',
        'metrics_scope': 'macro average conditional on an evaluable user; missing slots have zero precision credit',
        'serving_match': f'raw mean genres; Euclidean ball_tree top-{protocol.candidate_count}; seeded sample of {protocol.k}; seen movies retained',
        'production_defaults_match': protocol.k == 10
        and protocol.candidate_count == 20,
        'coverage': {
            'test_users': len(test_users),
            'known_history_users': sum(u in history_counts for u in test_users),
            'users_without_history': sum(u not in history_counts for u in test_users),
            'users_below_minimum_history': sum(
                0 < history_counts[u] < protocol.min_history for u in test_users
            ),
            'history_qualified_returning_users': len(returning),
            'sampled_returning_users': len(chosen),
            'evaluated_users': len(eligible),
            'sampled_without_relevant_movies': len(chosen) - len(eligible),
            'evaluated_fraction_of_test_users': len(eligible) / len(test_users),
            'catalog_movies': len(catalog),
            'boundary_training_rows_excluded': boundary_rows,
            'unknown_metadata_training_rows': unknown_train,
            'sampled_positive_rows': selected_positive_rows,
            'sampled_positive_rows_outside_catalog': unknown_positive_rows,
            'sampled_positive_rows_already_seen': already_seen_positive_rows,
        },
        'train_parts': stats,
        'test_part': {'name': test_path.name, **test_stats},
        'metrics': summaries,
        'inputs': inputs,
        'source_commit': commit,
        'evaluator_sha256': digest(Path(__file__)),
        'versions': {
            'python': platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ['numpy', 'pandas', 'scikit-learn']
            },
        },
        'limitations': [
            'Observed ratings are not randomized exposure or proof of future satisfaction.',
            'Scores cover established returning users with eligible positives, not cold users.',
            'Movie metadata is assumed stable; catalog membership is restricted by past observations.',
            'No model selection or automatic deployment quality gate is performed.',
            'Unseen-only variants are counterfactual diagnostics, not current API behavior.',
        ],
    }
    return report, rows


def write_report(output, report, rows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if rows:
        with (output / 'per_user.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    coverage = report['coverage']
    lines = [
        '# Offline recommendation evaluation',
        '',
        f'Train parts 1–{report["protocol"]["train_through"]}; held-out part {report["protocol"]["test_part"]}.',
        f'Status: {report["status"]}. Evaluated {coverage["evaluated_users"]} of {coverage["test_users"]} held-out users.',
        f'History-qualified returning users: {coverage["history_qualified_returning_users"]}; sampled: {coverage["sampled_returning_users"]}.',
        '',
        '| Method | Recall@K | NDCG@K | Precision@K | Hit rate | Seen fraction |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, metrics in report['metrics'].items():
        values = [
            'n/a' if metrics[key] is None else f'{metrics[key]:.6f}'
            for key in ['recall', 'ndcg', 'precision', 'hit_rate', 'seen_fraction']
        ]
        lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    lines.extend(
        [
            '',
            'These are retrospective, conditional cohort scores, not live-model or online-quality claims.',
            'The unseen variants change eligibility policy; the serving variant retains already-rated movies.',
            'Full protocol, coverage, input hashes and library versions are in report.json.',
        ]
    )
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings-dir', type=Path, required=True)
    parser.add_argument('--movies-csv', type=Path, required=True)
    parser.add_argument('--train-through', type=int, default=6)
    parser.add_argument('--test-part', type=int, default=7)
    parser.add_argument('--min-history', type=int, default=20)
    parser.add_argument('--positive-rating', type=float, default=4.0)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument(
        '--max-users',
        type=int,
        default=1000,
        help='0 evaluates all qualified returning users',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk-size', type=int, default=200_000)
    parser.add_argument(
        '--final', action='store_true', help='acknowledge use of held-out part 8'
    )
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.test_part == 8 and not args.final:
        parser.error(
            'Part 8 is the final holdout: use --final only after freezing the protocol'
        )
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            'Output directory must be new or empty; preserve previous evaluation evidence'
        )
    protocol = Protocol(
        **{
            key: getattr(args, key)
            for key in Protocol.__dataclass_fields__
            if hasattr(args, key)
        }
    )
    report, rows = evaluate(args.ratings_dir, args.movies_csv, protocol)
    write_report(args.output, report, rows)
    print(
        f'{report["status"]}: {report["coverage"]["evaluated_users"]} users; report: {args.output / "summary.md"}'
    )


if __name__ == '__main__':
    main()
