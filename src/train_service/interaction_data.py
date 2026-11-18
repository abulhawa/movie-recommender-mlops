"""Historical interaction data shared by ALS and the retrieval two-tower."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.evaluation.recommendations import chunks, digest, statistics


@dataclass
class InteractionData:
    user_ids: np.ndarray
    movie_ids: np.ndarray
    features: np.ndarray
    genres: list
    events: pd.DataFrame
    starts: np.ndarray
    seen: list
    dislikes: list
    counts: dict
    inputs: list
    cutoff: int

    def contexts(self, positions, length):
        # Strictly earlier interactions only; equal-time events cannot reveal targets.
        positions = np.asarray(positions)
        users = self.events.u.to_numpy()[positions]
        timestamps = self.events.timestamp.to_numpy()
        ends = np.array(
            [
                np.searchsorted(
                    timestamps[self.starts[u] : self.starts[u + 1]],
                    timestamps[p],
                    side='left',
                )
                + self.starts[u]
                for u, p in zip(users, positions)
            ]
        )
        index = ends[:, None] - np.arange(length, 0, -1)[None, :]
        valid = index >= self.starts[users, None]
        index = np.maximum(index, 0)
        ids = self.events.i.to_numpy()[index] + 1  # zero is padding
        weights = (self.events.rating.to_numpy()[index] - 3.0) / 2.0
        # A repeated target must never be its own history feature.
        valid &= ids != (self.events.i.to_numpy()[positions, None] + 1)
        return np.where(valid, ids, 0), np.where(valid, weights, 0).astype('float32')

    def inference_contexts(self, users, length):
        users = np.asarray(users)
        index = self.starts[users + 1, None] - np.arange(length, 0, -1)[None, :]
        valid = index >= self.starts[users, None]
        index = np.maximum(index, 0)
        return (
            np.where(valid, self.events.i.to_numpy()[index] + 1, 0),
            np.where(
                valid, (self.events.rating.to_numpy()[index] - 3.0) / 2.0, 0
            ).astype('float32'),
        )

    def confidence_matrix(self, alpha=20.0, positive=4.0, negative=2.0):
        e = self.events
        # Signed confidence: low ratings are observed dislikes; neutral ratings remain
        # context and seen-item exclusions, without becoming positive preferences.
        values = np.where(
            e.rating >= positive,
            1 + alpha * (e.rating - positive + 1),
            np.where(e.rating <= negative, -(1 + alpha * (negative - e.rating + 1)), 0),
        )
        mask = values != 0
        return csr_matrix(
            (
                values[mask].astype('float32'),
                (e.u.to_numpy()[mask], e.i.to_numpy()[mask]),
            ),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )


def load_interactions(
    ratings_dir,
    movies_csv,
    train_through=6,
    test_part=7,
    *,
    use_year=False,
    max_train_users=0,
    seed=42,
    chunk_size=200_000,
):
    if not 1 <= train_through < 8 or test_part != train_through + 1:
        raise ValueError('Require consecutive cumulative historical/development parts')
    if max_train_users < 0:
        raise ValueError('max_train_users must be nonnegative')
    paths = [
        Path(ratings_dir) / f'ratings_{n:02d}.csv' for n in range(1, train_through + 1)
    ]
    test_path = Path(ratings_dir) / f'ratings_{test_part:02d}.csv'
    all_paths = [Path(movies_csv), *paths, test_path]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in all_paths}
    cutoff = statistics(test_path, chunk_size)[0]['min_timestamp']
    metadata = pd.read_csv(movies_csv)
    if (
        not {'movieId', 'title', 'genres'} <= set(metadata)
        or metadata.empty
        or metadata.movieId.duplicated().any()
        or metadata[['movieId', 'genres']].isna().any().any()
    ):
        raise ValueError('Require unique movie IDs, titles and genres')
    frames, previous_max = [], None
    for path in paths:
        low, high = None, None
        for frame in chunks(path, chunk_size):
            if frame.empty:
                continue
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
            if high > cutoff:
                raise ValueError('Chronological leakage in training inputs')
            frames.append(
                frame[(frame.timestamp < cutoff) & frame.movieId.isin(metadata.movieId)]
            )
        if low is None or previous_max is not None and previous_max > low:
            raise ValueError('Empty or out-of-order historical parts')
        previous_max = high
    events = pd.concat(frames, ignore_index=True)
    if events.empty:
        raise ValueError('No historical interactions')
    counts = events.userId.value_counts().to_dict()
    user_ids = np.sort(events.userId.unique())
    if max_train_users and len(user_ids) > max_train_users:
        user_ids = np.sort(
            np.random.default_rng(seed).choice(user_ids, max_train_users, replace=False)
        )
        events = events[events.userId.isin(user_ids)]
    movie_ids = np.sort(events.movieId.unique())
    metadata = metadata.set_index('movieId').loc[movie_ids]
    genre_frame = metadata.genres.str.get_dummies(sep='|')
    features = genre_frame.to_numpy(dtype='float32')
    if use_year:
        years = pd.to_numeric(
            metadata.title.str.extract(r'\((\d{4})\)$', expand=False), errors='coerce'
        )
        known = years.notna().to_numpy(dtype='float32')
        centered = (
            (years.fillna(years.median()) - years.median())
            / max(float(years.std()), 1.0)
        ).fillna(0)
        features = np.column_stack(
            [features, centered.to_numpy(dtype='float32'), known]
        )
    # Duplicate user/movie events use the most recent historical rating. Stable
    # equal-time ties use input row order; targets are never copied into context.
    events = (
        events.sort_values('timestamp', kind='stable')
        .drop_duplicates(['userId', 'movieId'], keep='last')
        .copy()
    )
    events['u'] = np.searchsorted(user_ids, events.userId).astype('int32')
    events['i'] = np.searchsorted(movie_ids, events.movieId).astype('int32')
    events = events.sort_values(
        ['u', 'timestamp', 'movieId'], kind='stable'
    ).reset_index(drop=True)
    starts = np.r_[0, np.cumsum(np.bincount(events.u, minlength=len(user_ids)))]
    seen, dislikes = [], []
    for u in range(len(user_ids)):
        group = events.iloc[starts[u] : starts[u + 1]]
        seen.append(set(map(int, group.i)))
        dislikes.append(group.loc[group.rating <= 2, 'i'].to_numpy())
    inputs = [
        {'name': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size}
        for p in all_paths
    ]
    if any((p.stat().st_size, p.stat().st_mtime_ns) != before[p] for p in all_paths):
        raise ValueError('Inputs changed while loading interactions')
    return InteractionData(
        user_ids,
        movie_ids,
        features,
        genre_frame.columns.tolist(),
        events,
        starts,
        seen,
        dislikes,
        counts,
        inputs,
        cutoff,
    )


def sample_negatives(data, users, rng, explicit_probability=0.5):
    """Unobserved is a weak comparison, never a confirmed dislike.

    Known low ratings are explicit comparisons. Other observed ratings are never
    drawn as unlabeled negatives; future held-out labels are never consulted.
    """
    negatives, confidence = [], []
    for u in users:
        low = data.dislikes[u]
        if len(low) and rng.random() < explicit_probability:
            negatives.append(int(rng.choice(low)))
            confidence.append(1.0)
            continue
        if len(data.seen[u]) == len(data.movie_ids):
            if not len(low):
                raise ValueError('No valid negative comparisons for user')
            negatives.append(int(rng.choice(low)))
            confidence.append(1.0)
            continue
        # Bounded rejection, with an exact complement fallback for dense users.
        for _ in range(32):
            item = int(rng.integers(len(data.movie_ids)))
            if item not in data.seen[u]:
                break
        else:
            item = int(
                rng.choice(
                    np.setdiff1d(np.arange(len(data.movie_ids)), list(data.seen[u]))
                )
            )
        negatives.append(item)
        confidence.append(0.25)
    return np.asarray(negatives), np.asarray(confidence, dtype='float32')
