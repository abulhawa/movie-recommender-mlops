"""Exact unseen-item retrieval and an MLflow-compatible frozen serving bundle."""

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz


def rank_unseen(scores, seen, movie_ids, k=10):
    if k < 1:
        raise ValueError('k must be positive')
    scores = np.asarray(scores).copy()
    if scores.shape != (len(movie_ids),) or not np.isfinite(scores).all():
        raise ValueError('Invalid retrieval scores')
    scores[list(seen)] = -np.inf
    # Exact deterministic top-K; ties are ordered by stable original movie IDs.
    order = np.lexsort((movie_ids, -scores))
    return order[np.isfinite(scores[order])][:k]


def save_bundle(directory, data, user_vectors, movie_vectors):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / 'vectors.npz',
        user_ids=data.user_ids,
        movie_ids=data.movie_ids,
        user_vectors=user_vectors,
        movie_vectors=movie_vectors,
    )
    rows, cols = [], []
    for u, items in enumerate(data.seen):
        rows.extend([u] * len(items))
        cols.extend(items)
    save_npz(
        directory / 'seen.npz',
        csr_matrix(
            (np.ones(len(rows), dtype='uint8'), (rows, cols)),
            shape=(len(data.user_ids), len(data.movie_ids)),
        ),
    )


class RetrievalModel(mlflow.pyfunc.PythonModel):
    """Frozen known-user embeddings; unknown users require a new encoded profile.

    Deliberately separate from the legacy API's KNN artifact contract.
    """

    def load_context(self, context):
        with np.load(Path(context.artifacts['bundle']) / 'vectors.npz') as values:
            self.user_ids = values['user_ids']
            self.movie_ids = values['movie_ids']
            self.user_vectors = values['user_vectors']
            self.movie_vectors = values['movie_vectors']
        self.seen = load_npz(Path(context.artifacts['bundle']) / 'seen.npz')
        self.lookup = {int(uid): u for u, uid in enumerate(self.user_ids)}

    def predict(self, context, model_input, params=None):
        results = []
        for row in model_input.to_dict('records'):
            uid, k = int(row['user_id']), int(row.get('k', 10))
            if uid not in self.lookup:
                raise ValueError(f'Unknown historical user: {uid}')
            u = self.lookup[uid]
            scores = self.movie_vectors @ self.user_vectors[u]
            positions = rank_unseen(scores, self.seen[u].indices, self.movie_ids, k)
            results.append(
                {
                    'user_id': uid,
                    'recommendations': json.dumps(self.movie_ids[positions].tolist()),
                }
            )
        return pd.DataFrame(results)
