"""Hybrid retrieval trained on real interactions with pairwise BPR loss.

The old two_tower module is retained only as a genre-cosine historical experiment.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.train_service.interaction_data import sample_negatives


class InteractionTwoTower(nn.Module):
    def __init__(
        self, n_users, movie_features, dimension=32, hidden=64, temperature=0.1
    ):
        super().__init__()
        if dimension < 1 or hidden < 1 or temperature <= 0:
            raise ValueError('Invalid tower dimensions/temperature')
        self.temperature = temperature
        self.user_ids = nn.Embedding(n_users, dimension)
        self.movie_ids = nn.Embedding(len(movie_features) + 1, dimension, padding_idx=0)
        self.register_buffer(
            'features',
            torch.cat(
                [
                    torch.zeros(1, movie_features.shape[1]),
                    torch.as_tensor(movie_features),
                ]
            ),
        )
        self.user_tower = nn.Sequential(
            nn.Linear(dimension * 2 + movie_features.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
        )
        self.movie_tower = nn.Sequential(
            nn.Linear(dimension + movie_features.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
        )

    def encode_users(self, users, history_ids, history_weights):
        denominator = history_weights.abs().sum(1, keepdim=True).clamp_min(1)
        history = (self.movie_ids(history_ids) * history_weights.unsqueeze(-1)).sum(
            1
        ) / denominator
        preferences = (self.features[history_ids] * history_weights.unsqueeze(-1)).sum(
            1
        ) / denominator
        return F.normalize(
            self.user_tower(
                torch.cat([self.user_ids(users), history, preferences], dim=1)
            ),
            dim=1,
        )

    def encode_movies(self, items):
        ids = items + 1
        return F.normalize(
            self.movie_tower(
                torch.cat([self.movie_ids(ids), self.features[ids]], dim=1)
            ),
            dim=1,
        )

    def forward(self, users, histories, weights, items):
        return (
            self.encode_users(users, histories, weights) * self.encode_movies(items)
        ).sum(1) / self.temperature


def fit(
    data,
    *,
    dimension=32,
    hidden=64,
    epochs=3,
    batch_size=512,
    learning_rate=0.001,
    weight_decay=0.0001,
    history_length=32,
    seed=42,
    max_pairs=0,
    on_epoch=None,
):
    if epochs < 1 or batch_size < 1 or history_length < 1 or max_pairs < 0:
        raise ValueError('Invalid training sizes')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = InteractionTwoTower(len(data.user_ids), data.features, dimension, hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    valid_users = [
        u
        for u in range(len(data.user_ids))
        if len(data.seen[u]) < len(data.movie_ids) or len(data.dislikes[u])
    ]
    positives = np.flatnonzero(
        ((data.events.rating >= 4) & data.events.u.isin(valid_users)).to_numpy()
    )
    if not len(positives):
        raise ValueError('No positive interactions with valid comparisons')
    losses = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(positives)
        if max_pairs:
            order = order[:max_pairs]
        total = 0.0
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            users = data.events.u.to_numpy()[positions]
            items = data.events.i.to_numpy()[positions]
            history, weights = data.contexts(positions, history_length)
            negatives, confidence = sample_negatives(data, users, rng)
            users, history, weights = (
                torch.as_tensor(users, dtype=torch.long),
                torch.as_tensor(history),
                torch.as_tensor(weights),
            )
            user_vectors = model.encode_users(users, history, weights)
            positive_vectors = model.encode_movies(
                torch.as_tensor(items, dtype=torch.long)
            )
            negative_vectors = model.encode_movies(
                torch.as_tensor(negatives, dtype=torch.long)
            )
            difference = (user_vectors * (positive_vectors - negative_vectors)).sum(
                1
            ) / model.temperature
            strength = torch.as_tensor(
                (data.events.rating.to_numpy()[positions] - 3) / 2, dtype=torch.float32
            )
            loss = (
                F.softplus(-difference) * torch.as_tensor(confidence) * strength
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite BPR loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(positions)
        losses.append(total / len(order))
        print(
            f'two_tower epoch={epoch + 1} bpr_loss={losses[-1]:.6f} pairs={len(order)}',
            flush=True,
        )
        if on_epoch:
            on_epoch(epoch + 1, {'train_bpr_loss': losses[-1]})
    model.eval()
    return model, losses


def vectors(model, data, *, history_length=32, batch_size=512):
    with torch.inference_mode():
        items = torch.cat(
            [
                model.encode_movies(
                    torch.arange(start, min(start + batch_size, len(data.movie_ids)))
                )
                for start in range(0, len(data.movie_ids), batch_size)
            ]
        ).numpy()
        users = []
        for start in range(0, len(data.user_ids), batch_size):
            ids = np.arange(start, min(start + batch_size, len(data.user_ids)))
            histories, weights = data.inference_contexts(ids, history_length)
            users.append(
                model.encode_users(
                    torch.as_tensor(ids),
                    torch.as_tensor(histories),
                    torch.as_tensor(weights),
                ).numpy()
            )
    return np.concatenate(users), items
