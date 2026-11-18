"""Behavioral tests for real-interaction learning and leakage-free retrieval."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('implicit')
import torch

from src.evaluation.retrieval import evaluate_retrieval
from src.train_service.interaction_data import load_interactions, sample_negatives
from src.train_service.models import als, interaction_two_tower
from src.train_service.retrieval import RetrievalModel, rank_unseen, save_bundle
from src.train_service.train_retrieval import main


@pytest.fixture
def dataset(tmp_path):
    movies = tmp_path / 'movies.csv'
    pd.DataFrame(
        {
            'movieId': [10, 20, 30, 40, 50, 60],
            'title': ['A (1990)', 'B (2000)', 'C', 'D', 'E', 'F'],
            'genres': ['Action', 'Drama', 'Action', 'Drama', 'Action', 'Drama'],
        }
    ).to_csv(movies, index=False)
    train = [
        (1, 10, 5, 1),
        (1, 20, 1, 2),
        (1, 30, 4, 3),
        (2, 20, 5, 1),
        (2, 40, 4, 2),
        (2, 10, 1, 3),
        (3, 50, 5, 1),
        (3, 60, 4, 2),
        (3, 30, 2, 3),
    ]
    holdout = [
        (1, 50, 5, 10),
        (2, 60, 5, 11),
        (3, 20, 4, 12),
        (99, 50, 5, 13),
        (1, 999, 5, 14),
    ]
    for number, rows in [(1, train), (2, holdout)]:
        pd.DataFrame(rows, columns=['userId', 'movieId', 'rating', 'timestamp']).to_csv(
            tmp_path / f'ratings_{number:02d}.csv', index=False
        )
    return tmp_path, movies


def load(dataset):
    return load_interactions(*dataset, train_through=1, test_part=2)


def test_contexts_strictly_precede_target_and_ignore_future_labels(dataset):
    data = load(dataset)
    positions = np.arange(len(data.events))
    histories, weights = data.contexts(positions, 4)
    for p, ids in enumerate(histories):
        event = data.events.iloc[p]
        earlier = data.events[
            (data.events.u == event.u) & (data.events.timestamp < event.timestamp)
        ]
        assert set(ids) - {0} <= set(earlier.i + 1)
        assert event.i + 1 not in ids
    assert (weights < 0).any()  # disliked ratings survive in the profile
    future = pd.read_csv(dataset[0] / 'ratings_02.csv')
    future.rating = 1
    future.to_csv(dataset[0] / 'ratings_02.csv', index=False)
    changed = load(dataset)
    pd.testing.assert_frame_equal(data.events, changed.events)
    np.testing.assert_array_equal(data.features, changed.features)
    np.testing.assert_array_equal(
        data.contexts(positions, 4)[0], changed.contexts(positions, 4)[0]
    )


def test_negatives_never_label_known_positives_as_missing(dataset):
    data = load(dataset)
    users = np.tile(np.arange(3), 100)
    items, confidence = sample_negatives(data, users, np.random.default_rng(42))
    for u, i, weight in zip(users, items, confidence):
        if weight == 1:
            assert i in data.dislikes[u]
        else:
            assert weight == 0.25
            assert i not in data.seen[u]
    values = data.confidence_matrix().toarray()
    assert values[0, 0] > 0
    assert values[0, 1] < 0
    assert values[0, 5] == 0


def test_bpr_learns_observed_preferences_and_is_reproducible(dataset):
    data = load(dataset)
    torch.set_num_threads(1)
    settings = {
        'dimension': 8,
        'hidden': 16,
        'epochs': 40,
        'batch_size': 6,
        'learning_rate': 0.01,
        'history_length': 4,
        'seed': 42,
    }
    model, losses = interaction_two_tower.fit(data, **settings)
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    other, repeated = interaction_two_tower.fit(data, **settings)
    assert losses == repeated
    u, i = interaction_two_tower.vectors(model, data, history_length=4)
    u2, i2 = interaction_two_tower.vectors(other, data, history_length=4)
    np.testing.assert_array_equal(u, u2)
    np.testing.assert_array_equal(i, i2)
    assert (u[0] @ i[0]) > (u[0] @ i[1])
    assert model.user_ids.weight.grad is not None
    assert model.movie_ids.weight.grad is not None


def test_als_bundle_roundtrip_and_shared_evaluation(dataset, tmp_path):
    data = load(dataset)
    model = als.fit(data, factors=4, iterations=3, threads=1)
    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (6, 4)
    scorer = lambda u: model.item_factors @ model.user_factors[u]
    metrics, rows, coverage = evaluate_retrieval(
        data,
        dataset[0] / 'ratings_02.csv',
        {'als': scorer},
        k=2,
        min_history=1,
        max_users=0,
    )
    assert coverage['users_without_history'] == 1
    assert coverage['sampled_positive_rows_outside_catalog'] == 1
    assert coverage['evaluated_users'] == 3
    assert 0 <= metrics['als']['ndcg'] <= 1
    for row in rows:
        u = np.searchsorted(data.user_ids, row['user_id'])
        assert not set(json.loads(row['recommendations'])) & set(
            data.movie_ids[list(data.seen[u])]
        )
    directory = tmp_path / 'bundle'
    save_bundle(directory, data, model.user_factors, model.item_factors)
    wrapper = RetrievalModel()
    wrapper.load_context(SimpleNamespace(artifacts={'bundle': str(directory)}))
    result = wrapper.predict(None, pd.DataFrame({'user_id': [1], 'k': [2]}))
    expected = data.movie_ids[
        rank_unseen(scorer(0), data.seen[0], data.movie_ids, 2)
    ].tolist()
    assert json.loads(result.iloc[0].recommendations) == expected
    with pytest.raises(ValueError, match='Unknown historical user'):
        wrapper.predict(None, pd.DataFrame({'user_id': [999]}))


def test_ties_seen_items_chronology_and_final_protection(dataset, tmp_path):
    assert rank_unseen(np.ones(4), {1}, np.array([40, 30, 20, 10]), 2).tolist() == [
        3,
        2,
    ]
    frame = pd.read_csv(dataset[0] / 'ratings_01.csv')
    frame.loc[0, 'timestamp'] = 99
    frame.to_csv(dataset[0] / 'ratings_01.csv', index=False)
    with pytest.raises(ValueError, match='Chronological leakage'):
        load(dataset)
    with pytest.raises(SystemExit):
        main(
            [
                '--ratings-dir',
                str(dataset[0]),
                '--movies-csv',
                str(dataset[1]),
                '--output',
                str(tmp_path / 'output'),
                '--test-part',
                '8',
            ]
        )


def test_local_mlflow_registration_and_loaded_predictions(
    dataset, tmp_path, monkeypatch
):
    import mlflow

    from src.evaluation import tracking
    from src.train_service.train_retrieval import track

    uri = f'sqlite:///{tmp_path / "tracking.db"}'
    monkeypatch.setattr(tracking, 'PERSONAL_URI', uri)
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    mlflow.MlflowClient(tracking_uri=uri).create_experiment(
        'movie-interaction-retrieval', artifact_location=str(tmp_path / 'artifacts')
    )
    data = load(dataset)
    rng = np.random.default_rng(1)
    directory = tmp_path / 'bundle'
    save_bundle(directory, data, rng.normal(size=(3, 4)), rng.normal(size=(6, 4)))
    report = {
        'protocol': {'train_through': 1, 'test_part': 2},
        'source_commit': 'test',
        'source_working_tree_dirty': True,
        'metrics': {'interaction_als': {'ndcg': 0.25, 'recall': 0.5}},
        'training': {'interaction_als': {'train_seconds': 1.0}},
    }
    (tmp_path / 'report.json').write_text(json.dumps(report))
    (tmp_path / 'summary.md').write_text('Aggregate result')
    try:
        receipt = track(tmp_path, report, {'interaction_als': directory})
        assert receipt['interaction_als']['status'] == 'verified'
        assert receipt['interaction_als']['model_version'] == '1'
        loaded = mlflow.pyfunc.load_model('models:/movie-interaction-als/1')
        assert len(loaded.predict(pd.DataFrame({'user_id': [1]}))) == 1
        with pytest.raises(ValueError, match='receipt already exists'):
            track(tmp_path, report, {'interaction_als': directory})
    finally:
        mlflow.set_tracking_uri(None)
        mlflow.set_registry_uri(None)


def test_equal_time_contexts_are_empty_and_year_is_optional(dataset):
    path = dataset[0] / 'ratings_01.csv'
    frame = pd.read_csv(path)
    frame.timestamp = 1
    frame.to_csv(path, index=False)
    data = load_interactions(*dataset, train_through=1, test_part=2, use_year=True)
    histories, weights = data.contexts(np.arange(len(data.events)), 4)
    assert not histories.any()
    assert not weights.any()
    assert data.features.shape[1] == len(data.genres) + 2
    assert np.isfinite(data.features).all()
