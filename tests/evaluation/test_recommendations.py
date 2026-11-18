"""Tiny chronological fixtures verify leakage boundaries and metric definitions."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from src.evaluation.recommendations import (
    Protocol,
    evaluate,
    main,
    ranking_metrics,
    sample_candidates,
    write_report,
)


@pytest.fixture
def inputs(tmp_path):
    movies = tmp_path / 'movies.csv'
    pd.DataFrame(
        {
            'movieId': range(1, 13),
            'title': [f'Movie {i}' for i in range(1, 13)],
            'genres': ['Action'] * 6 + ['Drama'] * 6,
        }
    ).to_csv(movies, index=False)
    # User 99 supplies the historical catalog; duplicate events are intentional.
    train = [(99, mid, 4.0, 10) for mid in range(1, 13)] + [(99, 9, 5.0, 11)] * 3
    train += [
        (1, 1, 1.0, 12),
        (1, 2, 2.0, 13),
        (2, 3, 5.0, 14),
        (4, 1, 3.0, 15),
        (4, 2, 3.0, 16),
    ]
    test = [
        (1, 9, 5.0, 20),
        (1, 9, 4.0, 21),
        (1, 1, 5.0, 22),
        (2, 9, 5.0, 23),
        (3, 9, 5.0, 24),
        (4, 9, 2.0, 25),
    ]
    for number, rows in [(1, train), (2, test)]:
        pd.DataFrame(rows, columns=['userId', 'movieId', 'rating', 'timestamp']).to_csv(
            tmp_path / f'ratings_{number:02d}.csv', index=False
        )
    return (
        tmp_path,
        movies,
        Protocol(
            train_through=1, test_part=2, min_history=2, k=2, max_users=0, chunk_size=3
        ),
    )


def test_metric_values_match_definition_and_sklearn():
    actual = ranking_metrics([1, 2], {1, 3}, 2)
    assert actual['recall'] == actual['precision'] == 0.5
    assert actual['hit_rate'] == 1
    assert actual['ndcg'] == pytest.approx(ndcg_score([[1, 0, 1]], [[3, 2, 1]], k=2))
    assert ranking_metrics([1], {1}, 2)['precision'] == 0.5
    with pytest.raises(ValueError):
        ranking_metrics([1, 1], {1}, 2)
    with pytest.raises(ValueError):
        ranking_metrics([1], set(), 2)


def test_cohort_coverage_and_deduplicated_future_truth(inputs):
    folder, movies, protocol = inputs
    report, rows = evaluate(folder, movies, protocol)
    c = report['coverage']
    assert c['test_users'] == 4
    assert c['users_without_history'] == 1
    assert c['users_below_minimum_history'] == 1
    assert c['history_qualified_returning_users'] == 2
    assert c['evaluated_users'] == c['sampled_without_relevant_movies'] == 1
    assert {row['user_id'] for row in rows} == {1}
    assert {row['relevant_movies'] for row in rows} == {
        1
    }  # repeated future movie 9 counted once
    for row in rows:
        if row['method'].endswith('_unseen'):
            assert row['seen_fraction'] == 0
            assert not set(json.loads(row['recommendations'])) & {1, 2}
    popularity = next(row for row in rows if row['method'] == 'popularity')
    assert json.loads(popularity['recommendations'])[0] == 9


def test_future_rating_labels_do_not_change_recommendations(inputs):
    folder, movies, protocol = inputs
    before, first = evaluate(folder, movies, protocol)
    test = pd.read_csv(folder / 'ratings_02.csv')
    test.loc[(test.userId == 1) & (test.movieId == 9), 'movieId'] = 10
    test.to_csv(folder / 'ratings_02.csv', index=False)
    after, second = evaluate(folder, movies, protocol)
    assert [r['recommendations'] for r in first] == [
        r['recommendations'] for r in second
    ]
    assert (
        before['metrics']['popularity']['recall']
        != after['metrics']['popularity']['recall']
    )


def test_chunk_size_and_user_seed_are_reproducible(inputs):
    folder, movies, protocol = inputs
    first, rows = evaluate(folder, movies, protocol)
    second, other = evaluate(folder, movies, replace(protocol, chunk_size=10000))
    assert first['metrics'] == second['metrics'] and rows == other
    ids = list(range(20))
    assert (
        sample_candidates(ids, 10, 42, 7)
        == np.random.default_rng([42, 7]).choice(ids, 10, replace=False).tolist()
    )


def test_overlap_is_rejected_and_boundary_ties_are_removed(inputs):
    folder, movies, protocol = inputs
    train = pd.read_csv(folder / 'ratings_01.csv')
    train.loc[0, 'timestamp'] = 30
    train.to_csv(folder / 'ratings_01.csv', index=False)
    with pytest.raises(ValueError, match='Chronological leakage'):
        evaluate(folder, movies, protocol)
    train.loc[0, 'timestamp'] = 20
    train.to_csv(folder / 'ratings_01.csv', index=False)
    report, _ = evaluate(folder, movies, protocol)
    assert report['coverage']['boundary_training_rows_excluded'] == 1


def test_unknown_future_items_are_outside_historical_catalog(inputs):
    folder, movies, protocol = inputs
    metadata = pd.read_csv(movies)
    metadata.loc[len(metadata)] = [999, 'Future movie', 'Action']
    metadata.to_csv(movies, index=False)
    test = pd.read_csv(folder / 'ratings_02.csv')
    test.loc[(test.userId == 1) & (test.movieId == 9), 'movieId'] = 999
    test.to_csv(folder / 'ratings_02.csv', index=False)
    report, rows = evaluate(folder, movies, protocol)
    assert report['status'] == 'no_evaluable_users' and not rows
    assert report['coverage']['sampled_positive_rows_outside_catalog'] == 2
    assert report['metrics']['popularity']['recall'] is None


def test_report_is_finite_and_final_holdout_requires_acknowledgement(inputs, tmp_path):
    folder, movies, protocol = inputs
    report, rows = evaluate(folder, movies, protocol)
    output = tmp_path / 'out'
    write_report(output, report, rows)
    assert json.loads((output / 'report.json').read_text())['status'] == 'evaluated'
    assert (output / 'per_user.csv').exists() and '# Offline' in (
        output / 'summary.md'
    ).read_text()
    with pytest.raises(SystemExit):
        main(
            [
                '--ratings-dir',
                str(folder),
                '--movies-csv',
                str(movies),
                '--train-through',
                '7',
                '--test-part',
                '8',
                '--output',
                str(tmp_path / 'final'),
            ]
        )


def test_changed_input_is_detected_before_report(inputs, monkeypatch):
    import src.evaluation.recommendations as module

    folder, movies, protocol = inputs
    original = module.digest

    def mutate_after_read(path):
        value = original(path)
        if path.name == 'ratings_01.csv':
            with path.open('a') as stream:
                stream.write('\n')
        return value

    monkeypatch.setattr(module, 'digest', mutate_after_read)
    with pytest.raises(ValueError, match='Input files changed'):
        evaluate(folder, movies, protocol)
