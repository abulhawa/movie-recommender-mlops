"""Train interaction ALS/two-tower candidates without changing deployed KNN."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits

from src.evaluation.recommendations import digest, write_report
from src.evaluation.retrieval import evaluate_retrieval
from src.train_service.interaction_data import load_interactions
from src.train_service.retrieval import RetrievalModel, save_bundle


def track(output, report, bundles):
    import mlflow
    import pandas as pd
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    from src.evaluation.tracking import PERSONAL_URI

    load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
    uri = os.getenv('MLFLOW_TRACKING_URI', PERSONAL_URI)
    if uri.rstrip('/') != PERSONAL_URI:
        raise ValueError('Only personal DagsHub writes are allowed')
    if not os.getenv('MLFLOW_TRACKING_PASSWORD') and os.getenv('DAGSHUB_TOKEN'):
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['DAGSHUB_TOKEN']
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment('movie-interaction-retrieval')
    if (output / 'mlflow_upload.json').exists():
        raise ValueError(
            'Upload receipt already exists; inspect recorded runs before retrying'
        )
    receipt = {}
    for name, directory in bundles.items():
        with mlflow.start_run(run_name=name) as run:
            # Persist immediately, so an interrupted upload can be inspected rather than retrained.
            receipt[name] = {'run_id': run.info.run_id, 'status': 'uploading'}
            (output / 'mlflow_upload.json').write_text(json.dumps(receipt, indent=2))
            mlflow.log_params(report['protocol'])
            mlflow.set_tags(
                {
                    'model_type': name,
                    'label_type': 'observed_rating_relevance',
                    'evaluation.scope': 'historical_development',
                    'deployment_eligible': 'false',
                    'source_commit': report['source_commit'],
                    'source_working_tree_dirty': str(
                        report['source_working_tree_dirty']
                    ),
                }
            )
            metrics = {
                key: value
                for key, value in report['metrics'][name].items()
                if value is not None
            }
            metrics.update(report['training'][name])
            for baseline in ['popularity_unseen', 'genre_knn_unseen']:
                for metric, value in report['metrics'].get(baseline, {}).items():
                    if value is not None:
                        metrics[f'baseline.{baseline}.{metric}'] = value
            mlflow.log_metrics(metrics)
            if name == 'interaction_two_tower':
                for epoch, loss in enumerate(report['training_losses'], 1):
                    mlflow.log_metric('train_bpr_loss', loss, step=epoch)
            mlflow.log_artifact(str(output / 'report.json'), artifact_path='evaluation')
            mlflow.log_artifact(str(output / 'summary.md'), artifact_path='evaluation')
            with tempfile.TemporaryDirectory() as temporary:
                model_path = Path(temporary) / 'model'
                # Package only the serving wrapper, never unrelated app/account files.
                package = Path(temporary) / 'src'
                (package / 'train_service').mkdir(parents=True)
                (package / '__init__.py').write_text('')
                (package / 'train_service/__init__.py').write_text('')
                shutil.copyfile(
                    Path(__file__).with_name('retrieval.py'),
                    package / 'train_service/retrieval.py',
                )
                with np.load(directory / 'vectors.npz') as values:
                    example = pd.DataFrame(
                        {'user_id': [int(values['user_ids'][0])], 'k': [10]}
                    )
                mlflow.pyfunc.save_model(
                    path=str(model_path),
                    python_model=RetrievalModel(),
                    artifacts={'bundle': str(directory)},
                    code_paths=[str(package)],
                    input_example=example,
                    signature=ModelSignature(
                        inputs=Schema(
                            [
                                ColSpec('long', 'user_id'),
                                ColSpec('long', 'k', required=False),
                            ]
                        ),
                        outputs=Schema(
                            [
                                ColSpec('long', 'user_id'),
                                ColSpec('string', 'recommendations'),
                            ]
                        ),
                    ),
                    pip_requirements=[
                        f'mlflow=={importlib.metadata.version("mlflow")}',
                        f'numpy=={np.__version__}',
                        f'pandas=={importlib.metadata.version("pandas")}',
                        f'scipy=={importlib.metadata.version("scipy")}',
                    ],
                )
                mlflow.log_artifacts(str(model_path), artifact_path='model')
            model_uri = f'runs:/{run.info.run_id}/model'
            version = mlflow.register_model(
                model_uri, f'movie-{name.replace("_", "-")}'
            )
            receipt[name].update(
                model_version=str(version.version), model_uri=model_uri
            )
            (output / 'mlflow_upload.json').write_text(json.dumps(receipt, indent=2))
        client = mlflow.MlflowClient()
        saved = client.get_run(run.info.run_id)
        if saved.info.status != 'FINISHED' or any(
            saved.data.metrics.get(key) != value for key, value in metrics.items()
        ):
            raise RuntimeError('Remote metric readback failed')
        # Load the registered artifact and compare recommendations against the local bundle.
        loaded = mlflow.pyfunc.load_model(
            f'models:/movie-{name.replace("_", "-")}/{version.version}'
        )
        local = RetrievalModel()
        local.load_context(
            type('Context', (), {'artifacts': {'bundle': str(directory)}})()
        )
        with np.load(directory / 'vectors.npz') as vectors:
            sample_user = int(vectors['user_ids'][0])
        query = pd.DataFrame({'user_id': [sample_user], 'k': [10]})
        if not loaded.predict(query).equals(local.predict(None, query)):
            raise RuntimeError('Registered model prediction readback failed')
        receipt[name]['status'] = 'verified'
        (output / 'mlflow_upload.json').write_text(json.dumps(receipt, indent=2))
        print(
            f'Verified {name}: run={run.info.run_id}, version={version.version}',
            flush=True,
        )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings-dir', type=Path)
    parser.add_argument('--movies-csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-through', type=int, default=6)
    parser.add_argument('--test-part', type=int, default=7)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--als-iterations', type=int, default=10)
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--history-length', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-train-users', type=int, default=0)
    parser.add_argument(
        '--max-pairs', type=int, default=0, help='per-epoch positive cap; 0 uses all'
    )
    parser.add_argument('--max-users', type=int, default=1000)
    parser.add_argument('--min-history', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--use-year', action='store_true')
    parser.add_argument('--log-mlflow', action='store_true')
    parser.add_argument(
        '--upload-only',
        action='store_true',
        help='upload saved reports and bundles without training again',
    )
    args = parser.parse_args(argv)
    if args.upload_only:
        report = json.loads((args.output / 'report.json').read_text())
        return track(
            args.output,
            report,
            {
                'interaction_als': args.output / 'als_bundle',
                'interaction_two_tower': args.output / 'two_tower_bundle',
            },
        )
    if args.ratings_dir is None or args.movies_csv is None:
        parser.error('--ratings-dir and --movies-csv are required for training')
    if args.test_part == 8 and not args.final:
        parser.error(
            'Part 8 is reserved: explicit --final required after freezing protocol'
        )
    if (
        min(
            args.epochs,
            args.als_iterations,
            args.factors,
            args.history_length,
            args.batch_size,
            args.threads,
            args.min_history,
        )
        < 1
        or min(args.max_users, args.max_pairs, args.max_train_users) < 0
    ):
        parser.error('Invalid training/evaluation sizes')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new/empty output directory to preserve evidence')
    import torch

    from src.train_service.models import als, interaction_two_tower

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=1, user_api='blas'):
        print('Loading historical interactions', flush=True)
        data = load_interactions(
            args.ratings_dir,
            args.movies_csv,
            args.train_through,
            args.test_part,
            use_year=args.use_year,
            max_train_users=args.max_train_users,
            seed=args.seed,
        )
        print(
            f'History: {len(data.events)} pairs, {len(data.user_ids)} users, {len(data.movie_ids)} movies',
            flush=True,
        )
        training, bundles, scores = {}, {}, {}
        started = time.perf_counter()
        als_model = als.fit(
            data,
            factors=args.factors,
            iterations=args.als_iterations,
            seed=args.seed,
            threads=args.threads,
        )
        training['interaction_als'] = {'train_seconds': time.perf_counter() - started}
        user_vectors, movie_vectors = als_model.user_factors, als_model.item_factors
        scores['interaction_als'] = lambda u: movie_vectors @ user_vectors[u]
        bundles['interaction_als'] = args.output / 'als_bundle'
        save_bundle(bundles['interaction_als'], data, user_vectors, movie_vectors)
        started = time.perf_counter()
        tower, losses = interaction_two_tower.fit(
            data,
            dimension=args.factors,
            epochs=args.epochs,
            batch_size=args.batch_size,
            history_length=args.history_length,
            seed=args.seed,
            max_pairs=args.max_pairs,
        )
        training['interaction_two_tower'] = {
            'train_seconds': time.perf_counter() - started
        }
        tower_users, tower_movies = interaction_two_tower.vectors(
            tower, data, history_length=args.history_length
        )
        scores['interaction_two_tower'] = lambda u: tower_movies @ tower_users[u]
        bundles['interaction_two_tower'] = args.output / 'two_tower_bundle'
        save_bundle(bundles['interaction_two_tower'], data, tower_users, tower_movies)
        # Save learnable weights and mappings locally for research/retraining.
        torch.save(tower.state_dict(), bundles['interaction_two_tower'] / 'weights.pt')
        (bundles['interaction_two_tower'] / 'architecture.json').write_text(
            json.dumps(
                {
                    'dimension': args.factors,
                    'hidden': 64,
                    'temperature': 0.1,
                    'history_length': args.history_length,
                    'genres': data.genres,
                    'use_year': args.use_year,
                },
                indent=2,
            )
        )
        popularity = np.bincount(
            data.events.loc[data.events.rating >= 4, 'i'], minlength=len(data.movie_ids)
        ).astype('float32')
        scores['popularity_unseen'] = lambda u: popularity
        # Preserve the legacy sampling policy while excluding all historical ratings.
        from sklearn.neighbors import NearestNeighbors

        from src.evaluation.recommendations import sample_candidates

        genre_features = data.features[:, : len(data.genres)]
        knn = NearestNeighbors(algorithm='ball_tree').fit(genre_features)

        def knn_scores(u):
            group = data.events.iloc[data.starts[u] : data.starts[u + 1]]
            profile = genre_features[group.i].mean(axis=0, keepdims=True)
            positions = knn.kneighbors(
                profile,
                n_neighbors=min(len(data.movie_ids), 20 + len(data.seen[u])),
                return_distance=False,
            )[0]
            pool = [i for i in positions if i not in data.seen[u]][:20]
            selected = sample_candidates(pool, 10, args.seed, int(data.user_ids[u]))
            values = np.full(len(data.movie_ids), -1e10)
            for rank, i in enumerate(selected):
                values[i] = 10 - rank
            return values

        scores['genre_knn_unseen'] = knn_scores
        metrics, rows, coverage = evaluate_retrieval(
            data,
            args.ratings_dir / f'ratings_{args.test_part:02d}.csv',
            scores,
            min_history=args.min_history,
            max_users=args.max_users,
            seed=args.seed,
        )
    for name, directory in bundles.items():
        training[name]['bundle_bytes'] = sum(
            p.stat().st_size for p in directory.iterdir()
        )
    # Recheck all sources after training/scoring, not only after data preparation.
    source_paths = [
        args.movies_csv,
        *[
            args.ratings_dir / f'ratings_{n:02d}.csv'
            for n in range(1, args.test_part + 1)
        ],
    ]
    if any(
        digest(path) != source['sha256']
        for path, source in zip(source_paths, data.inputs)
    ):
        raise ValueError('Source data changed during experiment')
    report = {
        'status': 'evaluated' if coverage['evaluated_users'] else 'no_evaluable_users',
        'protocol': {
            key: value
            for key, value in vars(args).items()
            if key
            not in {'ratings_dir', 'movies_csv', 'output', 'log_mlflow', 'upload_only'}
        },
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'source_working_tree_dirty': bool(
            subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
        ),
        'inputs': data.inputs,
        'cutoff_timestamp': data.cutoff,
        'coverage': coverage,
        'metrics': metrics,
        'training': training,
        'training_losses': losses,
        'versions': {
            name: importlib.metadata.version(name)
            for name in ['torch', 'implicit', 'numpy', 'pandas', 'scipy']
        },
        'source_hashes': {
            str(p.relative_to(Path(__file__).resolve().parents[2])): digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name('interaction_data.py'),
                Path(__file__).with_name('retrieval.py'),
                Path(als.__file__),
                Path(interaction_two_tower.__file__),
                Path(__file__).resolve().parents[1] / 'evaluation/retrieval.py',
            ]
        },
        'limitations': [
            'Part 7 is development, not an untouched final test.',
            'Fixed epochs; no early stopping or tuning against part 7 in this command.',
            'Unobserved comparisons are weak assumptions, not dislikes.',
            'Latency is sequential local warm scoring/filtering/sorting; not p95 under API load.',
            'Coverage is catalog exposure, not proof of semantic diversity.',
            'Frozen user embeddings require re-encoding/retraining to reflect new ratings.',
            'Missing metadata/cold users and future-only items are excluded from warm metrics.',
        ],
    }
    write_report(args.output, report, rows)
    if args.log_mlflow:
        track(args.output, report, bundles)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
