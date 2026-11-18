"""Snapshot the two processed CSVs that belong to one registered serving bundle."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

FILES = ('movie_matrix.csv', 'user_matrix.csv')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(files):
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def sample_user_id(directory):
    with (Path(directory) / 'user_matrix.csv').open(newline='') as stream:
        row = next(csv.DictReader(stream), None)
    if not row or 'userId' not in row:
        raise ValueError('Serving snapshot requires a nonempty user_matrix with userId')
    return int(row['userId'])


def snapshot_processed_data(source, destination):
    """Copy once before training; train and export from the same private snapshot.

    The source pair must not be edited concurrently. This snapshots the exact
    files consumed by training, not a transaction across upstream preparation jobs.
    """
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(source / name, destination / name)
    files = {
        name: {
            'sha256': sha256(destination / name),
            'size': (destination / name).stat().st_size,
        }
        for name in FILES
    }
    manifest = {
        'format_version': 1,
        'files': files,
        'data_fingerprint': fingerprint(files),
        'sample_user_id': sample_user_id(destination),
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_snapshot(directory, expected_fingerprint):
    """Reject missing, modified or mismatched data before the model is served."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or set(manifest.get('files', {})) != set(
        FILES
    ):
        raise ValueError('Unsupported or incomplete serving-data manifest')
    for name in FILES:
        path = directory / name
        recorded = manifest['files'][name]
        if (
            path.stat().st_size != recorded['size']
            or sha256(path) != recorded['sha256']
        ):
            raise ValueError(f'Serving snapshot checksum mismatch: {name}')
    actual = fingerprint(manifest['files'])
    if actual != manifest.get('data_fingerprint') or actual != expected_fingerprint:
        raise ValueError('Model and serving snapshot fingerprints disagree')
    if manifest.get('sample_user_id') != sample_user_id(directory):
        raise ValueError('Serving snapshot sample user does not match the data')
    return manifest


def attach_snapshot(model_directory, snapshot_directory):
    """Publish profiles with the model; the schema declares the mandatory snapshot."""
    model_directory = Path(model_directory)
    manifest = json.loads((Path(snapshot_directory) / 'manifest.json').read_text())
    validate_snapshot(snapshot_directory, manifest['data_fingerprint'])
    shutil.copytree(snapshot_directory, model_directory / 'serving_data')
    path = model_directory / 'training_schema.json'
    schema = json.loads(path.read_text())
    schema['serving_data_fingerprint'] = manifest['data_fingerprint']
    path.write_text(json.dumps(schema, indent=2))
    return manifest
