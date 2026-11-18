"""Offline setup checks use temporary files and fake socket metadata only."""

import importlib.util
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'setup_local', Path(__file__).resolve().parents[2] / 'airflow/setup_local.py'
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class FakeSocket:
    def __init__(self, gid=None):
        self.gid = gid

    def stat(self):
        if self.gid is None:
            raise PermissionError('unavailable')
        return SimpleNamespace(st_gid=self.gid)


def complete_settings(root):
    values = {key: 'private-test-value' for key in setup.REQUIRED}
    values.update(
        MODEL_TYPE='knn',
        PREDICT_MODEL_TYPE='knn',
        PREDICT_MODEL_VERSION='latest',
        AIRFLOW_DOCKER_GID='321',
        AIRFLOW_DOCKER_PROJECT_DIR=root.as_posix(),
    )
    return values


def env_content(values):
    return ''.join(f'{key}={value}\n' for key, value in values.items())


def test_fills_missing_defaults_without_copying_example_credentials(tmp_path):
    example = (
        'MODEL_TYPE=knn\nPREDICT_MODEL_TYPE=knn\nPREDICT_MODEL_VERSION=latest\n'
        'DAGSHUB_BUCKET=public-bucket\nDAGSHUB_TOKEN=do-not-copy-this-token\n'
        'DATA_COLLECTION_PASSWORD=do-not-copy-this-password\n'
        'PREDICT_SMOKE_PASSWORD=do-not-copy-this-password\n'
        'AUTH_SECRET_KEY=do-not-copy-this-key\n'
    )
    result = setup.fill_settings('', example, tmp_path, '321')
    values = setup.read_settings(result)
    assert values['DAGSHUB_BUCKET'] == 'public-bucket'
    assert values['PREDICT_MODEL_VERSION'] == 'latest'
    assert values['AIRFLOW_DOCKER_GID'] == '321'
    assert values['AIRFLOW_DOCKER_PROJECT_DIR'] == tmp_path.as_posix()
    assert len(values['AIRFLOW_FERNET_KEY']) == 44
    assert len(values['AUTH_SECRET_KEY']) == 64
    assert values['GRAFANA_ADMIN_PASSWORD']
    assert (
        not {'DAGSHUB_TOKEN', 'DATA_COLLECTION_PASSWORD', 'PREDICT_SMOKE_PASSWORD'}
        & values.keys()
    )
    assert 'do-not-copy' not in result


def test_preserves_existing_quoted_secrets_comments_and_custom_selections(tmp_path):
    original = (
        '# Keep this comment\n'
        'AUTH_SECRET_KEY="existing # secret" # Keep inline comment\n'
        "GRAFANA_ADMIN_PASSWORD='existing password'\n"
        'AIRFLOW_POSTGRES_PASSWORD=existing#password\n'
        'PREDICT_MODEL_VERSION=@my-owned-alias\n'
        'MODEL_TYPE=two_tower\n'
        'AIRFLOW_DOCKER_GID=321 # socket group\n'
        'export AIRFLOW_ADMIN_PASSWORD = "" # fill this one\n'
        'AIRFLOW_JWT_SECRET= # fill this too\n'
    )
    result = setup.fill_settings(original, 'MODEL_TYPE=knn\n', tmp_path, '321')
    for line in original.splitlines()[:7]:
        assert line + '\n' in result
    assert ' # socket group\n' in result
    assert ' # fill this one\n' in result
    assert ' # fill this too\n' in result
    values = setup.read_settings(result)
    assert values['AUTH_SECRET_KEY'] == 'existing # secret'
    assert values['AIRFLOW_POSTGRES_PASSWORD'] == 'existing#password'
    assert values['AIRFLOW_ADMIN_PASSWORD']
    assert values['AIRFLOW_JWT_SECRET']
    assert setup.fill_settings(result, '', tmp_path, '321') == result


def test_stale_socket_group_is_checked_then_repaired_without_rotation(tmp_path, capsys):
    original = env_content(complete_settings(tmp_path)).replace(
        'AIRFLOW_DOCKER_GID=321', 'AIRFLOW_DOCKER_GID="0" # retain comment'
    )
    env_path = tmp_path / '.env'
    env_path.write_text(original)
    assert setup.main(['--check'], root=tmp_path, socket_path=FakeSocket(321)) == 1
    assert 'AIRFLOW_DOCKER_GID' in capsys.readouterr().out
    assert env_path.read_text() == original
    assert setup.main([], root=tmp_path, socket_path=FakeSocket(321)) == 0
    updated = env_path.read_text()
    assert 'AIRFLOW_DOCKER_GID=321 # retain comment\n' in updated
    assert setup.read_settings(updated)['AUTH_SECRET_KEY'] == 'private-test-value'
    assert 'private-test-value' not in capsys.readouterr().out


def test_check_reports_missing_keys_without_writing_or_printing_values(
    tmp_path, capsys
):
    content = 'DAGSHUB_TOKEN="sensitive-test-token" # stays private\n'
    env_path = tmp_path / '.env'
    env_path.write_text(content)
    assert (
        setup.main(
            ['--check', '--deployment'], root=tmp_path, socket_path=FakeSocket(321)
        )
        == 1
    )
    output = capsys.readouterr().out
    assert 'MLFLOW_TRACKING_USERNAME' in output
    assert 'DATA_COLLECTION_PASSWORD' in output
    assert 'PREDICT_SMOKE_PASSWORD' in output
    assert 'PREDICT_MODEL_VERSION' in output
    assert 'sensitive-test-token' not in output
    assert env_path.read_text() == content
    assert not (tmp_path / 'airflow').exists()


def test_check_does_not_create_missing_env(tmp_path):
    assert setup.main(['--check'], root=tmp_path, socket_path=FakeSocket()) == 1
    assert list(tmp_path.iterdir()) == []


def test_setup_saves_generated_values_but_requires_real_api_credentials(
    tmp_path, capsys
):
    (tmp_path / '.env.example').write_text('MODEL_TYPE=knn\nPREDICT_MODEL_TYPE=knn\n')
    assert setup.main([], root=tmp_path, socket_path=FakeSocket()) == 1
    values = setup.read_settings((tmp_path / '.env').read_text())
    assert values['AUTH_SECRET_KEY']
    assert values['GRAFANA_ADMIN_PASSWORD']
    assert values['AIRFLOW_DOCKER_GID'] == '0'
    output = capsys.readouterr().out
    assert 'Docker socket group unavailable' in output
    assert 'DAGSHUB_TOKEN' in output
    assert values['AUTH_SECRET_KEY'] not in output


def test_unavailable_socket_preserves_existing_group(tmp_path):
    result = setup.fill_settings(
        'AIRFLOW_DOCKER_GID=765 # custom\n', '', tmp_path, None
    )
    assert 'AIRFLOW_DOCKER_GID=765 # custom\n' in result


def test_latest_is_standalone_only_and_alias_is_not_changed(tmp_path):
    values = complete_settings(tmp_path)
    assert setup.validate_settings(values, tmp_path, '321') == []
    errors = setup.validate_settings(values, tmp_path, '321', deployment=True)
    assert len(errors) == 1 and 'PREDICT_MODEL_VERSION' in errors[0]
    values['PREDICT_MODEL_VERSION'] = '@my-owned-alias'
    assert setup.validate_settings(values, tmp_path, '321', deployment=True) == []


@pytest.mark.parametrize('selector', ['', '@', '@alias with spaces', 'latest', '3'])
def test_deployment_requires_explicit_nonempty_alias(tmp_path, selector):
    values = complete_settings(tmp_path)
    values['PREDICT_MODEL_VERSION'] = selector
    assert any(
        'PREDICT_MODEL_VERSION' in error
        for error in setup.validate_settings(values, tmp_path, '321', deployment=True)
    )


@pytest.mark.parametrize(
    'key,value',
    [
        ('MODEL_TYPE', 'two_tower'),
        ('PREDICT_MODEL_TYPE', 'two_tower'),
        ('AIRFLOW_DOCKER_GID', '-1'),
        ('AIRFLOW_DOCKER_GID', 'docker'),
        ('PREDICT_MODEL_VERSION', '0'),
        ('PREDICT_MODEL_VERSION', 'unexpected'),
    ],
)
def test_rejects_invalid_demo_selections(tmp_path, key, value):
    values = complete_settings(tmp_path)
    values[key] = value
    errors = setup.validate_settings(values, tmp_path, None, deployment=True)
    assert any(key in error for error in errors)
    assert all('private-test-value' not in error for error in errors)


def test_project_path_must_exist_and_identify_this_root(tmp_path):
    values = complete_settings(tmp_path)
    other = tmp_path / 'other'
    other.mkdir()
    for path in (str(tmp_path / 'absent'), str(other), '.'):
        values['AIRFLOW_DOCKER_PROJECT_DIR'] = path
        assert any(
            'AIRFLOW_DOCKER_PROJECT_DIR' in error
            for error in setup.validate_settings(values, tmp_path, '321')
        )


def test_windows_daemon_path_keeps_drive_mapping_and_spaces():
    root = PureWindowsPath('D:/Git/Liora/Final Movie Reco')
    assert setup.docker_project_path(root, windows=True) == (
        '/run/desktop/mnt/host/d/Git/Liora/Final Movie Reco'
    )


def test_leading_hash_password_is_preserved_but_spaced_comment_is_empty(tmp_path):
    content = 'AUTH_SECRET_KEY=#existing-secret\nAIRFLOW_JWT_SECRET= # fill this\n'
    result = setup.fill_settings(content, '', tmp_path, None)
    assert 'AUTH_SECRET_KEY=#existing-secret\n' in result
    values = setup.read_settings(result)
    assert values['AUTH_SECRET_KEY'] == '#existing-secret'
    assert values['AIRFLOW_JWT_SECRET']
    assert ' # fill this\n' in result


def test_standalone_two_tower_remains_valid(tmp_path):
    values = complete_settings(tmp_path)
    values.update(MODEL_TYPE='two_tower', PREDICT_MODEL_TYPE='two_tower')
    assert setup.validate_settings(values, tmp_path, '321') == []
    assert setup.validate_settings(values, tmp_path, '321', deployment=True)


def test_non_ascii_version_is_rejected_before_model_loading(tmp_path):
    values = complete_settings(tmp_path)
    values['PREDICT_MODEL_VERSION'] = '١'
    assert any(
        'PREDICT_MODEL_VERSION' in error
        for error in setup.validate_settings(values, tmp_path, '321')
    )
