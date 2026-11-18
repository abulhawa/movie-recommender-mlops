"""Fill missing local settings and check readiness without printing secrets.

Use --check for read-only validation and --deployment to require the alias
selector used by the Airflow deployment DAG. No containers or APIs are started.
"""

import argparse
import base64
import os
import re
import secrets
from pathlib import Path

# Only public fields may be copied, even if the example contains a private value.
PUBLIC_DEFAULTS = {
    'GRAFANA_ADMIN_USER',
    'DAGSHUB_S3_ENDPOINT_URL',
    'DAGSHUB_BUCKET',
    'MODEL_TYPE',
    'PREDICT_MODEL_TYPE',
    'PREDICT_MODEL_NAME',
    'PREDICT_MODEL_VERSION',
    'MLFLOW_TRACKING_URI',
    'MLFLOW_TRACKING_USERNAME',
    'AIRFLOW_POSTGRES_USER',
    'AIRFLOW_POSTGRES_DB',
    'AIRFLOW_ADMIN_USER',
    'DATA_COLLECTION_USERNAME',
    'PREDICT_SMOKE_USERNAME',
}
REQUIRED = (
    'GRAFANA_ADMIN_USER',
    'GRAFANA_ADMIN_PASSWORD',
    'DAGSHUB_S3_ENDPOINT_URL',
    'DAGSHUB_BUCKET',
    'DAGSHUB_TOKEN',
    'MLFLOW_TRACKING_URI',
    'MLFLOW_TRACKING_USERNAME',
    'MODEL_TYPE',
    'PREDICT_MODEL_TYPE',
    'PREDICT_MODEL_VERSION',
    'AIRFLOW_POSTGRES_USER',
    'AIRFLOW_POSTGRES_DB',
    'AIRFLOW_POSTGRES_PASSWORD',
    'AIRFLOW_FERNET_KEY',
    'AIRFLOW_JWT_SECRET',
    'AIRFLOW_ADMIN_USER',
    'AIRFLOW_ADMIN_PASSWORD',
    'AIRFLOW_DOCKER_PROJECT_DIR',
    'AIRFLOW_DOCKER_GID',
    'DATA_COLLECTION_USERNAME',
    'DATA_COLLECTION_PASSWORD',
    'PREDICT_SMOKE_USERNAME',
    'PREDICT_SMOKE_PASSWORD',
    'AUTH_SECRET_KEY',
)
ASSIGNMENT = re.compile(r'^(\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*)(.*)$')


def split_value(raw, *, comment_at_start=False):
    """Separate a dotenv value from its trailing whitespace/comment."""
    if raw.startswith(('"', "'")):
        quote = raw[0]
        escaped = False
        for index, char in enumerate(raw[1:], 1):
            if char == quote and not escaped:
                value = raw[1:index]
                if quote == '"':
                    value = re.sub(r'\\(["\\])', r'\1', value)
                return value, raw[index + 1 :]
            escaped = char == '\\' and not escaped
        return '', ''
    comment = re.search(r'\s#', raw)
    if comment_at_start and raw.startswith('#'):
        comment = re.match(r'#', raw)
    value_end = comment.start() if comment else len(raw)
    value = raw[:value_end].rstrip()
    suffix = raw[len(value) :]
    if suffix.startswith('#'):
        suffix = ' ' + suffix
    return value, suffix


def read_settings(content):
    values = {}
    for line in content.splitlines():
        match = ASSIGNMENT.match(line)
        if match:
            values[match[2]] = split_value(
                match[3], comment_at_start=match[1][-1].isspace()
            )[0]
    return values


def set_value(content, key, value):
    """Change only the last assignment's value, keeping layout and comments."""
    lines = content.splitlines(keepends=True)
    matches = [
        i
        for i, line in enumerate(lines)
        if (match := ASSIGNMENT.match(line.rstrip('\r\n'))) and match[2] == key
    ]
    if matches:
        index = matches[-1]
        line = lines[index]
        newline = (
            '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
        )
        match = ASSIGNMENT.match(line.rstrip('\r\n'))
        _, suffix = split_value(match[3], comment_at_start=match[1][-1].isspace())
        lines[index] = f'{match[1]}{value}{suffix}{newline}'
        return ''.join(lines)
    separator = '' if not content or content.endswith('\n') else '\n'
    return f'{content}{separator}{key}={value}\n'


def docker_project_path(root, windows=None):
    if windows is None:
        windows = os.name == 'nt'
    if windows:
        drive = root.drive.rstrip(':').lower()
        relative = '/'.join(root.parts[1:])
        return f'/run/desktop/mnt/host/{drive}/{relative}'
    return root.as_posix()


def socket_group(socket_path):
    try:
        return str(socket_path.stat().st_gid)
    except OSError:
        return None


def fill_settings(content, example, root, gid):
    values = read_settings(content)
    candidates = {
        key: value
        for key, value in read_settings(example).items()
        if key in PUBLIC_DEFAULTS and value.strip()
    }
    candidates.update(
        {
            'AIRFLOW_FERNET_KEY': base64.urlsafe_b64encode(
                secrets.token_bytes(32)
            ).decode(),
            'AIRFLOW_JWT_SECRET': secrets.token_hex(32),
            'AIRFLOW_POSTGRES_PASSWORD': secrets.token_hex(24),
            'AIRFLOW_ADMIN_USER': 'airflow',
            'AIRFLOW_ADMIN_PASSWORD': secrets.token_urlsafe(24),
            'AUTH_SECRET_KEY': secrets.token_hex(32),
            'GRAFANA_ADMIN_PASSWORD': secrets.token_urlsafe(24),
            'AIRFLOW_DOCKER_PROJECT_DIR': docker_project_path(root),
            'AIRFLOW_DOCKER_GID': gid if gid is not None else '0',
        }
    )
    for key, value in candidates.items():
        if not values.get(key, '').strip():
            content = set_value(content, key, value)
    # Group membership is machine metadata; repair a stale group on this host.
    if gid is not None and read_settings(content).get('AIRFLOW_DOCKER_GID') != gid:
        content = set_value(content, 'AIRFLOW_DOCKER_GID', gid)
    return content


def validate_settings(values, root, gid, deployment=False):
    errors = [
        f'{key}: required for the full demo.'
        for key in REQUIRED
        if not values.get(key, '').strip()
    ]
    for key in ('MODEL_TYPE', 'PREDICT_MODEL_TYPE'):
        allowed = {'knn'} if deployment else {'knn', 'two_tower'}
        if values.get(key, '').strip() and values[key] not in allowed:
            guidance = (
                'select knn for Airflow deployment'
                if deployment
                else 'use knn or two_tower'
            )
            errors.append(f'{key}: {guidance}.')
    selected = values.get('PREDICT_MODEL_VERSION', '')
    alias = (
        selected.startswith('@')
        and bool(selected[1:].strip())
        and not any(char.isspace() for char in selected)
    )
    if deployment and not alias:
        errors.append(
            'PREDICT_MODEL_VERSION: set an owned @alias before deployment; '
            'latest is only for standalone serving.'
        )
    elif selected and not (
        alias
        or selected.lower() == 'latest'
        or selected.isascii()
        and selected.isdecimal()
        and int(selected) > 0
    ):
        errors.append(
            'PREDICT_MODEL_VERSION: use latest, a positive version, or @alias.'
        )
    group = values.get('AIRFLOW_DOCKER_GID', '')
    if group and not re.fullmatch(r'[0-9]+', group):
        errors.append('AIRFLOW_DOCKER_GID: must be a numeric group ID.')
    elif gid is not None and group and group != gid:
        errors.append(
            'AIRFLOW_DOCKER_GID: differs from the Docker socket; '
            'run setup without --check to update it.'
        )
    project = values.get('AIRFLOW_DOCKER_PROJECT_DIR', '')
    if project:
        # Docker Desktop's daemon path differs from the Windows filesystem path.
        if os.name == 'nt' and project == docker_project_path(root):
            local_path = root
        else:
            local_path = Path(project)
        try:
            same_root = (
                local_path.is_absolute()
                and local_path.is_dir()
                and (local_path.resolve() == root.resolve())
            )
        except (OSError, ValueError):
            same_root = False
        if not same_root:
            errors.append(
                'AIRFLOW_DOCKER_PROJECT_DIR: must identify this existing '
                'repository root (use its Docker Desktop daemon path on Windows).'
            )
    return errors


def main(argv=None, *, root=None, socket_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check', action='store_true', help='validate .env without writing'
    )
    parser.add_argument(
        '--deployment',
        action='store_true',
        help='also require PREDICT_MODEL_VERSION to use an @alias',
    )
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    socket_path = Path('/var/run/docker.sock') if socket_path is None else socket_path
    env_path = root / '.env'
    content = env_path.read_text(encoding='utf-8-sig') if env_path.exists() else ''
    gid = socket_group(socket_path)
    if gid is None:
        print(
            'Docker socket group unavailable; AIRFLOW_DOCKER_GID was not detected. '
            'Verify it on the Docker host; setup preserves the existing value or uses 0.'
        )
    if not args.check:
        example_path = root / '.env.example'
        example = (
            example_path.read_text(encoding='utf-8-sig')
            if example_path.exists()
            else ''
        )
        content = fill_settings(content, example, root, gid)
        env_path.write_text(content, encoding='utf-8')
        env_path.chmod(0o600)
        (root / 'airflow' / 'dags').mkdir(parents=True, exist_ok=True)
        print(
            'Missing local settings saved in root .env; existing secrets were preserved.'
        )
    errors = validate_settings(read_settings(content), root, gid, args.deployment)
    if errors:
        print('Setup needs attention:')
        for error in errors:
            print(f'- {error}')
        return 1
    print(
        'Local settings check passed. Account access and running services were not checked.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
