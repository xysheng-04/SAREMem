"""Small explicit dotenv reader. No shell execution or implicit .env loading."""
import os
from pathlib import Path


def load_config(path):
    path = Path(path).resolve(strict=True)
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, sep, value = line.partition('=')
        if not sep or not name.startswith('AML_') or name in values:
            raise ValueError('invalid configuration entry')
        values[name] = value.strip()
    if 'AML_STORAGE_PATH' in values:
        values['AML_STORAGE_PATH'] = str((path.parent/values['AML_STORAGE_PATH']).resolve())
    os.environ.update(values)
