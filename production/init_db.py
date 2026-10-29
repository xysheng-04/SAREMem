"""Explicit creation of a NEW empty database; never migrate or overwrite an existing one."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        parser.error('use an absolute dedicated persistence path')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    from aml_api.store import SQLiteStore
    SQLiteStore(path)
    print('Initialized a new empty AML database. No provider requests made.')


if __name__ == '__main__':
    main()
