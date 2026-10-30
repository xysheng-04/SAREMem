"""Explicit competition launcher: no ingestion, tests, model calls or DB initialization."""
import argparse
import os
from pathlib import Path


def configured_settings():
    from aml_api.config import Settings
    required = ['AML_STORAGE_PATH', 'AML_MEMORY_SYSTEM_KEY', 'AML_LLM_API_KEY',
                'AML_EMBEDDING_API_KEY', 'AML_LLM_BASE_URL', 'AML_EMBEDDING_BASE_URL',
                'AML_SEARCH_PIPELINE']
    if any(not os.environ.get(k) for k in required):
        raise ValueError('missing required competition configuration')
    if any('CHANGE_ME' in os.environ[k] for k in required):
        raise ValueError('replace all configuration placeholders')
    if os.environ['AML_SEARCH_PIPELINE'] != 'simple':
        raise ValueError('competition candidate requires explicit AML_SEARCH_PIPELINE=simple')
    if len(os.environ['AML_MEMORY_SYSTEM_KEY']) < 24:
        raise ValueError('Memory System Key must contain at least 24 characters')
    if not Path(os.environ['AML_STORAGE_PATH']).is_absolute():
        raise ValueError('AML_STORAGE_PATH must be an absolute persistent path')
    from urllib.parse import urlparse
    for name in ['AML_LLM_BASE_URL','AML_EMBEDDING_BASE_URL']:
        url = urlparse(os.environ[name])
        if url.scheme not in {'http','https'} or not url.hostname or url.username or url.password or url.query:
            raise ValueError('invalid provider URL')
    mode = os.environ.get('AML_SEED_RETRIEVAL_MODE', 'default')
    if mode not in {'default','legacy'}:
        raise ValueError('invalid seed retrieval mode')
    if mode == 'legacy' and not (Path(__file__).parent/'aml_api/legacy_seed/source.py').is_file():
        raise ValueError('legacy seed module unavailable')
    reorder_mode = os.environ.get('AML_SIMPLE_REORDER_MODE', 'stable_groups')
    if reorder_mode not in {'bounded','stable_groups'}:
        raise ValueError('invalid simple reorder mode')
    if not 1 <= int(os.environ.get('AML_SIMPLE_ASSESS_POOL', '40')) <= 80:
        raise ValueError('simple assessment pool must be within 1..80')
    if reorder_mode == 'stable_groups' and not (Path(__file__).parent/'aml_api/simple/reordering.py').is_file():
        raise ValueError('stable grouping module unavailable')
    neighbor_mode = os.environ.get('AML_SIMPLE_NEIGHBOR_MODE', 'session')
    if neighbor_mode not in {'off','session'}:
        raise ValueError('invalid simple neighbor mode')
    if neighbor_mode == 'session' and not (Path(__file__).parent/'aml_api/simple/neighbor_fill.py').is_file():
        raise ValueError('neighbor fill module unavailable')
    settings = Settings.from_env()
    if mode == 'legacy' and not 1 <= settings.retrieve_pool <= 80:
        raise ValueError('legacy seed pool must be within 1..80')
    if not settings.storage_path.is_file():
        raise ValueError('initialize a new dedicated database explicitly before starting')
    return settings


def build_app(*, service=None, settings=None):
    from aml_api.production.process import quiet
    quiet()
    from aml_api.app import create_app
    return create_app(settings or configured_settings(), service=service, production=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=os.environ.get('AML_BIND_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('AML_PORT','8000')))
    parser.add_argument('--config', help='explicit configuration file; never auto-loaded')
    args = parser.parse_args()
    if args.config:
        from configuration import load_config
        load_config(args.config)
        args.host = os.environ.get('AML_BIND_HOST', args.host)
        args.port = int(os.environ.get('AML_PORT',args.port))
    # Set before importing numpy or spawning workers. Never consume all CPU threads.
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    try:
        app = build_app()
    except Exception:
        raise SystemExit('Competition startup rejected: check required configuration and database.') from None
    import uvicorn
    from aml_api.production.http_transport import DeadlineH11Protocol
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_config=None,
                log_level='critical', server_header=False, workers=1,http=DeadlineH11Protocol)


if __name__ == '__main__':
    main()
