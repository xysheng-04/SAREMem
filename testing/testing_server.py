"""Optional diagnostic test launcher, using the actual competition app and service."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'production'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    from configuration import load_config
    from server import configured_settings, build_app
    from aml_api.production.service import ProductionService
    from aml_api.production.process import quiet
    load_config(args.config)
    settings = configured_settings()
    test_root = (ROOT.parent/'state/testing').resolve()
    if not settings.storage_path.is_relative_to(test_root):
        parser.error('test launcher requires state/testing persistence')
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):
        parser.error('diagnostic artifacts must be under testing/outputs')
    output.mkdir(parents=True,exist_ok=False)
    lock = threading.Lock()
    def record(event):
        with lock, (output/'requests.jsonl').open('a') as f:
            f.write(json.dumps(event,allow_nan=False)+'\n')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    quiet()
    app = build_app(settings=settings,service=ProductionService(settings,diagnostics=record))
    import uvicorn
    uvicorn.run(app, host=os.environ.get('AML_BIND_HOST','127.0.0.1'),
                port=int(os.environ.get('AML_PORT','8001')), access_log=False,
                log_config=None, log_level='critical', server_header=False)


if __name__ == '__main__':
    main()
