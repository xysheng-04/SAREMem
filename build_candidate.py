"""Allowlist-only, reproducible package creation. No databases, credentials or tests run."""
import argparse
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def payload(root):
    files = {}
    for name in ['README.md','VERSION.json','INHERITED_SHA256.json','build_candidate.py','LICENSE','.gitignore']:
        files[name] = (root/name).read_bytes()
    for folder in ['production','testing']:
        for p in sorted((root/folder).rglob('*')):
            if p.is_symlink():
                raise ValueError('symlink forbidden')
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            relative = p.relative_to(root)
            if folder=='testing' and len(relative.parts)!=2:
                continue
            if p.suffix not in {'.py','.json'} and p.name not in {'requirements.lock.txt','config.env.example'}:
                continue
            files[str(relative)] = p.read_bytes()
    forbidden = [b'/Users/', b'/opt/data/private/', b'BEGIN PRIVATE KEY', b'sk-proj-']
    # This build script names the scanner patterns, not a runtime dependency on them.
    for name, raw in files.items():
        if name != 'build_candidate.py' and any(value in raw for value in forbidden):
            raise ValueError('private path or credential pattern in '+name)
        if re.search(rb'v[-_ ]?mem', raw, re.I):
            raise ValueError('obsolete project branding in '+name)
    inherited = json.loads(files['INHERITED_SHA256.json'])
    for name, expected in inherited.items():
        if name in {'aml_api/app.py','aml_api/service_factory.py'}:
            continue
        if sha(files['production/'+name]) != expected:
            raise ValueError('frozen source changed: '+name)
    manifest = {name:sha(raw) for name,raw in sorted(files.items())}
    identity = sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
    files['MANIFEST.json'] = (json.dumps({'identity':identity,'files':manifest},indent=2)+'\n').encode()
    files['SHA256SUMS'] = ''.join(f'{sha(raw)}  {name}\n' for name,raw in sorted(files.items())).encode()
    return files, identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    files,identity=payload(Path(__file__).resolve().parent)
    version=json.loads(files['VERSION.json'])['version']
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as destination, gzip.GzipFile(fileobj=destination,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name,raw in sorted(files.items()):
                info=tarfile.TarInfo(version+'/'+name)
                info.size=len(raw);info.mode=0o644;info.mtime=0
                archive.addfile(info,io.BytesIO(raw))
    print(json.dumps({'identity':identity,'archive_sha256':sha(output.read_bytes()),'files':len(files)}))


if __name__=='__main__':
    main()
