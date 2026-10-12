"""Download completed FastGAN once; verify archive and every extracted file."""
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from datetime import datetime
from pakuj_wynik import sha

HERE = Path(__file__).resolve().parent
REMOTE = 'eden-cluster:smallvit_runs/2026-09-23_fastgan/'


def fetch(name, destination):
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', REMOTE + name, str(destination)], check=True)


def main():
    out = HERE / 'wyniki/eden_20260923_50k'
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / 'DOWNLOAD.json'
    fetch('DOWNLOAD.json', manifest_file.with_suffix('.json.part'))
    manifest = json.loads(manifest_file.with_suffix('.json.part').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or manifest['step'] != 50000 or manifest['archive'] != 'wynik_koncowy.tar':
        raise RuntimeError('Unexpected download manifest')
    bundle = out / 'wynik_koncowy.tar'
    if not bundle.exists() or sha(bundle) != manifest['archive_sha256']:
        part = bundle.with_suffix('.tar.part')
        fetch('wynik_koncowy.tar', part)
        if part.stat().st_size != manifest['archive_bytes'] or sha(part) != manifest['archive_sha256']:
            raise RuntimeError('Archive verification failed')
        os.replace(part, bundle)
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        if len(members) != len(manifest['files']) or {m.name for m in members} != set(manifest['files']):
            raise RuntimeError('Archive member list differs from manifest')
        for m in members:
            p = PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name:
                raise RuntimeError('Unsafe archive path')
            dest = out.joinpath(*p.parts)
            expected = manifest['files'][m.name]
            if dest.exists():
                if sha(dest) != expected['sha256']:
                    raise RuntimeError(f'Existing different file: {dest}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + '.part')
            with archive.extractfile(m) as src, part.open('wb') as dst:
                for chunk in iter(lambda: src.read(8 * 1024**2), b''):
                    dst.write(chunk)
            if part.stat().st_size != expected['bytes'] or sha(part) != expected['sha256']:
                raise RuntimeError(f'File verification failed: {m.name}')
            os.replace(part, dest)
    os.replace(manifest_file.with_suffix('.json.part'), manifest_file)
    check = dict(status='downloaded_verified', verified_at=datetime.now().astimezone().isoformat(),
                 checkpoint=str(out / 'results_50k/checkpoint_050000.pth'), files=len(manifest['files']),
                 archive_sha256=manifest['archive_sha256'])
    (out / 'POBRANIE_OK.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
