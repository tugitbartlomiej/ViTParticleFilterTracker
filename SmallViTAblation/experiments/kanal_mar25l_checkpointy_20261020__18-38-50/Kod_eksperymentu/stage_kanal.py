"""Lokalnie: zapisuje FREEZE.json (SHA Kod_eksperymentu + Opis) i paczkę tar.gz do wysłania na Eden.

Paczka trafia do Wyniki/Dane/pakiet/. Po zamrożeniu każda zmiana kodu lub opisu blokuje worker (nowy FREEZE = nowy bieg).
"""
import argparse
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kanal_io import file_sha, freeze_files, read_json, save_json  # noqa: E402


def stage(exp):
    cfg = read_json(HERE / 'config.json')
    files = freeze_files(exp / 'Kod_eksperymentu', exp / 'Opis')
    save_json(exp / 'Kod_eksperymentu' / 'FREEZE.json', dict(schema=1, experiment=cfg['experiment'],
              created_utc=datetime.now(timezone.utc).isoformat(), files=files))
    out = exp / 'Wyniki' / 'Dane' / 'pakiet'
    out.mkdir(parents=True, exist_ok=True)
    archive = out / 'pakiet_kod_opis.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for rel in sorted(list(files) + ['Kod_eksperymentu/FREEZE.json']):
            tar.add(exp / rel, arcname=rel, recursive=False)
    save_json(out / 'PAKIET.json', dict(archive=archive.name, sha256=file_sha(archive), bytes=archive.stat().st_size,
              freeze_sha256=file_sha(exp / 'Kod_eksperymentu' / 'FREEZE.json'), files=len(files) + 1,
              remote_root=cfg['remote_root']))
    print(archive, file_sha(archive))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--exp-root', type=Path, default=HERE.parent)
    stage(p.parse_args().exp_root.resolve())
