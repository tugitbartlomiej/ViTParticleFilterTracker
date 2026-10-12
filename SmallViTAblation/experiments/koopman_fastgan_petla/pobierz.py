"""Pobranie gotowego archiwum i bezpieczne rozpakowanie z kontrolą SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True,type=int);ap.add_argument('--mode',choices=['smoke','run'],default='run');ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=False)
    remote='smallvit_runs/2026-09-24_koopman_fastgan_petla';stem=f'{a.mode}_{a.job}'
    # Paths are fixed plus constrained integer/enum, never interpolated arbitrary shell text.
    command=f'~/venvs/rfdetr_nano_clean_1_10_1/bin/python ~/{{r}}/code/pakuj_wynik.py --run ~/{{r}}/results/{{s}} --out ~/{{r}}/{{s}}.tar'.format(r=remote,s=stem)
    subprocess.run(['ssh','eden-cluster',command],check=True)
    for suffix in ['.tar','.manifest.json']:
        subprocess.run(['scp',f'eden-cluster:{remote}/{stem}{suffix}',str(root)],check=True)
    archive_path=root/f'{stem}.tar';target=root/'results';target.mkdir()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            destination=(target/member.name).resolve()
            if not destination.is_relative_to(target) or not member.isfile():raise ValueError('Unsafe archive member')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,destination.open('xb') as dest:
                for b in iter(lambda:source.read(2**20),b''):dest.write(b)
    for entry in json.loads((root/f'{stem}.manifest.json').read_text()):
        path=(target/entry['file']).resolve()
        if not path.is_relative_to(target):raise ValueError('Unsafe manifest path')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for b in iter(lambda:stream.read(2**20),b''):h.update(b)
        if path.stat().st_size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:raise ValueError(f'Corrupted download: {path}')
    print(f'Verified: {target}')
if __name__=='__main__':main()
