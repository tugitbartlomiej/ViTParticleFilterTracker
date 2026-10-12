"""Pakuje raporty, obrazy, latent i końcowe wagi bez zmiany danych źródłowych."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.run).resolve();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    entries=[]
    with tarfile.open(out,'w') as archive:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            if p.suffix not in ['.json','.jsonl','.npz','.npy','.png','.log','.csv','.md'] and p.name!='FINAL.pth':continue
            arc=p.relative_to(root).as_posix();archive.add(p,arcname=arc,recursive=False)
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for b in iter(lambda:stream.read(2**20),b''):h.update(b)
            entries.append(dict(file=arc,bytes=p.stat().st_size,sha256=h.hexdigest()))
    out.with_suffix('.manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(out),files=len(entries),bytes=out.stat().st_size)))
if __name__=='__main__':main()
