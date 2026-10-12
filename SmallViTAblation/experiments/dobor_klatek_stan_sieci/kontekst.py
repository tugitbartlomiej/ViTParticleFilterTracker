"""Zapisuje oryginalne obrazy gęstego train bez etykiet, aby nie fałszować czasu.

Kontekst ma dokładnie ID istniejącego cache DINO; jego cechy nie są używane.
Wynik nie może zawierać grup val/test. Piksele są kopiowane bajt w bajt z ZIP.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import zipfile
import numpy as np

from .dane import sha256, write_json


def export_context(cache_dir, dataset_root, materialized_pool, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    from .dane import read_jsonl
    pool = read_jsonl(materialized_pool)
    groups = {r['video_id']: str(r['group_id']) for r in pool if r.get('split') == 'train'}
    nontrain = {str(r['group_id']) for r in pool if r.get('split') != 'train'}
    by = defaultdict(set)
    for chunk in sorted(Path(cache_dir).glob('*.npz')):
        with np.load(chunk, allow_pickle=False) as data:
            for fid in data['frame_ids']:
                *_, video, number = str(fid).split('/')
                if video not in groups or groups[video] in nontrain:
                    raise ValueError(f'context outside known train groups: {fid}')
                by[video].add(int(number))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.partial.tar')
    rows = []
    total_bytes = 0
    started = time.monotonic()
    with tarfile.open(tmp, 'x') as tar:
        for vi, (video,numbers) in enumerate(sorted(by.items())):
            archive = Path(dataset_root)/'3_Object_Tracking/annotations'/f'{video}.zip'
            with zipfile.ZipFile(archive) as z:
                csvname = [n for n in z.namelist() if PurePosixPath(n).name == 'annotation.csv']
                if len(csvname) != 1:
                    raise ValueError(f'{archive}: annotation.csv not unique')
                parent = PurePosixPath(csvname[0]).parent
                found = set()
                for raw in csv.DictReader(io.StringIO(z.read(csvname[0]).decode('utf-8-sig'))):
                    number = int(raw['frame_number'])
                    if number not in numbers:
                        continue
                    payload = z.read(str(parent/PurePosixPath(raw['filename'])))
                    name = f'images/{video}_{number:08d}.jpg'
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
                    total_bytes += len(payload)
                    found.add(number)
                    rows.append(dict(frame_id=f'cataract_lmm/tracking/{video}/{number:08d}',
                        path=name, split='train', video_id=video, group_id=groups[video],
                        frame_index=number, time_unit='frames'))
                if found != numbers:
                    raise ValueError(f'{video}: missing context frames {numbers-found}')
            print(f'[context] {vi+1}/{len(by)} clips, {len(rows)} frames, {time.monotonic()-started:.0f}s', flush=True)
        payload = ''.join(json.dumps(r)+'\n' for r in rows).encode()
        info = tarfile.TarInfo('pool_manifest.jsonl')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    tmp.rename(target)
    write_json(target.with_suffix('.json'), dict(frames=len(rows), videos=len(by), bytes=total_bytes,
        tar_sha256=sha256(target), pixels='original_archive_bytes', labels_exported=False,
        split='train', pool_sha256=sha256(materialized_pool), seconds=time.monotonic()-started))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir','dataset-root','materialized-pool','out'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    export_context(args.cache_dir,args.dataset_root,args.materialized_pool,args.out)


if __name__ == '__main__':
    main()
