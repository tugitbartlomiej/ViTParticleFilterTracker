"""Pełne pary obrazów i podział po filmach, bez masek ani poolingu."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / HERE.name
PAIRS = ROOT / 'experiments/maly_vit_test_orientacji_narzedzia_400'
SPLIT = ROOT / 'experiments/maly_vit_tozsamosc_czasteczki_pary_400/PODZIAL.json'
CKPT = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
PAPER = Path('F:/Studia/Articles/Zbiory/JEPA/PDF_all/kim18d.pdf')
SEED = 20261005
D = 600 * 768
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def js(p, obj):
    p = Path(p)
    assert p.resolve().is_relative_to(HERE) or p.resolve().is_relative_to(OUT)
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding='utf-8')
    for k in range(12):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(.1 * (k + 1))
    raise RuntimeError('Atomic JSON replacement failed: ' + str(p))
def load():
    pairs = read(PAIRS / 'PARY.json')
    rows = read(PAIRS / 'manifest.json')
    split = read(SPLIT)
    assert len(pairs) == len(rows) == len(split['records']) == 400
    records = []
    for i, (p, r, s) in enumerate(zip(pairs, rows, split['records'])):
        assert p['frame_id'] == r['frame_id'] == s['frame_id']
        assert s['source_index'] == p['comparison_token_row'] == i
        assert split['films'][r['video_id']] == s['split']
        for kind, y in [('original', 1), ('synthetic', -1)]:
            path = PAIRS / p[kind]
            assert path.exists(), path
            records.append(dict(index=len(records), pair_index=i, film=r['video_id'], frame_id=p['frame_id'], split=s['split'], kind=kind, label=y, path=str(path), sha256=sha(path)))
        assert records[-1]['sha256'] == p['synthetic_sha256']
    ids = {s: np.array([r['index'] for r in records if r['split'] == s]) for s in ['train', 'val', 'test']}
    films = {s: sorted({r['film'] for r in records if r['split'] == s}) for s in ids}
    assert [len(ids[s]) for s in ids] == [480, 150, 170]
    assert [len(films[s]) for s in ids] == [69, 23, 24]
    assert not (set(films['train']) & set(films['val']) or set(films['train']) & set(films['test']) or set(films['val']) & set(films['test']))
    return rows, pairs, records, ids, films
