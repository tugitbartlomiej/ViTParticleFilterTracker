"""Wybór prawdziwych klatek do dodania (PLAN.md).

koopman: obraz z FastGAN (target.png) każdej kotwicy → wyszukiwanie w indeksie DINOv2 puli (wycinki narzędzia,
         `koopman_fastgan_petla`, tylko odczyt) top-90 → klatki na przemian z list kotwic, po kolei wg podobieństwa;
losowe:  losowe klatki z tej samej puli.
Oba ramiona z tą samą regułą różnorodności: bez klatek bliżej niż ±30 klatek tego samego filmu od klatek bazy
i już dodanych (diagnoza poprzedniej pętli: wybór skupiał się na sąsiednich, prawie identycznych klatkach).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

import dane as D

OKNO, TOP_K = 30, 90


def _vf(fid):
    v, f = fid.split('/')[-2:]
    return v, int(f)


class Zajete:
    def __init__(self, fids):
        self.m = {}
        for f in fids:
            self.dodaj(f)

    def dodaj(self, fid):
        v, f = _vf(fid); self.m.setdefault(v, []).append(f)

    def blisko(self, fid):
        v, f = _vf(fid)
        return any(abs(f - g) <= OKNO for g in self.m.get(v, ()))


_KODER = {}


def zapytaj(png, wykluczone, out_json):
    """WERSJA EDEN: to samo co query_index z wyszukiwanie.py (ten sam DINOv2, przetwarzanie, ranking
    rank_candidates z różnorodnością grup), ale podobieństwo liczone numpy do wszystkich 10 637 cech indeksu —
    dokładnie jak IndexFlatIP, bez FAISS (lokalnie sprawdzone: top-90 FAISS = top-90 numpy)."""
    import json
    sys.path.insert(0, str(D.WEJ / 'wyszukiwanie'))
    import wyszukiwanie as WY
    if Path(out_json).exists():
        return json.loads(Path(out_json).read_text(encoding='utf-8'))['candidates']
    if 'k' not in _KODER:
        cfg = json.loads((D.INDEKS / 'config.json').read_text(encoding='utf-8'))
        zr = WY.resolve_model_source(torch_hub_dir=D.WEJ / 'dino', weights=D.WEJ / 'dinov2_vits14_pretrain.pth')
        assert zr['model_sha256'] == cfg['model_source']['model_sha256'], 'inny DINOv2 niż przy budowie indeksu'
        _KODER['k'] = WY.DinoEncoder(zr, device='cuda', pooling=cfg['pooling'])
        assert _KODER['k'].preprocessing == cfg['preprocessing']
        _KODER['cechy'] = WY.normalize_features(np.load(D.INDEKS / 'features.npy'))
        _KODER['meta'] = WY._read_jsonl(D.INDEKS / 'metadata.jsonl')
    from PIL import Image
    with Image.open(png) as im:
        crop, box = WY.prepare_query_image(im)
    q = _KODER['k'].encode([crop])[0]
    wyn = WY.rank_candidates(_KODER['meta'], _KODER['cechy'] @ q, exclude_ids=sorted(wykluczone), top_k=TOP_K,
                             diversify_groups=True)
    Path(out_json).write_text(json.dumps(dict(candidates=wyn, query_crop_xyxy=box), ensure_ascii=False), encoding='utf-8')
    return wyn


def koopman(listy, pula_byid, zajete, n):
    wyb, poz = [], [0] * len(listy)
    while len(wyb) < n and any(p < len(l) for p, l in zip(poz, listy)):
        for i, l in enumerate(listy):
            while poz[i] < len(l):
                c = l[poz[i]]; poz[i] += 1
                if c['frame_id'] in pula_byid and not zajete.blisko(c['frame_id']):
                    wyb.append(dict(frame_id=c['frame_id'], kotwica=i, podobienstwo=float(c['score']), pozycja=poz[i] - 1))
                    zajete.dodaj(c['frame_id'])
                    break
            if len(wyb) >= n:
                break
    return wyb


def kolejne(pula_wiersze, zrodla, n, rng):
    """Wersja 3, ramię `kolejne` (PLAN.md): kolejne klatki tych samych filmów. Dla każdej klatki źródłowej (baza
    i klatki już dodane w tym ramieniu) najbliższa w czasie NASTĘPNA klatka puli z tego samego filmu; wybór n klatek
    o najmniejszym odstępie (remisy losowo). Bez reguły ±30, bo sensem ramienia są klatki sąsiednie.
    pula_wiersze: dostępne klatki puli (bez bazy, wykluczonych i już dodanych); zrodla: frame_id klatek źródłowych."""
    import bisect
    wolne, fid = {}, {}
    for r in pula_wiersze:
        v, f = _vf(r['frame_id']); wolne.setdefault(v, []).append(f); fid[(v, f)] = r['frame_id']
    for v in wolne:
        wolne[v].sort()
    zr = [_vf(f) for f in zrodla]
    wyb = []
    while len(wyb) < n:
        kand = {}
        for v, f in zr:
            l = wolne.get(v, [])
            j = bisect.bisect_right(l, f)
            if j < len(l):
                key = (v, l[j]); gap = l[j] - f
                if key not in kand or gap < kand[key]:
                    kand[key] = gap
        if not kand:
            break
        los = rng.random(len(kand))
        for (key, gap), _ in sorted(zip(kand.items(), los), key=lambda t: (t[0][1], t[1])):
            v, f = key
            wolne[v].remove(f)
            wyb.append(dict(frame_id=fid[key], odstep=int(gap)))
            if len(wyb) >= n:
                break
    return wyb


def losowe(pula_wiersze, zajete, n, rng):
    wyb = []
    for j in rng.permutation(len(pula_wiersze)):
        f = pula_wiersze[j]['frame_id']
        if not zajete.blisko(f):
            wyb.append(dict(frame_id=f)); zajete.dodaj(f)
            if len(wyb) >= n:
                break
    return wyb
