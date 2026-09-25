#!/usr/bin/env python3
"""Tokeny 72 sond dla WSZYSTKICH pobranych checkpointow biegu 1810662 (kroki 5..800), do wlasnego katalogu.

Nie modyfikuje katalogu koopman_gesta_historia/wyniki/macierze_wiele_obrazow: istniejace TOKENS_*.npy (73 kroki)
sa kopiowane po sprawdzeniu ich dowodow SHA256, brakujace kroki liczone ta sama funkcja extract() z
macierz_kazde_przejscie.py (ten sam kod, ta sama kontrola zgodnosci z CZASTKI Edena), z OUT przestawionym na
wyniki/tokeny_pelne. Powstaje CONFIG.json w tym samym formacie, wiec licz.py / najlepszy.py dzialaja z --zrodlo.

    <venv rfdetr>/python experiments/koopman_tozsamosc_kierunki/ekstrakcja.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GH = HERE.parent / 'koopman_gesta_historia'
ZR = GH / 'wyniki/macierze_wiele_obrazow'
RUN = GH / 'wyniki/run_1810662'
OUT = HERE / 'wyniki/tokeny_pelne'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(GH))
    import macierz_kazde_przejscie as mk
    mk.OUT = OUT                                                            # zapis wylacznie do naszego katalogu
    stary = json.loads((ZR / 'CONFIG.json').read_text(encoding='utf-8'))
    manifest = {r['path']: r for r in json.loads((RUN / 'REMOTE_MANIFEST.json').read_text())}
    weights = []
    for p in sorted((RUN / 'pretrain').glob('model_*.pth')):
        step = int(p.stem.split('_')[-1]); oczek = manifest['pretrain/' + p.name]['sha256']
        if sha(p) != oczek:
            print('pomijam (zla suma, pobieranie w toku?)', p.name); continue
        weights.append(dict(step=step, path=str(p), sha256=oczek))
    assert all(b['step'] - a['step'] == 5 for a, b in zip(weights, weights[1:])), 'luka w checkpointach'
    print(f'checkpointow zweryfikowanych: {len(weights)} (kroki {weights[0]["step"]}..{weights[-1]["step"]})', flush=True)
    # kopie istniejacych tokenow z dowodami
    for w in weights:
        src = ZR / f'TOKENS_{w["step"]:07d}.npy'; proof = src.with_suffix('.json'); dst = OUT / src.name
        if src.exists() and proof.exists() and not dst.exists():
            rec = json.loads(proof.read_text())
            if rec['checkpoint_sha256'] == w['sha256'] and rec['sha256'] == sha(src):
                shutil.copyfile(src, dst); shutil.copyfile(proof, dst.with_suffix('.json'))
    shutil.copyfile(ZR / 'MASK_WEIGHTS.npy', OUT / 'MASK_WEIGHTS.npy')
    cfg = dict(stary); cfg['weights'] = weights; cfg['same_image_pairs'] = (len(weights) - 1) * 72
    cfg['purpose'] = 'tokeny pelnej historii 5..800 dla koopman_tozsamosc_kierunki'
    (OUT / 'CONFIG.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    brak = [w for w in weights if not (OUT / f'TOKENS_{w["step"]:07d}.npy').exists()]
    print(f'do policzenia: {len(brak)} checkpointow', flush=True)
    if brak:
        mk.extract(dict(cfg, weights=brak))
    print('gotowe:', len(list(OUT.glob('TOKENS_*.npy'))), 'plikow tokenow', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
