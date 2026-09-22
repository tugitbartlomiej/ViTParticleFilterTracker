#!/usr/bin/env python3
"""Co naprawde niesie osiem liczb kodu tozsamosci: sprawdzenie wzgledem etykiet (tylko do oceny).

Kod policzony na puli detektorem L0 (KOD_TOZSAMOSCI_pool_b0500.npz). Etykiety z pool_manifest.jsonl:
ramka prawdziwa (rozmiar, proporcja) i punkt koncowki narzedzia -> kat narzedzia wzgledem srodka ramki.
Predykcja grzbietowa z podzialem po FILMACH (model uczy sie na jednych filmach, sprawdza na innych),
zeby nie moglo wygrac samym rozpoznaniem filmu. Punkt odniesienia: stala srednia (skill 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')


def skill(x: np.ndarray, y: np.ndarray, video: np.ndarray, lam: float = 1.0, folds: int = 5, seed: int = 0) -> float:
    """Procent wyjasnionej zmiennosci poza filmami uzytymi do uczenia."""
    names = np.array(sorted(set(video.tolist())))
    fold = {v: i % folds for i, v in enumerate(np.random.default_rng(seed).permutation(names))}
    part = np.array([fold[v] for v in video])
    pred = np.zeros_like(y)
    x = np.hstack([x, np.ones((len(x), 1))])
    for f in range(folds):
        tr, te = part != f, part == f
        a = x[tr].T @ x[tr] + lam * np.eye(x.shape[1])
        pred[te] = x[te] @ np.linalg.solve(a, x[tr].T @ y[tr])
    return float(100 * (1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def main() -> int:
    d = np.load(ROOT / 'results/koopman_langevin_epochs/KOD_TOZSAMOSCI_pool_b0500.npz')
    code = d['code']
    at = {f: i for i, f in enumerate(d['frame_ids'].tolist())}
    rows, box, tip, video = [], [], [], []
    for line in (ROOT / 'feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        i = at.get(r['frame_id'])
        ann = r.get('annotations') or []
        if i is None or not ann or not np.isfinite(code[i]).all() or not r.get('tip_point'):
            continue
        rows.append(i); box.append(ann[0]['bbox']); tip.append(r['tip_point']); video.append(r['video_id'])
    rows = np.array(rows); box = np.array(box, float); tip = np.array(tip, float); video = np.array(video)
    x = code[rows]
    x = (x - x.mean(0)) / x.std(0)
    cx, cy = box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2
    cele = {
        'wielkosc narzedzia w kadrze (pole ramki)': np.log(box[:, 2] * box[:, 3]),
        'proporcja ramki (wskazuje nachylenie)': np.log(box[:, 2] / np.maximum(box[:, 3], 1)),
        'kat: koncowka wzgledem srodka ramki, poziomo': (tip[:, 0] - cx) / np.maximum(box[:, 2], 1),
        'kat: koncowka wzgledem srodka ramki, pionowo': (tip[:, 1] - cy) / np.maximum(box[:, 3], 1),
        'polozenie narzedzia w kadrze, poziomo': cx,
        'polozenie narzedzia w kadrze, pionowo': cy,
    }
    print(f'klatek z kodem i etykieta: {len(rows)}, filmow: {len(set(video.tolist()))}\n')
    print(f'{"co przewidujemy":48s} {"z kodu":>8s} {"z ramki":>9s}')
    geo = np.column_stack([np.log(box[:, 2] * box[:, 3]), np.log(box[:, 2] / np.maximum(box[:, 3], 1)), cx / 720, cy / 480])
    for nazwa, y in cele.items():
        s_kod = skill(x, y, video)
        s_geo = skill(np.delete(geo, list(cele).index(nazwa) if list(cele).index(nazwa) < 2 else 99, axis=1) if list(cele).index(nazwa) < 2 else geo[:, 2:], y, video)
        print(f'{nazwa:48s} {s_kod:7.1f}% {s_geo:8.1f}%')
    # ile z kodu to sama tozsamosc filmu
    names = sorted(set(video.tolist()))
    onehot = np.zeros((len(video), len(names)))
    onehot[np.arange(len(video)), [names.index(v) for v in video]] = 1
    mid = onehot.T @ x / np.maximum(onehot.sum(0)[:, None], 1)
    within = ((x - mid[[names.index(v) for v in video]]) ** 2).sum()
    print(f'\nudzial zmiennosci kodu MIEDZY filmami: {100 * (1 - within / (x ** 2).sum()):.0f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
