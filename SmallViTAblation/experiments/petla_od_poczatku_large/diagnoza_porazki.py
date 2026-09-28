#!/usr/bin/env python3
"""Diagnoza, dlaczego pętla koopman nie przyspiesza uczenia (DIAGNOZA_PORAZKI.md). Cztery pomiary na danych kampanii:

  1. prognoza   błąd prognozy Koopmana 3 obroty naprzód (96 kroków) wobec „brak zmiany” i „stała prędkość”,
                w jednostkach rozrzutu sond, pętla koopman, obroty startowe k1..k4; korelacja kierunku skalara
  2. wagi       względna zmiana wag ‖ΔW‖/‖W‖ per część sieci (kręgosłup DINOv2, projektor, dekoder, głowice):
                COCO → k00 i k00 → k06 (ziarno 42, pętla koopman; wagi lokalne)
  3. latent-mAP korelacja Spearmana skalara tożsamości i R z mAP@0,5 między pętlami w tym samym (ziarno, obrót)
  4. wybor      różnorodność dodanych klatek (koopman vs losowe): średni cos DINOv2 między dodanymi, sąsiedzi
                w filmie ≤ 30 klatek, mediana pola ramki, pokrycie puli (średni max cos klatki puli do zbioru uczącego)

    <venv rfdetr>/python diagnoza_porazki.py      ->  wyniki/DIAGNOZA_PORAZKI.json
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

import obrot_large as O

HERE = Path(__file__).resolve().parent
INDEKS = O.PETLA / 'wyniki/local_1810448/run/index'


def prognoza(mapa):
    out = []
    for s in (42, 43, 44):
        P = np.load(O.W / f's{s}' / 'kierunki.npz')['projection']
        for k in (1, 2, 3, 4):
            Y = O.historia(s, k, mapa)
            Yf = np.concatenate([Y] + [O.latent(s, 'koopman', j, mapa) @ P for j in range(k, k + 3)])
            prawda = Yf[-1] - Y[-1]; sk = Y[-1].std(0)
            V = np.diff(Y, axis=0); d = np.zeros((72, 5))
            for dims in ([0], [1, 2, 3, 4]):
                op, _ = O.dopasuj(V, dims)
                d[:, dims] = O.rollout(op, [V[t][:, dims] for t in range(V.shape[0] - O.L_PAM, V.shape[0])], 96)
            e = lambda x: float(np.sqrt((((prawda - x) / sk) ** 2).mean()))
            out.append(dict(s=s, k=k, ruch=e(0 * d), blad_koopman=e(d), blad_brak_zmiany=e(0 * d),
                            blad_stala_predkosc=e(V[-8:].mean(0) * 96),
                            korelacja_kierunku_skalara=float(np.corrcoef(d[:, 0], prawda[:, 0])[0, 1])))
    return out


def wagi():
    import torch

    def load(p):
        d = torch.load(p, map_location='cpu', weights_only=False)
        return d['model'] if 'model' in d else d

    def grupa(k):
        if k.startswith('backbone.0.encoder'):
            return 'kregoslup_dinov2'
        if k.startswith('backbone'):
            return 'projektor'
        if k.startswith('transformer'):
            return 'dekoder'
        return 'glowice'

    def zmiana(X, Y):
        s = collections.defaultdict(lambda: [0., 0., 0])
        for k, v in Y.items():
            if k in X and X[k].shape == v.shape and v.dtype.is_floating_point:
                g = grupa(k); s[g][0] += float((v - X[k]).float().norm() ** 2); s[g][1] += float(X[k].float().norm() ** 2)
                s[g][2] += v.numel()
        return {g: dict(wzgledna_zmiana=(a / b) ** .5, parametrow=n) for g, (a, b, n) in s.items()}
    A = load(HERE / 'modele/rf-detr-large.pth'); B = load(O.W / 's42/wspolny/k00/model_0000160.pth')
    C = load(O.W / 's42/koopman/k06/model_0000160.pth')
    return dict(coco_do_k00=zmiana(A, B), k00_do_k06=zmiana(B, C))


def latent_map():
    from scipy.stats import spearmanr
    diag = json.loads((O.EXP / 'domykanie_tozsamosci_large/wyniki/DIAGNOZA.json').read_text(encoding='utf-8'))['wynik']
    A = json.loads((O.W / 'ANALIZA.json').read_text(encoding='utf-8'))['ziarna']
    xs, rs, ys = [], [], []
    for s in ('42', '43', '44'):
        for k in range(1, 7):
            t = {}
            for p in ('koopman', 'losowe', 'bez'):
                e = [w for w in diag[s][p] if w['obrot'] == f'k{k:02d}']
                if k in A[s][p]['obroty'] and e:
                    t[p] = (A[s][p]['mAP50'][A[s][p]['obroty'].index(k)], e[0]['skalar_sredni'], e[0]['W2_R'])
            if len(t) < 2:
                continue
            m = np.mean(list(t.values()), axis=0)
            for v in t.values():
                ys.append(v[0] - m[0]); xs.append(v[1] - m[1]); rs.append(v[2] - m[2])
    a, b = spearmanr(xs, ys), spearmanr(rs, ys)
    return dict(n=len(ys), skalar_vs_mAP=dict(rho=float(a.statistic), p=float(a.pvalue)),
                R_vs_mAP=dict(rho=float(b.statistic), p=float(b.pvalue)))


def wybor():
    meta = [json.loads(l) for l in open(INDEKS / 'metadata.jsonl', encoding='utf-8')]
    F = np.load(INDEKS / 'features.npy').astype(np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True)
    row = {m['frame_id']: i for i, m in enumerate(meta)}
    out = {}
    for s in (42, 43, 44):
        for p in ('koopman', 'losowe', 'bez'):
            d = json.loads((O.W / 'selekcje' / f's{s}_{p}_k06.json').read_text(encoding='utf-8'))
            tr = [row[f] for f in d['frame_ids'] if f in row]
            pool = np.setdiff1d(np.arange(len(meta)), tr)
            wpis = dict(pokrycie_puli=float((F[pool] @ F[tr].T).max(1).mean()))
            if p != 'bez':
                r = [row[x['frame_id']] for x in d['dodane_szczegoly'] if x['frame_id'] in row]
                G = F[r] @ F[r].T
                fi = sorted((meta[i]['group_id'], meta[i]['frame_index']) for i in r)
                wpis.update(dodanych=len(r), cos_sredni=float(G[~np.eye(len(r), dtype=bool)].mean()),
                            sasiedzi_30_klatek=sum(a[0] == b[0] and abs(a[1] - b[1]) <= 30 for a, b in zip(fi, fi[1:])),
                            pole_ramki_mediana=float(np.median([np.prod(meta[i]['bbox_size_fraction']) for i in r])))
            out[f's{s}_{p}'] = wpis
    return out


def main() -> int:
    rows, mapa, fit = O.sondy()
    wyn = dict(opis=__doc__, prognoza=prognoza(mapa), wagi=wagi(), latent_mAP=latent_map(), wybor=wybor())
    (O.W / 'DIAGNOZA_PORAZKI.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    pr = wyn['prognoza']
    print('prognoza: blad Koopman / brak zmiany srednio %.3f / %.3f; Koopman gorszy w %d z %d' % (
        np.mean([x['blad_koopman'] for x in pr]), np.mean([x['blad_brak_zmiany'] for x in pr]),
        sum(x['blad_koopman'] > x['blad_brak_zmiany'] for x in pr), len(pr)))
    print('wagi k00->k06:', {g: round(v['wzgledna_zmiana'], 5) for g, v in wyn['wagi']['k00_do_k06'].items()})
    print('latent-mAP:', wyn['latent_mAP'])
    print('wybor:', {k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in wyn['wybor'].items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
