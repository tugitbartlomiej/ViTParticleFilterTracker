#!/usr/bin/env python3
"""Tożsamość narzędzia wyznaczona na parach „to samo miejsce z narzędziem i bez” (PLAN.md).

Sieci: stan po rozgrzewce (krok 1000 = 21 epok) ziaren 47–56 (`maly_vit_petla_sila_tozsamosci/wyniki_eden/petla_s*/
wspolny/k02.pt`, ta sama rozgrzewka co w wersjach 2 i 3).
Latent narzędzia: średnia 768 aktywacji po tokenach narzędzia sondy (jak w pętli). Zdjęcia bez narzędzia (`tla.py`):
ten sam latent na TYCH SAMYCH tokenach, ale z zamalowanym narzędziem (`oko`) albo na czarnym zdjęciu (`czarne`).

Wektory:
  stary  — jak w pętli (`kod/sila.py` wersji 3): narzędzie kontra tło sond (tokeny daleko od narzędzia) i obszar
           narzędzia w `empty_black`, z zapisanych latentów (średnia 5 zapisów przed krokiem 1000);
  nowy   — narzędzie kontra to samo miejsce bez narzędzia (`oko` + `czarne`), z jednego stanu k02.pt;
  razem  — negatywy starego i nowego naraz.
Miary: rozdzielenie d' = (średnia narzędzia − średnia negatywów) / sd negatywów, osobno dla każdego rodzaju negatywów;
AUC narzędzie kontra `oko` na wycinku wyłączonym z uczenia (12 razy, po jednym wycinku); cosinus wektorów; prognoza
siły tożsamości (Koopman, Langevin) na zapisanych latentach wersji 2 (ramiona `bez`, `losowe`, `koopman`, kroki
1000–4500, horyzont 500 kroków) względem „braku zmiany” — dla starego i nowego wektora.

    <venv>/python wektor_z_tlami.py  ->  wyniki/WEKTOR_Z_TLAMI.json, wyniki/WEKTOR_Z_TLAMI.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
V3 = EXP / 'maly_vit_petla_sila_tozsamosci'
V2 = EXP / 'maly_vit_wektor_tozsamosci'
PROBE = EXP / 'maly_vit_od_zera' / 'eden' / 'wejscia' / 'probe'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MALY_VIT_WEJSCIA', str(EXP / 'maly_vit_od_zera' / 'eden' / 'wejscia'))   # wymagane przy imporcie kod/generuj.py
sys.path.insert(0, str(V3 / 'kod'))
import model as M  # noqa: E402
import prognoza as PR  # noqa: E402
import sila as SI  # noqa: E402

S_SONDY = 176
RAMIONA = ('bez', 'losowe', 'koopman')
POCZATKI = list(range(1000, 4500, 250))
H = 10


def wczytaj(p):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (S_SONDY, S_SONDY), interpolation=cv2.INTER_AREA)


def tokeny(maska_p):
    """Tokeny narzędzia i tła jak w `dane.sondy` wersji 3."""
    g = S_SONDY // M.KROK
    mk = cv2.resize((cv2.imread(str(maska_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8), (S_SONDY, S_SONDY),
                    interpolation=cv2.INTER_NEAREST) > 0
    tn = mk.reshape(g, M.KROK, g, M.KROK).mean((1, 3)).reshape(-1) >= .25
    dyl = cv2.dilate(mk.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    tt = dyl.reshape(g, M.KROK, g, M.KROK).mean((1, 3)).reshape(-1) == 0
    return tn, tt


@torch.no_grad()
def latent(net, obrazy, wagi):
    """obrazy (N,176,176,3) uint8, wagi (N,121) — średni latent (N,768) na zaznaczonych tokenach."""
    net.eval(); net.zapisuj_latent(True)
    X = torch.from_numpy(np.stack(obrazy)).permute(0, 3, 1, 2).float().div(255)
    W = torch.from_numpy(np.stack(wagi).astype(np.float32))
    out = []
    for s in range(0, len(X), 96):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, lat = net(X[s:s + 96].cuda())
        a = W[s:s + 96].cuda()
        out.append(((lat.float() * a[..., None]).sum(1) / a.sum(1, keepdim=True).clamp(min=1)).cpu().numpy())
    net.zapisuj_latent(False)
    return np.concatenate(out).astype(np.float64)


def lda_wektor(poz, neg):
    X = np.concatenate([poz, neg]); y = np.r_[np.ones(len(poz)), np.zeros(len(neg))].astype(int)
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    w = SI.lda((X - mu) / sd, y); w /= np.linalg.norm(w)
    return dict(mu=mu, sd=sd, w=w)


def rzut(m, X):
    return ((X - m['mu']) / m['sd']) @ m['w']


def dprim(m, poz, neg):
    a, b = rzut(m, poz), rzut(m, neg)
    return float((a.mean() - b.mean()) / (b.std() + 1e-9))


def auc(a, b):
    """P(narzędzie > bez narzędzia) — pole pod krzywą ROC."""
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())


def bloki_v2(d, ramie):
    pl = [d / 'wspolny' / 'k01' / 'LATENT.npz', d / 'wspolny' / 'k02' / 'LATENT.npz'] + \
         [d / ramie / f'k{k:02d}' / 'LATENT.npz' for k in range(3, 10)]
    L = np.concatenate([np.load(p)['latent'] for p in pl])[1:]
    nb = len(L) // PR.B_BLOK
    return L[:nb * PR.B_BLOK].astype(np.float32).reshape(nb, PR.B_BLOK, *L.shape[1:]).mean(1)


def test_prognozy(Y):
    """Y (T,P,1): stosunek błędu prognozy do braku zmiany (sondy), mediana po chwilach startu, oba modele."""
    wyn = {m: [] for m in ('koopman', 'langevin')}
    for t in POCZATKI:
        b0 = t // 50 - 1
        if b0 + H >= len(Y):
            continue
        hist = Y[:b0 + 1]; real, teraz = Y[b0 + H, :, 0], hist[-1, :, 0]
        e0 = np.sqrt(np.mean((teraz - real) ** 2))
        for mod in wyn:
            p = PR.rollout(hist, PR.dopasuj(hist, mod), H, mod)[-1, :, 0]
            wyn[mod].append(float(np.sqrt(np.mean((p - real) ** 2)) / max(e0, 1e-12)))
    return {m: float(np.median(v)) for m, v in wyn.items()}


def main() -> int:
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    zrodlo = np.array([m['source_file'] for m in man])
    tok = [tokeny(PROBE / m['mask']) for m in man]
    tn = [t[0] for t in tok]; tt = [t[1] for t in tok]
    obr_s = [wczytaj(PROBE / m['image']) for m in man]
    tla = json.loads((HERE / 'tla' / 'manifest.json').read_text(encoding='utf-8'))
    obr_t = {(t['sonda'], t['typ']): wczytaj(HERE / 'tla' / t['image']) for t in tla}
    poz_idx = np.where(obecn)[0]
    net = M.MalyViT().cuda()
    wyn, rys = {}, {}
    for d in sorted((V3 / 'wyniki_eden').glob('petla_s*')):
        s = d.name.split('_s')[1]
        st = torch.load(d / 'wspolny' / 'k02.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(st['net'])
        nar = latent(net, [obr_s[i] for i in poz_idx], [tn[i] for i in poz_idx])                     # (180,768)
        tlo = latent(net, obr_s, tt)                                                                  # (192,768)
        pusta = latent(net, [obr_s[i] for i in np.where(~obecn)[0]], [tn[i] for i in np.where(~obecn)[0]])
        oko = latent(net, [obr_t[(i, 'oko')] for i in poz_idx], [tn[i] for i in poz_idx])
        cz = latent(net, [obr_t[(i, 'czarne')] for i in poz_idx], [tn[i] for i in poz_idx])
        L2 = V2 / 'wyniki_eden' / d.name
        Lk = np.load(L2 / 'wspolny' / 'k02' / 'LATENT.npz')['latent'][-5:].astype(np.float32).mean(0)
        stary = SI.wektor(Lk, obecn)                                                                  # jak w pętli
        nowy = lda_wektor(nar, np.concatenate([oko, cz]))
        razem = lda_wektor(nar, np.concatenate([oko, cz, tlo, pusta]))
        w_ = {}
        for nazwa, m in (('stary', stary), ('nowy', nowy), ('razem', razem)):
            w_[nazwa] = dict(dprim_oko=dprim(m, nar, oko), dprim_czarne=dprim(m, nar, cz), dprim_tlo=dprim(m, nar, tlo),
                             auc_oko=auc(rzut(m, nar), rzut(m, oko)))
        # wycinek wyłączony: nowy wektor bez jednego wycinka, sprawdzony na nim (narzędzie kontra oko)
        zr = zrodlo[poz_idx]; a_n, a_s = [], []
        for z in np.unique(zr):
            u = zr != z
            mz = lda_wektor(nar[u], np.concatenate([oko[u], cz[u]]))
            a_n.append(auc(rzut(mz, nar[~u]), rzut(mz, oko[~u])))
            a_s.append(auc(rzut(stary, nar[~u]), rzut(stary, oko[~u])))
        w_['auc_oko_wycinek_wylaczony'] = dict(stary=float(np.mean(a_s)), nowy=float(np.mean(a_n)))
        w_['cos_stary_nowy'] = float(stary['w'] @ nowy['w'])
        # prognoza siły na historii wersji 2 (te same sieci do kroku 1000, dalej ramiona)
        pr = {}
        for ramie in RAMIONA:
            Lb = bloki_v2(L2, ramie)
            pr[ramie] = {}
            for nazwa, m in (('stary', stary), ('nowy', nowy), ('razem', razem)):
                Y = rzut(m, Lb[:, obecn, 0].astype(np.float64))[:, :, None]
                pr[ramie][nazwa] = test_prognozy(Y)
                if ramie == 'bez':
                    rys.setdefault(nazwa, []).append(Y[:, :, 0].mean(1))
        w_['prognoza'] = pr
        wyn[s] = w_
        print(s, f"d' oko: stary {w_['stary']['dprim_oko']:.1f}, nowy {w_['nowy']['dprim_oko']:.1f} | AUC oko (wycinek "
              f"wyłączony): stary {w_['auc_oko_wycinek_wylaczony']['stary']:.3f}, nowy {w_['auc_oko_wycinek_wylaczony']['nowy']:.3f}"
              f" | cos {w_['cos_stary_nowy']:.2f} | prognoza bez/L: stary {pr['bez']['stary']['langevin']:.3f}, "
              f"nowy {pr['bez']['nowy']['langevin']:.3f}", flush=True)
    Z = list(wyn.values())
    pod = {k: {w: float(np.mean([z[w][k] for z in Z])) for w in ('stary', 'nowy', 'razem')}
           for k in ('dprim_oko', 'dprim_czarne', 'dprim_tlo', 'auc_oko')}
    pod['auc_oko_wycinek_wylaczony'] = {w: float(np.mean([z['auc_oko_wycinek_wylaczony'][w] for z in Z])) for w in ('stary', 'nowy')}
    pod['cos_stary_nowy'] = float(np.mean([z['cos_stary_nowy'] for z in Z]))
    pod['prognoza'] = {r: {w: {m: dict(mediana=float(np.median([z['prognoza'][r][w][m] for z in Z])),
                                       ponizej_1=int(sum(z['prognoza'][r][w][m] < 1 for z in Z)))
                               for m in ('koopman', 'langevin')} for w in ('stary', 'nowy', 'razem')} for r in RAMIONA}
    pod['nowy_lepiej_prognozowany'] = {r: {m: int(sum(z['prognoza'][r]['nowy'][m] < z['prognoza'][r]['stary'][m] for z in Z))
                                           for m in ('koopman', 'langevin')} for r in RAMIONA}
    pod['razem_lepiej_prognozowany'] = {r: {m: int(sum(z['prognoza'][r]['razem'][m] < z['prognoza'][r]['stary'][m] for z in Z))
                                            for m in ('koopman', 'langevin')} for r in RAMIONA}
    (HERE / 'wyniki' / 'WEKTOR_Z_TLAMI.json').write_text(json.dumps(dict(podsumowanie=pod, ziarna=wyn), indent=1,
                                                                    ensure_ascii=False), encoding='utf-8')
    print(json.dumps(pod, indent=1, ensure_ascii=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        kroki = 50 * np.arange(1, len(rys['stary'][0]) + 1)
        for a, nazwa in zip(ax, ('stary', 'razem')):
            for y in rys[nazwa]:
                a.plot(kroki, y, lw=1)
            a.axvline(1000, color='gray', ls=':')
            a.set_title(f'wektor {nazwa}: średnia siła tożsamości 180 sond, ramię bez dokładania (10 ziaren)', fontsize=9)
            a.set_xlabel('krok uczenia')
        fig.tight_layout(); fig.savefig(HERE / 'wyniki' / 'WEKTOR_Z_TLAMI.png', dpi=120)
    except Exception as e:  # wykres pomocniczy
        print('wykres:', e)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
