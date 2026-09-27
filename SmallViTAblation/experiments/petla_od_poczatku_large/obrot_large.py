#!/usr/bin/env python3
"""Lokalna część obrotu pętli dla RF-DETR LargeDeprecated (PLAN.md).

Etapy (wywoływane przez dyrygent_large.py):
  kierunki  --s S          kierunki tożsamości z latentu rozgrzewki (k00): w = średnia (narzędzie − tło) sond fit na
                           końcu rozgrzewki (znormalizowana), S_4 = 4 główne kierunki (narzędzie − tło) bez w;
                           projekcja 3072 → 5 jak w Nano (w na narzędziu i −w na tle, S_4 na narzędziu).
  prognoza  --s S --k K    historia pętli koopman (k00 + obroty 1..K−1), wspólny operator na 8 ostatnich przyrostach
                           (grzbiet λ = 0,1, osobno skalar i S_4), rollout każdej sondy o 3 obroty naprzód (96 kroków);
                           kotwice: 3 oryginały o największej przewidzianej drodze; zadanie.json dla fastgan_large.py.
  wybor     --s S --k K    po FastGAN: wyszukanie DINOv2 (obraz celu i, kontrolnie, obecny), kandydaci top-90 na
                           kotwicę; ocena „do nauczenia i jeszcze nienauczone”: trudność dla bieżącego modelu Large
                           minus trudność dla modelu odniesienia (Nano 1810662 po 800 aktualizacjach, tych klatek
                           nie widział); 134 najwyższe. Selekcje trzech pętli (koopman, losowe, bez) na obrót K.
Trudność klatki = 1 − max(pewność × IoU z ramką prawdziwą), predykcje z progiem 0,05.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
KT = EXP / 'koopman_tozsamosc_kierunki'
PETLA = EXP / 'koopman_fastgan_petla'
sys.path.insert(0, str(PETLA))
W = HERE / 'wyniki'
EDEN = '/home2/faculty/bpiotrowski/smallvit_runs/2026-09-27_petla_large'
EDEN_ROOT = '/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_gesta_historia'
EDEN_FASTGAN = '/home2/faculty/bpiotrowski/smallvit_runs/2026-09-23_fastgan/results_50k/checkpoint_050000.pth'
PROBE_MANIFEST = PETLA / 'wyniki/local_1810448/run/probe/manifest.json'
ROWS72 = KT / 'wyniki/tokeny_pelne/CONFIG.json'
INDEKS = PETLA / 'wyniki/local_1810448/run/index'
INDEX_CFG = INDEKS / 'config.json'
WYKL = KT / 'wyniki/dobor_klatek/wykluczone_pretrain_1500_plus_kotwice.json'
BAZA_SEL = KT / 'wyniki/dobor_klatek/trening5ep/selekcja_bez_dodania.json'
NANO_REF = EXP / 'koopman_gesta_historia/wyniki/run_1810662/pretrain/model_0000800.pth'
KROKI = 32            # zapisów latentu na obrót (co 5 aktualizacji)
PRZOD = 3             # obroty naprzód celu FastGAN
L_PAM, LAM = 8, 0.1
NOWYCH, TOP_K = 134, 90
C = 768


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def zj(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding='utf-8')


def sondy():
    rows = jz(ROWS72)['rows']
    idx = {r['image_sha256']: i for i, r in enumerate(jz(PROBE_MANIFEST))}
    mapa = np.array([idx[r['image_sha256']] for r in rows])
    fit = np.array([r['role'] == 'fit' for r in rows])
    return rows, mapa, fit


def latent(s, petla, k, mapa):
    d = W / f's{s}' / petla / f'k{k:02d}' / 'latent'
    kroki = range(0, 161, 5) if k == 0 else range(5, 161, 5)
    return np.stack([np.load(d / f'LATENT_{t:07d}.npz')['features'].astype(np.float64)[mapa] for t in kroki])


def etap_kierunki(s):
    rows, mapa, fit = sondy()
    D = latent(s, 'wspolny', 0, mapa)                              # (33, 72, 3072)
    diff = D[..., :C] - D[..., C:2 * C]                             # narzędzie − tło
    w = diff[-1, fit].mean(0); w /= np.linalg.norm(w)
    X = diff[:, fit].reshape(-1, C); X = X - X.mean(0); X = X - np.outer(X @ w, w)
    Q4 = np.linalg.svd(X, full_matrices=False)[2][:4].T               # (768, 4)
    P = np.zeros((4 * C, 5)); P[:C, 0] = w; P[C:2 * C, 0] = -w; P[:C, 1:] = Q4
    Y = D @ P
    skala = Y[-1].std(0)
    out = W / f's{s}' / 'kierunki.npz'
    np.savez(out, mean=np.zeros(4 * C), scale=np.ones(4 * C), projection=P, target_scale=skala)
    a = Y[..., 0]
    print(f's{s}: kierunki zapisane; skalar tożsamości w rozgrzewce: średnio {a[0].mean():.3f} -> {a[-1].mean():.3f} '
          f'(rozrzut {a[-1].std():.3f})', flush=True)
    return out


def historia(s, k, mapa):
    P = np.load(W / f's{s}' / 'kierunki.npz')['projection']
    czesci = [latent(s, 'wspolny', 0, mapa)] + [latent(s, 'koopman', j, mapa) for j in range(1, k)]
    return np.concatenate(czesci) @ P


def dopasuj(V, dims):
    X, y = [], []
    for t in range(L_PAM, V.shape[0]):
        X.append(np.concatenate([V[t - l][:, dims] for l in range(1, L_PAM + 1)], 1)); y.append(V[t][:, dims])
    X, y = np.concatenate(X), np.concatenate(y)
    mx, sx, my = X.mean(0), X.std(0) + 1e-12, y.mean(0)
    Z = (X - mx) / sx
    Wt = np.linalg.solve(Z.T @ Z + LAM * len(y) * np.eye(Z.shape[1]), Z.T @ (y - my))
    res = y - (my + Z @ Wt)
    return (mx, sx, my, Wt), float(1 - np.mean(res ** 2) / np.mean(y ** 2))


def rollout(op, hist, H):
    mx, sx, my, Wt = op
    hist = list(hist); suma = np.zeros_like(hist[-1])
    for _ in range(H):
        x = np.concatenate([hist[-l] for l in range(1, L_PAM + 1)], -1)
        v = my + ((x - mx) / sx) @ Wt
        suma = suma + v; hist.append(v)
    return suma


def etap_prognoza(s, k):
    rows, mapa, fit = sondy()
    Y = historia(s, k, mapa)                                          # (T, 72, 5)
    V = np.diff(Y, axis=0)
    delta, diag = np.zeros((72, 5)), {}
    for nazwa, dims in (('skalar', [0]), ('S4', [1, 2, 3, 4])):
        op, um = dopasuj(V, dims)
        delta[:, dims] = rollout(op, [V[t][:, dims] for t in range(V.shape[0] - L_PAM, V.shape[0])], PRZOD * KROKI)
        diag[nazwa] = dict(umiejetnosc_1krok_w_probie=um)
    skala = Y[-1].std(0)
    droga = np.linalg.norm(delta / skala, axis=1)
    orig = [i for i, r in enumerate(rows) if r['variant'] == 'original']
    kotwice = sorted(orig, key=lambda i: -droga[i])[:3]
    idx_man = {r['image']: i for i, r in enumerate(jz(PROBE_MANIFEST))}
    src = 'wspolny' if k == 1 else 'koopman'
    zad = dict(wagi=f'{EDEN}/results/s{s}/{src}/k{k - 1:02d}/trening/model_0000160.pth',
               kierunki=f'{EDEN}/fastgan/s{s}/kierunki.npz', manifest=f'{EDEN_ROOT}/probe/manifest.json',
               fastgan=EDEN_FASTGAN, vendor=f'{EDEN}/vendor/deepfake_gi_fastGAN', out=f'{EDEN}/fastgan/s{s}/k{k:02d}',
               poprzedni=(f'{EDEN}/fastgan/s{s}/k{k - 1:02d}' if k > 1 else None), obrot=k,
               kotwice=[dict(wiersz=int(i), obraz=rows[i]['image'], indeks_manifestu=idx_man[rows[i]['image']],
                             przyrost=delta[i].tolist()) for i in kotwice])
    lok = W / f's{s}' / 'fastgan' / f'k{k:02d}'
    zj(lok / 'zadanie.json', zad)
    rap = dict(s=s, k=k, punktow_historii=int(len(Y)), horyzont_krokow=PRZOD * KROKI, diagnostyka=diag,
               droga_w_odchyleniach={str(i): float(droga[i]) for i in orig}, kotwice=[int(i) for i in kotwice],
               przyrost_sredni_w_odchyleniach=(delta.mean(0) / skala).tolist())
    zj(lok / 'PROGNOZA.json', rap)
    print(f's{s} k{k}: historia {len(Y)} pkt, kotwice {kotwice}, droga {[round(float(droga[i]), 3) for i in kotwice]}', flush=True)


def trudnosc(model, rows, batch=16):
    import torch
    from PIL import Image
    import efekt_lokalny as E
    out = np.zeros(len(rows))
    for st in range(0, len(rows), batch):
        ims = []
        for r in rows[st:st + batch]:
            with Image.open(r['path']) as im:
                ims.append(im.convert('RGB'))
        with torch.inference_mode():
            dets = model.predict(ims, threshold=.05)
        for j, d in enumerate(dets if isinstance(dets, list) else [dets]):
            b = np.asarray(d.xyxy, np.float32).reshape(-1, 4); c = np.asarray(d.confidence, np.float32).reshape(-1)
            gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                           for a in rows[st + j]['annotations']], float)
            out[st + j] = 1 - (float((c[:, None] * E.iou_macierz(b, gt)).max()) if len(c) else 0.0)
    return out


def pula():
    from wyszukiwanie import read_manifest
    cfg = jz(INDEX_CFG)
    wyk = set(jz(WYKL))
    return [r for r in read_manifest(cfg['manifest_path'], cfg['data_root'])
            if r['split'] == 'train' and r['annotations'] and r['frame_id'] not in wyk]


def odniesienie(rows):
    f = W / 'nano_ref_trudnosc.npz'
    if f.exists():
        z = np.load(f)
        return dict(zip(z['frame_ids'].tolist(), z['trud'].tolist()))
    import torch
    from rfdetr import RFDETRNano
    model = RFDETRNano(pretrain_weights=str(NANO_REF), num_classes=1, resolution=384, amp=False, device='cuda', trust_checkpoint=True)
    t = trudnosc(model, rows, batch=32)
    np.savez(f, frame_ids=np.array([r['frame_id'] for r in rows]), trud=t)
    del model; torch.cuda.empty_cache()
    return dict(zip([r['frame_id'] for r in rows], t.tolist()))


def selekcja(nazwa, dodane, baza):
    ids = list(baza['frame_ids']) + [d['frame_id'] for d in dodane]
    assert len(ids) == len(set(ids))
    return dict(run_id=nazwa, baza=baza['baza'], dodane=len(dodane), frame_ids=ids, dodane_szczegoly=dodane)


def etap_wybor(s, k):
    sys.path.insert(0, str(KT))
    from wyszukiwanie import query_index
    lok = W / f's{s}' / 'fastgan' / f'k{k:02d}'
    zad = jz(lok / 'zadanie.json')
    baza = jz(BAZA_SEL)
    prev = {p: (jz(W / 'selekcje' / f's{s}_{p}_k{k - 1:02d}.json')['dodane_szczegoly'] if k > 1 else []) for p in ('koopman', 'losowe')}
    rows = pula(); byid = {r['frame_id']: r for r in rows}
    wyk = set(jz(WYKL)) | {d['frame_id'] for d in prev['koopman']}
    listy, pokrycie = {}, {}
    for kot in zad['kotwice']:
        kd = lok / f"kotwica_{kot['wiersz']:02d}"
        wyn = {}
        for arm in ('target', 'current'):
            f = kd / f'zapytanie_{arm}.json'
            if not f.exists():
                query_index(INDEKS, kd / f'{arm}.png', sorted(wyk), out=f, top_k=TOP_K, device='cuda')
            wyn[arm] = jz(f)['candidates']
        listy[kot['wiersz']] = wyn['target']
        a = [c['frame_id'] for c in wyn['target']]; b = [c['frame_id'] for c in wyn['current']]
        pokrycie[str(kot['wiersz'])] = dict(top45_wspolne=len(set(a[:45]) & set(b[:45])), top90_wspolne=len(set(a) & set(b)))
    kand = {}
    for w_, lst in listy.items():
        for poz, c in enumerate(lst):
            if c['frame_id'] in byid and c['frame_id'] not in kand:
                kand[c['frame_id']] = dict(frame_id=c['frame_id'], kotwica=int(w_), podobienstwo=float(c['score']),
                                           pozycja=poz, video=c.get('video_id'))
    ref = odniesienie(rows)
    import torch
    from rfdetr import RFDETRLargeDeprecated
    wagi = W / f's{s}' / ('wspolny' if k == 1 else 'koopman') / f'k{k - 1:02d}' / 'model_0000160.pth'
    model = RFDETRLargeDeprecated(pretrain_weights=str(wagi), num_classes=1, amp=False, device='cuda', trust_checkpoint=True)
    kl = list(kand)
    t0 = time.time()
    dcur = trudnosc(model, [byid[f] for f in kl], batch=8)
    del model; torch.cuda.empty_cache()
    for f, d in zip(kl, dcur):
        kand[f].update(trud_biezacy=float(d), trud_odniesienia=float(ref[f]), wynik_rho=float(d - ref[f]))
    wybrane = sorted(kand.values(), key=lambda x: (-x['wynik_rho'], x['pozycja']))[:NOWYCH]
    for d in wybrane:
        d['obrot'] = k
    zj(W / 'selekcje' / f's{s}_koopman_k{k:02d}.json', selekcja(f's{s}_koopman_k{k:02d}', prev['koopman'] + wybrane, baza))
    pool_ids = sorted(set(byid) - {d['frame_id'] for d in prev['losowe']})
    rng = np.random.default_rng(1000 * s + k)
    los = [dict(frame_id=pool_ids[j], obrot=k) for j in rng.choice(len(pool_ids), NOWYCH, replace=False)]
    zj(W / 'selekcje' / f's{s}_losowe_k{k:02d}.json', selekcja(f's{s}_losowe_k{k:02d}', prev['losowe'] + los, baza))
    zj(W / 'selekcje' / f's{s}_bez_k{k:02d}.json', selekcja(f's{s}_bez_k{k:02d}', [], baza))
    rap = dict(s=s, k=k, kandydatow=len(kand), pokrycie_cel_vs_obecny=pokrycie, sekundy_oceny=time.time() - t0,
               rho_wybranych=dict(srednia=float(np.mean([d['wynik_rho'] for d in wybrane])),
                                  trud_biezacy=float(np.mean([d['trud_biezacy'] for d in wybrane])),
                                  trud_odniesienia=float(np.mean([d['trud_odniesienia'] for d in wybrane]))),
               rho_wszystkich_kandydatow=float(np.mean([d['wynik_rho'] for d in kand.values()])))
    zj(lok / 'WYBOR.json', rap)
    print(f's{s} k{k}: kandydatów {len(kand)}, wybrano {len(wybrane)}, RHO wybranych {rap["rho_wybranych"]}, pokrycie {pokrycie}', flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('etap', choices=['kierunki', 'prognoza', 'wybor'])
    ap.add_argument('--s', type=int, required=True)
    ap.add_argument('--k', type=int, default=0)
    a = ap.parse_args(argv)
    {'kierunki': lambda: etap_kierunki(a.s), 'prognoza': lambda: etap_prognoza(a.s, a.k),
     'wybor': lambda: etap_wybor(a.s, a.k)}[a.etap]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
