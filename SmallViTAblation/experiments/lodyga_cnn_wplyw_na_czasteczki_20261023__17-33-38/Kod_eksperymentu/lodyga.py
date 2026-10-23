"""Czy łodyga CNN (4 konwolucje 3×3/2 z BatchNorm i GELU) uczy się razem z blokami i jak wpływa na ruch cząsteczek w blokach ViT.
A. Zmiana wag łodygi (konwolucje, BN γ/β, średnie kroczące BN) między checkpointami wobec bloków i głowy.
B. Dryf wejścia bloków (stan 0 = łodyga + kodowanie pozycji) na 10 klatkach TEST: cos do kroku 0, norma, Dirichlet, Fisher.
C. Zamiana łodygi między checkpointami (łodyga z A + bloki/głowa z B): IoU, pewność, Fisher po MLP 4 i 6, źródła uwagi bloku 4,
   rozrzut MLP 1, wskaźnik balistyczny, odległość stanów końcowych od układu pełnego.
Seedy: 3047 (klatki TEST z replikacji) i s47 (klatka TRAIN 0). Wynik: Wyniki/Dane/lodyga.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BAZA = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
REPL = BAZA / 'experiments/plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41'
sys.path.insert(0, str(REPL / 'Kod_eksperymentu'))
import wspolne as WS  # noqa: E402
from wspolne import T, H, W, dev  # noqa: E402

WY = Path(sys.argv[1]); WY.mkdir(parents=True, exist_ok=True)
S47 = BAZA / 'assets/MalyViT_35000_od_zera_s47/wagi'
idx = np.arange(T).reshape(H, W)
kraw = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1), np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])


def stan(sciezka):
    st = torch.load(sciezka, map_location='cpu', weights_only=False); return st.get('net', st)


def grupa(k):
    if k.startswith('lodyga.'):
        import re
        return 'lodyga_BN_stat' if 'running' in k else ('lodyga_BN_gb' if re.fullmatch(r'lodyga\.\d\.1\.(weight|bias)', k) else 'lodyga_conv')
    if k.startswith('bloki.'):
        return 'bloki'
    if k.startswith('glowa.'):
        return 'glowa'
    return 'inne'


def zmiana_wag(st0, st1):
    out = {}
    for k in st0:
        if 'num_batches' in k:
            continue
        a, b = st0[k].float().flatten(), st1[k].float().flatten()
        g = grupa(k); o = out.setdefault(g, dict(n0=0.0, d=0.0, dot=0.0, n1=0.0))
        o['n0'] += float((a ** 2).sum()); o['n1'] += float((b ** 2).sum()); o['d'] += float(((a - b) ** 2).sum()); o['dot'] += float((a * b).sum())
    return {g: dict(zmiana_wzgledna=round((o['d'] / max(o['n0'], 1e-12)) ** 0.5, 4), cos=round(o['dot'] / max((o['n0'] * o['n1']) ** 0.5, 1e-12), 4),
                    skala=round((o['n1'] / max(o['n0'], 1e-12)) ** 0.5, 4)) for g, o in out.items()}


def fisher(X, narz, tlo):
    a, b = X[narz], X[tlo]; d = a.mean(0) - b.mean(0)
    return float(d @ d / ((a.var(0).sum() * len(a) + b.var(0).sum() * len(b)) / (len(a) + len(b))))


def dirichlet(X):
    return float(((X[kraw[:, 0]] - X[kraw[:, 1]]) ** 2).sum(1).mean() / (2 * ((X - X.mean(0)) ** 2).sum(1).mean()))


def obserwable(P, udzial, gt):
    narz = udzial >= WS.PROG_NARZ; tlo = udzial == 0
    X = P['X'].float().cpu().numpy().astype(np.float64); Xc = X - X.mean(1, keepdims=True); rms = np.linalg.norm(Xc, axis=2).mean(1)
    dX = Xc[1:] - Xc[:-1]; en = (dX ** 2).sum(2)
    A = P['A'].float().cpu().numpy()
    return dict(iou=WS.iou(P['ramka'], gt), pewnosc=P['pewnosc'], fisher_wej=fisher(Xc[0], narz, tlo), fisher_m4=fisher(Xc[8], narz, tlo),
                fisher_m6=fisher(Xc[12], narz, tlo), zrodla_b4=float(np.median(1 / (A[3] ** 2).sum(2))), rozrzut_m1=float(rms[2] / rms[1]),
                balistyczny=float(np.median((dX.sum(0) ** 2).sum(1) / np.maximum(en.sum(0), 1e-12))), dirichlet_wej=dirichlet(Xc[0]),
                norma_wej=float(np.linalg.norm(X[0], axis=1).mean()), X0=X[0], Xk=X[12])


def zloz(st_lodyga, st_reszta):
    return {k: (st_lodyga[k] if k.startswith('lodyga.') else st_reszta[k]) for k in st_reszta}


wyn = {}
# ------------------------------------------------------------------ A. wagi
pary = {'s47': [(0, k) for k in (1094, 10940, 65640)], '3047': [(0, k) for k in (1000, 10000, 65640, 218750, 437500)] + [(218750, 437500)]}
sc = {'s47': lambda k: S47 / f'k{k:05d}.pt', '3047': WS.sciezka}
wyn['A_wagi'] = {s: {f'{a}->{b}': zmiana_wag(stan(sc[s](a)), stan(sc[s](b))) for a, b in pary[s]} for s in pary}
print(json.dumps(wyn['A_wagi'], indent=1), flush=True)
# ------------------------------------------------------------------ B i C: 3047 na 10 klatkach TEST
KL = json.loads((REPL / 'Wyniki/Dane/klatki.json').read_text(encoding='utf-8'))
OBR = [(WS.obraz(k['film'], k['klatka'])[0], WS.etykieta_cystotomu(k['film'], k['klatka'])['udzial'], k['gt']) for k in KL]
net = WS.siec()


def przebieg(st):
    net.load_state_dict(st, strict=True)
    return [obserwable(WS.przejscie(net, rgb), u, gt) for rgb, u, gt in OBR]


def srednie(R, klucze=('iou', 'pewnosc', 'fisher_wej', 'fisher_m4', 'fisher_m6', 'zrodla_b4', 'rozrzut_m1', 'balistyczny', 'dirichlet_wej', 'norma_wej')):
    return {k: round(float(np.mean([r[k] for r in R])), 4) for k in klucze}


S = {k: stan(WS.sciezka(k)) for k in (0, 1000, 10000, 65640, 218750, 437500)}
R = {k: przebieg(S[k]) for k in S}
wyn['B_dryf_wejscia_3047'] = {}
for k in S:
    cos0 = float(np.mean([np.mean(np.sum(r['X0'] * r0['X0'], 1) / (np.linalg.norm(r['X0'], axis=1) * np.linalg.norm(r0['X0'], axis=1))) for r, r0 in zip(R[k], R[0])]))
    wyn['B_dryf_wejscia_3047'][f'k{k}'] = dict(srednie(R[k]), cos_wejscia_do_k0=round(cos0, 4))
print('B', json.dumps(wyn['B_dryf_wejscia_3047'], indent=1), flush=True)
wyn['C_zamiana_3047'] = {}
for (la, lb) in ((0, 437500), (437500, 0), (65640, 437500), (218750, 437500), (0, 65640), (65640, 0)):
    Rz = przebieg(zloz(S[la], S[lb])); pelny = R[lb]
    odl = float(np.mean([np.linalg.norm(rz['Xk'] - rp['Xk']) / np.linalg.norm(rp['Xk'] - rp['Xk'].mean(0)) for rz, rp in zip(Rz, pelny)]))
    wyn['C_zamiana_3047'][f'lodyga_k{la}+reszta_k{lb}'] = dict(srednie(Rz), odleglosc_stanu_koncowego_od_pelnego=round(odl, 4))
    print('C', f'lodyga k{la} + reszta k{lb}', wyn['C_zamiana_3047'][f'lodyga_k{la}+reszta_k{lb}'], flush=True)
wyn['C_pelne_3047'] = {f'k{k}': srednie(R[k]) for k in S}
# ------------------------------------------------------------------ s47, klatka TRAIN 0
rgb47 = np.load(BAZA / 'LaserLatentSpaceApp/cache_uczenia/lasery/7992f2317c64a2e3/dane.npz')['rgb']
u47 = np.load(BAZA / 'LaserLatentSpaceApp/cache_uczenia/lasery/140854f7a6dbcbe5/etykieta_obrys.npz')['udzial']
gt47 = [194.0, 116.0, 334.67, 138.67]
S47st = {k: stan(S47 / f'k{k:05d}.pt') for k in (0, 65640)}


def jeden(st):
    net.load_state_dict(st, strict=True); r = obserwable(WS.przejscie(net, rgb47), (u47 >= 0.5).astype(float) * 0.5 + (u47 > 0) * 0.0 + u47 * 0, gt47)
    return r


def jeden47(st):
    net.load_state_dict(st, strict=True); P = WS.przejscie(net, rgb47)
    narz = u47 >= 0.5; tlo = u47 == 0
    X = P['X'].float().cpu().numpy().astype(np.float64); Xc = X - X.mean(1, keepdims=True); rms = np.linalg.norm(Xc, axis=2).mean(1)
    A = P['A'].float().cpu().numpy(); dX = Xc[1:] - Xc[:-1]; en = (dX ** 2).sum(2)
    return dict(iou=round(WS.iou(P['ramka'], gt47), 4), pewnosc=round(P['pewnosc'], 4), fisher_wej=round(fisher(Xc[0], narz, tlo), 3),
                fisher_m4=round(fisher(Xc[8], narz, tlo), 3), fisher_m6=round(fisher(Xc[12], narz, tlo), 3), zrodla_b4=round(float(np.median(1 / (A[3] ** 2).sum(2))), 1),
                rozrzut_m1=round(float(rms[2] / rms[1]), 3), balistyczny=round(float(np.median((dX.sum(0) ** 2).sum(1) / np.maximum(en.sum(0), 1e-12))), 3),
                dirichlet_wej=round(dirichlet(Xc[0]), 4), Xk=X[12])


wyn['D_s47_TRAIN0'] = {}
r_pelne = {k: jeden47(S47st[k]) for k in S47st}
for (la, lb) in ((0, 0), (65640, 65640), (0, 65640), (65640, 0)):
    r = jeden47(zloz(S47st[la], S47st[lb])); rp = r_pelne[lb]
    r['odleglosc_stanu_koncowego_od_pelnego'] = round(float(np.linalg.norm(r['Xk'] - rp['Xk']) / np.linalg.norm(rp['Xk'] - rp['Xk'].mean(0))), 4)
    wyn['D_s47_TRAIN0'][f'lodyga_k{la}+reszta_k{lb}'] = {k: v for k, v in r.items() if k != 'Xk'}
print('D', json.dumps(wyn['D_s47_TRAIN0'], indent=1), flush=True)
json.dump(wyn, open(WY / 'lodyga.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('zapisano', WY / 'lodyga.json')
