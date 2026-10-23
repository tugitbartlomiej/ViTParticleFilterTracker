"""Temperatura widoczna i niewidoczna dla głowy.
Przesunięcie tokenu w stanie końcowym (X[12]) między sąsiednimi checkpointami rozłożone na składową w podprzestrzeni wierszy jakobianu głowy
(norm -> Linear 192 -> GELU -> Linear 5) i resztę. Szczegóły: Opis/OPIS_EKSPERYMENTU.md.
Uruchomienie: PYTHONIOENCODING=utf-8 W:/Python/3.11/python.exe temperatura_glowy.py <katalog Wyniki/Dane>"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev, vmap

BAZA = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
REPL = BAZA / 'experiments/plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41'
sys.path.insert(0, str(REPL / 'Kod_eksperymentu'))
import wspolne as WS  # noqa: E402
from wspolne import T, C, dev  # noqa: E402

WY = Path(sys.argv[1]); WY.mkdir(parents=True, exist_ok=True)
TOL_SV = 1e-8
NA_EP = WS.NA_EPOKE
GRUPY = ['narz', 'tlo', 'wszystkie']
OKNA = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 100), (100, 200), (200, 300), (300, 400)]
_log = open(WY / 'log.txt', 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); _log.write(s + '\n'); _log.flush()


def zrob_glowe(net):
    """glowa_tokenu(x_i): R^192 -> R^5, dokładnie ścieżka modelu: net.glowa(net.norm(t)) dla jednego tokenu; kopia modułów w float64 na CPU."""
    norm, glowa = copy.deepcopy(net.norm).double().cpu(), copy.deepcopy(net.glowa).double().cpu()

    def glowa_tokenu(x_i):
        return glowa(norm(x_i))
    return glowa_tokenu


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ------------------------------------------------------------------ dane
KL = json.loads((REPL / 'Wyniki/Dane/klatki.json').read_text(encoding='utf-8'))
NF = len(KL)
OBR, UDZ, GT = [], [], []
for k in KL:
    OBR.append(WS.obraz(k['film'], k['klatka'])[0]); UDZ.append(WS.etykieta_cystotomu(k['film'], k['klatka'])['udzial']); GT.append(k['gt'])
MASKI = np.zeros((NF, 3, T), bool)
for f, u in enumerate(UDZ):
    MASKI[f, 0] = u >= WS.PROG_NARZ; MASKI[f, 1] = u == 0; MASKI[f, 2] = True
log('klatek', NF, 'tokenów narzędzia', MASKI[:, 0].sum(1).tolist(), 'tła', MASKI[:, 1].sum(1).tolist())
KR = WS.checkpointy(); NK = len(KR)
EP = np.array(KR) / NA_EP
log('checkpointów', NK, KR)

net = WS.siec()
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

# ------------------------------------------------------------------ przebieg 1: stany, jakobiany, bazy
X = np.zeros((NK, NF, T, C))
HJ = np.zeros((NK, NF, T, 5, C))
VH = np.zeros((NK, NF, T, 5, C))
SV = np.zeros((NK, NF, T, 5))
RANK = np.zeros((NK, NF, T), np.int8)
IOU = np.zeros((NK, NF)); PEW = np.zeros((NK, NF))
ERR_F32 = np.zeros((NK, NF)); ERR_MODEL = np.zeros((NK, NF)); ERR_F64 = np.zeros((NK, NF))
GLOWY = []
t0 = time.time()
for ik, k in enumerate(KR):
    WS.wczytaj(net, k)
    gt64 = zrob_glowe(net); GLOWY.append(gt64)
    for f in range(NF):
        P = WS.przejscie(net, OBR[f])
        IOU[ik, f] = WS.iou(P['ramka'], GT[f]); PEW[ik, f] = P['pewnosc']
        X12 = P['X'][12]                                                   # (600,192) float32 na dev
        with torch.no_grad():
            ref32 = net.glowa(net.norm(X12[None]))[0].double().cpu().numpy()                         # ścieżka modelu, partia
            x01 = torch.from_numpy(np.ascontiguousarray(OBR[f])).permute(2, 0, 1)[None].float().div(255).to(dev)
            pelny = net(x01)[0][0].reshape(T, 5).double().cpu().numpy()                              # pełny forward modelu
        X64 = X12.double().cpu()
        with torch.no_grad():
            wy64 = vmap(gt64)(X64).numpy()                                                          # glowa_tokenu na każdym tokenie (float64)
            x32 = X12                                                                               # ta sama funkcja w float32 na dev
            wy_g32 = net.glowa(net.norm(x32))                                                        # to samo, bez vmap
        ERR_F32[ik, f] = rel(wy64, ref32); ERR_MODEL[ik, f] = rel(wy64, pelny)
        # float64 vs float64: vmap(glowa_tokenu) vs partia w float64 (spójność funkcji jednotokenowej)
        with torch.no_grad():
            norm64 = copy.deepcopy(net.norm).double().cpu(); gl64 = copy.deepcopy(net.glowa).double().cpu()
            ERR_F64[ik, f] = rel(wy64, gl64(norm64(X64)).numpy())
        Hk = vmap(jacrev(gt64))(X64)                                                                # (600,5,192)
        U, S, Vh = torch.linalg.svd(Hk, full_matrices=False)
        maska = (S > TOL_SV * S.max(1, keepdim=True).values)
        X[ik, f] = X64.numpy(); HJ[ik, f] = Hk.numpy(); VH[ik, f] = (Vh * maska[..., None]).numpy()  # wiersze odrzucone zerowane
        SV[ik, f] = S.numpy(); RANK[ik, f] = maska.sum(1).numpy()
    log(f'k{k:>6d} ep {EP[ik]:6.1f}: IoU {IOU[ik].mean():.3f} pewn {PEW[ik].mean():.3f} blad glowy (vs model f32 / pelny forward / f64 partia) '
        f'{ERR_F32[ik].max():.2e} / {ERR_MODEL[ik].max():.2e} / {ERR_F64[ik].max():.2e} | wymiar efektywny min {RANK[ik].min()} sr {RANK[ik].mean():.3f} | {time.time() - t0:.0f}s')

R2 = np.zeros((NK, NF))
for ik in range(NK):
    for f in range(NF):
        R2[ik, f] = ((X[ik, f] - X[ik, f].mean(0)) ** 2).sum(1).mean()

# ------------------------------------------------------------------ losowe podprzestrzenie (jedna na token, seed 0)
gen = torch.Generator().manual_seed(0)
QR = torch.linalg.qr(torch.randn(T, C, 5, generator=gen, dtype=torch.float64))[0].numpy()          # (600,192,5) ortonormalne kolumny
log('losowa baza: maks |QᵀQ − I|', float(np.abs(np.einsum('tcr,tcs->trs', QR, QR) - np.eye(5)).max()))


def rozklad(d, Vh_, Q_):
    """d (600,192); Vh_ (600,5,192) wiersze ortonormalne (odrzucone = 0); Q_ (600,192,5). Zwraca energie na token."""
    coef = np.einsum('trc,tc->tr', Vh_, d)
    d_vis = np.einsum('trc,tr->tc', Vh_, coef); d_inv = d - d_vis
    cr = np.einsum('tcr,tc->tr', Q_, d)
    return (d ** 2).sum(1), (d_vis ** 2).sum(1), (d_inv ** 2).sum(1), (cr ** 2).sum(1)


def mediany(Etot, Evis, Einv, Erand, mask):
    out = []
    for g in range(3):
        m = mask[g]
        et, ev, ei, er = Etot[m], Evis[m], Einv[m], Erand[m]
        out.append([np.median(et), np.median(ev), np.median(ei), np.median(er), np.median(ev / np.maximum(et, 1e-300)), np.median(er / np.maximum(et, 1e-300))])
    return out


NAZWY_E = ['E_tot', 'E_vis', 'E_inv', 'E_rand', 'u_vis', 'u_rand']
# ------------------------------------------------------------------ przebieg 2: pierwsze różnice
NP = NK - 1
MET = np.zeros((NP, NF, 3, 6)); LIN = np.zeros((NP, NF, 3)); LIN_E = np.zeros((NP, NF, 3))
for ik in range(NP):
    for f in range(NF):
        d = X[ik + 1, f] - X[ik, f]
        Et, Ev, Ei, Er = rozklad(d, VH[ik, f], QR)
        for g, v in enumerate(mediany(Et, Ev, Ei, Er, MASKI[f])):
            MET[ik, f, g] = v
        pred = np.einsum('trc,tc->tr', HJ[ik, f], d)                                                # H_k d (linearyzacja)
        with torch.no_grad():
            fakt = (vmap(GLOWY[ik])(torch.from_numpy(X[ik + 1, f])) - vmap(GLOWY[ik])(torch.from_numpy(X[ik, f]))).numpy()
        r = np.linalg.norm(fakt, axis=1) / np.maximum(np.linalg.norm(pred, axis=1), 1e-300)
        for g in range(3):
            LIN[ik, f, g] = np.median(r[MASKI[f, g]])
# ------------------------------------------------------------------ przebieg 3: druga różnica
NJ = NK - 2
JET = np.zeros((NJ, NF, 3, 6))
for ik in range(1, NK - 1):
    for f in range(NF):
        j = X[ik, f] - 0.5 * (X[ik - 1, f] + X[ik + 1, f])
        Et, Ev, Ei, Er = rozklad(j, VH[ik, f], QR)
        for g, v in enumerate(mediany(Et, Ev, Ei, Er, MASKI[f])):
            JET[ik - 1, f, g] = v

KRa = np.array(KR); dkrok = np.diff(KRa).astype(float)
ep_para = 0.5 * (EP[:-1] + EP[1:])
ep_trojka = EP[1:-1]
odstep_l = KRa[1:-1] - KRa[:-2]; odstep_p = KRa[2:] - KRa[1:-1]
rowny = (np.abs(odstep_l - odstep_p) / odstep_p < 0.01)
log('trójki o równych odstępach (różnica < 1 %):', int(rowny.sum()), 'z', NJ, ' środki (ep):', np.round(ep_trojka[rowny][[0, -1]], 1).tolist())

# normalizacje: energie (kolumny 0..3) na 1000 kroków i / R²_k
E_1000 = MET[..., :4] * 1000.0 / dkrok[:, None, None, None]
E_1000_nr = E_1000 / R2[:NP, :, None, None]
J_nr = JET[..., :4] / R2[1:-1, :, None, None]

np.savez(WY / 'temperatura_glowy.npz',
         krok=KRa, epoka=EP, dkrok=dkrok, epoka_pary=ep_para, epoka_trojki=ep_trojka, rowny_odstep=rowny, grupy=np.array(GRUPY),
         met_nazwy=np.array(NAZWY_E), met_pary=MET, met_pary_na1000=E_1000, met_pary_na1000_nr=E_1000_nr,
         lin_stosunek=LIN, met_trojki=JET, met_trojki_nr=J_nr, iou=IOU, pewnosc=PEW, promien2_chmury=R2, wymiar_efektywny=RANK,
         wart_osobliwe=SV.astype(np.float32), blad_glowy_f32=ERR_F32, blad_glowy_pelny_forward=ERR_MODEL, blad_glowy_f64=ERR_F64,
         maski_grup=MASKI, zero_izotropia=5 / 192, tol_sv=TOL_SV)

# ------------------------------------------------------------------ tabele okien
def w_oknie(ep, a, b):
    return (ep >= a) & (ep < b)


def agreguj(maska_par, arr, ks):
    """mediana po (pary w oknie × klatki) dla kolumny ks tablicy arr (pary, klatki, grupy, metryki)."""
    return [float(np.median(arr[maska_par][:, :, g, ks])) for g in range(3)]


tab = dict(okna=[], jitter=[], werdykt={}, kontrola={})
for a, b in OKNA + [(0, 10), (200, 400)]:
    mp = w_oknie(ep_para, a, b)
    if not mp.any():
        continue
    wiersz = dict(okno=f'{a}-{b}', n_par=int(mp.sum()), iou=float(IOU[:NP][mp].mean()), iou_narz_tlo=None)
    for nazwa, ks, arr in (('E_tot_1000', 0, E_1000), ('E_vis_1000', 1, E_1000), ('E_inv_1000', 2, E_1000),
                           ('E_tot_1000_nr', 0, E_1000_nr), ('E_vis_1000_nr', 1, E_1000_nr), ('E_inv_1000_nr', 2, E_1000_nr)):
        wiersz[nazwa] = dict(zip(GRUPY, agreguj(mp, arr, ks)))
    wiersz['u_vis'] = dict(zip(GRUPY, agreguj(mp, MET, 4))); wiersz['u_rand'] = dict(zip(GRUPY, agreguj(mp, MET, 5)))
    wiersz['lin_stosunek'] = dict(zip(GRUPY, [float(np.median(LIN[mp][:, :, g])) for g in range(3)]))
    tab['okna'].append(wiersz)
OKNA_J = [(60, 100), (100, 200), (200, 300), (300, 400), (200, 400)]
for a, b in OKNA_J:
    mt = w_oknie(ep_trojka, a, b) & rowny
    if not mt.any():
        continue
    wiersz = dict(okno=f'{a}-{b}', n_trojek=int(mt.sum()), iou=float(IOU[1:-1][mt].mean()))
    for nazwa, ks, arr in (('J_tot', 0, JET), ('J_vis', 1, JET), ('J_inv', 2, JET), ('J_tot_nr', 0, J_nr), ('J_vis_nr', 1, J_nr), ('J_inv_nr', 2, J_nr)):
        wiersz[nazwa] = dict(zip(GRUPY, agreguj(mt, arr, ks)))
    wiersz['u_vis'] = dict(zip(GRUPY, agreguj(mt, JET, 4))); wiersz['u_rand'] = dict(zip(GRUPY, agreguj(mt, JET, 5)))
    tab['jitter'].append(wiersz)
okno = {w['okno']: w for w in tab['okna']}
for g in GRUPY:
    w1, w2, w0 = okno['1-10'], okno['200-400'], okno['0-10']
    for sfx in ('', '_nr'):
        spad_vis = w1['E_vis_1000' + sfx][g] / w2['E_vis_1000' + sfx][g]; spad_inv = w1['E_inv_1000' + sfx][g] / w2['E_inv_1000' + sfx][g]
        tab['werdykt'][f'{g}{sfx}'] = dict(spad_E_vis=spad_vis, spad_E_inv=spad_inv, stosunek_spadkow=spad_vis / spad_inv,
                                           a_spad_vis_ge_10=bool(spad_vis >= 10), b_stosunek_ge_3=bool(spad_vis / spad_inv >= 3))
    tab['werdykt'][g].update(u_vis_0_10=w0['u_vis'][g], u_vis_200_400=w2['u_vis'][g], u_rand_0_10=w0['u_rand'][g], u_rand_200_400=w2['u_rand'][g],
                             c_u_vis_nizszy_pozniej=bool(w2['u_vis'][g] < w0['u_vis'][g]), c_u_vis_ponad_zerem_losowym_pozno=bool(w2['u_vis'][g] > w2['u_rand'][g]))
tab['kontrola'] = dict(blad_glowy_f32_maks=float(ERR_F32.max()), blad_glowy_f32_sr=float(ERR_F32.mean()),
                       blad_glowy_pelny_forward_maks=float(ERR_MODEL.max()), blad_glowy_pelny_forward_sr=float(ERR_MODEL.mean()),
                       blad_glowy_f64_maks=float(ERR_F64.max()), wymiar_efektywny_min=int(RANK.min()), wymiar_efektywny_sr=float(RANK.mean()),
                       wymiar_efektywny_frakcja_5=float((RANK == 5).mean()), zero_izotropia=5 / 192)
(WY / 'temperatura_glowy_tabele.json').write_text(json.dumps(tab, indent=1, ensure_ascii=False), encoding='utf-8')
log('zapisano', WY, f'| czas {time.time() - t0:.0f}s')
