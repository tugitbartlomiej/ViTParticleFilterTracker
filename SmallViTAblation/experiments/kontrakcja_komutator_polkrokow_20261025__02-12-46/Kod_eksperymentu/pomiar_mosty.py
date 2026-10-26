"""Dwa „mosty” do fizyki układów cząstek dla 12 półkroków MałegoViT (lokalnie, bez treningu):
  (2) kontrakcja i parametr porządku na sferze LayerNorm w każdym półkroku (Criscitiello i in. 2024, Markdahl 2016, Isobe i in. 2026);
  (3) defekt komutatora uwaga–MLP w każdym bloku (Mudarisov i in. 2026, „Feed-Forward Steering in Transformer Residual Dynamics”).
Wzorzec przejścia: ../../schur_qr_jakobiany_mlp_20261024__10-03-40/Kod_eksperymentu/pomiar_schur_qr.py (ręczna pętla po blokach timm).

Stany: s = 0 (łodyga + kodowanie pozycji), s = 2b+1 (po uwadze bloku b), s = 2b+2 (po MLP bloku b), b = 0..5 (13 stanów);
dodatkowo stan „glowa” = net.norm(X_12) (wejście głowy, LayerNorm z parametrami afinicznymi), trzymany osobno (klucze *_glowa).

Uruchomienie (cwd dowolny):
  W:/Python/3.11/python.exe pomiar_mosty.py --test          # jeden checkpoint (ostatni), bez zapisu
  W:/Python/3.11/python.exe pomiar_mosty.py --szesc         # 6 checkpointów: pełny most 2 + most 3 + kontrola z hist.podkroki
  W:/Python/3.11/python.exe pomiar_mosty.py --wszystkie     # wszystkie checkpointy historii: skalary punktów 1–4 (checkpoint × stan)
Wyniki: ../Wyniki/Dane/."""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

KOD = Path(__file__).resolve().parent
APP = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp')
KAT = APP / 'cache_uczenia' / 'lasery' / '90f58d4f8b263d8b'
DANE = KOD.parent / 'Wyniki' / 'Dane'
KNN = 5
GRUPY = ('wszystkie', 'narzedzie', 'tlo')
WARIANTY = ('S_prawdziwa', 'R_odwrocona', 'P_rownolegla', 'tylko_uwaga', 'tylko_MLP', 'bez_bloku')

import torch                                                                  # noqa: E402

torch.set_grad_enabled(False)
sys.path.insert(0, str(APP))
from laser_app.compute import history as hist                                 # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------ przejście w przód
def wczytaj_wagi(net, h, krok):
    """Wczytuje k<krok>.pt (strict=True) z jednego odczytu pliku; zwraca SHA256 tych bajtów."""
    p = Path(h.wagi) / f'k{int(krok):05d}.pt'
    b = p.read_bytes()
    st = torch.load(io.BytesIO(b), map_location=DEV, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    return hashlib.sha256(b).hexdigest()


def A_(blk, t):
    return t + blk.ls1(blk.attn(blk.norm1(t)))


def M_(blk, t):
    return t + blk.ls2(blk.mlp(blk.norm2(t)))


def przejscie(net, x, Mm):
    """x: wejście znormalizowane (1, 3, 320, 480). Zwraca stany (13, 600, 192) float32, wejście głowy (600, 192), wyjście o (1, 20, 30, 5)."""
    f = x
    for modul in net.lodyga:
        f = modul(f)
    B, C, hh, ww = f.shape
    t = f.flatten(2).transpose(1, 2) + Mm.sincos_2d(hh, ww, C, x.device)
    S = [t]
    for blk in net.bloki:
        t = A_(blk, t); S.append(t)
        t = M_(blk, t); S.append(t)
    n = net.norm(t)
    return torch.cat(S, 0), n[0], net.glowa(n).view(1, hh, ww, 5)


def reszta(net, t, b0):
    """Bloki b0..5, norm, głowa na stanie t (1, 600, 192) → o (1, 20, 30, 5)."""
    for blk in net.bloki[b0:]:
        t = M_(blk, A_(blk, t))
    return net.glowa(net.norm(t)).view(1, hist.H, hist.W, 5)


def wyjscie(o, Mm, gt):
    """Mapa ciepła (600,) float64, maks. ciepło, pewność i ramka szczytu (dekoduj k=1, jak hist.podkroki), IoU z GT."""
    heat = o[0, ..., 0].float().sigmoid().reshape(-1).double().cpu().numpy()
    bx, sc = Mm.dekoduj(o.float().cpu(), k=1)
    ramka = bx[0, 0].numpy().astype(np.float64)
    return dict(heat=heat, max_cieplo=float(heat.max()), pewnosc=float(sc[0, 0]), ramka=ramka.tolist(),
                iou=float(hist.iou(ramka, gt)) if gt is not None else float('nan'))


# ------------------------------------------------------------------ miary mostu 2
def sfera(X):
    """Rzut LayerNorm bez parametrów afinicznych: u = (x − mean_c x)/‖x − mean_c x‖ (po 192 współrzędnych tokenu)."""
    c = X - X.mean(-1, keepdim=True)
    return c / c.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _odl(Y):
    return torch.cdist(Y, Y, compute_mode='donot_use_mm_for_euclid_dist')


def skalary(Xs, n):
    """Xs (S, 600, 192) float64 na GPU, n maska narzędzia (600,) bool (GPU). Punkty 1–3 (i ranga efektywna z punktu 4, CPU):
    słownik tablic (S,)."""
    t = ~n
    nn_, nt_ = int(n.sum()), int(t.sum())
    N = Xs.shape[1]
    out = {}
    U = sfera(Xs)
    for nazwa, Y in (('X', Xs), ('u', U)):
        D = _odl(Y)                                                             # (S, 600, 600), przekątna 0
        out[f'D_{nazwa}_wszystkie'] = D.sum((1, 2)) / (N * (N - 1))
        out[f'D_{nazwa}_narzedzie'] = D[:, n][:, :, n].sum((1, 2)) / (nn_ * (nn_ - 1))
        out[f'D_{nazwa}_tlo'] = D[:, t][:, :, t].sum((1, 2)) / (nt_ * (nt_ - 1))
        out[f'D_{nazwa}_narz_tlo'] = D[:, n][:, :, t].mean((1, 2))
        mn, mt = Y[:, n].mean(1), Y[:, t].mean(1)
        dm = mn - mt
        nd = dm.norm(dim=-1)
        Vn = ((Y[:, n] - mn[:, None]) ** 2).sum(-1).mean(1)
        Vt = ((Y[:, t] - mt[:, None]) ** 2).sum(-1).mean(1)
        out[f'sep_{nazwa}'] = nd / ((Vn + Vt) / 2).sqrt()
        w = dm / nd[:, None].clamp_min(1e-30)
        vn = (((Y[:, n] - mn[:, None]) * w[:, None]).sum(-1) ** 2).mean(1)
        vt = (((Y[:, t] - mt[:, None]) * w[:, None]).sum(-1) ** 2).mean(1)
        out[f'sep_os_{nazwa}'] = nd / ((vn + vt) / 2).sqrt()
        out[f'odl_srednich_{nazwa}'] = nd
        out[f'war_wewn_{nazwa}'] = (Vn + Vt) / 2
    mu_all, mu_n, mu_t = U.mean(1), U[:, n].mean(1), U[:, t].mean(1)
    out['r_wszystkie'] = mu_all.norm(dim=-1)
    out['r_narzedzie'] = mu_n.norm(dim=-1)
    out['r_tlo'] = mu_t.norm(dim=-1)
    out['cos_srednich_narz_tlo'] = (mu_n * mu_t).sum(-1) / (mu_n.norm(dim=-1) * mu_t.norm(dim=-1)).clamp_min(1e-30)
    out['norma_srednia_X'] = Xs.norm(dim=-1).mean(1)
    out = {k: v.cpu().numpy() for k, v in out.items()}
    nc = n.cpu().numpy()
    for nazwa, Y in (('X', Xs), ('u', U)):
        Yc = Y.cpu().numpy()
        for g, m in (('wszystkie', np.ones_like(nc)), ('narzedzie', nc), ('tlo', ~nc)):
            out[f'PR_{nazwa}_{g}'] = np.array([ranga_efektywna(Yc[s][m]) for s in range(Yc.shape[0])])
    return out


def ranga_efektywna(Y):
    """Participation ratio (Σλ)²/Σλ² wartości własnych kowariancji chmury (numpy, CPU)."""
    Yc = Y - Y.mean(0)
    lam = np.clip(np.linalg.eigvalsh(Yc.T @ Yc / Y.shape[0]), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def polkroki(Xs, n):
    """Punkty 5–6 dla 12 półkroków s → s+1. Xs (13, 600, 192) float64 (GPU). Zwraca tablice per token (12, 600) i k-NN."""
    X0, X1 = Xs[:-1], Xs[1:]
    dx = X1 - X0
    c = X0 - X0.mean(-1, keepdim=True)
    nc_ = c.norm(dim=-1)
    u = c / nc_[..., None]
    C = Xs.shape[-1]
    n2 = (dx ** 2).sum(-1)
    sr2 = C * dx.mean(-1) ** 2                                                   # składowa wzdłuż 1 (LayerNorm jej nie widzi)
    rad = (u * dx).sum(-1)                                                       # składowa radialna wzdłuż u (zmienia ‖c‖)
    sty2 = (n2 - sr2 - rad ** 2).clamp_min(0)                                    # składowa styczna do sfery (zmienia kierunek u)
    U = sfera(Xs)
    kat = torch.rad2deg(torch.arccos((U[:-1] * U[1:]).sum(-1).clamp(-1, 1)))     # kąt obrotu tokenu na sferze
    tok = dict(norma_dx=n2.sqrt(), udz_srednia=sr2 / n2, udz_radialna=rad ** 2 / n2, udz_styczna=sty2 / n2,
               radialna_wzgl=rad / nc_, kat_u_deg=kat)
    tok = {k: v.cpu().numpy() for k, v in tok.items()}
    zbior = dict(udz_styczna_suma=(sty2.sum(-1) / n2.sum(-1)).cpu().numpy(),     # Σ‖Δ_styczna‖²/Σ‖Δx‖² (ważone normą)
                 udz_radialna_suma=((rad ** 2).sum(-1) / n2.sum(-1)).cpu().numpy(),
                 udz_srednia_suma=(sr2.sum(-1) / n2.sum(-1)).cpu().numpy())
    for g, m in (('narzedzie', n), ('tlo', ~n)):
        zbior[f'udz_styczna_suma_{g}'] = (sty2[:, m].sum(-1) / n2[:, m].sum(-1)).cpu().numpy()
    # stabilność sąsiedztwa: odsetek k = 5 najbliższych sąsiadów (na u, i dla porównania na X) zachowanych po półkroku
    for nazwa, Y in (('u', U), ('X', Xs)):
        D = _odl(Y)
        D = D + torch.diag(torch.full((Y.shape[1],), float('inf'), device=Y.device, dtype=Y.dtype))
        idx = D.topk(KNN, dim=-1, largest=False).indices                         # (13, 600, 5)
        zach = (idx[:-1, :, :, None] == idx[1:, :, None, :]).any(-1).double().mean(-1)   # (12, 600)
        tok[f'knn_zachowane_{nazwa}'] = zach.cpu().numpy()
    return tok, zbior


# ------------------------------------------------------------------ most 3: komutator
def komutator(net, Xs, n, Mm, gt, heat_S):
    """Dla każdego bloku b z wejściem X = stan 2b: S = M(A(X)), R = A(M(X)), P = X + attn(LN1 X) + mlp(LN2 X).
    Zwraca (słownik per blok, tablice per token (6, 600) dla ilorazów i cos)."""
    nc = n.cpu().numpy()
    bloki, tok = [], {k: [] for k in ('SR_do_SX', 'SP_do_SX', 'cos_SX_PX', 'cos_SX_RX')}
    for b, blk in enumerate(net.bloki):
        X = Xs[2 * b][None].float()
        AX = A_(blk, X)
        MX = M_(blk, X)
        S = M_(blk, AX)
        R = A_(blk, MX)
        P = X + blk.ls1(blk.attn(blk.norm1(X))) + blk.ls2(blk.mlp(blk.norm2(X)))
        zgodnosc_S = float((S[0].double() - Xs[2 * b + 2]).abs().max() / Xs[2 * b + 2].abs().max())
        X64, S64, R64, P64 = (a[0].double() for a in (X, S, R, P))
        dS = S64 - X64; nS = dS.norm(dim=-1)
        sr = ((S64 - R64).norm(dim=-1) / nS).cpu().numpy()
        sp = ((S64 - P64).norm(dim=-1) / nS).cpu().numpy()
        cp = ((dS * (P64 - X64)).sum(-1) / (nS * (P64 - X64).norm(dim=-1))).cpu().numpy()
        cr = ((dS * (R64 - X64)).sum(-1) / (nS * (R64 - X64).norm(dim=-1))).cpu().numpy()
        for k, v in (('SR_do_SX', sr), ('SP_do_SX', sp), ('cos_SX_PX', cp), ('cos_SX_RX', cr)):
            tok[k].append(v)
        r = dict(blok=b, zgodnosc_S_ze_stanem=zgodnosc_S,
                 globalnie=dict(SR_do_SX=float((S64 - R64).norm() / dS.norm()), SP_do_SX=float((S64 - P64).norm() / dS.norm())))
        for k, v in (('SR_do_SX', sr), ('SP_do_SX', sp), ('cos_SX_PX', cp), ('cos_SX_RX', cr)):
            r[k] = {g: kw(v[m]) for g, m in (('wszystkie', np.ones_like(nc)), ('narzedzie', nc), ('tlo', ~nc))}
        r['warianty'] = {}
        for nazwa, Y in (('S_prawdziwa', S), ('R_odwrocona', R), ('P_rownolegla', P), ('tylko_uwaga', AX), ('tylko_MLP', MX),
                         ('bez_bloku', X)):
            w = wyjscie(reszta(net, Y, b + 1), Mm, gt)
            hm = w.pop('heat')
            w['korelacja_mapy_z_S'] = float(np.corrcoef(hm, heat_S)[0, 1]) if hm.std() > 0 and heat_S.std() > 0 else float('nan')
            w['maks_roznica_mapy_od_S'] = float(np.abs(hm - heat_S).max())
            w['argmax_mapy'] = int(hm.argmax())
            r['warianty'][nazwa] = w
        bloki.append(r)
    return bloki, {k: np.stack(v) for k, v in tok.items()}


# ------------------------------------------------------------------ narzędzia
def kw(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, q)) for q in (10, 50, 90)] if v.size else None


def przygotuj():
    h = hist.History(KAT)
    Mm = hist.modul_modelu()
    net = Mm.MalyViT().to(DEV).eval().requires_grad_(False)
    x = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(DEV)
    x = (x - Mm.MEAN.to(DEV)) / Mm.STD.to(DEV)
    n = torch.as_tensor(np.asarray(h.narz, bool), device=DEV)
    return h, Mm, net, x, n


def szesc_krokow(h):
    K = len(h.kroki)
    idx = sorted(set(np.linspace(0, K - 1, 6).round().astype(int).tolist()))
    return idx, [int(h.kroki[i]) for i in idx]


def jeden_pelny(h, Mm, net, x, n, krok, kontrola=True):
    t0 = time.time()
    sha = wczytaj_wagi(net, h, krok)
    X, Xg, o = przejscie(net, x, Mm)
    t1 = time.time()
    Xs = X.double()
    sk = skalary(torch.cat([Xs, Xg.double()[None]]), n)
    t2 = time.time()
    tok, zb = polkroki(Xs, n)
    t3 = time.time()
    wS = wyjscie(o, Mm, h.gt)
    heat_S = wS.pop('heat')
    bloki, tokK = komutator(net, Xs, n, Mm, h.gt, heat_S)
    t4 = time.time()
    kontr = {}
    if kontrola:                                                                # zgodność z hist.podkroki (aplikacja)
        d = hist.podkroki(h.wagi, krok, h.rgb, urzadzenie=DEV.type, baza=None, lodyga=False)
        G = net.glowa(net.norm(X)).float().cpu().numpy()
        kontr = dict(glowa_maks_roznica=float(np.abs(G - d['glowa']).max()),
                     ramka_maks_roznica=float(np.abs(np.asarray(wS['ramka']) - d['ramka']).max()),
                     pewnosc_roznica=float(abs(wS['pewnosc'] - d['pewnosc'])),
                     mapa_maks_roznica=float(np.abs(heat_S - d['mapa']).max()))
        wczytaj_wagi(net, h, krok)                                              # (podkroki używa własnej sieci; nasza bez zmian)
    t5 = time.time()
    czasy = dict(wczytanie_i_przejscie_s=t1 - t0, skalary_s=t2 - t1, polkroki_s=t3 - t2, komutator_s=t4 - t3, kontrola_podkroki_s=t5 - t4)
    return dict(sha=sha, sk=sk, tok=tok, zb=zb, wS=wS, heat_S=heat_S, bloki=bloki, tokK=tokK, kontrola=kontr, czasy=czasy)


def podsumuj_pelny(r, nc):
    """JSON: stany (13 + glowa), półkroki (12), komutator (6 bloków)."""
    sk = r['sk']
    stany = []
    for s in range(14):
        stany.append({k: float(v[s]) for k, v in sk.items()})
    pk = []
    for s in range(12):
        e = dict(polkrok=s + 1, z_stanu=s, do_stanu=s + 1, rodzaj='uwaga' if s % 2 == 0 else 'MLP', blok=s // 2)
        for k in [k for k in sk if k.startswith('D_') or k.startswith('sep') or k.startswith('PR_')] + ['r_wszystkie', 'r_narzedzie',
                                                                                                        'r_tlo']:
            e[f'iloraz_{k}'] = float(sk[k][s + 1] / sk[k][s])
        for k in ('sep_u', 'sep_X', 'sep_os_u', 'sep_os_X', 'r_wszystkie', 'r_narzedzie', 'r_tlo', 'cos_srednich_narz_tlo'):
            e[f'zmiana_{k}'] = float(sk[k][s + 1] - sk[k][s])
        for k, v in r['tok'].items():
            e[k] = {g: kw(v[s][m]) for g, m in (('wszystkie', np.ones_like(nc)), ('narzedzie', nc), ('tlo', ~nc))}
        for k, v in r['zb'].items():
            e[k] = float(v[s])
        pk.append(e)
    return dict(stany=stany, polkroki=pk, komutator=r['bloki'], wyjscie_S=r['wS'], kontrola_podkroki=r['kontrola'],
                czasy_s=r['czasy'], wagi_sha256=r['sha'])


def tryb_szesc(arg):
    h, Mm, net, x, n = przygotuj()
    idx, kroki = szesc_krokow(h)
    if arg.test:
        idx, kroki = [len(h.kroki) - 1], [int(h.kroki[-1])]
    nc = np.asarray(h.narz, bool)
    print(f'historia {KAT.name}: K={len(h.kroki)}, {h.etykieta}, narz={int(nc.sum())}, gt={h.gt}, checkpointy {kroki}', flush=True)
    podsum = dict(historia=str(KAT), etykieta_narzedzia=h.etykieta, n_narz=int(nc.sum()), n_tlo=int((~nc).sum()), gt=h.gt,
                  kroki=kroki, indeksy=idx, na_epoke=hist.NA_EPOKE, knn=KNN, checkpointy={})
    for krok in kroki:
        t0 = time.time()
        r = jeden_pelny(h, Mm, net, x, n, krok)
        ps = podsumuj_pelny(r, nc)
        print(f'k{krok}: {time.time() - t0:.1f} s; kontrola {r["kontrola"]}; IoU S {r["wS"]["iou"]:.3f}; '
              f'zgodność S {max(b["zgodnosc_S_ze_stanem"] for b in r["bloki"]):.1e}', flush=True)
        if arg.test:
            sk = r['sk']
            for k in ('D_u_wszystkie', 'D_u_narzedzie', 'D_u_tlo', 'r_wszystkie', 'sep_u', 'PR_u_wszystkie'):
                print(f'  {k:16s}', np.round(sk[k], 3))
            print('  udz_styczna_suma', np.round(r['zb']['udz_styczna_suma'], 3))
            for b in r['bloki']:
                print('  blok', b['blok'], 'SR', np.round(b['SR_do_SX']['wszystkie'], 3), 'SP', np.round(b['SP_do_SX']['wszystkie'], 3),
                      {k: round(v['iou'], 3) for k, v in b['warianty'].items()})
            print('  czasy', r['czasy'])
            continue
        podsum['checkpointy'][f'k{krok}'] = ps
        zapis = dict(krok=krok, narz=nc, heat_S=r['heat_S'], **{f'stan_{k}': v for k, v in r['sk'].items()},
                     **{f'tok_{k}': v for k, v in r['tok'].items()}, **{f'polkrok_{k}': v for k, v in r['zb'].items()},
                     **{f'komutator_{k}': v for k, v in r['tokK'].items()})
        np.savez_compressed(DANE / f'mosty_k{krok}.npz', **zapis)
    if arg.test:
        return
    (DANE / 'mosty_6_checkpointow.json').write_text(json.dumps(podsum, indent=1, ensure_ascii=False), encoding='utf-8')
    _csv_komutator(podsum)
    print('zapisano mosty_6_checkpointow.json, komutator.csv, mosty_k*.npz')


def _csv_komutator(podsum):
    kol = ['krok', 'blok', 'wariant', 'max_cieplo', 'pewnosc', 'iou', 'korelacja_mapy_z_S', 'maks_roznica_mapy_od_S', 'argmax_mapy',
           'SR_do_SX_med', 'SP_do_SX_med', 'cos_SX_PX_med', 'SR_do_SX_med_narz', 'SR_do_SX_med_tlo', 'SP_do_SX_med_narz',
           'SP_do_SX_med_tlo', 'cos_SX_PX_med_narz', 'cos_SX_PX_med_tlo']
    with open(DANE / 'komutator.csv', 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(kol) + '\n')
        for kk, c in podsum['checkpointy'].items():
            for b in c['komutator']:
                for wn, w in b['warianty'].items():
                    row = [kk[1:], b['blok'], wn, w['max_cieplo'], w['pewnosc'], w['iou'], w['korelacja_mapy_z_S'],
                           w['maks_roznica_mapy_od_S'], w['argmax_mapy'],
                           b['SR_do_SX']['wszystkie'][1], b['SP_do_SX']['wszystkie'][1], b['cos_SX_PX']['wszystkie'][1],
                           b['SR_do_SX']['narzedzie'][1], b['SR_do_SX']['tlo'][1], b['SP_do_SX']['narzedzie'][1],
                           b['SP_do_SX']['tlo'][1], b['cos_SX_PX']['narzedzie'][1], b['cos_SX_PX']['tlo'][1]]
                    fh.write(','.join(f'{v:.8g}' if isinstance(v, float) else str(v) for v in row) + '\n')


def tryb_wszystkie(arg):
    h, Mm, net, x, n = przygotuj()
    kroki = [int(k) for k in h.kroki]
    if arg.test:
        kroki = kroki[:3]
    K = len(kroki)
    wyn, wyj, sha, czasy = {}, {k: [] for k in ('max_cieplo', 'pewnosc', 'iou')}, [], []
    t_start = time.time()
    for i, krok in enumerate(kroki):
        t0 = time.time()
        sha.append(wczytaj_wagi(net, h, krok))
        X, Xg, o = przejscie(net, x, Mm)
        sk = skalary(torch.cat([X.double(), Xg.double()[None]]), n)
        w = wyjscie(o, Mm, h.gt)
        for k in wyj:
            wyj[k].append(w[k])
        for k, v in sk.items():
            wyn.setdefault(k, np.zeros((K, 14)))[i] = v
        czasy.append(time.time() - t0)
        if i % 50 == 0 or i == K - 1:
            print(f'{i + 1}/{K} k{krok}: {czasy[-1]:.2f} s (łącznie {time.time() - t_start:.0f} s); r12 {sk["r_wszystkie"][12]:.3f} '
                  f'sep_u12 {sk["sep_u"][12]:.3f} IoU {w["iou"]:.3f}', flush=True)
    kr = np.array(kroki)
    zg = dict(pewnosc=float(np.nanmax(np.abs(np.array(wyj['pewnosc']) - np.asarray(h.pew)[:K]))),
              iou=float(np.nanmax(np.abs(np.array(wyj['iou']) - np.asarray(h.iou)[:K]))))
    print('zgodność z historią (pew, iou):', zg)
    if arg.test:
        return
    zapis = dict(kroki=kr, epoki=kr / hist.NA_EPOKE, stany=np.array([str(s) for s in range(13)] + ['glowa']),
                 **{k: v[:, :13] for k, v in wyn.items()}, **{f'{k}_glowa': v[:, 13] for k, v in wyn.items()},
                 **{k: np.array(v) for k, v in wyj.items()}, wagi_sha256=np.array(sha))
    np.savez_compressed(DANE / 'wszystkie_checkpointy.npz', **zapis)
    with open(DANE / 'wszystkie_checkpointy_skalary.csv', 'w', encoding='utf-8', newline='') as fh:
        nazwy = list(wyn)
        fh.write(','.join(['krok', 'epoka', 'stan'] + nazwy) + '\n')
        for i in range(K):
            for s in range(14):
                fh.write(','.join([str(kr[i]), f'{kr[i] / hist.NA_EPOKE:.4f}', 'glowa' if s == 13 else str(s)]
                                  + [f'{wyn[k][i, s]:.8g}' for k in nazwy]) + '\n')
    with open(DANE / 'wagi_sha256_wszystkie.csv', 'w', encoding='utf-8', newline='') as fh:
        fh.write('krok,plik,sha256\n')
        for k, s in zip(kroki, sha):
            fh.write(f'{k},k{k:05d}.pt,{s}\n')
    log = dict(K=K, czas_laczny_s=time.time() - t_start, czas_na_checkpoint_s=kw(czasy), zgodnosc_z_historia=zg)
    (DANE / 'log_wszystkie.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    print('zapisano wszystkie_checkpointy.npz/.csv', log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--szesc', action='store_true')
    ap.add_argument('--wszystkie', action='store_true')
    arg = ap.parse_args()
    DANE.mkdir(parents=True, exist_ok=True)
    if arg.szesc or (arg.test and not arg.wszystkie):
        tryb_szesc(arg)
    if arg.wszystkie:
        tryb_wszystkie(arg)


if __name__ == '__main__':
    main()
