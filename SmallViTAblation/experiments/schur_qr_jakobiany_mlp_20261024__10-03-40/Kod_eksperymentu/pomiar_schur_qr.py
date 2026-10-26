"""Pełne jakobiany 192 × 192 pola MLP (6 bloków × 600 tokenów) jako tensor gradientu prędkości (Keylock 2018):
rozkład Schura (Λ + N), S/Ω, niezmienniki Q, R (i rozbicie Q_S, Q_Ω, Keylock), widmo; to samo dla kompresji J3 = P J Pᵀ (3 × 3)
z klasycznym diagramem Q–R. Lokalnie, bez treningu. Wzorzec jakobianu i przejścia: diagnostyki_rzutu.py (sesja 2026-10-24).

Uruchomienie (cwd dowolny):  W:/Python/3.11/python.exe pomiar_schur_qr.py [--test] [--tylko-ostatni] [--procesy 12]
Wyniki: ../Wyniki/Dane/miary_k<krok>.npz, miary_k<krok>_tokeny.csv (ostatni checkpoint), podsumowanie.json, log_czasow.json."""
import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KOD = Path(__file__).resolve().parent
sys.path.insert(0, str(KOD))
import schur_miary as sm                                                     # noqa: E402

APP = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp')
KAT = APP / 'cache_uczenia' / 'lasery' / '90f58d4f8b263d8b'
DANE = KOD.parent / 'Wyniki' / 'Dane'


def przejscie_i_jakobiany(h, krok, baza, dev):
    """Przejście w przód checkpointu `krok` (hist.podkroki + ręczna pętla po blokach) → J (6, 600, 192, 192) float64 (numpy),
    F_surowe i F_srodkowane (6, 600, 192), ciepło głowy (600,), P (3, 192), zgodność J3 z jak_n aplikacji (6,)."""
    import torch
    from laser_app.compute import history as hist
    torch.set_grad_enabled(False)
    d = hist.podkroki(h.wagi, krok, h.rgb, urzadzenie=dev.type, baza=baza, lodyga=False)
    net = hist._siec[dev.type]
    M = hist.modul_modelu()
    P = torch.as_tensor(d['baza']['P_n'], device=dev, dtype=torch.float64)
    x = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
    x = (x - M.MEAN.to(dev)) / M.STD.to(dev)
    f = x
    for modul in net.lodyga:
        f = modul(f)
    B, C, hh, ww = f.shape
    t = f.flatten(2).transpose(1, 2) + M.sincos_2d(hh, ww, C, dev)
    XM = []
    for blk in net.bloki:
        t = t + blk.ls1(blk.attn(blk.norm1(t)))
        XM.append(t[0].double())
        t = t + blk.ls2(blk.mlp(blk.norm2(t)))
    Js, Fr, Fc, zg = [], [], [], []
    jak_app = np.asarray(d['jak_n'])
    I = torch.eye(C, device=dev, dtype=torch.float64)
    for b, (blk, xm) in enumerate(zip(net.bloki, XM)):
        ln = blk.norm2
        W1 = blk.mlp.fc1.weight.double(); b1 = blk.mlp.fc1.bias.double(); W2 = blk.mlp.fc2.weight.double()
        g = ln.weight.double(); be = ln.bias.double()
        mu = xm.mean(-1, keepdim=True); sig = (xm.var(-1, unbiased=False, keepdim=True) + ln.eps).sqrt(); xh = (xm - mu) / sig
        a = (xh * g + be) @ W1.T + b1
        z = torch.nn.functional.gelu(a)
        Fc.append(((z - z.mean(0, keepdim=True)) @ W2.T).cpu().numpy())
        Fr.append((z @ W2.T + blk.mlp.fc2.bias.double()).cpu().numpy())
        gp = 0.5 * (1 + torch.erf(a / math.sqrt(2.0))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
        JLN = ((I - 1.0 / C)[None] - xh[:, :, None] * xh[:, None, :] / C) * (g / sig)[:, :, None]
        H = torch.einsum('cd,jde->jce', W1, JLN) * gp[:, :, None]
        J = torch.einsum('fc,jce->jfe', W2, H)
        J3 = torch.einsum('ad,jde,eb->jab', P, J, P.T).cpu().numpy()
        zg.append(float(np.abs(J3 - jak_app[b]).max() / np.abs(jak_app[b]).max()))
        Js.append(J.cpu().numpy()); del H, JLN, J
    # kontrola: F_surowe = prawdziwy skok MLP z modułu timm
    sprawdz_F = float(max(np.abs(Fr[b] - blk_out).max() / np.abs(blk_out).max()
                          for b, blk_out in enumerate(_mlp_prawdziwe(net, XM))))
    cieplo = 1.0 / (1.0 + np.exp(-np.asarray(d['glowa'][12, :, 0], np.float64)))
    return (np.stack(Js), np.stack(Fr), np.stack(Fc), cieplo, np.asarray(d['baza']['P_n'], np.float64), zg, sprawdz_F,
            np.asarray(d['mapa'], np.float64))


def _mlp_prawdziwe(net, XM):
    out = []
    for blk, xm in zip(net.bloki, XM):
        out.append(blk.mlp(blk.norm2(xm.float())).double().cpu().numpy())
    return out


def licz_wsad(pula, Js, porcja):
    """Js (k, n, n) → (miary (k, m), λ (k, n)) równolegle w puli procesów."""
    kaw = [Js[i:i + porcja] for i in range(0, len(Js), porcja)]
    wyn = list(pula.map(sm.miary_wsad, kaw))
    return np.concatenate([w[0] for w in wyn]), np.concatenate([w[1] for w in wyn])


def kw(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, q)) for q in (10, 50, 90)] if v.size else None


def sha256(p):
    hs = hashlib.sha256()
    with open(p, 'rb') as fh:
        for blok in iter(lambda: fh.read(1 << 20), b''):
            hs.update(blok)
    return hs.hexdigest()


def typ_znakow(n_dod, n_uj):
    """1 = wszystkie niezerowe Re λ < 0 (stabilny), 2 = mieszane (siodło), 3 = wszystkie > 0 (niestabilny)."""
    return np.where(n_dod == 0, 1, np.where(n_uj == 0, 3, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='tylko blok 0, 24 tokeny, bez zapisu')
    ap.add_argument('--tylko-ostatni', action='store_true')
    ap.add_argument('--procesy', type=int, default=12)
    ap.add_argument('--porcja', type=int, default=25)
    arg = ap.parse_args()
    import torch
    sys.path.insert(0, str(APP))
    from laser_app.compute import history as hist, passage_store
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h = hist.History(KAT)
    narz = np.asarray(h.narz, bool)
    baza = passage_store.wczytaj_baze(KAT / 'przejscia')
    K = len(h.kroki)
    idx = sorted(set(np.linspace(0, K - 1, 6).round().astype(int).tolist()))
    kroki = [int(h.kroki[i]) for i in idx]
    if arg.tylko_ostatni or arg.test:
        kroki = [int(h.kroki[-1])]
    print(f'historia {KAT.name}: K={K}, etykieta: {h.etykieta}, narz={int(narz.sum())}, checkpointy {kroki}', flush=True)
    DANE.mkdir(parents=True, exist_ok=True)
    czasy = {}
    podsum = dict(historia=str(KAT), etykieta_narzedzia=h.etykieta, tryb_etykiety=h.tryb_etykiety, n_narz=int(narz.sum()),
                  n_tlo=int((~narz).sum()), kroki=kroki, indeksy_checkpointow=idx if len(kroki) > 1 else [K - 1],
                  tol_zera=f'|.| <= {sm.TOL_WZGL:g} * max(||J||_F, 1)', miary=sm.NAZWY, checkpointy={})
    with ProcessPoolExecutor(max_workers=arg.procesy, initializer=_init_watku) as pula:
        for krok in kroki:
            t0 = time.time()
            Js, Fr, Fc, cieplo, P, zg, spF, mapa = przejscie_i_jakobiany(h, krok, baza, dev)
            t1 = time.time()
            if arg.test:
                Js = Js[:1, :24]
            nb, nt = Js.shape[:2]
            J3 = np.einsum('ad,bjde,ce->bjac', P, Js, P)
            m192, lam192 = licz_wsad(pula, Js.reshape(-1, 192, 192), arg.porcja)
            t2 = time.time()
            m3, lam3 = sm.miary_wsad(J3.reshape(-1, 3, 3))
            t3 = time.time()
            m192 = m192.reshape(nb, nt, -1); lam192 = lam192.reshape(nb, nt, 192)
            m3 = m3.reshape(nb, nt, -1); lam3 = lam3.reshape(nb, nt, 3)
            print(f'k{krok}: przejście+J {t1 - t0:.1f} s, Schur/eig/SVD 192 {t2 - t1:.1f} s, 3×3 {t3 - t2:.1f} s; '
                  f'zgodność J3 z jak_n {max(zg):.1e}, F z timm {spF:.1e}', flush=True)
            if arg.test:
                for k, v in zip(sm.NAZWY, m192[0].T):
                    print(f'  {k:22s} {np.percentile(v, [0, 50, 100])}')
                mod = np.abs(lam192[0]); print('  najmniejsze |λ|:', np.sort(mod, 1)[:, :4][:5])
                print('  3x3:', {k: float(np.median(v)) for k, v in zip(sm.NAZWY, m3[0].T)})
                return
            czasy[f'k{krok}'] = dict(przejscie_i_jakobiany_s=t1 - t0, schur_eig_svd_192_s=t2 - t1, miary_3x3_s=t3 - t2)
            # klasy Q–R 3D, typy znakowe
            iQ, iR = sm.NAZWY.index('Q'), sm.NAZWY.index('R')
            D3, kl3 = sm.klasa_qr3(m3[..., iQ], m3[..., iR])
            iP, iU = sm.NAZWY.index('n_re_dod'), sm.NAZWY.index('n_re_uj')
            typ192 = typ_znakow(m192[..., iP], m192[..., iU]); typ3 = typ_znakow(m3[..., iP], m3[..., iU])
            zesp192 = m192[..., sm.NAZWY.index('n_im')] > 0; zesp3 = m3[..., sm.NAZWY.index('n_im')] > 0
            nF = np.linalg.norm(Fr, axis=2); nFc = np.linalg.norm(Fc, axis=2)
            zapis = dict(krok=krok, nazwy=np.array(sm.NAZWY), m192=m192, m3=m3, lam3=lam3, J3=J3, P=P, narz=narz, cieplo=cieplo,
                         mapa=mapa, normF_surowe=nF, normF_srodkowane=nFc, delta_QR3=D3, klasa_QR3=kl3, typ192=typ192, typ3=typ3,
                         zgodnosc_J3_z_jak_n=np.array(zg))
            if krok == int(h.kroki[-1]):
                zapis['lam192'] = lam192
            np.savez_compressed(DANE / f'miary_k{krok}.npz', **zapis)
            if krok == int(h.kroki[-1]):
                _csv(DANE / f'miary_k{krok}_tokeny.csv', m192, m3, narz, cieplo, nF, nFc, D3, kl3, typ192, typ3)
            podsum['checkpointy'][f'k{krok}'] = _podsumuj(m192, m3, narz, cieplo, nF, nFc, kl3, typ192, typ3, zesp192, zesp3, D3,
                                                          zg, spF)
            print(f'  zapisano miary_k{krok}.npz ({time.time() - t0:.1f} s łącznie)', flush=True)
    # trend: mediany na blok w funkcji kroku
    podsum['trend_mediany'] = {grupa: {zr: {k: [[podsum['checkpointy'][f'k{kr}'][f'blok_{b}'][zr][grupa][k][1]
                                                  for kr in kroki] for b in range(6)]
                                          for k in sm.NAZWY} for zr in ('J192', 'J3')}
                               for grupa in ('wszystkie', 'narzedzie', 'tlo')}
    podsum['czasy_s'] = czasy
    podsum['wagi_sha256'] = {f'k{kr}': sha256(Path(h.wagi) / f'k{kr:05d}.pt') for kr in kroki}
    (DANE / 'podsumowanie.json').write_text(json.dumps(podsum, indent=1, ensure_ascii=False), encoding='utf-8')
    (DANE / 'log_czasow.json').write_text(json.dumps(czasy, indent=1), encoding='utf-8')
    print('zapisano podsumowanie.json')


def _init_watku():
    import os
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _podsumuj(m192, m3, narz, cieplo, nF, nFc, kl3, typ192, typ3, zesp192, zesp3, D3, zg, spF):
    from scipy.stats import spearmanr
    out = {}
    for b in range(m192.shape[0]):
        r = dict(zgodnosc_J3_z_jak_n=zg[b])
        for zr, m in (('J192', m192[b]), ('J3', m3[b])):
            r[zr] = {g: {k: kw(m[mask, i]) for i, k in enumerate(sm.NAZWY)}
                     for g, mask in (('wszystkie', np.ones_like(narz)), ('narzedzie', narz), ('tlo', ~narz))}
            sp = {}
            for i, k in enumerate(sm.NAZWY):
                v = m[:, i]
                if np.ptp(v) == 0:
                    sp[k] = None; continue
                sp[k] = dict(cieplo=float(spearmanr(v, cieplo)[0]), normF_surowe=float(spearmanr(v, nF[b])[0]),
                             normF_srodkowane=float(spearmanr(v, nFc[b])[0]))
            r[zr]['spearman'] = sp
        nazwy_kl = ['SFS_ognisko_stab_rozciaganie', 'UFC_ognisko_niestab_sciskanie', 'UNSS_wezel_niestab_siodlo',
                    'SNSS_wezel_stab_siodlo']
        r['klasy_QR3'] = {g: {nazwy_kl[k]: int(((kl3[b] == k) & mask).sum()) for k in range(4)}
                          for g, mask in (('wszystkie', np.ones_like(narz)), ('narzedzie', narz), ('tlo', ~narz))}
        r['typ_znakow'] = {zr: {g: {n_: int(((t[b] == k) & mask).sum()) for k, n_ in ((1, 'stabilny'), (2, 'siodlo'),
                                                                                       (3, 'niestabilny'))}
                                for g, mask in (('wszystkie', np.ones_like(narz)), ('narzedzie', narz), ('tlo', ~narz))}
                           for zr, t in (('J192', typ192), ('J3', typ3))}
        r['zespolone'] = dict(J192=int(zesp192[b].sum()), J3=int(zesp3[b].sum()),
                              J3_ognisko_Delta_dodatnia=int((D3[b] > 0).sum()),
                              zgodnosc_Delta_z_widmem_J3=int(((D3[b] > 0) == zesp3[b]).sum()))
        r['tabela_klasaQR3_x_typ192'] = [[int(((kl3[b] == i) & (typ192[b] == k)).sum()) for k in (1, 2, 3)] for i in range(4)]
        r['tabela_typ3_x_typ192'] = [[int(((typ3[b] == i) & (typ192[b] == k)).sum()) for k in (1, 2, 3)] for i in (1, 2, 3)]
        out[f'blok_{b}'] = r
    out['sprawdzenie_F_z_timm'] = spF
    return out


def _csv(p, m192, m3, narz, cieplo, nF, nFc, D3, kl3, typ192, typ3):
    kol = (['blok', 'token', 'narzedzie', 'cieplo', 'normF_surowe', 'normF_srodkowane'] + [f'J192_{k}' for k in sm.NAZWY]
           + [f'J3_{k}' for k in sm.NAZWY] + ['J3_delta_QR', 'J3_klasa_QR', 'typ192', 'typ3'])
    with open(p, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(kol) + '\n')
        for b in range(m192.shape[0]):
            for j in range(m192.shape[1]):
                w = ([b, j, int(narz[j]), cieplo[j], nF[b, j], nFc[b, j]] + list(m192[b, j]) + list(m3[b, j])
                     + [D3[b, j], int(kl3[b, j]), int(typ192[b, j]), int(typ3[b, j])])
                fh.write(','.join(f'{v:.10g}' if isinstance(v, float) else str(v) for v in w) + '\n')


if __name__ == '__main__':
    main()
