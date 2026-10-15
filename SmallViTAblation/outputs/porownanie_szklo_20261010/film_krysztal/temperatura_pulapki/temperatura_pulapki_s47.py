"""Temperatura w pułapce laserowej MLP bloku 6 (bieg MalyViT_35000_od_zera_s47, zdjęcie TRAIN 0, 600 tokenów), 2026-10-13.

A. Temperatura reflektorów (GELU jako obsadzenie Fermiego). GELU(a) = a·Φ(a), a Φ(a) ≈ 1/(1 + exp(−1,702·a)) to funkcja Fermiego
   z kT₀ = 1/1,702 ≈ 0,59 w jednostkach pobudzenia a. Liczy się szerokość bramki względem rozrzutu pobudzeń:
   T_laser(c) = kT₀ / sd_j(a_jc) (mediana po laserach), T_glob = kT₀ / sd(a) po wszystkich (j, c), odsetek par (j, c) w oknie
   cieplnym 0,1 < Φ(a) < 0,9 (|a| < 1,2816), entropia obsadzenia (bity), średnie GELU′, udział okna w Σ_c GELU′² |v_c|² |s_c|²
   (wkład laserów do ‖J‖², bez wyrazów krzyżowych). Wszystkie 806 checkpointów; stany x z pulapka_17_cache.npz (float16),
   w 4 checkpointach kontrola z nowym przejściem modelu.
B. Termometr wirtualny (cząstka Browna w pułapce). Szum δ = s·σ_j·ε, ε ~ N(0, I₁₉₂), σ_j = odchylenie tokenu w LayerNorm
   (s bez wymiaru: szum na kanał względem rozrzutu kanałów tokenu). Jakobian pułapki w temperaturze T_j = (s σ_j)²:
       J_T = E_δ[J(x + δ)]  (dokładnie: gradient pola wygładzonego gaussowsko; lemat Steina: J_T = E[F(x + δ) δᵀ] / T).
   Liczony analitycznie jako średnia po N próbkach, z kontrolą Steina z tych samych próbek. Miary jak w pkt 4.4 opisu układu:
   udział ujemnych wartości własnych S_T, ‖A_T‖/‖S_T‖, ‖S_T‖²/‖J_T‖², ślad/192, ‖J_T‖₂, max Re λ(J_T), ρ(I + J_T);
   nieliniowość: ‖F(x+δ) − F(x) − J₀δ‖ / ‖F(x+δ) − F(x)‖; podgrzanie bramek: sd szumu w pobudzeniu a (s_a) i szerokość
   bramki po uśrednieniu E[Φ(a + η)] = Φ(a / √(1 + s_a²)).
C. Test otwarty 16 (ZASADY): ile wiru tłumaczą pojedyncze lasery. Dla J₀ = Σ_c g′_c v_c s̃_cᵀ (s̃_c = P₁(γ ⊙ w_c)/σ, bez członu
   radialnego): przewidywanie „bez wyrazów krzyżowych” R₁ = √(Σ m_c (1 − κ_c²) / Σ m_c (1 + κ_c²)), m_c = g′_c² |v_c|² |s̃_c|²,
   κ_c = cos(s̃_c, v_c), wobec zmierzonego ‖A₀‖/‖S₀‖.
Uruchom (z katalogu film_krysztal): F:/Python/3.13/python.exe -X utf8 temperatura_pulapki/temperatura_pulapki_s47.py [proba]
→ temperatura_pulapki/TEMPERATURA_PULAPKI_S47.json, log w tym samym katalogu.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

HERE = Path(__file__).resolve().parent
FK = HERE.parent
ROOT = FK.parents[2]
sys.path.insert(0, str(FK))
from lasery_wspolne import dane, oko                                                # noqa: E402

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
LOG = open(HERE / ('temperatura_proba.log' if PROBA else 'temperatura_pulapki_s47.log'), 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a) + f' [{time.time() - t0:.0f} s]'
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
LATENT = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
CACHE = FK / 'pulapka_17_cache.npz'
EP, N, EPS = 1094, 192, 1e-6
KT0 = 1 / 1.702
OKNO = 1.2815516                                                                   # Φ(a) ∈ (0,1; 0,9)
KROKI_B = [0, 1094, 10940, 65640]
S_LISTA = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
NPROB = 64 if PROBA else 256
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f4 = lambda x: float(np.round(float(x), 4))
q = lambda x: dict(med=f4(np.median(x)), p10=f4(np.percentile(x, 10)), p90=f4(np.percentile(x, 90)))

D = dane(0)
kroki, narz = D['kroki'].astype(int), D['narz'].astype(bool)
SIATKA = np.stack(np.divmod(np.arange(600), 30)[::-1], 1).astype(np.float64)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - SIATKA[narz].mean(0), axis=1))])


def wczytaj(k):
    return torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']


def wagi(s):
    g = lambda nm: s[f'bloki.5.{nm}'].double().numpy()
    return dict(W1=g('mlp.fc1.weight'), b1=g('mlp.fc1.bias'), W2=g('mlp.fc2.weight'), b2=g('mlp.fc2.bias'),
                gam=g('norm2.weight'), bet=g('norm2.bias'))


def pobudzenia(x, w):
    u = x - x.mean(1, keepdims=True); sd = np.sqrt((u * u).mean(1) + EPS)
    return (w['gam'] * u / sd[:, None] + w['bet']) @ w['W1'].T + w['b1'], u, sd


def miary_A(a, w, sd):
    """Temperatura reflektorów z pobudzeń a (600 × 768)."""
    Phi = 0.5 * (1 + erf(a / np.sqrt(2)))
    p = np.clip(Phi, 1e-12, 1 - 1e-12)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    sd_c = a.std(0)
    g = dgelu(a)
    sv = np.linalg.norm(w['W2'], axis=0) ** 2                                       # |v_c|²
    st = w['gam'][None, :] * w['W1']; st = st - st.mean(1, keepdims=True)
    ss = (np.linalg.norm(st, axis=1) ** 2)[None, :] / sd[:, None] ** 2               # |s̃_c|² dla tokenu j
    m = g ** 2 * sv[None, :] * ss
    ok = np.abs(a) < OKNO
    jasne = a > OKNO
    return dict(T_laser_med=f4(np.median(KT0 / np.maximum(sd_c, 1e-9))), T_glob=f4(KT0 / a.std()),
                sd_a_glob=f4(a.std()), sd_a_laser_med=f4(np.median(sd_c)), srednie_a=f4(a.mean()),
                okno_cieplne=f4(ok.mean()), zapalone=f4(jasne.mean()), zgaszone=f4((a < -OKNO).mean()),
                entropia_bity=f4(H.mean()), gelu_prim_srednie=f4(g.mean()),
                udzial_okna_w_J2=f4((m * ok).sum() / m.sum()), udzial_zapalonych_w_J2=f4((m * jasne).sum() / m.sum()))


wynik = dict(opis=__doc__.split('\n\n')[0], bieg='MalyViT_35000_od_zera_s47', blok=6, zdjecie='TRAIN 0', token_narzedzia=FOK,
             kT0=f4(KT0), okno_cieplne_a=OKNO, proba=PROBA)

# ---------------------------------------------------------------- A: wszystkie checkpointy
log('A: wczytuję cache', CACHE.name)
X = np.load(CACHE)['X']
assert X.shape[0] == len(kroki), (X.shape, len(kroki))
kroki_A = list(range(0, len(kroki), 40)) + [len(kroki) - 1] if PROBA else list(range(len(kroki)))
A = {}
for i in kroki_A:
    k = int(kroki[i]); w = wagi(wczytaj(k))
    a, u, sd = pobudzenia(X[i].astype(np.float64), w)
    A[str(k)] = dict(epoka=round(k / EP, 3), **miary_A(a, w, sd))
    if i % 50 == 0 or i == len(kroki) - 1:
        log('A', k, A[str(k)])
wynik['A_reflektory'] = A

# ---------------------------------------------------------------- B, C: termometr wirtualny
sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402

xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float() / 255.
net = M.MalyViT().eval(); net.zapisuj_latent(True)
zap = {}
net.bloki[5].norm2.register_forward_hook(lambda m, i, o: zap.__setitem__('x', i[0].detach()))
ZL = np.load(LATENT, mmap_mode='r')
gen = torch.Generator(device=dev).manual_seed(47)
SQ2 = np.sqrt(2.0)


def Ft(xx, W):
    """Pole F dla stanów xx (..., 192) w torch (float64): zwraca F, a, u, sd."""
    u = xx - xx.mean(-1, keepdim=True); sd = torch.sqrt((u * u).mean(-1, keepdim=True) + EPS)
    a = (W['gam'] * u / sd + W['bet']) @ W['W1'].T + W['b1']
    z = 0.5 * a * (1 + torch.erf(a / SQ2))
    return z @ W['W2'].T + W['b2'], a, u, sd


def dg_t(a):
    return 0.5 * (1 + torch.erf(a / SQ2)) + a * torch.exp(-a * a / 2) / np.sqrt(2 * np.pi)


def miary_J(J):
    """J (T, 192, 192) → miary jak w pkt 4.4 opisu (na tokenach); wartości własne w float64 na CPU (na GPU są bardzo wolne)."""
    J = J.double().cpu()
    S, Aa = (J + J.transpose(1, 2)) / 2, (J - J.transpose(1, 2)) / 2
    ev = torch.linalg.eigvalsh(S)
    lam = torch.from_numpy(np.linalg.eigvals(J.numpy()))
    nS, nA, nJ = (torch.linalg.matrix_norm(t) for t in (S, Aa, J))
    return dict(ujemne=(ev < 0).double().mean(1), wir=nA / nS, sym=nS ** 2 / nJ ** 2,
                slad=torch.diagonal(J, dim1=1, dim2=2).sum(1) / N, norma=torch.linalg.matrix_norm(J, 2),
                maxre=lam.real.max(1).values, promien=(1 + lam).abs().max(1).values,
                lmin=ev[:, 0], lmax=ev[:, -1])


B, C = {}, {}
for k in KROKI_B:
    s_ = wczytaj(k); w = wagi(s_)
    net.load_state_dict(s_)
    with torch.no_grad():
        _, lat = net(xin)
    x = zap['x'][0].double().numpy()
    a0, u0, sd0 = pobudzenia(x, w)
    zr = np.asarray(ZL[int(np.searchsorted(kroki, k)), 0]).astype(np.float64)
    r2 = 1 - ((gelu(a0) - zr) ** 2).sum() / ((zr - zr.mean()) ** 2).sum()
    i_c = int(np.searchsorted(kroki, k))
    a_c, _, sd_c = pobudzenia(X[i_c].astype(np.float64), w)
    zgod_cache = float(1 - ((a_c - a0) ** 2).sum() / ((a0 - a0.mean()) ** 2).sum())
    W = {nm: torch.from_numpy(v).to(dev) for nm, v in w.items()}
    xt = torch.from_numpy(x).to(dev)
    F0, a0t, u0t, sd0t = Ft(xt, W)
    Mw = W['W1'] * W['gam'][None, :]                                                # W₁ diag(γ): 768 × 192
    P1 = torch.eye(N, dtype=torch.float64, device=dev) - 1.0 / N
    # J₀ dokładnie (s = 0) — do nieliniowości i do testu C
    g0 = dg_t(a0t)                                                                  # 600 × 768
    J0 = torch.einsum('dc,jc,ce->jde', W['W2'], g0 / sd0t, Mw @ P1) - \
        torch.einsum('dc,jc,je->jde', W['W2'], g0 * (u0t @ Mw.T) / (N * sd0t ** 3), u0t)
    # C: pojedyncze lasery
    st = (Mw @ P1)                                                                  # 768 × 192 (P₁(γ⊙w_c))ᵀ wierszami
    vn = torch.linalg.norm(W['W2'], dim=0); sn = torch.linalg.norm(st, dim=1)
    kap = (st * W['W2'].T).sum(1) / (vn * sn)
    m = (g0 / sd0t) ** 2 * (vn * sn)[None, :] ** 2
    R1 = torch.sqrt((m * (1 - kap ** 2)).sum(1) / (m * (1 + kap ** 2)).sum(1))
    mj0 = miary_J(J0)
    C[str(k)] = dict(epoka=round(k / EP, 2), kappa_med=f4(kap.median().item()),
                     kappa_wazone=f4(((m * kap[None, :]).sum(1) / m.sum(1)).median().item()),
                     wir_przewidywany_bez_wyrazow_krzyzowych=q(R1.cpu().numpy()), wir_zmierzony=q(mj0['wir'].cpu().numpy()),
                     wir_przewidywany_tlo=f4(R1.cpu()[torch.from_numpy(~narz)].median().item()),
                     wir_zmierzony_tlo=f4(mj0['wir'][torch.from_numpy(~narz)].median().item()))
    log('C', k, C[str(k)])
    Bk = dict(epoka=round(k / EP, 2), R2_latent=round(float(r2), 6), zgodnosc_a_cache_R2=round(zgod_cache, 6), poziomy={})
    # część z szumem w float32 (GPU), średnie i miary w float64
    f32 = torch.float32
    W32 = {nm: v.to(f32) for nm, v in W.items()}
    x32, F032, a032, sd032, J032 = xt.to(f32), F0.to(f32), a0t.to(f32), sd0t.to(f32), J0.to(f32)
    Mw32, P132 = Mw.to(f32), P1.to(f32)
    for s in S_LISTA:
        if s == 0:
            J, stein_bl, nielin, s_a, extra = J0, 0.0, 0.0, 0.0, {}
        else:
            acc1 = torch.zeros(600, 768, dtype=f32, device=dev)
            acc2 = torch.zeros(600, 768, N, dtype=f32, device=dev)
            accS = torch.zeros(600, N, N, dtype=f32, device=dev)
            nl_num, nl_den, sa = [], [], []
            for p0 in range(0, NPROB, 32):
                nb = min(32, NPROB - p0)
                eps = torch.randn(600, nb, N, dtype=f32, device=dev, generator=gen)
                d = s * sd032[:, None, :] * eps                                    # (600, nb, 192)
                Fn, an, un, sdn = Ft(x32[:, None, :] + d, W32)
                gn = dg_t(an)
                acc1 += (gn / sdn).sum(1)
                acc2 += torch.einsum('jnc,jne->jce', gn * (un @ Mw32.T) / (N * sdn ** 3), un)
                dF = Fn - F032[:, None, :]
                lin = torch.einsum('jde,jne->jnd', J032, d)
                accS += torch.einsum('jnd,jne->jde', dF - lin, d)                  # Stein z odjętą częścią liniową (zmienna kontrolna)
                nl_num.append(torch.linalg.norm(dF - lin, dim=-1)); nl_den.append(torch.linalg.norm(dF, dim=-1))
                sa.append((an - a032[:, None, :]).std(1))
            acc1 /= NPROB; acc2 /= NPROB
            J = torch.einsum('dc,jc,ce->jde', W32['W2'], acc1, Mw32 @ P132) - torch.einsum('dc,jce->jde', W32['W2'], acc2)
            T_j = (s * sd032[:, 0]) ** 2
            Js = J032 + accS / NPROB / T_j[:, None, None]
            stein_bl = (torch.linalg.matrix_norm(Js - J) / torch.linalg.matrix_norm(J)).median().item()
            nielin = (torch.cat(nl_num, 1) / torch.cat(nl_den, 1).clamp_min(1e-30)).median().item()
            s_a = torch.stack(sa).mean(0).median().item()
            extra = dict(szerokosc_bramki_wzgledna=f4(np.sqrt(1 + s_a ** 2)))
        mj = miary_J(J)
        sel = {'wszystkie': slice(None), 'tlo': torch.from_numpy(~narz), 'narzedzie': torch.from_numpy(narz)}
        rec = {grp: {nm: f4(v[ix].median().item()) for nm, v in mj.items()} for grp, ix in sel.items()}
        rec['token_316'] = {nm: f4(v[FOK].item()) for nm, v in mj.items()}
        rec['p10_p90_wir'] = [f4(torch.quantile(mj['wir'], .1).item()), f4(torch.quantile(mj['wir'], .9).item())]
        rec['p10_p90_ujemne'] = [f4(torch.quantile(mj['ujemne'], .1).item()), f4(torch.quantile(mj['ujemne'], .9).item())]
        rec['frakcja_tokenow_ujemne_ponad_60proc'] = f4((mj['ujemne'] > 0.6).double().mean().item())
        rec.update(blad_stein_wzgledny=f4(stein_bl), nieliniowosc_med=f4(nielin), sd_szumu_w_pobudzeniu_a=f4(s_a), **extra)
        Bk['poziomy'][str(s)] = rec
        log('B', k, 's', s, {kk: rec['wszystkie'][kk] for kk in ('ujemne', 'wir', 'sym', 'slad', 'norma', 'promien')},
            'stein', rec['blad_stein_wzgledny'], 'nielin', rec['nieliniowosc_med'], 's_a', rec['sd_szumu_w_pobudzeniu_a'])
    B[str(k)] = Bk
    wynik['B_termometr'] = B; wynik['C_pojedyncze_lasery'] = C
    json.dump(wynik, open(HERE / ('TEMPERATURA_PROBA.json' if PROBA else 'TEMPERATURA_PULAPKI_S47.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

wynik['B_termometr'] = B; wynik['C_pojedyncze_lasery'] = C
json.dump(wynik, open(HERE / ('TEMPERATURA_PROBA.json' if PROBA else 'TEMPERATURA_PULAPKI_S47.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
log('zapisano')
