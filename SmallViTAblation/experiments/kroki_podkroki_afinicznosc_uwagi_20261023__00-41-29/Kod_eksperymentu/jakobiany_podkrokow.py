"""Jakobiany 12 podkroków dla wybranych cząsteczek w wybranych checkpointach biegu s47 (klatka TRAIN 0 historii 7992f2317c64a2e3).

Dla cząsteczki i i podkroku s liczony jest autogradem (torch.func.jacrev):
  * uwaga:  J_pelny[s] = ∂x_i^(s) / ∂X^(s−1)  (192 × 600 × 192): wpływ WSZYSTKICH cząsteczek na przesunięcie cząsteczki i, oraz
            J_zamr    = to samo przy zamrożonych wagach uwagi A (A liczone z odłączonego X): część „barycentryczna”, czyli afiniczna
            w chmurze; reszta J_pelny − J_zamr przechodzi przez softmax (zależność wag od położeń);
            J_wl[s]   = ∂x_i^(s) / ∂x_i^(s−1) (192 × 192): mapa własna cząsteczki przy ustalonych innych (pole w miejscu cząsteczki);
  * MLP:    J_wl[s]   = ∂x_i^(s) / ∂x_i^(s−1) = I + W₂ diag(GELU′) W₁ J_LN (dokładnie funkcja własnego położenia).
Kontrola: f(x_i) odtwarza zapisany stan po podkroku (błąd względny podawany w JSON).

Z map własnych: wartości osobliwe J_wl, wartości własne części symetrycznej (ile kierunków kurczy: < 1), stosunek części antysymetrycznej
do symetrycznej J_wl − I, log|det|; mapa bloku M_b = J_MLP · J_uwaga; iloczyn po głębokości Π_s J_wl (wartości osobliwe po każdym podkroku:
czy złożenie siodeł kurczy, choć pojedyncze siodła nie — test „silnego ogniskowania”); nakładanie podprzestrzeni kurczących kolejnych
bloków tr(P_b P_{b+1}) wobec losowego k_b·k_{b+1}/192.
Z jakobianów pełnych uwagi: udział własny ‖J_ii − I‖² wobec Σ_{j≠i} ‖J_ij‖², efektywna liczba cząsteczek wpływających
(Σ_j w_j)²/Σ_j w_j², w_j = ‖J_ij‖_F², oraz udział części barycentrycznej ‖J_zamr‖²/‖J_pelny‖² i cos między nimi.
Wynik: Wyniki/Dane/jakobiany_podkrokow.npz, jakobiany_podkrokow.json.
Uruchomienie: python jakobiany_podkrokow.py <katalog Wyniki/Dane>
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BAZA = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
HIST = BAZA / 'LaserLatentSpaceApp/cache_uczenia/lasery/7992f2317c64a2e3'
HIST_ETY = BAZA / 'LaserLatentSpaceApp/cache_uczenia/lasery/140854f7a6dbcbe5'
WAGI = BAZA / 'assets/MalyViT_35000_od_zera_s47/wagi'
KOD_MODELU = BAZA / 'experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py'
WY = Path(sys.argv[1])
KROKI = [0, 1094, 5470, 10940, 21880, 43760, 65640]
T, C, H, W = 600, 192, 20, 30
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

spec = importlib.util.spec_from_file_location('maly_vit_35000_model', KOD_MODELU)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
net = M.MalyViT().to(dev).eval().requires_grad_(False)
rgb = np.load(HIST / 'dane.npz')['rgb']
udzial = np.load(HIST_ETY / 'etykieta_obrys.npz')['udzial']
r, c = np.divmod(np.arange(T), W)
dotk = np.where(udzial > 0)[0]
dist_do_narz = np.min(np.sqrt((r[:, None] - r[dotk][None]) ** 2 + (c[:, None] - c[dotk][None]) ** 2), 1)
rng = np.random.default_rng(0)
tlo_kand = np.where((udzial == 0) & (dist_do_narz >= 5) & (r >= 2) & (r <= 17) & (c >= 3) & (c <= 26))[0]
TOKENY = [223, 260, 224, 226] + sorted(rng.choice(tlo_kand, 3, replace=False).tolist())
RODZAJ = ['narzędzie', 'narzędzie', 'narzędzie', 'mieszana 0,35'] + ['tło'] * 3
print('tokeny', TOKENY, RODZAJ, flush=True)

x0 = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
x0 = (x0 - M.MEAN.to(dev)) / M.STD.to(dev)


def stany(krok):
    st = torch.load(WAGI / f'k{krok:05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    with torch.no_grad():
        f = x0
        for modul in net.lodyga:
            f = modul(f)
        _, _, h, w = f.shape
        t = f.flatten(2).transpose(1, 2) + M.sincos_2d(h, w, C, dev)
        X = [t[0]]
        for blk in net.bloki:
            t = t + blk.ls1(blk.attn(blk.norm1(t))); X.append(t[0])
            t = t + blk.ls2(blk.mlp(blk.norm2(t))); X.append(t[0])
    return torch.stack(X)                                                        # (13, 600, 192)


def f_uwaga(blk, Xp, i, zamroz=False):
    at = blk.attn; nh = at.num_heads; hd = C // nh

    def f(Xi):
        xn = blk.norm1(Xi)
        qkv = at.qkv(xn).reshape(T, 3, nh, hd).permute(1, 2, 0, 3)
        q, k, v = at.q_norm(qkv[0]), at.k_norm(qkv[1]), qkv[2]
        if zamroz:
            q, k = q.detach(), k.detach()
        a = ((q[:, i:i + 1] * at.scale) @ k.transpose(-1, -2)).softmax(-1)[:, 0]  # (nh, 600)
        o = torch.einsum('hj,hjd->hd', a, v).reshape(-1)
        return Xi[i] + blk.ls1(at.proj(o))
    return f


def f_uwaga_wlasna(blk, Xp, i):
    g = f_uwaga(blk, Xp, i)
    return lambda xi: g(torch.cat([Xp[:i], xi[None], Xp[i + 1:]], 0))


def f_mlp(blk):
    return lambda xi: xi + blk.ls2(blk.mlp(blk.norm2(xi[None])))[0]


def sym_asym(Jm):
    S, A = 0.5 * (Jm + Jm.T), 0.5 * (Jm - Jm.T)
    return S, A


wyn = dict(krok=KROKI, tokeny=TOKENY, rodzaj=RODZAJ)
nK, nT = len(KROKI), len(TOKENY)
sv_wl = np.zeros((nK, nT, 12, C)); eigS = np.zeros((nK, nT, 12, C)); asym = np.zeros((nK, nT, 12)); logdet = np.zeros((nK, nT, 12))
sv_prod = np.zeros((nK, nT, 12, C)); sv_blok = np.zeros((nK, nT, 6, C)); eigS_blok = np.zeros((nK, nT, 6, C))
nakl = np.zeros((nK, nT, 5, 3)); nakl_mlp = np.zeros((nK, nT, 5, 3))
odch_wl = np.zeros((nK, nT, 12))
uw_self = np.zeros((nK, nT, 6)); uw_inni = np.zeros((nK, nT, 6)); uw_neff = np.zeros((nK, nT, 6)); uw_zamr = np.zeros((nK, nT, 6))
uw_cos = np.zeros((nK, nT, 6)); uw_zamr_inni = np.zeros((nK, nT, 6)); uw_wagi_sas = np.zeros((nK, nT, 6)); uw_wagi_narz = np.zeros((nK, nT, 6))
blad = np.zeros((nK, nT, 12))
sas8 = (np.abs(r[:, None] - r[None]) <= 1) & (np.abs(c[:, None] - c[None]) <= 1); np.fill_diagonal(sas8, False)
narz = udzial >= 0.5
t0 = time.time()
for a_, krok in enumerate(KROKI):
    X = stany(krok).double(); net.double()
    for b_, i in enumerate(TOKENY):
        Jwl = []
        for s in range(12):
            blk = net.bloki[s // 2]; Xp = X[s]
            if s % 2 == 0:
                fw = f_uwaga_wlasna(blk, Xp, i)
                J = torch.func.jacrev(fw)(Xp[i])
                Jf = torch.func.jacrev(f_uwaga(blk, Xp, i))(Xp)                   # (192, 600, 192)
                Jz = torch.func.jacrev(f_uwaga(blk, Xp, i, zamroz=True))(Xp)
                with torch.no_grad():
                    blad[a_, b_, s] = float((fw(Xp[i]) - X[s + 1][i]).norm() / X[s + 1][i].norm())
                    b6 = s // 2
                    wj = (Jf ** 2).sum((0, 2)); wj_self = ((Jf[:, i, :] - torch.eye(C, device=dev, dtype=Jf.dtype)) ** 2).sum()
                    wj_inni = wj.clone(); wj_inni[i] = 0
                    uw_self[a_, b_, b6] = float(wj_self); uw_inni[a_, b_, b6] = float(wj_inni.sum())
                    uw_neff[a_, b_, b6] = float(wj_inni.sum() ** 2 / (wj_inni ** 2).sum())
                    uw_wagi_sas[a_, b_, b6] = float(wj_inni[torch.from_numpy(sas8[i]).to(dev)].sum() / wj_inni.sum())
                    uw_wagi_narz[a_, b_, b6] = float(wj_inni[torch.from_numpy(narz).to(dev)].sum() / wj_inni.sum())
                    Jf_i = Jf.clone(); Jf_i[:, i, :] -= torch.eye(C, device=dev, dtype=Jf.dtype)
                    Jz_i = Jz.clone(); Jz_i[:, i, :] -= torch.eye(C, device=dev, dtype=Jf.dtype)
                    uw_zamr[a_, b_, b6] = float((Jz_i ** 2).sum() / (Jf_i ** 2).sum())
                    uw_cos[a_, b_, b6] = float((Jz_i * Jf_i).sum() / (Jz_i.norm() * Jf_i.norm()))
                    m = torch.ones(T, dtype=torch.bool, device=dev); m[i] = False
                    uw_zamr_inni[a_, b_, b6] = float((Jz[:, m, :] ** 2).sum() / (Jf[:, m, :] ** 2).sum())
            else:
                fm = f_mlp(blk)
                J = torch.func.jacrev(fm)(Xp[i])
                with torch.no_grad():
                    blad[a_, b_, s] = float((fm(Xp[i]) - X[s + 1][i]).norm() / X[s + 1][i].norm())
            Jn = J.detach().cpu().numpy(); Jwl.append(Jn)
            S, A = sym_asym(Jn - np.eye(C))
            sv_wl[a_, b_, s] = np.linalg.svd(Jn, compute_uv=False)
            eigS[a_, b_, s] = np.linalg.eigvalsh(S + np.eye(C))
            asym[a_, b_, s] = np.linalg.norm(A) / max(np.linalg.norm(S), 1e-300)
            logdet[a_, b_, s] = np.linalg.slogdet(Jn)[1]
            odch_wl[a_, b_, s] = np.linalg.norm(Jn - np.eye(C))
        P = np.eye(C); Pb = []; Pm = []
        for s in range(12):
            P = Jwl[s] @ P
            sv_prod[a_, b_, s] = np.linalg.svd(P, compute_uv=False)
        for b6 in range(6):
            Mb = Jwl[2 * b6 + 1] @ Jwl[2 * b6]
            sv_blok[a_, b_, b6] = np.linalg.svd(Mb, compute_uv=False)
            w_, V_ = np.linalg.eigh(0.5 * (Mb + Mb.T)); eigS_blok[a_, b_, b6] = w_
            Vk = V_[:, w_ < 1]; Pb.append(Vk @ Vk.T)
            w2, V2 = np.linalg.eigh(0.5 * (Jwl[2 * b6 + 1] + Jwl[2 * b6 + 1].T)); Vk2 = V2[:, w2 < 1]; Pm.append(Vk2 @ Vk2.T)
        for b6 in range(5):
            for tab, PP in ((nakl, Pb), (nakl_mlp, Pm)):
                k1, k2 = np.trace(PP[b6]), np.trace(PP[b6 + 1])
                tab[a_, b_, b6] = (np.trace(PP[b6] @ PP[b6 + 1]), k1, k2)
        print(f'krok {krok} token {i}: błąd maks {blad[a_, b_].max():.1e}, {time.time() - t0:.0f} s', flush=True)
    net.float()

WY.mkdir(parents=True, exist_ok=True)
np.savez_compressed(WY / 'jakobiany_podkrokow.npz', krok=np.array(KROKI), tokeny=np.array(TOKENY), sv_wl=sv_wl, eigS=eigS, asym=asym,
                    logdet=logdet, sv_prod=sv_prod, sv_blok=sv_blok, eigS_blok=eigS_blok, nakl=nakl, nakl_mlp=nakl_mlp, odch_wl=odch_wl,
                    uw_self=uw_self, uw_inni=uw_inni, uw_neff=uw_neff, uw_zamr=uw_zamr, uw_cos=uw_cos, uw_zamr_inni=uw_zamr_inni,
                    uw_wagi_sas=uw_wagi_sas, uw_wagi_narz=uw_wagi_narz, blad=blad)
f = lambda a: np.round(a, 4).tolist()
wyn.update(blad_maks=float(blad.max()),
           kurczy_wl=f((eigS < 1).mean(3)), kurczy_prod=f((sv_prod < 1).mean(3)), kurczy_blok=f((eigS_blok < 1).mean(3)),
           log_svmax_prod=f(np.log(sv_prod[..., 0])), log_svmin_prod=f(np.log(sv_prod[..., -1])), log_svmax_wl=f(np.log(sv_wl[..., 0])),
           log_svmin_wl=f(np.log(sv_wl[..., -1])), asym=f(asym), logdet=f(logdet), odch_wl=f(odch_wl),
           nakl_blok=f(nakl[..., 0] / np.maximum(np.minimum(nakl[..., 1], nakl[..., 2]), 1)),
           nakl_blok_losowe=f(np.maximum(nakl[..., 1], nakl[..., 2]) / C),
           nakl_mlp=f(nakl_mlp[..., 0] / np.maximum(np.minimum(nakl_mlp[..., 1], nakl_mlp[..., 2]), 1)),
           nakl_mlp_losowe=f(np.maximum(nakl_mlp[..., 1], nakl_mlp[..., 2]) / C),
           uw_self=f(uw_self), uw_inni=f(uw_inni), uw_neff=f(uw_neff), uw_zamr=f(uw_zamr), uw_cos=f(uw_cos), uw_zamr_inni=f(uw_zamr_inni),
           uw_wagi_sas=f(uw_wagi_sas), uw_wagi_narz=f(uw_wagi_narz))
json.dump(wyn, open(WY / 'jakobiany_podkrokow.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('zapisano', time.time() - t0, flush=True)
