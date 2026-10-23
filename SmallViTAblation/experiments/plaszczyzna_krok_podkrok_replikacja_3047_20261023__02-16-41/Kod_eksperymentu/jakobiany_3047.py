"""Jakobiany 12 podkroków dla seedu 3047: 10 checkpointów × 10 klatek TEST × 6 tokenów (3 narzędzia o największym udziale, 3 tła losowe),
mapy własne (192 × 192, autograd float64, inne cząsteczki ustalone); pełny jakobian uwagi (192 × 600 × 192) i wersja z zamrożonymi A
dla pierwszego tokenu narzędzia i pierwszego tokenu tła. Miary jak w jakobiany_podkrokow.py eksperymentu
kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29. Wynik: <WY>/jakobiany_3047.npz, jakobiany_3047.json.
Uruchomienie: python jakobiany_3047.py <katalog Wyniki/Dane> (wymaga klatki.json z pomiar_3047.py)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wspolne as WS  # noqa: E402
from wspolne import C, H, T, W, dev  # noqa: E402

WY = Path(sys.argv[1])
KROKI = [0, 300, 1000, 3000, 10000, 24000, 65640, 109375, 218750, 437500]
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
KL = json.loads((WY / 'klatki.json').read_text(encoding='utf-8'))
r, c = np.divmod(np.arange(T), W)
sas8 = (np.abs(r[:, None] - r[None]) <= 1) & (np.abs(c[:, None] - c[None]) <= 1); np.fill_diagonal(sas8, False)


def tokeny_klatki(udzial, ziarno):
    narz = np.argsort(-udzial)[:3]
    dotk = np.where(udzial > 0)[0]
    dist = np.min(np.sqrt((r[:, None] - r[dotk][None]) ** 2 + (c[:, None] - c[dotk][None]) ** 2), 1)
    kand = np.where((udzial == 0) & (dist >= 5) & (r >= 2) & (r <= 17) & (c >= 3) & (c <= 26))[0]
    tlo = np.random.default_rng(ziarno).choice(kand, 3, replace=False)
    return [int(v) for v in narz] + sorted(int(v) for v in tlo)


def f_uwaga(blk, i, zamroz=False):
    at = blk.attn; nh = at.num_heads; hd = C // nh

    def f(Xi):
        xn = blk.norm1(Xi)
        qkv = at.qkv(xn).reshape(T, 3, nh, hd).permute(1, 2, 0, 3)
        q, k, v = at.q_norm(qkv[0]), at.k_norm(qkv[1]), qkv[2]
        if zamroz:
            q, k = q.detach(), k.detach()
        a = ((q[:, i:i + 1] * at.scale) @ k.transpose(-1, -2)).softmax(-1)[:, 0]
        o = torch.einsum('hj,hjd->hd', a, v).reshape(-1)
        return Xi[i] + blk.ls1(at.proj(o))
    return f


def f_uwaga_wlasna(blk, Xp, i):
    g = f_uwaga(blk, i)
    return lambda xi: g(torch.cat([Xp[:i], xi[None], Xp[i + 1:]], 0))


def f_mlp(blk):
    return lambda xi: xi + blk.ls2(blk.mlp(blk.norm2(xi[None])))[0]


nK, nF, nT = len(KROKI), len(KL), 6
Z = lambda *s: np.full(s, np.nan)
sv_wl, eigS, sv_prod = Z(nK, nF, nT, 12, C), Z(nK, nF, nT, 12, C), Z(nK, nF, nT, 12, C)
eigS_blok, sv_blok = Z(nK, nF, nT, 6, C), Z(nK, nF, nT, 6, C)
asym, logdet, odch, blad = Z(nK, nF, nT, 12), Z(nK, nF, nT, 12), Z(nK, nF, nT, 12), Z(nK, nF, nT, 12)
nakl, nakl_mlp = Z(nK, nF, nT, 5, 3), Z(nK, nF, nT, 5, 3)
uw_self, uw_inni, uw_neff, uw_zamr, uw_zamr_inni, uw_sas, uw_narz = (Z(nK, nF, 2, 6) for _ in range(7))
TOK = np.zeros((nF, nT), int)
net = WS.siec().double(); t0 = time.time()
OBR = []
for f_, k_ in enumerate(KL):
    rgb, sha = WS.obraz(k_['film'], k_['klatka']); assert sha == k_['sha256_jpg']
    u = WS.etykieta_cystotomu(k_['film'], k_['klatka'])['udzial']
    OBR.append((rgb, u)); TOK[f_] = tokeny_klatki(u, k_['klatka'])
for a_, krok in enumerate(KROKI):
    WS.wczytaj(net, krok)
    for f_, (rgb, u) in enumerate(OBR):
        X = WS.przejscie(net, rgb)['X']                                       # (13, 600, 192) float64
        narz = torch.from_numpy(u >= WS.PROG_NARZ).to(dev)
        for b_, i in enumerate(TOK[f_]):
            Jwl = []
            for s in range(12):
                blk = net.bloki[s // 2]; Xp = X[s]
                if s % 2 == 0:
                    fw = f_uwaga_wlasna(blk, Xp, i); J = torch.func.jacrev(fw)(Xp[i])
                    with torch.no_grad():
                        blad[a_, f_, b_, s] = float((fw(Xp[i]) - X[s + 1][i]).norm() / X[s + 1][i].norm())
                    if b_ in (0, 3):
                        g_ = 0 if b_ == 0 else 1; b6 = s // 2
                        Jf = torch.func.jacrev(f_uwaga(blk, i))(Xp); Jz = torch.func.jacrev(f_uwaga(blk, i, zamroz=True))(Xp)
                        with torch.no_grad():
                            I_ = torch.eye(C, device=dev, dtype=Jf.dtype)
                            wj = (Jf ** 2).sum((0, 2)); wj_inni = wj.clone(); wj_inni[i] = 0
                            uw_self[a_, f_, g_, b6] = float(((Jf[:, i, :] - I_) ** 2).sum()); uw_inni[a_, f_, g_, b6] = float(wj_inni.sum())
                            uw_neff[a_, f_, g_, b6] = float(wj_inni.sum() ** 2 / (wj_inni ** 2).sum())
                            uw_sas[a_, f_, g_, b6] = float(wj_inni[torch.from_numpy(sas8[i]).to(dev)].sum() / wj_inni.sum())
                            uw_narz[a_, f_, g_, b6] = float(wj_inni[narz].sum() / wj_inni.sum())
                            Jf_i = Jf.clone(); Jf_i[:, i, :] -= I_; Jz_i = Jz.clone(); Jz_i[:, i, :] -= I_
                            uw_zamr[a_, f_, g_, b6] = float((Jz_i ** 2).sum() / (Jf_i ** 2).sum())
                            m = torch.ones(T, dtype=torch.bool, device=dev); m[i] = False
                            uw_zamr_inni[a_, f_, g_, b6] = float((Jz[:, m, :] ** 2).sum() / (Jf[:, m, :] ** 2).sum())
                        del Jf, Jz
                else:
                    fm = f_mlp(blk); J = torch.func.jacrev(fm)(Xp[i])
                    with torch.no_grad():
                        blad[a_, f_, b_, s] = float((fm(Xp[i]) - X[s + 1][i]).norm() / X[s + 1][i].norm())
                Jn = J.detach().cpu().numpy(); Jwl.append(Jn)
                S, A_ = 0.5 * (Jn + Jn.T) - np.eye(C), 0.5 * (Jn - Jn.T)
                sv_wl[a_, f_, b_, s] = np.linalg.svd(Jn, compute_uv=False); eigS[a_, f_, b_, s] = np.linalg.eigvalsh(S + np.eye(C))
                asym[a_, f_, b_, s] = np.linalg.norm(A_) / max(np.linalg.norm(S), 1e-300)
                logdet[a_, f_, b_, s] = np.linalg.slogdet(Jn)[1]; odch[a_, f_, b_, s] = np.linalg.norm(Jn - np.eye(C))
            P = np.eye(C); Pb, Pm = [], []
            for s in range(12):
                P = Jwl[s] @ P; sv_prod[a_, f_, b_, s] = np.linalg.svd(P, compute_uv=False)
            for b6 in range(6):
                Mb = Jwl[2 * b6 + 1] @ Jwl[2 * b6]; sv_blok[a_, f_, b_, b6] = np.linalg.svd(Mb, compute_uv=False)
                w_, V_ = np.linalg.eigh(0.5 * (Mb + Mb.T)); eigS_blok[a_, f_, b_, b6] = w_; Vk = V_[:, w_ < 1]; Pb.append(Vk @ Vk.T)
                w2, V2 = np.linalg.eigh(0.5 * (Jwl[2 * b6 + 1] + Jwl[2 * b6 + 1].T)); Vk2 = V2[:, w2 < 1]; Pm.append(Vk2 @ Vk2.T)
            for b6 in range(5):
                for tab, PP in ((nakl, Pb), (nakl_mlp, Pm)):
                    tab[a_, f_, b_, b6] = (np.trace(PP[b6] @ PP[b6 + 1]), np.trace(PP[b6]), np.trace(PP[b6 + 1]))
        print(f'krok {krok} klatka {f_ + 1}/{nF}: błąd maks {np.nanmax(blad[a_, f_]):.1e}, {time.time() - t0:.0f} s', flush=True)
np.savez_compressed(WY / 'jakobiany_3047.npz', krok=np.array(KROKI), tokeny=TOK, sv_wl=sv_wl, eigS=eigS, sv_prod=sv_prod, eigS_blok=eigS_blok,
                    sv_blok=sv_blok, asym=asym, logdet=logdet, odch=odch, blad=blad, nakl=nakl, nakl_mlp=nakl_mlp, uw_self=uw_self,
                    uw_inni=uw_inni, uw_neff=uw_neff, uw_zamr=uw_zamr, uw_zamr_inni=uw_zamr_inni, uw_sas=uw_sas, uw_narz=uw_narz)
f4 = lambda a: np.round(np.nanmean(a, axis=1), 4).tolist()                     # średnia po klatkach
wyn = dict(krok=KROKI, tokeny=TOK.tolist(), blad_maks=float(np.nanmax(blad)),
           kurczy_wl=f4((eigS < 1).mean(4)), kurczy_prod=f4((sv_prod < 1).mean(4)), kurczy_blok=f4((eigS_blok < 1).mean(4)),
           log_svmax_prod=f4(np.log(sv_prod[..., 0])), log_svmin_prod=f4(np.log(sv_prod[..., -1])), logdet=f4(logdet), logdet_suma=f4(logdet.sum(3)),
           asym=f4(asym), odch=f4(odch),
           nakl_blok=f4(nakl[..., 0] / np.maximum(np.minimum(nakl[..., 1], nakl[..., 2]), 1)), nakl_blok_losowe=f4(np.maximum(nakl[..., 1], nakl[..., 2]) / C),
           uw_self=f4(uw_self), uw_inni=f4(uw_inni), uw_neff=f4(uw_neff), uw_zamr=f4(uw_zamr), uw_zamr_inni=f4(uw_zamr_inni), uw_sas=f4(uw_sas), uw_narz=f4(uw_narz))
json.dump(wyn, open(WY / 'jakobiany_3047.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('zapisano', time.time() - t0, flush=True)
