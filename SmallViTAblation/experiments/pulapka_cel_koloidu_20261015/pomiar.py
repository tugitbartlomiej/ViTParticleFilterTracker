"""Cel koloidu w uczeniu i fizyka pułapki: pomiar na wszystkich checkpointach historii jednego zdjęcia (bieg s47, 806 ckpt).

Koloid = stan tokenu po 6 blokach (strumień rezydualny, 192 wymiary). Cel tokenu j jest znany z kodu uczenia (model.strata):
głowa(LN(x_j)) ma dać mapę środków heat_j i odległości ramki ltrb_j. Dla każdego checkpointu t (głowa H_t, stany X_t):
  siła błędu g = ∂L/∂x (−g = tam, gdzie błąd chce mieć koloid), na stanie końcowym i przed MLP bloku 6;
  pchnięcie pułapki (MLP bloku 6): cos(pchnięcie, −g_przed) i zmiana błędu tokenu od samego pchnięcia;
  ruch w uczeniu ΔX = X_{t+1} − X_t: cos(ΔX, −g);
  zmiana błędu: sam ruch koloidu L(H_t, X_{t+1}) − L(H_t, X_t), sam ruch celu L(H_{t+1}, X_t) − L(H_t, X_t), oba;
  zapadanie na podprzestrzeń głowy (jak neural regression collapse): udział wariancji LN(X) w 5 głównych składowych
  i w 5-wymiarowej podprzestrzeni odczytu głowy (wiersze średniego jakobianu ∂o/∂LN(x)); losowo 5/192 = 2,6 %.
Zapis: outputs/pulapka_cel_koloidu_20261015/pomiar.npz (tablice (K, 600) i statystyki; stany i siły tokenów w ramce do analizy opóźnienia i dyfuzji).
Uruchomienie: python pomiar.py <katalog app> <katalog historii>
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

APP, KAT = sys.argv[1], sys.argv[2]
WY = Path(__file__).resolve().parents[2] / 'outputs' / 'pulapka_cel_koloidu_20261015'; WY.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, APP)
import lasery_dane as LD

dev = torch.device('cuda')
M = LD.modul_modelu()
h = LD.Historia(KAT)
net = M.MalyViT().to(dev).eval()
for p_ in net.parameters():
    p_.requires_grad_(False)
heat, ltrb, poz = M.cele([[[float(v) for v in h.gt]]], 20, 30)      # jedno zdjęcie, jedna ramka xyxy
heat, ltrb, poz = heat.view(600).to(dev), ltrb.view(600, 4).to(dev), poz.view(600).to(dev)
pos_m = heat.eq(1).float(); n_pos = pos_m.sum().clamp(min=1); mb = poz > 0
WYBRANE = np.union1d(np.flatnonzero(mb.cpu().numpy()), [226])        # tokeny w ramce i cząsteczka 226


def strata_tokenow(o):
    """o (600, 5) → wkład każdego tokenu do straty (suma = strata całego zdjęcia, jak model.strata)."""
    p = o[:, 0].sigmoid().clamp(1e-4, 1 - 1e-4)
    l = (-(torch.log(p) * (1 - p) ** 2 * pos_m) - torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * (1 - pos_m)) / n_pos
    pr = o[:, 1:].exp()
    gi = M._giou(pr[mb], ltrb[mb])
    lb = torch.zeros_like(l); lb[mb] = 2 * (1 - gi) * poz[mb] / poz[mb].sum()
    return l + lb


def glowa(Hp, X):
    nw, nb, w0, b0, w2, b2 = Hp
    F_ = torch.nn.functional
    return F_.linear(F_.gelu(F_.linear(F_.layer_norm(X, (X.shape[1],), nw, nb, 1e-6), w0, b0)), w2, b2)


x_img = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
f_in = (x_img - M.MEAN.to(dev)) / M.STD.to(dev)
K = h.K
XF = np.zeros((K, 600, 192), np.float16); GF = np.zeros((K, 600, 192), np.float16); HP = []
T = {n: np.zeros((K, 600), np.float32) for n in ('cos_push', 'dL_push', 'L_tok', 'push_n', 'g_n')}
nrc_pca5 = np.zeros(K, np.float32); nrc_glowa = np.zeros(K, np.float32); nrc_226 = np.zeros(K, np.float32)
t0 = time.time()
for i in range(K):
    st = torch.load(Path(h.wagi) / f'k{int(h.kroki[i]):05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    with torch.no_grad():
        f = net.lodyga(f_in); B, C, hh, ww = f.shape
        t = f.flatten(2).transpose(1, 2) + M.sincos_2d(hh, ww, C, dev)
        for blk in net.bloki[:-1]:
            t = blk(t)
        blk = net.bloki[-1]
        t = t + blk.ls1(blk.attn(blk.norm1(t)))
    Hp = [x.detach().clone() for x in (net.norm.weight, net.norm.bias, net.glowa[0].weight, net.glowa[0].bias,
                                         net.glowa[2].weight, net.glowa[2].bias)]
    HP.append(Hp)
    Xp = t[0].clone().requires_grad_(True)
    push = blk.ls2(blk.mlp(blk.norm2(Xp)))
    Xf = Xp + push; Xf.retain_grad()
    lt = strata_tokenow(glowa(Hp, Xf)); lt.sum().backward()
    gp, gf = Xp.grad, Xf.grad
    with torch.no_grad():
        lt0 = strata_tokenow(glowa(Hp, Xp))
        T['cos_push'][i] = torch.nn.functional.cosine_similarity(push, -gp, dim=1).cpu().numpy()
        T['dL_push'][i] = (lt - lt0).cpu().numpy(); T['L_tok'][i] = lt.cpu().numpy()
        T['push_n'][i] = push.norm(dim=1).cpu().numpy(); T['g_n'][i] = gf.norm(dim=1).cpu().numpy()
        XF[i] = Xf.cpu().numpy().astype(np.float16); GF[i] = gf.cpu().numpy().astype(np.float16)
        # zapadanie na podprzestrzeń głowy: Y = LN(X) z afinicznym (wejście głowy)
        Y = torch.nn.functional.layer_norm(Xf, (192,), Hp[0], Hp[1], 1e-6); Yc = Y - Y.mean(0)
        s = torch.linalg.svdvals(Yc) ** 2; nrc_pca5[i] = float(s[:5].sum() / s.sum())
    with torch.enable_grad():                                            # średni jakobian głowy ∂o/∂y (5 × 192) po tokenach
        Yv = Y.detach().clone().requires_grad_(True)
        F_ = torch.nn.functional
        o = F_.linear(F_.gelu(F_.linear(Yv, Hp[2], Hp[3])), Hp[4], Hp[5])
        Jm = torch.stack([torch.autograd.grad(o[:, c].sum(), Yv, retain_graph=True)[0].mean(0) for c in range(5)])
    with torch.no_grad():
        Q, _ = torch.linalg.qr(Jm.T)                                     # (192, 5) baza podprzestrzeni odczytu
        nrc_glowa[i] = float((Yc @ Q).pow(2).sum() / Yc.pow(2).sum())
        d226 = Yc[226]; nrc_226[i] = float((d226 @ Q).pow(2).sum() / d226.pow(2).sum().clamp_min(1e-30))
    if i % 100 == 0:
        print(f'{i}/{K} {time.time() - t0:.0f} s', flush=True)

D = {n: np.zeros((K - 1, 600), np.float32) for n in ('cos_ucz', 'dL_koloid', 'dL_cel', 'dL_razem')}
with torch.no_grad():
    for i in range(K - 1):
        Xa, Xb = torch.from_numpy(XF[i].astype(np.float32)).to(dev), torch.from_numpy(XF[i + 1].astype(np.float32)).to(dev)
        Laa = strata_tokenow(glowa(HP[i], Xa)); Lab = strata_tokenow(glowa(HP[i], Xb)); Lba = strata_tokenow(glowa(HP[i + 1], Xa))
        Lbb = strata_tokenow(glowa(HP[i + 1], Xb))
        D['dL_koloid'][i] = (Lab - Laa).cpu().numpy(); D['dL_cel'][i] = (Lba - Laa).cpu().numpy(); D['dL_razem'][i] = (Lbb - Laa).cpu().numpy()
        D['cos_ucz'][i] = torch.nn.functional.cosine_similarity(Xb - Xa, -torch.from_numpy(GF[i].astype(np.float32)).to(dev), dim=1).cpu().numpy()
XF32 = XF.astype(np.float32)
rms = np.sqrt((XF32 ** 2).mean((1, 2)))
odl_kon = np.linalg.norm(XF32 - XF32[-1:], axis=2) / rms[:, None]
np.savez_compressed(WY / 'pomiar.npz', kroki=h.kroki, heat=heat.cpu().numpy(), poz=poz.cpu().numpy(), rms=rms, odl_kon=odl_kon,
                    nrc_pca5=nrc_pca5, nrc_glowa=nrc_glowa, nrc_226=nrc_226, wybrane=WYBRANE,
                    X_wyb=XF32[:, WYBRANE], G_wyb=GF[:, WYBRANE].astype(np.float32), **T, **D)
print('zapisano', WY / 'pomiar.npz', f'{time.time() - t0:.0f} s')
