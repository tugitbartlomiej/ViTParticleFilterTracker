"""Skąd pochodzi siła (wektor c latentu) patcha i: ścisły rozkład przepływu sygnału w małym ViT, od obrazu przez
łodygę, kolejne bloki i głowy atencji aż do neuronu c ukrytej warstwy MLP ostatniego bloku.

Model (model_snapshot.MalyViT): łodyga konwolucyjna (4 warstwy, krok 16 px) + położenie sin-cos, 6 bloków timm
(pre-LN, 3 głowy, wymiar 192, MLP 768). Latent = GELU(fc1(LN2(x_mid))) ostatniego bloku, więc siła c patcha i to
GELU(przed[i, c]), przed = fc1(LN2(x_mid_5))[i, c].

Rozkład (atrybucja obwodów, Elhage i in. 2021): przy wagach atencji A i skalach LayerNorm sigma ustalonych z tego
przebiegu (wartości z tego obrazu, nie przybliżenie) wszystko między liśćmi a przed[i, c] jest liniowe:
  liście = wyjście łodygi (obraz), położenie sin-cos, wyjścia MLP bloków 0..4 (nieliniowe, więc są źródłem nowego
  sygnału), stałe (biasy). Wtedy przed[i, c] = suma_liści <g_liść, liść> + stałe DOKŁADNIE (sprawdzane: reszta ~1e-5).
Wielkości:
  S_l[k]   = <g_l[k], x_l[k]>: ile sygnału siedzi w tokenie k na wejściu bloku l (mapa 20 x 30 na każdej głębokości);
  F_lh[k,j] = A_lh[k, j] <g_mid_l[k], v_lh[j] W_O,h>: ile głowa h bloku l przeniosła z patcha j do patcha k;
  M_l[k]   = <g_(l+1)[k], MLP_l[k]>: ile nowego sygnału MLP bloku l wstrzyknął w tokenie k.
Atencja tylko przenosi sygnał między tokenami (suma się zachowuje z dokładnością do biasów), MLP go tworzy.
Obraz: wkład łodygi w pikselach metodą Integrated Gradients (Sundararajan i in. 2017; łodyga jest nieliniowa,
więc to jedyne przybliżenie, z kontrolą zupełności: suma pikseli = wkład obrazu − wkład obrazu bazowego).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _ln(x, mod, sig=None):
    """LayerNorm ze skalą sigma podaną (ustaloną) albo liczoną; zwraca (wynik, sigma)."""
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    if sig is None:
        sig = (xc.pow(2).mean(-1, keepdim=True) + mod.eps).sqrt()
    return xc / sig * mod.weight + mod.bias, sig


def _wejscie(net, rgb480, dev):
    """Obraz 480 x 320 uint8 -> znormalizowany tensor (1, 3, 320, 480) jak w MalyViT.forward."""
    import model_snapshot as M
    x = torch.from_numpy(np.ascontiguousarray(rgb480)).permute(2, 0, 1)[None].float().to(dev) / 255
    return (x - M.MEAN.to(dev)) / M.STD.to(dev)


def _lodyga(net, xn):
    f = net.lodyga(xn)
    return f.flatten(2).transpose(1, 2)[0]                                   # [600, 192]


def _przebieg(net, e, poz, stale=None, liscie_mlp=None):
    """Przebieg bloków od osadzenia e + poz. stale: dict A [blok] i sigmy (ustalone); bez: liczone i zwracane.
    liscie_mlp: lista tensorów wyjść MLP bloków 0..4 (liście); bez: liczone z przebiegu."""
    zap = dict(A=[], s1=[], s2=[], x=[], xm=[], mlp=[], przed=None) if stale is None else None
    x = e + poz
    xs, xms, mlps = [], [], []
    for l, b in enumerate(net.bloki):
        at = b.attn
        a, s1 = _ln(x, b.norm1, None if stale is None else stale['s1'][l])
        N, C = a.shape
        H = at.num_heads; d = C // H
        qkv = at.qkv(a).reshape(N, 3, H, d).permute(1, 2, 0, 3)               # [3, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        if stale is None:
            A = ((q * at.scale) @ k.transpose(-2, -1)).softmax(-1)
        else:
            A = stale['A'][l]
        o = (A @ v).transpose(0, 1).reshape(N, C)
        xm = x + at.proj(o)
        h_in, s2 = _ln(xm, b.norm2, None if stale is None else stale['s2'][l])
        przed = b.mlp.fc1(h_in)                                               # [600, 768] przed GELU
        xs.append(x); xms.append(xm)
        if zap is not None:
            zap['A'].append(A.detach()); zap['s1'].append(s1.detach()); zap['s2'].append(s2.detach())
        if l < len(net.bloki) - 1:
            m = b.mlp.fc2(F.gelu(przed)) if liscie_mlp is None else liscie_mlp[l]
            mlps.append(m)
            x = xm + m
        else:
            ost = przed
    if zap is not None:
        zap.update(x=xs, xm=xms, mlp=mlps, przed=ost)
        return zap
    return dict(x=xs, xm=xms, mlp=mlps, przed=ost)


def licz(net, rgb480, tok, kan, kroki_ig=32):
    """Rozkład siły kan patcha tok. Zwraca słownik tablic numpy (opis w docstringu modułu i w kluczach)."""
    dev = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        xn = _wejscie(net, rgb480, dev)
        e0 = _lodyga(net, xn)
        import model_snapshot as M
        poz = M.sincos_2d(20, 30, e0.shape[1], dev).to(e0.dtype)
        z = _przebieg(net, e0, poz)                                           # pełny przebieg: A, sigmy, liście MLP
        latent = F.gelu(z['przed'])
    stale = dict(A=z['A'], s1=z['s1'], s2=z['s2'])
    # liniowy przebieg z ustalonymi A i sigmami: liście z gradientem
    e = e0.clone().requires_grad_(True)
    pz = poz.clone().requires_grad_(True)
    ml = [m.detach().clone().requires_grad_(True) for m in z['mlp']]
    r = _przebieg(net, e, pz, stale, ml)
    for t in r['x'] + r['xm']:
        t.retain_grad()
    przed = r['przed'][tok, kan]
    net.zero_grad(set_to_none=True)
    przed.backward()
    g_x = [t.grad.detach() for t in r['x']]                                   # g na wejściu bloku l
    g_xm = [t.grad.detach() for t in r['xm']]                                 # g po atencji bloku l
    with torch.no_grad():
        wklad_obraz = float((e.grad * e0).sum()); wklad_poz = float((pz.grad * poz).sum())
        M_l = [(g.grad * m).sum(-1) for g, m in zip(ml, z['mlp'])]            # [600] na blok 0..4
        wklad_mlp = [float(m.sum()) for m in M_l]
        przed_v = float(przed.detach())
        stale_v = przed_v - wklad_obraz - wklad_poz - sum(wklad_mlp)          # biasy (b_V, b_O, beta LN, fc1)
        # kontrola: ten sam liniowy przebieg z wyzerowanymi liśćmi daje dokładnie stałe
        r0 = _przebieg(net, torch.zeros_like(e0), torch.zeros_like(poz), stale, [torch.zeros_like(m) for m in z['mlp']])
        stale_kontrola = float(r0['przed'][tok, kan])
        S = [(g * x).sum(-1) for g, x in zip(g_x, r['x'])]                    # sygnał w tokenach na wejściu bloku l
        Sm = [(g * x).sum(-1) for g, x in zip(g_xm, r['xm'])]                 # po atencji bloku l
        # przeniesienia głów: F[l, h, k, j] = A[k, j] <g_xm[k], v_h[j] W_O,h>
        przen = []
        for l, b in enumerate(net.bloki):
            at = b.attn; H = at.num_heads
            a, _ = _ln(r['x'][l].detach(), b.norm1, stale['s1'][l])
            N, C = a.shape; d = C // H
            v = at.qkv(a).reshape(N, 3, H, d)[:, 2]                           # [N, H, d]
            Wo = at.proj.weight                                               # [C, C]: wyjście = o @ Wo^T
            Fl = []
            for h in range(H):
                U = v[:, h] @ Wo[:, h * d:(h + 1) * d].T                       # [N, C]: co patch j wnosi przez głowę h
                G = g_xm[l] @ U.T                                             # [k, j]
                Fl.append(stale['A'][l][h] * G)
            przen.append(torch.stack(Fl))                                     # [H, N, N]
        Fp = torch.stack(przen).cpu().numpy().astype(np.float32)             # [L, H, N, N]
    ig = _ig_lodyga(net, xn, g_x[0], kroki_ig)
    return dict(tok=int(tok), kan=int(kan), przed=przed_v, wartosc=float(latent[tok, kan]),
                latent_patcha=latent[tok].cpu().numpy(),
                wklad_obraz=wklad_obraz, wklad_poz=wklad_poz, wklad_mlp=np.array(wklad_mlp), stale=stale_v,
                stale_kontrola=stale_kontrola,
                S=np.stack([s.cpu().numpy() for s in S]), Sm=np.stack([s.cpu().numpy() for s in Sm]),
                M=np.stack([m.cpu().numpy() for m in M_l]), F=Fp,
                A=np.stack([a.cpu().numpy() for a in stale['A']]).astype(np.float32),
                ig=ig['mapa'], ig_suma=ig['suma'], ig_cel=ig['cel'],
                wklad_obraz_tokeny=(e.grad * e0).sum(-1).detach().cpu().numpy(),
                wklad_poz_tokeny=(pz.grad * poz).sum(-1).detach().cpu().numpy())


def _ig_lodyga(net, xn, g0, kroki):
    """Integrated Gradients wkładu łodygi f(obraz) = sum_j <g0[j], łodyga(obraz)[j]> względem pikseli; baza = obraz
    średni (zero po normalizacji). Zwraca mapę [320, 480] (suma po kanałach), sumę i cel f(x) − f(baza)."""
    g0 = g0.detach()
    f = lambda x: (_lodyga(net, x) * g0).sum()
    baza = torch.zeros_like(xn)
    calk = torch.zeros_like(xn)
    for a in (np.arange(kroki) + .5) / kroki:                                 # reguła punktu środkowego
        x = (baza + float(a) * (xn - baza)).requires_grad_(True)
        gr, = torch.autograd.grad(f(x), x)
        calk += gr
    ig = (xn - baza) * calk / kroki
    with torch.no_grad():
        cel = float(f(xn) - f(baza))
    return dict(mapa=ig[0].sum(0).detach().cpu().numpy(), suma=float(ig.sum()), cel=cel)


def sprawdz_przebieg(net, rgb480):
    """Największa różnica latentu z ręcznego przebiegu i z MalyViT (powinna być ~1e-5)."""
    dev = next(net.parameters()).device
    with torch.no_grad():
        xn = _wejscie(net, rgb480, dev)
        e0 = _lodyga(net, xn)
        import model_snapshot as M
        poz = M.sincos_2d(20, 30, e0.shape[1], dev).to(e0.dtype)
        moj = F.gelu(_przebieg(net, e0, poz)['przed'])
        net.zapisuj_latent(True)
        try:
            x = torch.from_numpy(np.ascontiguousarray(rgb480)).permute(2, 0, 1)[None].float().to(dev) / 255
            _, lat = net(x)
        finally:
            net.zapisuj_latent(False)
    return float((moj - lat[0]).abs().max())
