"""Siła tożsamości narzędzia (PLAN.md, wersja 3).

Wektor tożsamości: LDA ze ściąganiem kowariancji Ledoit-Wolf (jak sklearn LinearDiscriminantAnalysis(solver='lsqr',
shrinkage='auto'), `maly_vit_wektor_tozsamosci/wektor_tozsamosci_21ep.py`) na latentach sond z końca rozgrzewki:
narzędzie (180 sond) kontra tło wszystkich 192 sond i obszar narzędzia w `empty_black`, po standaryzacji średnią i sd
negatywów. Na Edenie nie ma sklearn, więc numpy; zgodność z sklearn sprawdza `../test_lda_lokalnie.py`.
Siła tożsamości sondy = ((latent narzędzia − mu) / sd) · w; wektor stały od końca rozgrzewki do końca biegu.
Prognoza: Koopman i Langevin z `prognoza.py` na historii siły (bloki po 50 kroków); model wybierany testem wstecznym.
FastGAN: inwersja kotwicy, potem sterowanie samym wejściem z do siły startu + przewidziana zmiana
(`fastgan_odwrotnosc_mlp/wzdluz_tozsamosci.py`, wariant Z_1x).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import generuj as GN
import prognoza as PR

MODELE = ('koopman', 'langevin')
S_VIT, SIATKA, K_ZAB = 176, 11, 4


# ---------------------------------------------------------------- wektor tożsamości
def _ledoit_wolf(X):
    """Kowariancja ściągnięta Ledoit-Wolf (sklearn.covariance.ledoit_wolf, dane centrowane tutaj)."""
    X = X - X.mean(0)
    n, p = X.shape
    X2 = X ** 2
    tr = X2.sum(0) / n
    mu = tr.sum() / p
    beta_ = (X2.T @ X2).sum()
    delta_ = ((X.T @ X) ** 2).sum() / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * tr.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    a = 0.0 if beta == 0 else beta / delta
    C = (1 - a) * (X.T @ X) / n
    C.flat[::p + 1] += a * mu
    return C


def _cov(X):
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return sd[:, None] * _ledoit_wolf(X / sd) * sd[None]


def lda(Z, y):
    pri = [(y == k).mean() for k in (0, 1)]
    M = np.stack([Z[y == k].mean(0) for k in (0, 1)])
    C = sum(pri[k] * _cov(Z[y == k]) for k in (0, 1))
    coef = np.linalg.lstsq(C, M.T, rcond=None)[0].T
    return coef[1] - coef[0]


def wektor(Lk, obecn):
    """Lk: (192, 2, 768) latent sond (średnia kilku zapisów). Zwraca dict(mu, sd, w) — w o długości 1."""
    L = np.asarray(Lk, np.float64)
    X, y = [], []
    for i in range(len(L)):
        X.append(L[i, 0]); y.append(1 if obecn[i] else 0)            # empty_black: obszar narzędzia bez narzędzia
        X.append(L[i, 1]); y.append(0)                                 # tło sondy
    X, y = np.array(X), np.array(y)
    neg = X[y == 0]
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    Z = (X - mu) / sd
    w = lda(Z, y)
    w = w / np.linalg.norm(w)
    r = Z @ w
    return dict(mu=mu, sd=sd, w=w, sila_narzedzia=float(r[y == 1].mean()), sila_negatywow=float(r[y == 0].mean()),
                sd_negatywow=float(r[y == 0].std()))


def sila(nar, wek):
    """nar: (..., 768) latent narzędzia → siła tożsamości (...)."""
    return ((np.asarray(nar, np.float32) - wek['mu'].astype(np.float32)) / wek['sd'].astype(np.float32)) @ wek['w'].astype(np.float32)


# ---------------------------------------------------------------- prognoza siły
def _rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def prognozy(Sb, H):
    """Sb: (Tb, P, 1) historia siły w blokach. Prognoza H bloków naprzód obu modeli, test wsteczny (dopasowanie bez
    ostatnich H bloków) i model wybrany: ten o mniejszym błędzie testu wstecznego względem braku zmiany."""
    Sb = np.asarray(Sb, np.float64)
    tw = {}
    if len(Sb) > H + PR.L_PAM + 4:
        hist = Sb[:-H]
        e0 = _rmse(hist[-1] - Sb[-1])
        for m in MODELE:
            p = PR.rollout(hist, PR.dopasuj(hist, m), H, m)[-1]
            tw[m] = _rmse(p - Sb[-1]) / max(e0, 1e-12)
    if len(Sb) >= PR.L_PAM + 4:
        prog = {m: PR.rollout(Sb, PR.dopasuj(Sb, m), H, m)[-1, :, 0] for m in MODELE}
    else:                                                               # za krótka historia: brak zmiany
        prog = {m: Sb[-1, :, 0].copy() for m in MODELE}
    return dict(prognozy=prog, teraz=Sb[-1, :, 0].copy(), test_wsteczny=tw,
                model=min(tw, key=tw.get) if tw else 'langevin', punktow=int(len(Sb)))


def sprawdz(pp, real, kotw):
    """Prognoza z poprzedniego obrotu (pp z `prognozy`) kontra siła, która się naprawdę pojawiła (real, (P,))."""
    teraz = pp['teraz']
    d = real - teraz
    e0, e0k = _rmse(d), _rmse(d[kotw])
    duze = np.abs(d) >= np.median(np.abs(d))
    out = dict(brak_zmiany=e0, brak_zmiany_kotwice=e0k, zmiana_srednia=float(d.mean()), model_wybrany=pp['model'])
    for m, p in pp['prognozy'].items():
        dp = p - teraz
        out[m] = dict(sondy=_rmse(p - real) / max(e0, 1e-12), kotwice=_rmse(p[kotw] - real[kotw]) / max(e0k, 1e-12),
                      znak=float((np.sign(dp[duze]) == np.sign(d[duze])).mean()),
                      znak_kotwice=float((np.sign(dp[kotw]) == np.sign(d[kotw])).mean()))
    out['wybrany'] = out[pp['model']]
    return out


# ---------------------------------------------------------------- FastGAN wzdłuż wektora tożsamości
def tokeny(rgba):
    """Binarne wagi tokenów narzędzia (B,121) z maski obrazu, jak w sondach (dane.sondy)."""
    m = F.interpolate(rgba[:, 3:4].detach(), size=(S_VIT, S_VIT), mode='nearest').round()
    return (F.adaptive_avg_pool2d(m, (SIATKA, SIATKA)).flatten(1) >= .25).float()


def latent_nar(net, rgba, tok):
    x = F.interpolate(rgba[:, :3], size=(S_VIT, S_VIT), mode='area')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, lat = net(x)
    lat = lat.float()
    return (lat * tok[..., None]).sum(1) / tok.sum(1, keepdim=True).clamp(min=1e-3)


def zaburz(rgba, g):
    """K_ZAB zaburzonych kopii RGB (jasność/kontrast ±15 %, szum σ 0,02, lekkie rozmycie); maska bez zmian."""
    x = rgba[:, :3].repeat(K_ZAB, 1, 1, 1)
    dev = x.device
    a = torch.empty(x.shape[0], 2, device=dev).uniform_(.85, 1.15, generator=g)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * a[:, :1, None, None] + m * a[:, 1:, None, None]
    x = x + .02 * torch.randn(x.shape, device=dev, generator=g)
    k = torch.tensor([1., 2., 1.], device=dev); k = (k[:, None] * k[None]) / 16
    x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), k.expand(3, 1, 3, 3), groups=3)
    return torch.cat([x.clamp(0, 1), rgba[:, 3:4].repeat(K_ZAB, 1, 1, 1)], 1)


def _s_t(raw, wt):
    x = (raw - wt['mu']) / wt['sd']
    return x @ wt['w'], x


def _sila_odporna(net, o, tok, wt, seed=123):
    g = torch.Generator(device=o.device).manual_seed(seed)
    raw = latent_nar(net, zaburz(o, g), tok.repeat(K_ZAB, 1)).view(K_ZAB, o.shape[0], -1).mean(0)
    return _s_t(raw, wt)[0]


def _inwersja(G, img, msk, seed, kroki=160, starty=4):
    """z, dla którego G(z) odtwarza kotwice (L1 obrazu + L1 maski); kilka startów, najlepszy osobno dla każdej."""
    B, dev = img.shape[0], img.device
    najl = torch.full((B,), 1e9, device=dev); z_best = None
    for r in range(starty):
        z = torch.randn((B, 256), generator=torch.Generator().manual_seed(1000 * seed + r)).to(dev).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=.05)
        for it in range(kroki):
            o = GN.wyjscie(G, z, 42)
            L = (o[:, :3] - img).abs().mean((1, 2, 3)) + (o[:, 3:4] - msk).abs().mean((1, 2, 3))
            opt.zero_grad(); L.sum().backward(); opt.step()
            if it == 99:
                for gr in opt.param_groups:
                    gr['lr'] = .01
        with torch.no_grad():
            o = GN.wyjscie(G, z, 42)
            L = (o[:, :3] - img).abs().mean((1, 2, 3)) + (o[:, 3:4] - msk).abs().mean((1, 2, 3))
            lep = L < najl
            najl = torch.where(lep, L, najl)
            z_best = z.detach().clone() if z_best is None else torch.where(lep[:, None], z.detach(), z_best)
    return z_best, najl


def _steruj(net, G, z0, tok, wt, s_cel, x0, skala, kroki=300, lr=.01, bok=1.0):
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    najl = torch.full((z0.shape[0],), 1e9, device=z0.device); z_best = z0.clone()
    for _ in range(kroki):
        o = GN.wyjscie(G, z, 42)
        s, x = _s_t(latent_nar(net, o, tok), wt)
        dx = x - x0
        prost = dx - (dx @ wt['w'])[:, None] * wt['w'][None]           # zmiana latentu poza kierunkiem tożsamości
        Ls = ((s - s_cel) / skala) ** 2
        L = Ls + bok * (prost ** 2).mean(1) / skala ** 2 / 50
        with torch.no_grad():
            lep = Ls < najl
            najl = torch.where(lep, Ls, najl)
            z_best = torch.where(lep[:, None], z.detach(), z_best)
        opt.zero_grad(); L.sum().backward(); opt.step()
    return z_best


def generuj(u, G, kotwice, ds, wek, out, seed, sterowac=True):
    """kotwice: lista dict(obraz256 (1,3,256,256), maska256 (1,1,256,256)); ds: przewidziana zmiana siły każdej kotwicy.
    sterowac=True: obraz o sile startu + ds (ramię `tozsamosc`); False: sam obraz startowy z inwersji (`current`).
    Zapis w out/kotwica_XX: kotwica.png, start.png (inwersja), target.png (obraz do wyszukiwania) z maskami.
    Zwraca listę raportów i ścieżki obrazów do wyszukiwania (target.png, start.png)."""
    dev = u.dev
    net = u.net
    img = torch.cat([k['obraz256'] for k in kotwice]).to(dev)
    msk = torch.cat([k['maska256'] for k in kotwice]).to(dev)
    wt = {k: torch.tensor(wek[k], dtype=torch.float32, device=dev) for k in ('mu', 'sd', 'w')}
    ds_t = torch.tensor(np.asarray(ds, np.float64), dtype=torch.float32, device=dev)
    wymagane = [p.requires_grad for p in net.parameters()]
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval(); net.zapisuj_latent(True)
    try:
        z_inw, l_inw = _inwersja(G, img, msk, seed)
        with torch.no_grad():
            o0 = GN.wyjscie(G, z_inw, 42)
            tok = tokeny(o0)
            s0, x0 = _s_t(latent_nar(net, o0, tok), wt)
            s0r = _sila_odporna(net, o0, tok, wt)
        if sterowac:
            z = _steruj(net, G, z_inw, tok, wt, s0 + ds_t, x0, ds_t.abs().clamp(min=.02))
            with torch.no_grad():
                o = GN.wyjscie(G, z, 42)
        else:
            o = o0
        with torch.no_grad():
            s1, x1 = _s_t(latent_nar(net, o, tok), wt)
            s1r = _sila_odporna(net, o, tok, wt)
            dx = x1 - x0
            prost = dx - (dx @ wt['w'])[:, None] * wt['w'][None]
    finally:
        net.zapisuj_latent(False)
        for p, w_ in zip(net.parameters(), wymagane):
            p.requires_grad_(w_)
    out = Path(out)
    rap, sciezki = [], []
    for b in range(len(kotwice)):
        kd = out / f'kotwica_{b:02d}'; kd.mkdir(parents=True, exist_ok=True)
        GN._zapisz(kd / 'kotwica.png', img[b:b + 1]); GN._zapisz(kd / 'start.png', o0[b:b + 1, :3])
        GN._zapisz(kd / 'start_mask.png', o0[b:b + 1, 3:4]); GN._zapisz(kd / 'target.png', o[b:b + 1, :3])
        GN._zapisz(kd / 'target_mask.png', o[b:b + 1, 3:4])
        d = float(ds_t[b])
        r = dict(inwersja_l1=float(l_inw[b]), tokenow=int(tok[b].sum()), sila_start=float(s0[b]),
                 sila_start_odporna=float(s0r[b]), przewidziana_zmiana=d, sterowane=sterowac,
                 zmiana_osiagnieta=float(s1[b] - s0[b]), zmiana_osiagnieta_odporna=float(s1r[b] - s0r[b]),
                 zmiana_w_bok=float(prost[b].norm()), zmiana_obrazu_l1=float((o[b, :3] - o0[b, :3]).abs().mean()))
        if sterowac and abs(d) > 1e-6:
            r['czesc_zmiany'] = r['zmiana_osiagnieta'] / d
            r['czesc_zmiany_odporna'] = r['zmiana_osiagnieta_odporna'] / d
        (kd / 'RAPORT.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
        rap.append(r); sciezki.append((kd / 'target.png', kd / 'start.png'))
    return rap, sciezki
