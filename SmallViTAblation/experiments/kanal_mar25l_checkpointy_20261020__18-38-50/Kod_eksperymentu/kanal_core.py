"""Rdzeń testu kanału Mar25l (Martinelli i in. 2025, arXiv 2506.14951): czyste funkcje bez I/O.

Wszystkie obliczenia w float64. Oznaczenia dla neuronu k jednego MLP (pre-LN, blok timm):
  x̂   - wejście po LayerNorm bez afinicznego (średnia po cechach 0, ‖x̂‖≈√D),
  z_k  - preaktywacja = W1_k·(γ⊙x̂+β)+b1_k,   g_k = GELU(z_k),
  a_k  - kolumna k macierzy fc2.weight (wektor wyjściowy w R^D),  u_k = g_k a_k  (wkład neuronu).
Dla pary (i,j):  s=(a_i+a_j)/2,  m=(a_i-a_j)/2,  y = u_i+u_j = s(g_i+g_j) + m(g_i-g_j).
Kanał Mar25l: z_i≈z_j, ‖m‖ duże, s małe → y ≈ 2 s g(z̄) + m (z_i-z_j) g'(z̄)  (gated linear unit).
"""
from __future__ import annotations

import math

import numpy as np
import torch

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gelu(z):
    """Dokładny GELU (erf), jak nn.GELU(approximate='none')."""
    return 0.5 * z * (1.0 + torch.erf(z / SQRT2))


def gelu_prime(z):
    return 0.5 * (1.0 + torch.erf(z / SQRT2)) + z * torch.exp(-0.5 * z * z) * INV_SQRT_2PI


def effective_input(W1, b1, gamma, beta):
    """Wagi wejściowe po złożeniu afinicznego LayerNorm.

    z_k = (W1_k⊙γ)·x̂ + (W1_k·β + b1_k). Ponieważ Σ_d x̂_d = 0, składowa (W1_k⊙γ) wzdłuż wektora jedynek nie
    wpływa na z_k; usuwamy ją. Dla x̂ = √D·u (‖u‖=1) mamy z_k/√D = (W1_k⊙γ)^⊥·u + b̂_k/√D, stąd naturalny
    rozszerzony wektor θ_eff = [(W1⊙γ)^⊥, b̂/√D]. θ_raw = [W1, b1] to literalny wektor parametrów jak w Mar25l.
    """
    D = W1.shape[1]
    Wg = W1 * gamma[None, :]
    Wc = Wg - Wg.mean(1, keepdim=True)
    b_hat = b1 + W1 @ beta
    theta_eff = torch.cat([Wc, (b_hat / math.sqrt(D))[:, None]], 1)
    theta_raw = torch.cat([W1, b1[:, None]], 1)
    return theta_eff, theta_raw, b_hat


def cosine_matrix(X):
    n = torch.linalg.vector_norm(X, dim=1)
    Xn = X / torch.clamp(n, min=1e-300)[:, None]
    C = Xn @ Xn.T
    C.fill_diagonal_(1.0)
    return C


def gram_cosine(G):
    d = torch.sqrt(torch.clamp(torch.diagonal(G), min=0.0))
    C = G / torch.clamp(d[:, None] * d[None, :], min=1e-300)
    C.fill_diagonal_(1.0)
    return C


def upper_indices(n, device='cpu'):
    iu = torch.triu_indices(n, n, 1, device=device)
    return iu[0], iu[1]


def rank_average(x):
    """Rangi (od 0) z uśrednianiem remisów, jak w Spearmanie SciPy."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1]
    ends = np.r_[starts[1:], n] - 1
    group = np.repeat(np.arange(len(starts)), ends - starts + 1)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = (0.5 * (starts + ends))[group]
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    rx, ry = rank_average(x[ok]), rank_average(y[ok])
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float('nan')


def hist2d(x, y, bins, xrange=(-1.0, 1.0), yrange=(-1.0, 1.0)):
    """Histogram 2D z obcięciem do zakresu (wartości poza zakresem trafiają do skrajnych przedziałów); NaN pomijane."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    ex = np.linspace(xrange[0], xrange[1], bins + 1); ey = np.linspace(yrange[0], yrange[1], bins + 1)
    h, _, _ = np.histogram2d(np.clip(x[ok], *xrange), np.clip(y[ok], *yrange), bins=[ex, ey])
    return h.astype(np.int64), ex, ey


def enrichment(c_in, c_out, cout_lt, cin_gt):
    """P(c_in>τ | c_out<κ) / P(c_in>τ) dla wszystkich par; None, gdy mianownik pusty."""
    c_in = np.asarray(c_in); c_out = np.asarray(c_out)
    opp = c_out < cout_lt
    base = float((c_in > cin_gt).mean())
    if opp.sum() == 0 or base == 0:
        return dict(conditional=None if opp.sum() == 0 else float((c_in[opp] > cin_gt).mean()), base=base,
                    ratio=None, opposed=int(opp.sum()))
    cond = float((c_in[opp] > cin_gt).mean())
    return dict(conditional=cond, base=base, ratio=cond / base, opposed=int(opp.sum()))


def energy_cancellation(G, A):
    """Dokładne znoszenie energii wkładów pary sumowane po tokenach panelu.

    G = Σ_t g_t g_tᵀ (aktywacje po GELU), A = V Vᵀ (Gram kolumn fc2). Dla pary:
      Σ_t ‖u_i+u_j‖² = e_i + e_j + 2 G_ij A_ij,  e_k = G_kk A_kk,
      κ_ij = Σ‖u_i+u_j‖² / (e_i+e_j) = 1 + c_g · c_out · β,
    gdzie c_g = G_ij/√(G_ii G_jj) (współodpalanie), c_out = A_ij/√(A_ii A_jj), β = 2√(e_i e_j)/(e_i+e_j).
    κ=0 pełne znoszenie, 1 brak oddziaływania, 2 pełne wzmocnienie.
    """
    e = torch.diagonal(G) * torch.diagonal(A)
    c_g = gram_cosine(G)
    c_out = gram_cosine(A)
    se = torch.sqrt(torch.clamp(e, min=0.0))
    den = e[:, None] + e[None, :]
    beta = torch.where(den > 0, 2 * se[:, None] * se[None, :] / torch.clamp(den, min=1e-300),
                       torch.full_like(den, float('nan')))
    kappa = 1.0 + c_g * c_out * beta
    return kappa, c_g, c_out, beta, e


def correlation_from_moments(S, S2, n):
    mu = S / n
    cov = S2 / n - mu[:, None] * mu[None, :]
    sd = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0))
    R = cov / torch.clamp(sd[:, None] * sd[None, :], min=1e-300)
    R = torch.where((sd[:, None] > 0) & (sd[None, :] > 0), R, torch.full_like(R, float('nan')))
    R.fill_diagonal_(1.0)
    return R


def pair_constants(A, I, J):
    Aii, Ajj, Aij = A[I, I], A[J, J], A[I, J]
    return dict(Aii=Aii, Ajj=Ajj, Aij=Aij,
                s2=(Aii + Ajj + 2 * Aij) / 4, m2=(Aii + Ajj - 2 * Aij) / 4, sm=(Aii - Ajj) / 4)


TOKEN_SUMS = ('n', 'y_norm', 'u_sum', 'y2', 'u2', 'err2', 'taylor2', 'dg2', 'anti2', 'sym2', 'coact',
              'y_share', 'u_share', 'delta2', 'zsq', 'zi', 'zj', 'zi2', 'zj2', 'zizj',
              'y_norm_coact', 'u_sum_coact', 'n_coact')


def pair_token_sums(z, g, mlp_norm, I, J, const, thr):
    """Sumy po tokenach jednego obrazu dla par (I,J). Wszystko skalarne, bez materializacji wektorów.

    z, g: (T, N); mlp_norm: (T,) normy pełnego wyjścia MLP (z biasem fc2); thr: (N,) progi aktywności.
    Zwraca słownik tensorów (P,) z sumami TOKEN_SUMS.
    """
    zi, zj, gi, gj = z[:, I], z[:, J], g[:, I], g[:, J]
    Aii, Ajj, Aij = const['Aii'], const['Ajj'], const['Aij']
    y2 = torch.clamp(gi * gi * Aii + gj * gj * Ajj + 2 * gi * gj * Aij, min=0.0)
    y = torch.sqrt(y2)
    ui, uj = gi.abs() * torch.sqrt(Aii), gj.abs() * torch.sqrt(Ajj)
    zbar, delta = 0.5 * (zi + zj), zi - zj
    dg, sg = gi - gj, gi + gj
    alpha = sg - 2 * gelu(zbar)
    tay = dg - delta * gelu_prime(zbar)
    err2 = torch.clamp(alpha * alpha * const['s2'] + tay * tay * const['m2'] + 2 * alpha * tay * const['sm'], min=0.0)
    co = ((gi.abs() > thr[I]) & (gj.abs() > thr[J])).to(z.dtype)
    inv = 1.0 / torch.clamp(mlp_norm, min=1e-300)[:, None]
    return dict(n=torch.full_like(Aii, float(z.shape[0])), y_norm=y.sum(0), u_sum=(ui + uj).sum(0), y2=y2.sum(0),
                u2=(ui * ui + uj * uj).sum(0), err2=err2.sum(0), taylor2=(tay * tay).sum(0), dg2=(dg * dg).sum(0),
                anti2=(dg * dg * const['m2']).sum(0), sym2=(sg * sg * const['s2']).sum(0), coact=co.sum(0),
                y_share=(y * inv).sum(0), u_share=((ui + uj) * inv).sum(0), delta2=(delta * delta).sum(0),
                zsq=(0.5 * (zi * zi + zj * zj)).sum(0), zi=zi.sum(0), zj=zj.sum(0), zi2=(zi * zi).sum(0),
                zj2=(zj * zj).sum(0), zizj=(zi * zj).sum(0), y_norm_coact=(y * co).sum(0),
                u_sum_coact=((ui + uj) * co).sum(0), n_coact=co.sum(0))


def pair_vectors(z_tok, g_tok, V, I, J):
    """Wektory y = u_i+u_j i jego przybliżenie GLU na jednym tokenie (do interwencji w głowie). V: (N, D)."""
    ai, aj = V[I], V[J]
    s, m = 0.5 * (ai + aj), 0.5 * (ai - aj)
    zi, zj, gi, gj = z_tok[I], z_tok[J], g_tok[I], g_tok[J]
    zbar = 0.5 * (zi + zj)
    y = gi[:, None] * ai + gj[:, None] * aj
    y_glu = 2 * gelu(zbar)[:, None] * s + ((zi - zj) * gelu_prime(zbar))[:, None] * m
    return y, y_glu


def update_fractions(theta_prev, a_prev, theta_now, a_now, I, J, block_update_sq):
    """Ruch parametrów pary między checkpointami względem kierunków kanału z początku przedziału.

    p = [θ_i, θ_j, a_i, a_j] (surowe parametry: wiersze fc1 z biasem i kolumny fc2).
    e_m = [0, 0, m̂, -m̂]/√2  (rozbieganie przeciwnych wyjść),  e_Δ = [Δ̂, -Δ̂, 0, 0]/√2  (zbieganie wejść).
    f_ch = ((Δp·e_m)² + (Δp·e_Δ)²)/‖Δp‖². Dla losowego kierunku E[f_ch] = 2/dim(p).
    """
    dth_i, dth_j = theta_now[I] - theta_prev[I], theta_now[J] - theta_prev[J]
    da_i, da_j = a_now[I] - a_prev[I], a_now[J] - a_prev[J]
    m_prev = a_prev[I] - a_prev[J]
    d_prev = theta_prev[I] - theta_prev[J]
    mh = m_prev / torch.clamp(torch.linalg.vector_norm(m_prev, dim=1), min=1e-300)[:, None]
    dh = d_prev / torch.clamp(torch.linalg.vector_norm(d_prev, dim=1), min=1e-300)[:, None]
    proj_m = ((da_i - da_j) * mh).sum(1) / SQRT2
    proj_d = ((dth_i - dth_j) * dh).sum(1) / SQRT2
    total = (dth_i ** 2).sum(1) + (dth_j ** 2).sum(1) + (da_i ** 2).sum(1) + (da_j ** 2).sum(1)
    f_ch = (proj_m ** 2 + proj_d ** 2) / torch.clamp(total, min=1e-300)
    f_ch = torch.where(total > 0, f_ch, torch.full_like(f_ch, float('nan')))
    n_units = theta_now.shape[0]
    share = total / torch.clamp(block_update_sq, min=1e-300) * (n_units / 2.0)
    dim = 2 * theta_now.shape[1] + 2 * a_now.shape[1]
    return dict(f_channel=f_ch, proj_m=proj_m, proj_delta=proj_d, update_norm=torch.sqrt(total),
                share_vs_uniform=share, random_expectation=torch.full_like(f_ch, 2.0 / dim))


def pair_weight_metrics(theta_eff, theta_raw, V, I, J):
    """Geometria wag pary: kosinusy, normy, rozkład na s i m, odległość wejść i iloczyn Mar25l ‖m‖·‖Δθ‖."""
    def cos(X):
        a, b = X[I], X[J]
        den = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)
        return (a * b).sum(1) / torch.clamp(den, min=1e-300)
    ai, aj = V[I], V[J]
    s, m = 0.5 * (ai + aj), 0.5 * (ai - aj)
    nai, naj = torch.linalg.vector_norm(ai, dim=1), torch.linalg.vector_norm(aj, dim=1)
    d_eff = torch.linalg.vector_norm(theta_eff[I] - theta_eff[J], dim=1)
    d_raw = torch.linalg.vector_norm(theta_raw[I] - theta_raw[J], dim=1)
    ne_i, ne_j = torch.linalg.vector_norm(theta_eff[I], dim=1), torch.linalg.vector_norm(theta_eff[J], dim=1)
    nr_i, nr_j = torch.linalg.vector_norm(theta_raw[I], dim=1), torch.linalg.vector_norm(theta_raw[J], dim=1)
    ns, nm = torch.linalg.vector_norm(s, dim=1), torch.linalg.vector_norm(m, dim=1)
    pair_norm = torch.sqrt(nr_i ** 2 + nr_j ** 2 + nai ** 2 + naj ** 2)
    return dict(c_out=cos(V), c_in_eff=cos(theta_eff), c_in_raw=cos(theta_raw), a_norm_i=nai, a_norm_j=naj,
                a_balance=torch.minimum(nai, naj) / torch.clamp(torch.maximum(nai, naj), min=1e-300),
                s_norm=ns, m_norm=nm, sym_to_anti=ns / torch.clamp(nm, min=1e-300),
                d_in_eff=d_eff, d_in_eff_rel=d_eff / torch.clamp(0.5 * (ne_i + ne_j), min=1e-300),
                d_in_raw=d_raw, d_in_raw_rel=d_raw / torch.clamp(0.5 * (nr_i + nr_j), min=1e-300),
                m_times_d_eff=nm * d_eff, m_times_d_raw=nm * d_raw, pair_param_norm=pair_norm)


def literal_flags(wm, upd, cfg):
    """Literalne kryteria Mar25l (App. B.3) w adaptacji do wektorowych wyjść; zwraca maski i ich koniunkcję."""
    norm = wm['pair_param_norm'] > cfg['norm_gt']
    near = (1.0 - wm['c_in_raw']) < cfg['cos_distance_lt']
    sign = wm['c_out'] < cfg['cout_lt']
    if upd is None:
        upd_ok = torch.zeros_like(norm)
    else:
        upd_ok = torch.nan_to_num(upd['f_channel'], nan=0.0) > cfg['update_fraction_gt']
    return dict(norm=norm, input_near=near, opposite=sign, update=upd_ok, all=norm & near & sign & upd_ok)


def select_top(values, mask, k, largest):
    """Indeksy elementów maski uporządkowane po wartości; zwraca (indeksy, liczba spełniających, obcięte?)."""
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    total = int(idx.numel())
    if total == 0:
        return idx, 0, False
    order = torch.argsort(values[idx], descending=largest, stable=True)
    idx = idx[order]
    return idx[:k], total, total > k
