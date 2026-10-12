"""Prognoza na 100 kroków we WCZESNEJ fazie uczenia (kroki 0..500), 10 ziaren, latent sond co 10 kroków.

Rozstrzyga: czy w rozgrzewce, gdzie dryf jest kilkadziesiąt razy silniejszy od drgań, prosty model z własnej
historii osiąga cel 85 procent na 100 kroków naprzód. Współczynniki są skalarne (wspólne dla 768 współrzędnych
i wszystkich sond), zależne od kroku startu t (rozgrzewka lr trwa 300 kroków, dynamika zmienia się z t),
dopasowane na jednych ziarnach i sprawdzone na innych. Obala: wynik poniżej 0,85 na ziarnach testowych.

Dane: experiments/maly_vit_od_zera/wyniki_eden/petla_s47..s56/wspolny/k01/LATENT.npz
      (kroki 0..500 co 10; 192 sondy × 2 (narzędzie, tło) × 768, float16; wspólna rozgrzewka ziarna).
Uruchomienie: py -3.11 wczesna_faza_10_ziaren.py --wyniki ../wyniki/wczesna_faza
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ZR = Path(__file__).resolve().parents[2] / 'maly_vit_od_zera' / 'wyniki_eden'
ZIARNA = tuple(range(47, 57))
H = 10                       # 10 checkpointów = 100 kroków
LAGI = (1, 2, 3)             # 10, 20, 30 kroków wstecz
STARTY = tuple(range(3, 41)) # kroki 30..400, cel <= 500
KAPPA = np.concatenate([[0.0], np.logspace(-4, 0.5, 200)])


def wczytaj(z):
    d = np.load(ZR / f'petla_s{z}' / 'wspolny' / 'k01' / 'LATENT.npz')
    assert d['kroki'][1] - d['kroki'][0] == 10 and d['kroki'][-1] == 500
    return d['latent'].astype(np.float32).reshape(51, -1, 768)      # (T, 384, 768)


def sumy(Y):
    """Dla każdego startu t: FᵀF, FᵀΔ, ΔᵀΔ (cechy: prędkości z lagów; cel Δ_H)."""
    out = {}
    for t in STARTY:
        F = np.stack([Y[t] - Y[t - l] for l in LAGI])                 # (L, U, D)
        dl = Y[t + H] - Y[t]
        out[t] = dict(G=np.einsum('lud,mud->lm', F, F), b=np.einsum('lud,ud->l', F, dl),
                      c=float(np.einsum('ud,ud->', dl, dl)))
    return out


def dopasuj(S_list):
    beta = {}
    for t in STARTY:
        G = sum(S[t]['G'] for S in S_list); b = sum(S[t]['b'] for S in S_list)
        beta[t] = np.linalg.solve(G + 1e-6 * np.trace(G) * np.eye(len(LAGI)), b)
    return beta


def ocen(S_list, beta):
    sse_m = np.zeros(len(STARTY)); sse_p = np.zeros(len(STARTY))
    for i, t in enumerate(STARTY):
        for S in S_list:
            bt = beta[t]
            sse_m[i] += S[t]['c'] - 2 * bt @ S[t]['b'] + bt @ S[t]['G'] @ bt
            sse_p[i] += S[t]['c']
    return sse_m, sse_p


def ou_stale(Ys_fit, Ys_test):
    """OU ze średnią z ostatnich 5 checkpointów i jednym kappa (bez zależności od t), dla porównania."""
    k = 5
    def stat(Y):
        A = B = C = 0.0
        for t in STARTY:
            if t - k + 1 < 0: continue
            mu = Y[t - k + 1:t + 1].mean(0); d = mu - Y[t]; dl = Y[t + H] - Y[t]
            A += np.einsum('ud,ud->', dl, d); B += np.einsum('ud,ud->', d, d); C += np.einsum('ud,ud->', dl, dl)
        return A, B, C
    A, B, C = map(sum, zip(*[stat(Y) for Y in Ys_fit]))
    al = 1 - np.exp(-KAPPA * H)
    loss = C + 2 * al * A + al ** 2 * B
    a = al[int(np.argmin(loss))]
    At, Bt, Ct = map(sum, zip(*[stat(Y) for Y in Ys_test]))
    return float(1 - (Ct + 2 * a * At + a ** 2 * Bt) / Ct), float(KAPPA[int(np.argmin(loss))])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    Y = {z: wczytaj(z) for z in ZIARNA}
    S = {z: sumy(Y[z]) for z in ZIARNA}
    A, B = ZIARNA[:5], ZIARNA[5:]
    wyn = dict(meta=dict(zrodlo=str(ZR), ziarna=ZIARNA, horyzont_krokow=100, lagi_krokow=[10 * l for l in LAGI],
                         starty_krokow=[10 * t for t in STARTY], foldy=[A, B],
                         model='x(t+100) = x(t) + sum_l beta_l(t) * (x(t) - x(t - lag_l)); beta wspólne dla 768 '
                               'współrzędnych i 384 sond, osobne dla każdego kroku startu, dopasowane na innych ziarnach'))
    per_t_m = np.zeros(len(STARTY)); per_t_p = np.zeros(len(STARTY))
    for fit, test in ((A, B), (B, A)):
        beta = dopasuj([S[z] for z in fit])
        m, pp = ocen([S[z] for z in test], beta)
        per_t_m += m; per_t_p += pp
    skill_t = 1 - per_t_m / per_t_p
    wyn['skill_na_start'] = {str(10 * t): float(s) for t, s in zip(STARTY, skill_t)}
    for nazwa, sel in (('0-100', slice(0, 8)), ('100-200', slice(8, 18)), ('200-300', slice(18, 28)),
                       ('300-400', slice(28, 38))):
        wyn[f'skill_{nazwa}'] = float(1 - per_t_m[sel].sum() / per_t_p[sel].sum())
    wyn['skill_wszystkie_starty'] = float(1 - per_t_m.sum() / per_t_p.sum())
    ou = [ou_stale([Y[z] for z in fit], [Y[z] for z in test]) for fit, test in ((A, B), (B, A))]
    wyn['ou_stale_kappa_na_test'] = ou
    # ile dryf przewyższa drgania: |Δ_100|² względem |Δ_10|²·10 (białe drgania dałyby ~1, czysty dryf ~10)
    stos = {}
    for t in STARTY:
        d100 = sum(float(np.einsum('ud,ud->', Y[z][t + H] - Y[z][t], Y[z][t + H] - Y[z][t])) for z in ZIARNA)
        d10 = sum(float(np.einsum('ud,ud->', Y[z][t + 1] - Y[z][t], Y[z][t + 1] - Y[z][t])) for z in ZIARNA)
        stos[str(10 * t)] = d100 / (10 * d10)
    wyn['przesuniecie_100_do_10x_przesuniecie_10'] = stos
    (out / 'WCZESNA_FAZA.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in wyn.items() if k.startswith('skill_') and k != 'skill_na_start'})
    print('OU stałe:', ou)
    print('skill na start:', {k: round(v, 2) for k, v in wyn['skill_na_start'].items()})
    print('D100/(10 D10):', {k: round(v, 1) for k, v in stos.items()})


if __name__ == '__main__':
    main()
