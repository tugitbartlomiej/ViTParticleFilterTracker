"""Pułapka uczenia: koloid w czasie uczenia jako proces Ornsteina–Uhlenbecka wokół wolno dryfującego środka (jak pęseta).

Model: x_t = m_t + y_t, m_t wolny środek (średnia ruchoma), y_t drgania OU: y_{t+1} = φ y_t + szum, φ = exp(−Δt/τ).
Szacujemy φ (regresja AR(1) drgań, wspólna dla 192 wymiarów) i τ = −Δt / ln φ w krokach uczenia; sprawdzamy korelację kolejnych
przyrostów (dla OU próbkowanego rzadko → −(1 − φ)/2); „sztywność w jednostkach temperatury” κ/kT = 1 / wariancja drgań
(równopodział, jak w kalibracji pęsety); dryf środka: prędkość |Δm| / Δt i jej spadek w czasie.
Dwa zakresy: gęsty początek (co 10 kroków, kroki 0–1000) i późne uczenie (co 100 kroków, od 1100).
"""
from pathlib import Path

import numpy as np

KAT = Path(__file__).resolve().parents[2] / 'outputs' / 'pulapka_cel_koloidu_20261015'
f = np.load(KAT / 'pomiar.npz')
kr = f['kroki']; wyb = list(f['wybrane']); X = f['X_wyb']; rms = f['rms']
linie = []


def pisz(s=''):
    print(s); linie.append(s)


def okno(idx, w):
    """Średnia ruchoma (środek pułapki) po w checkpointach, wyśrodkowana."""
    Xs = X[idx]; m = np.empty_like(Xs); h = w // 2
    for i in range(len(idx)):
        a, b = max(0, i - h), min(len(idx), i + h + 1)
        m[i] = Xs[a:b].mean(0)
    return Xs, m


pisz('# Pułapka uczenia: koloid jako proces Ornsteina–Uhlenbecka wokół dryfującego środka')
for nazwa, maska, dt in (('gęsty początek (co 10 kroków, 0–1000)', (kr <= 1000) & (np.r_[np.diff(kr), 0] == 10), 10),
                         ('późne uczenie (co 100 kroków, od 1100)', (kr >= 1100) & (np.r_[np.diff(kr), 0] == 100), 100)):
    idx = np.flatnonzero(maska)
    for w in (11, 21):
        Xs, m = okno(idx, w)
        y = (Xs - m) / rms[idx][:, None, None]                          # drgania w jednostkach rms stanu
        for etyk, kol in (('226', [wyb.index(226)]), ('w ramce', list(range(len(wyb))))):
            a, b = y[:-1, kol].ravel(), y[1:, kol].ravel()
            phi = float((a * b).sum() / (a * a).sum())
            tau = -dt / np.log(phi) if 0 < phi < 1 else float('nan')
            dX = np.diff(X[idx][:, kol], axis=0)
            c1 = float(np.mean((dX[:-1] * dX[1:]).sum(-1) / np.maximum(np.linalg.norm(dX[:-1], axis=-1) * np.linalg.norm(dX[1:], axis=-1), 1e-30)))
            var = float(y[:, kol].var())
            dm = np.linalg.norm(np.diff(m[:, kol], axis=0), axis=-1).mean(-1) / rms[idx][1:] / dt
            pisz(f'- {nazwa}, okno {w}, {etyk}: φ = {phi:.3f} → τ ≈ {tau:.0f} kroków; korelacja kolejnych przyrostów {c1:+.3f} '
                 f'(OU: {-(1 - phi) / 2:+.3f}); wariancja drgań {var:.2e} rms² → κ/kT ≈ {1 / max(var, 1e-30):.3g}; '
                 f'prędkość środka {dm[:len(dm) // 3].mean():.2e} → {dm[-len(dm) // 3:].mean():.2e} rms/krok')
(KAT / 'PULAPKA_UCZENIA.md').write_text('\n'.join(linie) + '\n', encoding='utf-8')
print('zapisano', KAT / 'PULAPKA_UCZENIA.md')
