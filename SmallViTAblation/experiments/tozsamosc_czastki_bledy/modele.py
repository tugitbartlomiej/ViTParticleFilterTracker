"""Czy sygnal przestrzenny i dynamika dodaja informacje o bledzie poza statyka?

Rozstrzyga ocena na innych operacjach, nie dopasowanie tego modulu do danych A.
EDMD ma ustalony slownik; pamiec jest regresja opoznien, a Kalman nominalny.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def history_indices(data, length, step):
    if length < 1 or step < 1:
        raise ValueError('Historia i krok musza byc dodatnie')
    lookup = {}
    for i, (op, video, number) in enumerate(zip(data['operation'], data['video'], data['number'])):
        key = (str(op), str(video), int(number))
        if key in lookup:
            raise ValueError(f'Powtorzona klatka {key}')
        lookup[key] = i
    past = np.full((len(data['number']), length), -1, dtype=int)
    for key, i in lookup.items():
        for lag in range(1, length + 1):
            past[i, lag - 1] = lookup.get((key[0], key[1], key[2] - step * lag), -1)
    return past


def fit_direction(tokens, mask, rows, ridge):
    """Wazona LDA tylko z A. Tlo: komorki bez segmentacji; kontekst jawny."""
    k = tokens.shape[-1]
    counts, sums, seconds = np.zeros(2), np.zeros((2, k)), np.zeros((2, k, k))
    for i in rows:
        z = tokens[i].astype(float)
        for c, weights in enumerate((mask[i], (mask[i] == 0).astype(float))):
            counts[c] += weights.sum()
            sums[c] += weights @ z
            seconds[c] += (z * weights[:, None]).T @ z
    if (counts < 2).any():
        raise ValueError('Za malo tokenow narzedzia lub tla w A')
    means = sums / counts[:, None]
    cov = sum(seconds[c] / counts[c] - np.outer(means[c], means[c]) for c in range(2)) / 2
    w = np.linalg.solve(cov + ridge * np.eye(k), means[0] - means[1])
    if np.linalg.norm(w) < 1e-12:
        raise ValueError('Nie wyznaczono kierunku narzedzia')
    return w / np.linalg.norm(w)


def descriptors(data, direction):
    """Momenty wzgledem maski i koncowki, bez twierdzenia o materialnym sledzeniu."""
    gh, gw = data['grid']
    yy, xx = np.meshgrid((np.arange(gh) + .5) / gh, (np.arange(gw) + .5) / gw, indexing='ij')
    coords = np.c_[xx.ravel(), yy.ravel()]
    signals, residuals, geometry = [], [], []
    for i, z in enumerate(data['tokens']):
        z = z.astype(float)
        weights = data['mask'][i].astype(float)
        weights /= weights.sum()
        background = data['mask'][i] == 0
        a = z @ direction
        center = weights @ coords
        delta = coords - center
        spread = np.sqrt(weights @ (delta ** 2)) + 1e-6
        bg_mean, bg_sd = a[background].mean(), a[background].std() + 1e-6
        mean = weights @ a
        sd = np.sqrt(weights @ ((a - mean) ** 2))
        pixels = coords * data['size'][i]
        distances = np.linalg.norm(pixels - data['tip'][i], axis=1)
        inside = np.flatnonzero(weights > 0)
        ordered = inside[np.argsort(distances[inside])]
        tip = ordered[:max(1, len(ordered) // 3)]
        shaft = ordered[max(1, len(ordered) // 3):]
        tip_mean = np.average(a[tip], weights=weights[tip])
        shaft_mean = np.average(a[shaft], weights=weights[shaft])
        moments = (weights[:, None] * (a - mean)[:, None] * delta / spread).sum(0)
        signals.append([mean, sd, bg_mean, bg_sd, (mean - bg_mean) / bg_sd,
                        tip_mean, shaft_mean, *moments])
        residual = z - a[:, None] * direction[None]
        residuals.append(weights @ residual)
        b = data['gt_box'][i] / np.tile(data['size'][i], 2)
        geometry.append([*b, *center, *spread, float(data['mask'][i].mean())])
    return np.asarray(signals), np.asarray(residuals), np.asarray(geometry)


def iou_xywh(a, b):
    area = np.maximum(np.minimum(a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:]) -
                      np.maximum(a[:, :2], b[:, :2]), 0).prod(1)
    union = a[:, 2:].clip(0).prod(1) + b[:, 2:].clip(0).prod(1) - area
    return area / np.maximum(union, 1e-12)


def dictionary(x):
    return np.concatenate([x, x ** 2], axis=1)


def memory_design(s, past, rows):
    return s[past[rows]].reshape(len(rows), -1)


def kalman_features(s, past, transition, offset, q, observation_variance):
    """Innowacja przed aktualizacja, Joseph; zerowanie przy przerwie nagrania.

Q z reszt regresji, R ustalone. Nie sa rozdzielnie zidentyfikowane z obserwacji.
"""
    n, dim = s.shape
    identity = np.eye(dim)
    r = observation_variance * identity
    output = np.full((n, 3), np.nan)
    states, covariances = {}, {}
    for i in range(n):
        previous = int(past[i, 0])
        if previous < 0:
            states[i], covariances[i] = s[i].copy(), identity.copy()
            continue
        if previous >= i:
            raise ValueError('Dane musza byc uporzadkowane rosnaco wewnatrz filmu')
        predicted = transition @ states[previous] + offset
        p = transition @ covariances[previous] @ transition.T + q
        innovation = s[i] - predicted
        variance = p + r
        gain = np.linalg.solve(variance, p.T).T
        output[i] = [np.linalg.norm(innovation), np.trace(variance),
                     innovation @ np.linalg.solve(variance, innovation) / dim]
        states[i] = predicted + gain @ innovation
        covariances[i] = (identity - gain) @ p @ (identity - gain).T + gain @ r @ gain.T
        # Kazda klatka ma najwyzej jednego bezposredniego nastepnika przy stalym kroku.
        states.pop(previous)
        covariances.pop(previous)
    return output


def dynamic_features(signal, residual, past, fit_rows, cfg):
    scaler = StandardScaler().fit(signal[fit_rows])
    s = scaler.transform(signal)
    valid = np.flatnonzero((past >= 0).all(1))
    fit = np.intersect1d(valid, fit_rows)
    if len(fit) < cfg['min_frames_per_partition']:
        raise ValueError('Za malo dokladnych sekwencji w A; zmien dane, nie sklejaj przerw')
    reg = cfg['ridge'] * len(fit)
    lifted = dictionary(s)
    edmd = Ridge(alpha=reg).fit(lifted[past[fit, 0]], lifted[fit])
    # Dodana stala uwzglednia intercept. Pomijamy trywialna funkcje stala.
    operator = np.zeros((lifted.shape[1] + 1, lifted.shape[1] + 1))
    operator[:-1, :-1] = edmd.coef_.T
    operator[-1, :-1] = edmd.intercept_
    operator[-1, -1] = 1
    eigenvalues, vectors = np.linalg.eig(operator)
    nonconstant = np.flatnonzero(np.linalg.norm(vectors[:-1], axis=0) > 1e-8)
    chosen = nonconstant[np.argsort(np.abs(eigenvalues[nonconstant] - 1))[:cfg['modes']]]
    modes = np.c_[lifted, np.ones(len(s))] @ vectors[:, chosen]
    markov = Ridge(alpha=reg).fit(s[past[fit, 0]], s[fit])
    memory = Ridge(alpha=reg).fit(memory_design(s, past, fit), s[fit])
    residual_scaler = StandardScaler().fit(residual[fit_rows])
    remaining = residual_scaler.transform(residual)
    augmented = np.c_[s, remaining]
    conditioned = Ridge(alpha=reg).fit(memory_design(augmented, past, fit), s[fit])
    k = np.full((len(s), s.shape[1] + 2 * len(chosen)), np.nan)
    l = np.full((len(s), 3 * s.shape[1]), np.nan)
    k[valid] = np.c_[s[valid] - edmd.predict(lifted[past[valid, 0]])[:, :s.shape[1]],
                      modes[valid].real, modes[valid].imag]
    prediction = memory.predict(memory_design(s, past, valid))
    conditional_prediction = conditioned.predict(memory_design(augmented, past, valid))
    l[valid] = np.c_[s[valid] - conditional_prediction,
                      prediction - markov.predict(s[past[valid, 0]]),
                      conditional_prediction - prediction]
    errors = s[fit] - markov.predict(s[past[fit, 0]])
    q = np.atleast_2d(np.cov(errors, rowvar=False)) + 1e-6 * np.eye(s.shape[1])
    f = kalman_features(s, past, markov.coef_, markov.intercept_, q, cfg['observation_variance'])
    spectral = dict(real=eigenvalues.real.tolist(), imag=eigenvalues.imag.tolist(),
                    selected=chosen.tolist(), operator='EDMD [s,s^2,1]; bez trywialnego modu stalego',
                    eigenfunction_note='Przyblizone funkcje wlasne modelu EDMD, nie dowod tozsamosci.',
                    markov_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(markov.coef_)))),
                    nominal_q=q.tolist(), observation_variance=cfg['observation_variance'])
    return k, l, f, spectral


def circular_placebo(features, data, rows, seed, history):
    """Cyklicznie przesuwa dodatek dynamiki w kazdym filmie; zachowuje jego rozklad."""
    out = features.copy()
    rng = np.random.default_rng(seed)
    for video in np.unique(data['video'][rows]):
        ix = rows[data['video'][rows] == video]
        ix = ix[np.argsort(data['number'][ix])]
        if len(ix) <= 2 * history + 2:
            raise ValueError('Film za krotki dla cyklicznej kontroli')
        shift = int(rng.integers(history + 1, len(ix) - history))
        out[ix] = features[np.roll(ix, shift)]
    return out
