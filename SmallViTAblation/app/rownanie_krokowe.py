"""Równanie ruchu cząstki krok po kroku: w każdym kroku t uczenia prognoza ruchu cząstki w pełnych 768 wymiarach
i niedopasowanie tej prognozy do prawdziwego ruchu.

Modele (te same co w rownanie_czastki, bez Koopmana):
- Langevin (oscylator): v(t+1) = beta v(t) − kappa (x(t) − m(t)),
- GLE z pamięcią: v(t+1) = sum_{l<L} a_l v(t−l) − kappa (x(t) − m(t)) + c,
  v(t) = x(t) − x(t−1), m(t) = średnia ostatnich `okno` stanów (z bieżącym); parametry wspólne dla 768 kanałów.

Dopasowanie:
- kroczące (domyślne, uczciwe): parametry w kroku t z najmniejszych kwadratów na ostatnich W przykładach przed t,
  czyli tylko z par (stan s, ruch s -> s+1) dla s + 1 <= t. Prognoza z kroku t widzi wyłącznie stany do t.
- stałe okno od..do (jak panel równania): jedne parametry dla wszystkich kroków; dla kroków przed `do` model zna
  już późniejszy przebieg, więc to tylko porównanie, nie prognoza.

Niedopasowanie w kroku t (prawdziwy krok d = x(t+1) − x(t), przewidziany krok p = v̂(t+1)):
- blad1 = ||p − d|| / ||d||: 0 = idealnie, 1 = tak źle jak „bez ruchu” (p = 0), więcej = gorzej niż bez ruchu;
- cos = cos(p, d): czy równanie trafia w kierunek ruchu;
- wyjasnia = 1 − blad1^2: część energii kroku wyjaśniona przez równanie (reszta = szum, głównie pchnięcie paczki);
- bladh = ||x̂(t+h) − x(t+h)|| / ||x(t) − x(t+h)||: to samo po h krokach prognozy (h <= okno).
"""
import numpy as np

import analiza_uczenia as AU

MODELE = ['Langevin (oscylator)', 'GLE z pamięcią (jądro L kroków)']
DOPASOWANIA = ['kroczące: tylko przeszłość (W ostatnich kroków)', 'stałe okno od..do panelu (zagląda w przyszłość)']


def _cechy(x, model, L, okno):
    """Cechy A [K, D, P] w kroku t do przewidzenia v(t+1), cel [K, D] = v(t+1) (0 w ostatnim kroku), v, m."""
    K = len(x)
    v = np.zeros_like(x); v[1:] = np.diff(x, axis=0)
    m = AU.srednia_kroczaca(x, okno)
    L = 1 if model == 0 else L
    kol = []
    for l in range(L):
        vl = np.zeros_like(v); vl[l:] = v[:K - l]
        kol.append(vl)
    kol.append(-(x - m))
    if model == 1:
        kol.append(np.ones_like(x))
    cel = np.zeros_like(x); cel[:-1] = v[1:]
    return np.stack(kol, -1), cel, v, m, L


def licz(x, model=1, dopasowanie=0, W=100, L=10, okno=50, h=10, od=300, do=700, ridge=1e-6):
    """Prognoza i niedopasowanie cząstki o torze x [K, 768] w każdym kroku. Zwraca słownik tablic [K] (NaN tam,
    gdzie nie ma jeszcze danych do dopasowania albo prawdziwego ruchu do porównania) i przewidziane kroki p [K, 768]."""
    x = np.asarray(x, np.float64)
    K = len(x)
    h = int(min(h, okno))
    A, cel, v, m, L = _cechy(x, model, L, okno)
    P = A.shape[-1]
    s0 = max(L, 1)                                         # pierwszy przykład z pełną historią prędkości
    At = A.transpose(0, 2, 1)
    G = At @ A                                             # wkład kroku s do równań normalnych [K, P, P]
    r = (At @ cel[..., None])[..., 0]
    G[:s0] = 0; r[:s0] = 0; G[K - 1:] = 0; r[K - 1:] = 0  # ostatni krok nie ma ruchu do przodu
    w = np.full((K, P), np.nan)
    if dopasowanie == 1:
        a, b = max(od, s0), min(do, K - 1)
        Gs, rs = G[a:b].sum(0), r[a:b].sum(0)
        w[:] = np.linalg.solve(Gs + ridge * (np.trace(Gs) / P + 1e-12) * np.eye(P), rs)
    else:
        CG = np.concatenate([np.zeros((1, P, P)), np.cumsum(G, 0)])
        Cr = np.concatenate([np.zeros((1, P)), np.cumsum(r, 0)])
        t = np.arange(K)
        lo = np.clip(t - W, 0, None)                       # przykłady s w [t − W, t − 1]: ruch s -> s+1 znany w t
        Gw, rw = CG[t] - CG[lo], Cr[t] - Cr[lo]
        n = np.clip(t, s0, None) - np.clip(lo, s0, None)   # ile przykładów naprawdę weszło do okna
        ok = n >= max(8, 2 * P)
        I = np.eye(P)[None] * (ridge * (np.trace(Gw, axis1=1, axis2=2) / P + 1e-12))[:, None, None]
        w[ok] = np.linalg.solve(Gw[ok] + I[ok], rw[ok][..., None])[..., 0]
    p = (A @ np.nan_to_num(w)[..., None])[..., 0]          # przewidziany krok v̂(t+1) ze stanów do t
    p[np.isnan(w[:, 0])] = np.nan
    dn = np.linalg.norm(cel, axis=1)
    blad1 = np.linalg.norm(p - cel, axis=1) / np.maximum(dn, 1e-12)
    cos = (p * cel).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * dn, 1e-12)
    blad1[K - 1] = cos[K - 1] = np.nan
    # prognoza na h kroków ze wszystkich startów naraz (parametry startu stoją; m przesuwa się o przewidziane stany,
    # a stany wypadające z okna średniej są prawdziwe, bo h <= okno)
    bladh = np.full(K, np.nan)
    T = np.flatnonzero(~np.isnan(w[:, 0]) & (np.arange(K) + h <= K - 1) & (np.arange(K) >= okno - 1))
    if len(T) and h >= 1:
        wt = w[T]
        xs = x[T].copy(); ms = m[T].copy()
        V = [np.where((T - l >= 0)[:, None], v[np.clip(T - l, 0, None)], 0) for l in range(L)]
        for j in range(1, h + 1):
            kol = V + [-(xs - ms)] + ([np.ones_like(xs)] if model == 1 else [])
            vn = sum(c * wt[:, i:i + 1] for i, c in enumerate(kol))
            xs = xs + vn
            ms = ms + (xs - x[T + j - okno]) / okno
            V = [vn] + V[:-1]
        bladh[T] = np.linalg.norm(xs - x[T + h], axis=1) / np.maximum(np.linalg.norm(x[T] - x[T + h], axis=1), 1e-12)
    wyn = dict(blad1=blad1, cos=cos, wyjasnia=1 - blad1 ** 2, bladh=bladh, w=w, p=p.astype(np.float32),
               model=model, L=L, h=h, P=P)
    wyn['bezwladnosc'] = w[:, :L].sum(1)                   # beta (Langevin) albo suma a_l (GLE)
    wyn['sprezyna'] = w[:, L]                              # kappa
    return wyn


def wygladz(y, n):
    """Średnia krocząca z n ostatnich wartości (pomija NaN); tylko do rysunku."""
    y = np.asarray(y, np.float64)
    if n <= 1:
        return y
    ok = ~np.isnan(y)
    c = np.cumsum(np.concatenate([[0], np.where(ok, y, 0)])); k = np.cumsum(np.concatenate([[0], ok]))
    t = np.arange(len(y)); lo = np.clip(t - n + 1, 0, None)
    ile = k[t + 1] - k[lo]
    return np.where(ile > 0, (c[t + 1] - c[lo]) / np.maximum(ile, 1), np.nan)
