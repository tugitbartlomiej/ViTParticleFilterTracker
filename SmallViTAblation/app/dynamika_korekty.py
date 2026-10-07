"""Modele dynamiki na prawdziwie zapisanych sekwencjach stanowiska korekcji (tor optymalizacji korekty oracle albo tor
uczenia sieci) i obserwator Kalmana jako sprawdzian mniejszego kosztu pomiaru.

Każda sekwencja ma jawną oś czasu (time_axis 'latent_optimization' albo 'network_training') i rzeczywisty interwał
zapisu. Dopasowanie (redukcja wymiaru, skalowanie, parametry) wyłącznie na części treningowej [0, do); ocena prognoz
z punktów startu s >= do, każda prognoza widzi tylko stany do s. Miara: rel(h) = suma ||x̂(s+h) − x(s+h)||^2 /
suma ||x(s) − x(s+h)||^2 (persystencja = 1, mniej = lepiej). Za mało startów (< min_startow) -> „brak danych do oceny”.

Modele:
  persystencja       x̂(s+h) = x(s);
  stała prędkość     x̂(s+h) = x(s) + h (x(s) − x(s−1));
  Langevin           rownanie_czastki model 0 (oscylator: v(t+1) = β v(t) − κ (x(t) − m(t)), parametry wspólne);
  GLE z pamięcią L   v(t+1) = Σ_{l<L} a_l v(t−l) − κ (x(t) − m(t)) + c, to samo równanie co rownanie_czastki model 1,
                     ale najmniejsze kwadraty z sum Grama zamiast pełnej macierzy [kroki · D, L + 2] (ten sam wynik,
                     sprawdzone w testach; pełna macierz nie mieści się w pamięci dla 600 x 768);
  Koopman            rownanie_czastki model 2 (Hankel-DMD w PCA-r części treningowej);
  GLE + wejście      jak GLE z dodatkowym członem b · u(t), u(t) znane w chwili t (np. gradient celu korekty w stanie t
                     albo krok Adama z momentów); przyszłe wejścia nie są znane, więc tylko h = 1.
Reszta dopasowania to błąd modelu (bez sprawdzenia rozkładu i autokorelacji nie nazywamy jej białym szumem ani
temperaturą); współczynniki opóźnień nie są wyprowadzonym jądrem Mori-Zwanziga.
"""
import numpy as np

import analiza_uczenia as AU
import rownanie_czastki as RC

MODELE = ['persystencja', 'stała prędkość', 'Langevin', 'GLE', 'Koopman']


# ---------------------------------------------------------------------------------------------------- GLE (Gram)
def _serie(X, okno):
    V = np.zeros_like(X); V[1:] = X[1:] - X[:-1]
    Y = X - AU.srednia_kroczaca(X, okno)
    return V, Y


def gle_dopasuj(X, do, L=5, okno=20, U=None, stala=True, ridge=1e-6, od=None):
    """Współczynniki [a_0..a_{L−1}, κ, (b), (c)] z przykładów t w [od, do−1) (cel v(t+1) <= x(do)).
    L = 0 oznacza model bez pamięci prędkości (tylko sprężyna, wejście i stała)."""
    X = np.asarray(X, np.float64); K, D = X.shape
    V, Y = _serie(X, okno)
    od = max(L, 1) if od is None else max(od, L, 1)
    ts = np.arange(od, do - 1)
    if len(ts) < 3:
        raise ValueError('za mało kroków do dopasowania')
    F = [lambda t, l=l: V[t - l] for l in range(L)] + [lambda t: -Y[t]]
    if U is not None:
        Uu = np.asarray(U, np.float64); F.append(lambda t: Uu[t])
    if stala:
        F.append(lambda t: np.ones((len(t), D)))
    P = len(F)
    G = np.zeros((P, P)); b = np.zeros(P)
    for a in range(0, len(ts), 64):                         # porcje kroków: bez pełnej macierzy cech
        t = ts[a:a + 64]
        cz = [f(t) for f in F]
        cel = V[t + 1]
        for i in range(P):
            b[i] += (cz[i] * cel).sum()
            for j in range(i, P):
                G[i, j] += (cz[i] * cz[j]).sum()
    G = G + np.triu(G, 1).T
    w = np.linalg.solve(G + ridge * np.eye(P), b)
    return dict(w=w, L=L, okno=okno, wejscie=U is not None, stala=stala, do=do, od=od)


def gle_prognoza(X, mdl, starty, h, U=None):
    """Prognoza z każdego startu s (tylko stany do s; wejście tylko u(s), więc h > 1 wymaga modelu bez wejścia).
    Średnia krocząca m(s) stoi w czasie prognozy, jak w rownanie_czastki.prognozy. Zwraca x̂(s + h) [n, D]."""
    X = np.asarray(X, np.float64)
    L, w = mdl['L'], mdl['w']
    if mdl['wejscie'] and h > 1:
        raise ValueError('przyszłe wejścia nieznane: model z wejściem tylko dla h = 1')
    V, Y = _serie(X, mdl['okno'])
    s = np.asarray(starty, int)
    xs = X[s].copy()
    m = X[s] - Y[s]
    hist = [V[np.maximum(s - l, 0)] for l in range(L)]
    for _ in range(h):
        vn = sum(w[l] * hist[l] for l in range(L)) if L else np.zeros_like(xs)
        vn = vn - w[L] * (xs - m)
        k = L + 1
        if mdl['wejscie']:
            vn = vn + w[k] * np.asarray(U, np.float64)[s]; k += 1
        if mdl['stala']:
            vn = vn + w[k]
        xs = xs + vn
        if L:
            hist = [vn] + hist[:-1]
    return xs


# ---------------------------------------------------------------------------------------------------- ocena
def _rel(P, X, s, h):
    e = ((P - X[s + h]) ** 2).sum(); ep = ((X[s] - X[s + h]) ** 2).sum()
    return float(e / max(ep, 1e-30))


def ocen_modele(X, frac_fit=0.6, hs=(1, 5, 20), L=5, okno=20, r=8, d=4, min_startow=5, porcja=8, wyjscie=None,
                n_wyjscia=20, postep=None):
    """Tabela {model: {h: dict(rel, n) albo dict(brak='...')}} dla sekwencji X [K, D] (oś i interwał podaje wołający).
    wyjscie: opcjonalna funkcja x -> (strata, iou_top) (tylko oś optymalizacji korekty, gdzie końcówka jest zamrożona);
    wtedy także błąd wyniku detektora prognozy na n_wyjscia startach."""
    X = np.asarray(X, np.float64); K = len(X)
    do = int(round(frac_fit * K))
    okno = int(max(2, min(okno, do // 2)))
    wyn = {m: {} for m in MODELE}; det = {m: {} for m in MODELE}
    mdl = {}
    try:
        mdl['Langevin'] = RC.dopasuj(X, 0, od=1, do=do, okno=okno)
    except Exception as e:
        mdl['Langevin'] = e
    try:
        mdl['GLE'] = gle_dopasuj(X, do, L=L, okno=okno)
    except Exception as e:
        mdl['GLE'] = e
    try:
        mdl['Koopman'] = RC.dopasuj(X, 2, od=d + 1, do=do, okno=okno, d=d, r=r)
    except Exception as e:
        mdl['Koopman'] = e

    def prog(nazwa, s, h):
        if nazwa == 'persystencja':
            return X[s]
        if nazwa == 'stała prędkość':
            return X[s] + h * (X[s] - X[s - 1])
        m = mdl[nazwa]
        if isinstance(m, Exception):
            raise m
        if nazwa == 'GLE':
            return gle_prognoza(X, m, s, h)
        return np.concatenate([RC.prognozy(m, s[a:a + porcja], h)[:, -1] for a in range(0, len(s), porcja)])

    for ih, h in enumerate(hs):
        s = np.arange(max(do, 2), K - h)
        for nazwa in MODELE:
            if len(s) < min_startow:
                wyn[nazwa][h] = dict(brak=f'brak danych do oceny: {len(s)} startów (< {min_startow})'); continue
            try:
                P = prog(nazwa, s, h)
                wyn[nazwa][h] = dict(rel=_rel(P, X, s, h), n=int(len(s)))
                if wyjscie is not None:
                    wyb = np.unique(np.linspace(0, len(s) - 1, min(n_wyjscia, len(s))).astype(int))
                    a = [wyjscie(P[i]) for i in wyb]; b = [wyjscie(X[s[i] + h]) for i in wyb]
                    dl = [abs(x[0] - y[0]) for x, y in zip(a, b)]
                    di = [abs(x[1] - y[1]) for x, y in zip(a, b) if x[1] is not None and y[1] is not None]
                    det[nazwa][h] = dict(strata=float(np.mean(dl)), iou=float(np.mean(di)) if di else None, n=len(wyb))
            except Exception as e:
                wyn[nazwa][h] = dict(brak=f'błąd modelu: {e}')
        if postep:
            postep(int(100 * (ih + 1) / len(hs)), f'horyzont {h}')
    par = {}
    for k, m in mdl.items():
        if isinstance(m, Exception):
            par[k] = str(m)
        elif k == 'Langevin':
            par[k] = dict(beta=m['beta'], kappa=m['kappa'])
        elif k == 'GLE':
            par[k] = dict(a=m['w'][:L].tolist(), kappa=float(m['w'][L]), c=float(m['w'][-1]))
        else:
            par[k] = dict(modul_lambda=np.abs(m['lam']).tolist(), wariancja_r=float(m['war']), R2=m['R2'])
    return dict(tabela=wyn, detektor=det if wyjscie is not None else None, parametry=par, do=do, K=K, okno=okno, L=L,
                r=r, d=d, hs=list(hs), opis='rel = błąd / błąd persystencji (1 = jak bez ruchu); dopasowanie na krokach '
                                           f'0..{do - 1}, starty prognoz od {do}')


def blad_modelu(X, frac_fit=0.6, L=5, okno=20, nmax=20):
    """Błąd jednego kroku GLE na części testowej rzutowany na pierwszą składową PCA części treningowej: autokorelacja
    (z pasmem ±2/sqrt(n)) i kurtoza. Opisuje błąd modelu; nie przesądza, że to biały szum."""
    X = np.asarray(X, np.float64); K = len(X); do = int(round(frac_fit * K))
    okno = int(max(2, min(okno, do // 2)))
    m = gle_dopasuj(X, do, L=L, okno=okno)
    s = np.arange(max(do, 2), K - 1)
    if len(s) < 8:
        return None
    e = gle_prognoza(X, m, s, 1) - X[s + 1]
    Xf = X[:do] - X[:do].mean(0)
    _, _, Vt = np.linalg.svd(Xf, full_matrices=False)
    p = e @ Vt[0]; p = p - p.mean()
    acf = np.array([(p[:len(p) - k] * p[k:]).sum() for k in range(min(nmax, len(p) - 2) + 1)]) / max((p * p).sum(), 1e-30)
    from scipy import stats
    return dict(acf=acf, pasmo=2 / np.sqrt(len(p)), kurtoza=float(stats.kurtosis(p)), n=len(p))


def porownaj_wejscie(X, Ue, frac_fit=0.6, Ls=(0, 1, 5), okno=20, min_startow=5):
    """h = 1: GLE bez wejścia i z wejściem u(t) dla kilku długości pamięci L. Odpowiada, czy informacja o
    optymalizatorze zmniejsza potrzebę długiej historii latentu."""
    X = np.asarray(X, np.float64); K = len(X); do = int(round(frac_fit * K))
    okno = int(max(2, min(okno, do // 2)))
    s = np.arange(max(do, 2), K - 1)
    if len(s) < min_startow:
        return dict(brak=f'brak danych do oceny: {len(s)} startów')
    out = {}
    for L in Ls:
        for z_wej in (False, True):
            try:
                m = gle_dopasuj(X, do, L=L, okno=okno, U=Ue if z_wej else None)
                P = gle_prognoza(X, m, s, 1, U=Ue if z_wej else None)
                out[f'L={L}{" + wejście" if z_wej else ""}'] = dict(rel=_rel(P, X, s, 1), n=int(len(s)),
                                                                  w=m['w'].tolist())
            except Exception as e:
                out[f'L={L}{" + wejście" if z_wej else ""}'] = dict(brak=str(e))
    return out


def krok_adama(G, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Krok Adama −lr m̂/(sqrt(v̂)+eps) odtworzony z gradientów zapisanych w każdym kroku (G [K, D]); u(t) zależy tylko
    od gradientów do t."""
    m = np.zeros(G.shape[1]); v = np.zeros(G.shape[1]); out = np.zeros_like(G, dtype=np.float64)
    for t in range(len(G)):
        g = G[t].astype(np.float64)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (t + 1)); vh = v / (1 - b2 ** (t + 1))
        out[t] = -lr * mh / (np.sqrt(vh) + eps)
    return out


# ---------------------------------------------------------------------------------------------------- Kalman
def obserwator(X, frac_fit=0.6, r=6, ulamek_tokenow=0.25, co_ile=1, ziarno=0, n_tok=None, ridge=1e-6, kal=1.645):
    """Liniowy filtr Kalmana w małym modelu stanu: mody PCA-r części treningowej, stan s(t) = [y(t), y(t−1)],
    y(t+1) = A1 y(t) + A2 y(t−1) + b (najmniejsze kwadraty i kowariancja błędu Q z części treningowej).
    Obserwacja: tylko wybrane tokeny (ułamek ulamek_tokenow, stały losowy wybór z ziarnem) co co_ile kroków,
    sprowadzone do modów najmniejszymi kwadratami: ŷ_O = argmin ||x_O − c_O − U_O y||, model ŷ_O = y + v; pełna
    kowariancja R błędu v z części treningowej (obcięcie do r modów i brak tokenów; aktywacje są dokładne, więc to
    błąd reprezentacji, nie szum przyrządu). Prawda odniesienia: pełny zapis. Na części testowej: prognoza bez
    obserwacji (otwarta pętla), prognoza przed obserwacją kroku (prior), estymacja po niej (posterior) i najlepszy
    rzut na r modów. Latent odtwarzany przyrostowo od ostatniego pełnego pomiaru x(do − 1): x̂ = x(do−1) + (ŷ − y(do−1)) U^T.
    Strojenie nie używa ukrytych przyszłych stanów. X [K, n_tok * C] albo [K, n_tok, C]."""
    X = np.asarray(X, np.float64)
    if X.ndim == 3:
        n_tok = X.shape[1]; X = X.reshape(len(X), -1)
    K, D = X.shape
    n_tok = n_tok or 1
    C = D // n_tok
    do = int(round(frac_fit * K))
    if do < 2 * r + 5 or K - do < 5:
        return dict(brak=f'brak danych do oceny: {do} kroków treningowych, {K - do} testowych')
    c = X[:do].mean(0)
    _, S, Vt = np.linalg.svd(X[:do] - c, full_matrices=False)
    r = min(r, len(S)); U = Vt[:r].T                                      # [D, r]
    Y = (X - c) @ U                                                       # mody (baza tylko z części treningowej)
    Z0 = np.concatenate([Y[1:do - 1], Y[:do - 2], np.ones((do - 2, 1))], 1); Z1 = Y[2:do]
    W = np.linalg.solve(Z0.T @ Z0 + ridge * np.eye(Z0.shape[1]), Z0.T @ Z1)   # [2r+1, r]
    A1, A2, bb = W[:r].T, W[r:2 * r].T, W[2 * r]
    E = Z1 - Z0 @ W
    A = np.block([[A1, A2], [np.eye(r), np.zeros((r, r))]]); b = np.concatenate([bb, np.zeros(r)])
    Q = np.zeros((2 * r, 2 * r)); Q[:r, :r] = np.atleast_2d(np.cov(E.T)) + 1e-12 * np.eye(r)
    rng = np.random.default_rng(ziarno)
    n_obs = max(1, int(round(ulamek_tokenow * n_tok)))
    tok = np.sort(rng.choice(n_tok, n_obs, replace=False))
    idx = (tok[:, None] * C + np.arange(C)[None]).reshape(-1)
    Uo = U[idx]
    Pinv = np.linalg.pinv(Uo)                                             # [r, n_obs C]
    yo = (X[:, idx] - c[idx]) @ Pinv.T                                    # mody z obserwowanych tokenów
    Rm = np.atleast_2d(np.cov((yo[:do] - Y[:do]).T)) + 1e-12 * np.eye(r)
    H = np.concatenate([np.eye(r), np.zeros((r, r))], 1)
    Ri = np.linalg.inv(Rm)
    HtRiH = H.T @ Ri @ H
    Wo = np.zeros((2 * r, 2 * r)); Ak = np.eye(2 * r)
    for _ in range(2 * r):
        Wo += Ak.T @ HtRiH @ Ak; Ak = A @ Ak
    ew = np.linalg.eigvalsh((Wo + Wo.T) / 2)
    tol_o = ew.max() * 2 * r * np.finfo(float).eps
    s = np.concatenate([Y[do - 1], Y[do - 2]]); P = np.eye(2 * r) * 1e-12
    so = s.copy()
    prior, post, otw, Pp, Ppo, czy = [], [], [], [], [], []
    for t in range(do, K):
        s = A @ s + b; P = A @ P @ A.T + Q
        so = A @ so + b
        prior.append(s[:r].copy()); Pp.append(np.diag(P)[:r].copy()); otw.append(so[:r].copy())
        obs = (t - do) % co_ile == 0
        if obs:
            Sk = H @ P @ H.T + Rm
            Kg = P @ H.T @ np.linalg.inv(Sk)
            s = s + Kg @ (yo[t] - H @ s); P = (np.eye(2 * r) - Kg @ H) @ P; P = (P + P.T) / 2
        post.append(s[:r].copy()); Ppo.append(np.diag(P)[:r].copy()); czy.append(obs)
    prior, post, otw = np.array(prior), np.array(post), np.array(otw)
    Pp, Ppo = np.array(Pp), np.array(Ppo)
    czy = np.array(czy)
    Yt = Y[do:]; Xt = X[do:]; x0 = X[do - 1]; y0 = Y[do - 1]
    blad_x = lambda y: np.linalg.norm(x0 + (y - y0) @ U.T - Xt, axis=1)
    pokrycie = lambda y, p, m: float((np.abs(y[m] - Yt[m]) <= kal * np.sqrt(np.maximum(p[m], 1e-30))).mean()) if m.any() else None
    wszystkie = np.ones(len(Yt), bool)
    return dict(
        kroki_testu=np.arange(do, K), do=do, r=r, tokeny_obserwowane=tok, n_tok=n_tok, czy_obserwacja=czy,
        blad_modow=dict(otwarta_petla=np.linalg.norm(otw - Yt, axis=1), prior=np.linalg.norm(prior - Yt, axis=1),
                        posterior=np.linalg.norm(post - Yt, axis=1),
                        sam_pomiar=np.where(czy, np.linalg.norm(yo[do:] - Yt, axis=1), np.nan)),
        blad_latentu=dict(otwarta_petla=blad_x(otw), prior=blad_x(prior), posterior=blad_x(post), rzut_r_modow=blad_x(Yt),
                          persystencja=np.linalg.norm(Xt - x0, axis=1)),
        pokrycie_90=dict(prior=pokrycie(prior, Pp, wszystkie), posterior=pokrycie(post, Ppo, czy)),
        obserwowalnosc=dict(wartosci=ew, rzad=int((ew > tol_o).sum()), wymiar_stanu=2 * r,
                            uwarunkowanie=float(ew.max() / max(ew.min(), 1e-300))),
        koszt=dict(ulamek_tokenow=n_obs / n_tok, ulamek_krokow=float(czy.mean()), lacznie=n_obs / n_tok * float(czy.mean())),
        wariancja_r_modow=float((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-30)),
        opis='prior = prognoza przed obserwacją kroku, posterior = estymacja po niej; otwarta pętla = bez obserwacji; '
             'sam pomiar = mody z obserwowanych tokenów bez modelu; R z części treningowej')
