"""Równanie ruchu jednej cząstki (tokenu) z samej jej historii: Langevin, GLE z pamięcią i Koopman (Hankel-DMD).

Tor cząstki x [K, 768] (latent tokenu w kolejnych krokach uczenia albo klatkach). Bez siły paczki, bez wag: model
widzi tylko przeszłe stany tej cząstki. Dopasowanie na krokach od..do, ocena na krokach od do do końca.

Modele (dyskretny czas, krok = 1):
- oscylator (Langevin z bezwładnością, AR(2)): v(t+1) = beta v(t) − kappa (z(t) − m(t)) + eta(t+1),
  m = średnia ostatnich `okno` stanów; beta, kappa wspólne dla 768 kanałów (analiza_uczenia.oscylator_toru);
- GLE z pamięcią (dyskretne uogólnione równanie Langevina, Mori-Zwanzig; Lin i Lu 2021):
  v(t+1) = sum_{l=0}^{L-1} a_l v(t−l) − kappa (y(t) − m(t)) + c + eta(t+1), w PCA-r własnego toru cząstki.
  Jądro pamięci K(l) = delta_{l0} − a_l (tarcie z pamięcią); K(l) liczone też z korelacji (odwrócenie Volterry,
  Jung i in. 2017, Tepper i in. 2024) do porównania;
- Koopman (Hankel-DMD, Arbabi i Mezić 2017): stan H(t) = [y(t), y(t−1), ..., y(t−d+1)], H(t+1) = A H(t);
  wartości własne A to częstości i tłumienia modów.
Prognoza w PCA-r przenoszona do 768 kanałów jako przyrost: x̂ = x(s) + (ŷ − y(s)) U^T (reszta spoza U stoi).
Szum eta = to, czego model z historii nie przewiduje; w uczeniu to głównie pchnięcie nowej paczki (31-35 % ruchu,
experiments/maly_vit_liniowy_ruch_czastek/WYNIKI.md), którego z historii przewidzieć się nie da.
"""
import numpy as np

import analiza_uczenia as AU

MODELE = ['Langevin (oscylator)', 'GLE z pamięcią (jądro L kroków)', 'Koopman (Hankel-DMD)']


def _pca_toru(x, od, do, r):
    X = np.asarray(x[od:do], np.float64)
    c = X.mean(0)
    _, S, Vt = np.linalg.svd(X - c, full_matrices=False)
    r = min(r, len(S))
    return c, Vt[:r].T, (S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-30)


def dopasuj(x, model, od=300, do=700, okno=50, L=10, d=8, r=8, ridge=1e-6):
    """Model ruchu cząstki z jej toru x [K, 768], dopasowany na krokach od..do. Zwraca słownik z typem i parametrami."""
    x = np.asarray(x, np.float64)
    K = len(x); do = min(do, K - 2); od = min(od, max(L + d + 1, do // 2))
    if model == 0:
        b, k = AU.oscylator_toru(x, od=od, do=do, okno=okno)
        return dict(model=0, beta=b, kappa=k, okno=okno, od=od, do=do, M=AU.srednia_kroczaca(x, okno), x=x)
    c, U, war = _pca_toru(x, od, do, r)
    if model == 1:                                                    # pełne 768 kanałów, jedno jądro dla wszystkich
        y = x - c
        m = AU.srednia_kroczaca(y, okno)
        v = np.concatenate([np.zeros_like(y[:1]), np.diff(y, axis=0)])
        ts = np.arange(max(od, L), do)
        kol = [v[ts - l] for l in range(L)] + [-(y[ts] - m[ts]), np.ones_like(y[ts])]
        A = np.stack([k.reshape(-1) for k in kol], 1)                  # [len(ts) * r, L + 2]
        yy = v[ts + 1].reshape(-1)
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ yy)
        a, kappa, c0 = w[:L], w[L], w[L + 1]
        R2 = 1 - ((yy - A @ w) ** 2).sum() / max((yy ** 2).sum(), 1e-30)
        return dict(model=1, a=a, kappa=float(kappa), c0=float(c0), L=L, okno=okno, od=od, do=do, c=c, U=None, u1=U[:, 0],
                    war=war, y=y, v=v, M=m, R2=float(R2), x=x)
    y = (x - c) @ U                                                   # [K, r]
    reszta = x - c - y @ U.T                                          # ruch poza r składowymi: oscylator
    br, kr = AU.oscylator_toru(reszta, od=od, do=do, okno=okno)
    Hs = np.concatenate([y[d - 1 - j:K - j] for j in range(d)], 1)    # H(t) dla t = d-1..K-1: [K - d + 1, r d]
    t0 = max(od, d - 1) - (d - 1); t1 = do - (d - 1)
    X0, X1 = Hs[t0:t1 - 1], Hs[t0 + 1:t1]
    A = np.linalg.solve(X0.T @ X0 + ridge * np.eye(X0.shape[1]), X0.T @ X1).T
    lam = np.linalg.eigvals(A)
    R2 = 1 - ((X1 - X0 @ A.T)[:, :r] ** 2).sum() / max(((X1[:, :r] - X0[:, :r]) ** 2).sum(), 1e-30)   # przyrostów
    return dict(model=2, A=A, lam=lam, d=d, r=r, od=od, do=do, c=c, U=U, war=war, y=y, Hs=Hs, R2=float(R2), x=x,
                reszta=reszta, M_reszty=AU.srednia_kroczaca(reszta, okno), beta_r=br, kappa_r=kr)


def prognozy(mdl, starty, h):
    """Tory z wielu kroków startu naraz [n, h + 1, 768], każdy tylko ze stanów do swojego startu."""
    x = mdl['x']; s = np.asarray(starty, int)
    if mdl['model'] == 0:
        return AU.prognozy_toru(x, mdl['M'], s, h, mdl['beta'], mdl['kappa'])
    U, y = mdl['U'], mdl['y']
    if mdl['model'] == 1:
        L, a = mdl['L'], mdl['a']
        hist = [mdl['v'][np.maximum(s - l, 0)] for l in range(L)]       # ostatnie prędkości, najnowsza pierwsza
        yy = y[s]; m = mdl['M'][s]
        tor = [yy]
        for _ in range(h):
            vn = a[0] * hist[0] - mdl['kappa'] * (yy - m) + mdl['c0']
            for l in range(1, L):
                vn += a[l] * hist[l]
            yy = yy + vn
            hist.insert(0, vn); hist.pop()
            tor.append(yy)
    else:
        d, r, A = mdl['d'], mdl['r'], mdl['A']
        s = np.maximum(s, d - 1)
        H = mdl['Hs'][s - (d - 1)].copy()                              # [n, r d]
        tor = [H[:, :r].copy()]
        for _ in range(h):
            H = H @ A.T
            tor.append(H[:, :r].copy())
    Y = np.stack(tor, 1)                                               # [n, h + 1, r]
    if U is None:                                                      # GLE na pełnych kanałach
        return x[s][:, None, :] + (Y - Y[:, :1])
    R = AU.prognozy_toru(mdl['reszta'], mdl['M_reszty'], s, h, mdl['beta_r'], mdl['kappa_r'])
    return x[s][:, None, :] + (Y - Y[:, :1]) @ U.T + (R - R[:, :1])


def ocena(mdl, od_testu, hs=(1, 5, 20, 50)):
    """{h: błąd prognozy / błąd persystencji} na startach od od_testu do K − h (mniej = lepiej, 1 = jak „bez ruchu”)."""
    x = mdl['x']; K = len(x); wynik = {}
    for h in hs:
        s = np.arange(max(od_testu, 60), K - h)
        if len(s) < 5:
            continue
        P = prognozy(mdl, s, h)[:, -1]
        e = ((P - x[s + h]) ** 2).sum(); ep = ((x[s] - x[s + h]) ** 2).sum()
        wynik[h] = float(e / max(ep, 1e-30))
    return wynik


def analiza_gle(mdl, nmax=30):
    """Części równania GLE w PCA-r cząstki: szum eta, jego autokorelacja, jądro pamięci (regresja i Volterra), VACF
    danych i modelu, rozkład kroku na bezwładność, pamięć, sprężynę i szum (na pierwszej składowej toru)."""
    y, v, m, a, L, kap = mdl['y'], mdl['v'], mdl['M'], mdl['a'], mdl['L'], mdl['kappa']
    K = len(y)
    t = np.arange(L, K - 1)
    bezw = a[0] * v[t]
    pamiec = sum(a[l] * v[t - l] for l in range(1, L)) if L > 1 else np.zeros_like(v[t])
    spr = -kap * (y[t] - m[t])
    eta = v[t + 1] - bezw - pamiec - spr - mdl['c0']
    od, do = mdl['od'], mdl['do']
    w = (t >= od) & (t < do)
    e = eta[w]; e = e - e.mean(0)
    acf = np.array([(e[:len(e) - k] * e[k:]).sum() for k in range(nmax + 1)]); acf = acf / max(acf[0], 1e-30)
    vv = v[od:do] - v[od:do].mean(0); xx = (y - m)[od:do]
    Cvv = np.array([(vv[:len(vv) - k] * vv[k:]).sum() / (len(vv) - k) for k in range(nmax + 2)])
    Cxv = np.array([(xx[k:] * vv[:len(vv) - k]).sum() / (len(vv) - k) for k in range(nmax + 1)])
    Kv = np.zeros(nmax + 1)
    for n in range(nmax + 1):                                          # odwrócenie Volterry (dyskretne)
        Kv[n] = (Cvv[n] - Cvv[n + 1] - kap * Cxv[n] - sum(Kv[j] * Cvv[n - j] for j in range(n))) / max(Cvv[0], 1e-30)
    Kreg = np.zeros(nmax + 1); Kreg[:L] = -a; Kreg[0] += 1
    # VACF modelu: symulacja z szumem białym o tej samej wariancji
    rng = np.random.default_rng(0)
    T = 20000; r = 1; sig = np.ones(1)
    ys = np.zeros((T, r)); vs = np.zeros((T, r)); ms = np.zeros(r)
    for i in range(L, T - 1):
        vs[i + 1] = sum(a[l] * vs[i - l] for l in range(L)) - kap * (ys[i] - ms) + rng.standard_normal(r) * sig
        ys[i + 1] = ys[i] + vs[i + 1]
        ms = ms + (ys[i + 1] - ms) / mdl['okno']
    vm = vs[500:] - vs[500:].mean(0)
    Cm = np.array([(vm[:len(vm) - k] * vm[k:]).sum() / (len(vm) - k) for k in range(nmax + 1)])
    u1 = mdl['u1']
    rozklad = dict(t=t, bezwladnosc=bezw @ u1, pamiec=pamiec @ u1, sprezyna=spr @ u1, szum=eta @ u1, v=v[t + 1] @ u1)
    udz = {nazwa: float((czesc * v[t + 1]).sum() / max((v[t + 1] ** 2).sum(), 1e-30))
           for nazwa, czesc in (('bezwładność', bezw), ('pamięć', pamiec), ('sprężyna', spr), ('szum', eta))}
    from scipy import stats
    return dict(eta=e, acf=acf, K_reg=Kreg, K_volterra=Kv, vacf=Cvv[:nmax + 1] / max(Cvv[0], 1e-30),
                vacf_model=Cm / max(Cm[0], 1e-30), rozklad=rozklad, udzialy=udz, sigma=float(e.std()),
                kurtoza=float(stats.kurtosis(e.reshape(-1))), pasmo=2 / np.sqrt(len(e)))
