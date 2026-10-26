"""Miary rozkładu Schura (Keylock 2018) i niezmienników Q, R dla wsadu macierzy rzeczywistych n × n (n = 192 albo 3).
Moduł bez torcha: importowany przez procesy robocze puli (Windows spawn), liczy wszystko w numpy/LAPACK na procesorze.

Dla macierzy J (n × n):
  Schur zespolony J = Z T Z^H, T = Λ + N (Λ = diag T = wartości własne, N ściśle górnotrójkątna); LAPACK zgees bez wektorów Schura.
  ‖J‖_F² = ‖Λ‖_F² + ‖N‖_F² (Z unitarna); ‖N‖_F = odchylenie od normalności Henriciego.
  S = (J + Jᵀ)/2, Ω = (J − Jᵀ)/2.
  A = J − (tr J / n) I (część bezśladowa), μ_i = λ_i − tr J / n, S_A = (A + Aᵀ)/2 (Ω_A = Ω).
  Q = −½ tr(A²), R = −⅓ tr(A³), Q_S = −½ ‖S_A‖_F², Q_Ω = +½ ‖Ω‖_F² (Q = Q_S + Q_Ω).
  Q_Λ = −½ Re Σ μ_i², R_Λ = −⅓ Re Σ μ_i³ (tożsamościowo równe Q, R, bo tr(A^k) = Σ μ_i^k; różnica = błąd numeryczny).
  Rozbicie Keylocka (w bazie Schura, J rzeczywista ⇒ J^H = Jᵀ):
    ‖S_A‖_F² = Σ (Re μ_i)² + ½ ‖N‖_F²,   ‖Ω‖_F² = Σ (Im μ_i)² + ½ ‖N‖_F²
    Q_S = Q_S^Λ + Q_S^N,  Q_S^Λ = −½ Σ (Re μ_i)²,  Q_S^N = −¼ ‖N‖_F²
    Q_Ω = Q_Ω^Λ + Q_Ω^N,  Q_Ω^Λ = +½ Σ (Im μ_i)²,  Q_Ω^N = +¼ ‖N‖_F²   (Q^N = Q_S^N + Q_Ω^N = 0)
"""
import numpy as np
from scipy.linalg import lapack

TOL_WZGL = 1e-6          # próg „zera” dla wartości własnych: |·| ≤ TOL_WZGL · max(‖J‖_F, 1); dwa kierunki zerowe LayerNormu (1 i x̂) dają |λ| ~ 1e-16 i ~ 1e-8, następne ≥ 1e-4

# nazwy miar zwracanych przez miary_wsad (kolejność kolumn)
NAZWY = ['nJ', 'nLam', 'nN', 'nN_henrici', 'N_do_J', 'kappa', 'nS', 'nOm', 'Om_do_S', 'nSA',
         'trJ', 'Q', 'R', 'Q_S', 'Q_Om', 'Q_Lam', 'R_Lam', 'Q_minus_QLam', 'R_minus_RLam', 'QS_plus_QOm_minus_Q',
         'Q_S_Lam', 'Q_S_N', 'Q_Om_Lam', 'Q_Om_N', 'udzN_w_S2', 'udzN_w_Om2', 'resid_QS_keylock',
         'n_re_dod', 'n_re_uj', 'n_zero', 'n_im', 'max_re', 'rho', 'max_im', 'n_mod_gt1', 'n_1pl_gt1', 'max_mod_1pl',
         'norm2_J', 'norm2_IpJ']


def schur_T(M):
    """Zespolona postać Schura T (bez Z): zgees(jobvs='N')."""
    n = M.shape[0]
    T, sdim, w, vs, work, info = lapack.zgees(lambda x: None, M.astype(np.complex128), compute_v=0, sort_t=0,
                                              lwork=max(64 * n, 1), overwrite_a=0)
    if info != 0:
        raise RuntimeError(f'zgees info={info}')
    return T


def miary_jednej(M):
    """Słownik miar (NAZWY) i wartości własne λ (n,) dla jednej macierzy M (n × n, float64)."""
    n = M.shape[0]
    T = schur_T(M)
    lam = np.diag(T).copy()
    Nn = np.linalg.norm(np.triu(T, 1))
    nJ = np.linalg.norm(M)
    nLam = np.linalg.norm(lam)
    S = 0.5 * (M + M.T); Om = 0.5 * (M - M.T)
    tr = float(np.trace(M))
    A = M - (tr / n) * np.eye(n)
    SA = 0.5 * (A + A.T)
    A2 = A @ A
    Q = -0.5 * float(np.trace(A2)); R = -float(np.trace(A2 @ A)) / 3.0
    nSA = np.linalg.norm(SA); nOm = np.linalg.norm(Om); nS = np.linalg.norm(S)
    QS = -0.5 * nSA ** 2; QOm = 0.5 * nOm ** 2
    mu = lam - tr / n
    QL = -0.5 * float(np.real(np.sum(mu ** 2))); RL = -float(np.real(np.sum(mu ** 3))) / 3.0
    QSL = -0.5 * float(np.sum(mu.real ** 2)); QSN = -0.25 * Nn ** 2
    QOL = 0.5 * float(np.sum(mu.imag ** 2)); QON = 0.25 * Nn ** 2
    tol = TOL_WZGL * max(nJ, 1.0)
    re, im, mod = lam.real, lam.imag, np.abs(lam)
    m1 = np.abs(1.0 + lam)
    sv = np.linalg.svd(M, compute_uv=False)
    sv1 = np.linalg.svd(np.eye(n) + M, compute_uv=False)
    r = dict(nJ=nJ, nLam=nLam, nN=Nn, nN_henrici=np.sqrt(max(nJ ** 2 - float(np.sum(mod ** 2)), 0.0)),
             N_do_J=Nn / nJ, kappa=Nn / max(nLam, 1e-300), nS=nS, nOm=nOm, Om_do_S=nOm / max(nS, 1e-300), nSA=nSA,
             trJ=tr, Q=Q, R=R, Q_S=QS, Q_Om=QOm, Q_Lam=QL, R_Lam=RL, Q_minus_QLam=Q - QL, R_minus_RLam=R - RL,
             QS_plus_QOm_minus_Q=QS + QOm - Q,
             Q_S_Lam=QSL, Q_S_N=QSN, Q_Om_Lam=QOL, Q_Om_N=QON,
             udzN_w_S2=0.5 * Nn ** 2 / max(nSA ** 2, 1e-300), udzN_w_Om2=0.5 * Nn ** 2 / max(nOm ** 2, 1e-300),
             resid_QS_keylock=(QSL + QSN) - QS,
             n_re_dod=int((re > tol).sum()), n_re_uj=int((re < -tol).sum()), n_zero=int((mod <= tol).sum()),
             n_im=int((np.abs(im) > tol).sum()), max_re=float(re.max()), rho=float(mod.max()), max_im=float(np.abs(im).max()),
             n_mod_gt1=int((mod > 1.0 + tol).sum()),
             n_1pl_gt1=int((2.0 * re + mod ** 2 > 2.0 * tol).sum()),     # |1+λ|² − 1 = 2 Re λ + |λ|² > 0
             max_mod_1pl=float(m1.max()), norm2_J=float(sv[0]), norm2_IpJ=float(sv1[0]))
    return r, lam


def miary_wsad(Ms):
    """Ms (k, n, n) float64 → (tablica (k, len(NAZWY)) float64, λ (k, n) complex128)."""
    import os
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    Ms = np.asarray(Ms, np.float64)
    out = np.empty((Ms.shape[0], len(NAZWY)), np.float64)
    lams = np.empty(Ms.shape[:2], np.complex128)
    for i, M in enumerate(Ms):
        r, lam = miary_jednej(M)
        out[i] = [r[k] for k in NAZWY]
        lams[i] = lam
    return out, lams


def klasa_qr3(Q, R):
    """Klasyczny diagram Q–R (Chong, Perry & Cantwell 1990) dla bezśladowej części 3 × 3: Δ = 27R²/4 + Q³.
    0 = Δ>0, R<0: ognisko stabilne / rozciąganie (SFS); 1 = Δ>0, R>0: ognisko niestabilne / ściskanie (UFC);
    2 = Δ<0, R<0: węzeł niestabilny / siodło / siodło (UNSS); 3 = Δ<0, R>0: węzeł stabilny / siodło / siodło (SNSS)."""
    Q = np.asarray(Q); R = np.asarray(R)
    D = 27.0 * R ** 2 / 4.0 + Q ** 3
    k = np.where(D > 0, np.where(R < 0, 0, 1), np.where(R < 0, 2, 3))
    return D, k
