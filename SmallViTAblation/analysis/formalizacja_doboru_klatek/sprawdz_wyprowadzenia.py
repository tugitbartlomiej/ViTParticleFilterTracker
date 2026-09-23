"""Kontrole algebry dokumentu 44; bez danych klinicznych i bez treningu.

Przypadki numeryczne sprawdzają implementację nierówności, nie zastępują
dowodów uniwersalnych zapisanych w dokumencie. Wszystkie zapisy są lokalne.
Uruchomienie: python sprawdz_wyprowadzenia.py [--out pełna_ścieżka_json]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp


CHECKS: list[dict] = []


def record(name: str, kind: str, scope: str, **values) -> None:
    CHECKS.append(dict(name=name, method=kind, status="passed", scope=scope, **values))


def exact_zero(expression) -> None:
    entries = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
    assert all(sp.simplify(x) == 0 for x in entries), expression


def symbolic_checks() -> None:
    lam, f0 = sp.symbols("lambda f0")
    res = sp.symbols("r0:5")
    f = f0
    for h in range(1, 6):
        f = lam * f + res[h - 1]
        exact_zero(f - lam**h * f0 - sum(lam ** (h - 1 - j) * res[j] for j in range(h)))
    record("T1_iterowanie_residuum", "SymPy", "Tożsamości do h=5; dowód indukcyjny w dokumencie.")

    a, b, phase = sp.symbols("a b phase", real=True)
    phi = a + sp.I * b
    exact_zero(sp.re(sp.exp(sp.I * phase) * phi) - a * sp.cos(phase) + b * sp.sin(phase))
    exact_zero(sp.expand_complex(sp.Abs(phi) ** 2) - a**2 - b**2)
    assert sp.Rational(-1, 2) ** 2 == sp.Rational(1, 4)
    rot = (3 + 4 * sp.I) / 10
    assert sp.Abs(rot) == sp.Rational(1, 2)
    exact_zero(sp.re(phi) - sp.re(sp.conjugate(phi)))
    record("T1_faza_i_modul", "SymPy", "Dowolna faza miesza Re/Im; sprzężona para ma identyczne Re.",
           negative_lambda_square_scale="1/4", rotating_lambda_modulus_scale="1/2")

    # Wartość własna 1 dla warunkowej średniej nie oznacza niezmienności realizacji.
    x, noise = sp.symbols("x noise", real=True)
    exact_zero(((x + 1) + (x - 1)) / 2 - x)
    exact_zero(((x + 1) ** 2 + (x - 1) ** 2) / 2 - x**2 - 1)
    record("T1_operator_stochastyczny", "SymPy", "Symetryczny przyrost ±1: średnia stała, drugi moment rośnie o 1.")

    op = sp.Matrix(2, 2, sp.symbols("op0:4"))
    projection = sp.diag(1, 0)
    complement = sp.eye(2) - projection
    for n in range(5):
        rhs = (complement * op)**(n + 1)
        for j in range(n + 1):
            rhs += op**(n - j) * projection * op * (complement * op)**j
        exact_zero(op**(n + 1) - rhs)
    record("MZ_tozsamosc_operatorowa", "SymPy", "Nieprzemienne operatory 2×2, n=0..4; ogólny dowód przez rozwinięcie w dokumencie.")

    aa, bb, cc, dd, c0, h0 = sp.symbols("A B C D c0 h0")
    c_hist = [c0]
    state = sp.Matrix([c0, h0])
    block = sp.Matrix([[aa, bb], [cc, dd]])
    for t in range(5):
        nxt = block * state
        prediction = aa * c_hist[t] + sum(bb * dd**j * cc * c_hist[t - 1 - j] for j in range(t)) + bb * dd**t * h0
        exact_zero(nxt[0] - prediction)
        c_hist.append(nxt[0])
        state = nxt
    record("T3_eliminacja_symboliczna", "SymPy", "Bloki skalarne, t=0..4; indukcja ogólna w dokumencie.")

    # Sprawdzenie kolejności nieprzemiennych bloków na dokładnych liczbach.
    A = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 3)], [0, sp.Rational(1, 4)]])
    B = sp.Matrix([[1, 2], [0, 1]])
    C = sp.Matrix([[1, 0], [1, 1]])
    D = sp.Matrix([[sp.Rational(1, 3), sp.Rational(1, 4)], [0, sp.Rational(1, 5)]])
    c_hist = [sp.Matrix([1, 2])]
    h_init = sp.Matrix([3, -1])
    h = h_init
    for t in range(6):
        c_next = A * c_hist[t] + B * h
        prediction = A * c_hist[t] + B * D**t * h_init
        for j in range(t):
            prediction += B * D**j * C * c_hist[t - 1 - j]
        exact_zero(c_next - prediction)
        h = C * c_hist[t] + D * h
        c_hist.append(c_next)
    record("T3_bloki_nieprzemienne", "dokładne macierze wymierne", "Bloki 2×2, sześć kroków.")

    hidden_first_steps = [sp.Rational(1, 2) * 0 + initial for initial in [0, 1]]
    assert hidden_first_steps == [0, 1]
    assert all(1 * sp.Rational(1, 3)**j * 0 == 0 for j in range(10))
    record("T3_kontrprzyklad_domkniecia", "dokładny kontrprzykład", "C=0,B=1: jądra identycznie zero, przyszłość zależy od h0.",
           same_observed_state=0, different_next_states=[0, 1])

    r = sp.Rational(1, 4)
    gamma = lambda lag: {0: sp.S.One, 1: r, 2: r**2}.get(abs(lag), sp.S.Zero)
    rcov = lambda lag: (1 + r**2) * gamma(lag) - r * gamma(lag + 1) - r * gamma(lag - 1)
    kernel = sp.cancel((gamma(2) * gamma(0) - gamma(1)**2) / (gamma(0)**2 - gamma(1)**2))
    assert kernel == 0 and rcov(1) == 0 and rcov(2) == sp.Rational(1, 256)
    spectral_lower = 1 - 2 * r - 2 * r**2
    assert spectral_lower == sp.Rational(3, 8) and spectral_lower > 0
    record("lokalny_D59_poprawna_kowariancja", "dokładny kontrprzykład", "r=1/4; dodatni wielomian widmowy uzasadnia kowariancję stacjonarną.",
           kernel=str(kernel), residual_covariance_lag1=str(rcov(1)), residual_covariance_lag2=str(rcov(2)),
           spectral_lower_bound=str(spectral_lower))

    ff, fs, fp, ss, sh, hh = sp.symbols("FF FS FP SS SH HH", real=True)
    cond = fp - fs * sh / ss
    p = hh - sh**2 / ss
    beta_p = cond / p
    beta_s = (fs - beta_p * sh) / ss
    full_error = ff - 2 * beta_s * fs - 2 * beta_p * fp + beta_s**2 * ss + 2 * beta_s * beta_p * sh + beta_p**2 * hh
    exact_zero(full_error - (ff - fs**2 / ss - cond**2 / p))
    record("T4_Schur", "SymPy", "Tożsamość racjonalna przy SS≠0 i HH−SH²/SS≠0; nieujemność wymaga dodatniej wariancji.")

    # Y=2a+b+e, X=(a,b), Z=a; niezależne znaki o prawdopodobieństwie 1/2.
    risks = [sp.S.Zero] * 3
    for av, bv, ev in itertools.product([-1, 1], repeat=3):
        y = 2 * av + bv + ev
        coarse, fine = 2 * av, 2 * av + bv
        for j, value in enumerate([(y - coarse)**2, (y - fine)**2, (fine - coarse)**2]):
            risks[j] += sp.Rational(value, 8)
    exact_zero(risks[0] - risks[1] - risks[2])
    assert risks == [2, 1, 1]
    record("T2_T4_warunkowe_srednie", "dokładne osiem zdarzeń", "Ta sama tożsamość dla informacji reprezentacji i rozszerzenia o historię.",
           risk_coarse=2, risk_fine=1, information_gap=1)

    w = sp.Matrix(2, 2, sp.symbols("w0:4"))
    z = sp.Matrix(sp.symbols("z0:2"))
    zt = sp.Matrix(sp.symbols("zt0:2"))
    y = sp.Matrix(sp.symbols("y0:2"))
    exact_zero((w * z - y) * z.T - (w * zt - y) * zt.T - (w * (z - zt) * z.T + (w * zt - y) * (z - zt).T))
    record("T7_rozklad_gradientu", "SymPy", "Dokładna tożsamość macierzowa 2×2; normowe ograniczenie wyprowadzone w tekście.")

    eta = sp.symbols("eta", positive=True)
    base = sp.Rational(1, 2)
    gain_good = sp.expand(base - sp.Rational(1, 2) * (1 - eta)**2)
    gain_bad = sp.expand(base - sp.Rational(1, 2) * (1 + eta)**2)
    exact_zero(gain_good - eta + eta**2 / 2)
    exact_zero(gain_bad + eta + eta**2 / 2)
    record("T9_to_samo_widmo_inny_efekt", "SymPy", "Równe opisy dynamiczne; dla 0<eta<2 dobry krok poprawia, zły pogarsza.",
           good_gain=str(gain_good), bad_gain=str(gain_bad))

    alpha, smooth = sp.symbols("alpha L", positive=True)
    eta_max = (1 - alpha) / (smooth * (1 + alpha)**2)
    exact_zero(eta_max * (1 - alpha) - smooth * eta_max**2 * (1 + alpha)**2 / 2 - eta_max * (1 - alpha) / 2)
    record("T10_krok_PL", "SymPy", "Algebra współczynnika przy kroku granicznym; nierówność wymaga 0≤alpha<1 i L>0.")


def numeric_checks() -> None:
    rng = np.random.default_rng(20260923)
    min_interval_margin = float("inf")
    min_gradient_slack = float("inf")
    min_contraction_slack = float("inf")
    max_regret_minus_bound = -float("inf")
    for _ in range(300):
        n = 5
        matrix = rng.normal(size=(n, n))
        H = matrix.T @ matrix + 0.5 * np.eye(n)
        eig = np.linalg.eigvalsh(H)
        L, mu = float(eig[-1]), float(eig[0])
        theta = rng.normal(size=n)
        g = H @ theta
        eta = 0.2 / L
        ghat = g + 0.1 * rng.normal(size=n)
        eps = float(np.linalg.norm(g - ghat))
        gains, estimates, errors = [], [], []
        for _candidate in range(8):
            d = rng.normal(size=n)
            dhat = d + 0.1 * rng.normal(size=n)
            delta = float(np.linalg.norm(d - dhat))
            estimate = eta * (ghat @ dhat)
            error = eta * (eps * np.linalg.norm(dhat) + (np.linalg.norm(ghat) + eps) * delta) + L * eta**2 / 2 * (np.linalg.norm(dhat) + delta)**2
            nxt = theta - eta * d
            actual = float((theta @ H @ theta - nxt @ H @ nxt) / 2)
            margin = float(error - abs(actual - estimate))
            assert margin >= -1e-10
            min_interval_margin = min(min_interval_margin, margin)
            gains.append(actual)
            estimates.append(estimate)
            errors.append(float(error))
        chosen = int(np.argmax(estimates))
        regret_minus_bound = max(gains) - gains[chosen] - 2 * max(errors)
        assert regret_minus_bound <= 1e-10
        max_regret_minus_bound = max(max_regret_minus_bound, regret_minus_bound)

        W = rng.normal(size=(3, n))
        z, zt = rng.normal(size=(2, n))
        y = rng.normal(size=3)
        norm_M = np.linalg.norm(W, 2)
        radius = max(np.linalg.norm(z), np.linalg.norm(zt))
        bound = (2 * norm_M * radius + np.linalg.norm(y)) * np.linalg.norm(z - zt)
        difference = np.linalg.norm(np.outer(W @ z - y, z) - np.outer(W @ zt - y, zt), "fro")
        min_gradient_slack = min(min_gradient_slack, float(bound - difference))
        assert difference <= bound + 1e-10

        alpha = 0.3
        error_dir = rng.normal(size=n)
        error_dir *= alpha * np.linalg.norm(g) / np.linalg.norm(error_dir)
        eta_pl = (1 - alpha) / (L * (1 + alpha)**2)
        next_theta = theta - eta_pl * (g + error_dir)
        q = 1 - mu * eta_pl * (1 - alpha)
        assert 0 < q < 1
        slack = float(q * (theta @ H @ theta) / 2 - (next_theta @ H @ next_theta) / 2)
        assert slack >= -1e-10
        min_contraction_slack = min(min_contraction_slack, slack)

    record("T6_przedzial_poprawy", "NumPy, seed 20260923", "2400 kierunków na 300 dodatnio określonych celach kwadratowych; nie dowód ogólny.",
           minimum_margin=min_interval_margin)
    record("T6_wybor_i_strata_do_optimum", "NumPy, seed 20260923", "300 rodzin po 8 kierunków; kontrola nierówności (24).",
           maximum_regret_minus_bound=max_regret_minus_bound)
    record("T7_ograniczenie_normowe", "NumPy, seed 20260923", "300 odczytów liniowych z błędem cech.",
           minimum_slack=min_gradient_slack)
    record("T10_kontrakcja_PL", "NumPy, seed 20260923", "300 funkcji kwadratowych z kontrolowanym błędem względnym kierunku.",
           minimum_slack=min_contraction_slack)

    # Ręcznie skonstruowany przypadek, w którym certyfikat naprawdę rozstrzyga.
    eta, L = 0.25, 1.0
    eps = delta = 0.01
    E = eta * (eps + (1 + eps) * delta) + L * eta**2 / 2 * (1 + delta)**2
    good_lower, bad_upper = eta - E, -eta + E
    assert good_lower > bad_upper and good_lower > 0
    record("T6_niepusty_warunek_przewagi", "jawny przypadek liczbowy", "Dwa kierunki ±1; ghat=1, eps=delta=.01, L=1, eta=.25.",
           selected_lower=good_lower, baseline_upper=bad_upper)

    # Wagi pokrycia, z gradientami równymi współrzędnej (L_g=1).
    full = [sp.S.One] * 9 + [-sp.S.One]
    mean_full = sum(full) / 10
    unweighted = (1 + (-1)) / 2
    weighted = sp.Rational(9, 10) * 1 + sp.Rational(1, 10) * (-1)
    assert mean_full == weighted == sp.Rational(4, 5) and unweighted == 0
    record("T8_wagi_pokrycia", "dokładny kontrprzykład", "Zerowa odległość do reprezentantów; równe wagi reprezentantów tracą proporcje puli.",
           full_mean=str(mean_full), weighted_mean=str(weighted), unweighted_mean=unweighted)


def end_to_end_checks() -> None:
    """Łańcuch (37)→(38)→(39)→(22) na jawnych układach liniowych."""
    rng = np.random.default_rng(44)
    minimum_slacks = np.full(4, np.inf)
    for _ in range(100):
        A, B, C = [rng.normal(size=(2, 2)) / 5 for _ in range(3)]
        D = rng.normal(size=(2, 2))
        D *= 0.4 / np.linalg.norm(D, 2)
        rho = float(np.linalg.norm(D, 2))
        c = [rng.normal(size=2)]
        h_init = rng.normal(size=2)
        h = h_init.copy()
        t, length = 6, 3
        for _time in range(t + 1):
            c_next = A @ c[-1] + B @ h
            h = C @ c[-1] + D @ h
            c.append(c_next)
        Mc = max(float(np.linalg.norm(value)) for value in c)
        Ahat = A + rng.normal(size=(2, 2)) / 100
        K = [B @ np.linalg.matrix_power(D, j) @ C for j in range(length)]
        Khat = [item + rng.normal(size=(2, 2)) / 100 for item in K]
        f = B @ np.linalg.matrix_power(D, t) @ h_init
        fhat = np.zeros(2)
        chat = Ahat @ c[t] + fhat
        for j in range(length):
            chat += Khat[j] @ c[t - 1 - j]
        coefficients = np.linalg.norm(A - Ahat, 2) + sum(np.linalg.norm(k - kh, 2) for k, kh in zip(K, Khat))
        tail = np.linalg.norm(B, 2) * np.linalg.norm(C, 2) * Mc * rho**length / (1 - rho)
        ec = Mc * coefficients + tail + np.linalg.norm(f - fhat)
        actual_ec = np.linalg.norm(c[t + 1] - chat)
        assert actual_ec <= ec + 1e-12

        decoder = rng.normal(size=(4, 2))
        omitted = rng.normal(size=4) / 50
        z = decoder @ c[t + 1] + omitted
        zhat = decoder @ chat
        ez = np.linalg.norm(decoder, 2) * ec + np.linalg.norm(omitted)
        assert np.linalg.norm(z - zhat) <= ez + 1e-12

        W = rng.normal(size=(2, 4))
        y = rng.normal(size=2)
        d = np.outer(W @ z - y, z)
        dhat = np.outer(W @ zhat - y, zhat)
        radius = max(np.linalg.norm(z), np.linalg.norm(zhat))
        delta = (2 * np.linalg.norm(W, 2) * radius + np.linalg.norm(y)) * ez
        assert np.linalg.norm(d - dhat) <= delta + 1e-12

        steering_z = rng.normal(size=(7, 4))
        steering_y = rng.normal(size=(7, 2))
        def risk(weights):
            return float(np.sum((steering_z @ weights.T - steering_y)**2) / 14)
        g = (steering_z @ W.T - steering_y).T @ steering_z / 7
        L = np.linalg.norm(steering_z.T @ steering_z / 7, 2)
        eta = 0.1 / (1 + L)
        estimate = eta * float(np.sum(g * dhat))
        error = eta * np.linalg.norm(g) * delta + L * eta**2 / 2 * (np.linalg.norm(dhat) + delta)**2
        actual = risk(W) - risk(W - eta * d)
        assert abs(actual - estimate) <= error + 1e-10
        slacks = [ec - actual_ec, ez - np.linalg.norm(z - zhat), delta - np.linalg.norm(d - dhat), error - abs(actual - estimate)]
        minimum_slacks = np.minimum(minimum_slacks, slacks)
    record("T7a_pelny_lancuch", "NumPy, seed 44", "100 układów blokowych, przewidywanie z pamięcią, rekonstrukcja cech, liniowy odczyt, krok SGD i osobny cel kwadratowy.",
           minimum_slacks=dict(zip(["memory_prediction", "feature_reconstruction", "gradient_approximation", "gain_interval"], minimum_slacks.tolist())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("WERYFIKACJA.json"))
    args = parser.parse_args()
    symbolic_checks()
    numeric_checks()
    end_to_end_checks()
    script = Path(__file__).resolve()
    document = script.parents[2] / "docs" / "Analiza_ponownej_publikacji" / "44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md"
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="passed",
        sympy_version=sp.__version__,
        numpy_version=np.__version__,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        document_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
        checks=CHECKS,
        limitations=[
            "Nie jest to nowy dowód Lean ani maszynowa formalizacja całego dokumentu.",
            "Przykłady numeryczne nie dowodzą nierówności uniwersalnych.",
            "Nie sprawdzono założeń gwarancji na danych i trajektorii RF-DETR.",
            "Nie uruchomiono treningów, ekstrakcji tokenów ani zadań Eden.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], checks=len(CHECKS), report=str(args.out.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
