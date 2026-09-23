"""Kontrole dokumentu 45: algebra i układy kontrolne; bez danych i treningu.

SymPy sprawdza tożsamości. Przykłady numeryczne nie zastępują dowodów
uniwersalnych ani empirycznej kalibracji kowariancji na filmach.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp


CHECKS: list[dict] = []
SEED = 20260923


def record(name, method, scope, **values):
    CHECKS.append(dict(name=name, method=method, scope=scope, status="passed", **values))


def zero(x):
    assert sp.simplify(x) == 0, x


def symbolic():
    z, origin, d, jp, w, event, n = sp.symbols("z origin d jp w event n", real=True)
    r0 = z - origin - d - jp - w - event
    zero((z - (origin + d + jp + w + event + n)) - (r0 - n))
    record("dwie_reszty", "SymPy", "Dokładna tożsamość przed i po poprawce N.")

    b, sigma, c = sp.symbols("b sigma c", real=True)
    residuals = (b + sigma, b - sigma)
    zero(sum(e**2 for e in residuals) / 2 - b**2 - sigma**2)
    zero(sum(e**2 - (e - c)**2 for e in residuals) / 2 - (2*b*c - c**2))
    zero(2*b*c - c**2 - (b**2 - (b-c)**2))
    record("R1_R2", "SymPy", "Dokładne średnie dla symetrycznego szumu ±sigma; dowód ogólny w dokumencie.")

    mu, h, ref, noise = sp.symbols("mu h ref noise", real=True)
    difference = (mu+noise-h)**2 - (mu+noise-ref)**2
    zero(difference - ((mu-h)**2 - (mu-ref)**2 + 2*noise*(ref-h)))
    vals = [difference.subs({mu: 0, h: 0, ref: 1, noise: e}) for e in (-1, 1)]
    assert sum(vals) / 2 == -1 and vals[1] == 1
    record("R4_wybor_po_reszcie", "SymPy", "Wybór po wyniku odwraca znak pozornej korzyści.",
           unconditional_gain=-1, gain_selected_by_noise=1)

    p, r = sp.symbols("p r", positive=True)
    k = p / (p+r)
    posterior = (1-k)**2 * p + k**2*r
    zero(posterior - p*r/(p+r))
    zero(p-posterior - p**2/(p+r))
    record("K1_skalarny", "SymPy", "Postać Josepha i redukcja wariancji dla p,r>0.")

    # Całkowita wariancja mieszaniny modeli: dwa równie prawdopodobne parametry.
    m1, m2, v1, v2 = sp.symbols("m1 m2 v1 v2", real=True)
    mean = (m1+m2)/2
    total = (v1+m1**2+v2+m2**2)/2 - mean**2
    zero(total - (v1+v2)/2 - (m1-m2)**2/4)
    record("calkowita_wariancja", "SymPy", "Dokładna tożsamość dla mieszaniny dwóch modeli.")

    bias = sp.Matrix(sp.symbols("b0:2", real=True))
    factor = sp.Matrix(2, 2, sp.symbols("f0:4", real=True))
    aa, ab, bb = sp.symbols("aa ab bb", real=True)
    precision = sp.Matrix([[aa, ab], [ab, bb]])
    covariance = factor*factor.T
    moment = 0
    for u in (-1, 1):
        for v in (-1, 1):
            innovation = bias + factor*sp.Matrix([u, v])
            moment += (innovation.T*precision*innovation)[0] / 4
    zero(moment-sp.trace(precision*covariance)-(bias.T*precision*bias)[0])
    record("R3_znormalizowana_reszta", "SymPy",
           "Dokładna średnia formy kwadratowej w 2D, z korelacją: ślad iloczynu kowariancji i precyzji plus błąd średniej.")


def covariance_checks(rng):
    smallest = float("inf")
    worst = 0.0
    for _ in range(300):
        a = rng.normal(size=(5, 5))
        p = a @ a.T + .2*np.eye(5)
        h = rng.normal(size=(3, 5))
        b = rng.normal(size=(3, 3))
        r = b @ b.T + .1*np.eye(3)
        s = h@p@h.T + r
        k = np.linalg.solve(s, h@p).T
        residual_map = np.eye(5)-k@h
        posterior = residual_map@p@residual_map.T+k@r@k.T
        reduction = k@s@k.T
        err = float(np.max(np.abs(p-posterior-reduction)))
        worst = max(worst, err)
        assert np.allclose(p-posterior, reduction, atol=1e-10, rtol=1e-10)
        smallest = min(smallest, float(np.linalg.eigvalsh(p-posterior).min()))
        assert np.linalg.eigvalsh(posterior).min() > 0
        assert np.linalg.eigvalsh(p-posterior).min() > -1e-10
        task = rng.normal(size=(2, 5))
        assert np.trace(task@(p-posterior)@task.T) >= -1e-10
    record("K1_macierzowy", "NumPy", "300 układów SPD; postać Josepha, PSD redukcji, projekcja zadaniowa.",
           cases=300, maximum_identity_error=worst, minimum_reduction_eigenvalue=smallest)

    p = np.diag([100., 1.])
    h = np.array([[1., 0.]])
    s = h@p@h.T+np.array([[1.]])
    reduction = p@h.T@np.linalg.solve(s, h@p)
    task = np.array([[0., 1.]])
    assert np.trace(reduction) > 99
    assert float((task@reduction@task.T).item()) == 0
    record("tlo_bez_zadania", "NumPy", "Pomiar tła redukuje ogólną wariancję, nie wariancję zadania.",
           full_trace_reduction=float(np.trace(reduction)), task_reduction=0.)

    # Cztery identyczne błędy cząstek: uśrednienie nie redukuje wariancji.
    cross = np.ones((4, 4))
    weights = np.full(4, .25)
    full = float(weights @ cross @ weights)
    diagonal_only = float(weights @ np.diag(np.diag(cross)) @ weights)
    assert full == 1.0 and diagonal_only == .25
    record("wspolny_blad_czastek", "NumPy", "Pomijanie korelacji zaniża wariancję czterokrotnie.",
           correct_variance=full, incorrectly_independent_variance=diagonal_only)

    # Mała/zerowa kowariancja nie kontroluje systematycznego błędu średniej.
    errors = np.array([9., 11.])
    assert float(np.mean(errors**2)) == 101.
    assert float(np.var(errors)) == 1.
    record("obciazenie_w_K3", "NumPy", "Drugi moment to wariancja plus kwadrat średniej, nie sama wariancja.",
           second_moment=101., variance=1., squared_bias=100.)


def existing_implementation(repo):
    sys.path.insert(0, str(repo))
    from SmallViTAblation.pipeline.koopman_langevin.kalman_state import kalman

    params = np.array([2., 1., 0., 1.])
    smooth = np.tile(np.arange(20, dtype=float), (3, 1))
    jumps = smooth.copy()
    jumps[:, 5:] += np.array([100., -70., 500.])[:, None]
    _, v0, s0, _ = kalman(smooth, params)
    _, v1, s1, _ = kalman(jumps, params)
    assert np.array_equal(s0, s1)
    assert not np.allclose(v0, v1)
    record("K2_istniejacy_kod", "Wywołanie lokalnego kalman_state.kalman",
           "Zmiana treści pomiarów zmienia innowacje, lecz nie nominalną wariancję przy ustalonych parametrach.",
           observations_per_sequence=20, sequences=3)


def full_chain(rng):
    """Deterministyczne ograniczenia na zdarzeniu elipsoidalnym; nie estymacja częstości pokrycia."""
    worst_ratio = 0.
    for _ in range(100):
        particles = rng.normal(size=(4, 3))
        errors, radii = [], []
        for _p in range(4):
            a = rng.normal(size=(3, 3)) / 10
            cov = a@a.T + .001*np.eye(3)
            w = rng.normal(size=3)
            w *= 2 * rng.uniform() / np.linalg.norm(w)
            error = np.linalg.cholesky(cov)@w
            bound = 2*np.sqrt(np.linalg.eigvalsh(cov).max())
            assert np.linalg.norm(error) <= bound + 1e-12
            errors.append(error)
            radii.append(bound)
        errors = np.array(errors)
        weights = rng.dirichlet(np.ones(4))
        q = weights@particles
        qhat = weights@(particles-errors)
        feature_bound = float(weights@np.array(radii))
        assert np.linalg.norm(q-qhat) <= feature_bound + 1e-12

        model = rng.normal(size=(2, 3))
        target = rng.normal(size=2)
        d = np.outer(model@q-target, q)
        dhat = np.outer(model@qhat-target, qhat)
        radius = max(np.linalg.norm(q), np.linalg.norm(qhat))
        sensitivity = 2*np.linalg.norm(model, 2)*radius + np.linalg.norm(target)
        delta = sensitivity * feature_bound
        assert np.linalg.norm(d-dhat) <= delta + 1e-10

        xval = rng.normal(size=(7, 3))
        yval = rng.normal(size=(7, 2))
        def loss(w):
            return .5*np.square(xval@w.T-yval).sum()/len(xval)
        grad = (xval@model.T-yval).T@xval/len(xval)
        lipschitz = np.linalg.norm(xval, 2)**2/len(xval)
        eta = .01
        predicted = eta*float(np.sum(grad*dhat))
        bound = eta*np.linalg.norm(grad)*delta + lipschitz*eta**2/2*(np.linalg.norm(dhat)+delta)**2
        actual = loss(model)-loss(model-eta*d)
        discrepancy = abs(actual-predicted)
        assert discrepancy <= bound + 1e-10
        worst_ratio = max(worst_ratio, float(discrepancy/bound))
    record("K3_czastki_gradient_poprawa", "NumPy",
           "100 liniowych odczytów: elipsoida błędu → cząstki → gradient → przedział rzeczywistej poprawy; brak twierdzenia o kalibracji na obrazach.",
           cases=100, maximum_error_to_bound_ratio=worst_ratio)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve()
    repo = here.parents[3]
    document = repo / "SmallViTAblation/docs/Analiza_ponownej_publikacji/45_RESZTA_LANGEVINA_KALMAN_I_DOBOR_KLATEK.md"
    kalman_source = repo / "SmallViTAblation/pipeline/koopman_langevin/kalman_state.py"
    rng = np.random.default_rng(SEED)
    symbolic()
    covariance_checks(rng)
    existing_implementation(repo)
    full_chain(rng)
    report = dict(status="passed", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  seed=SEED, python=sys.version, numpy=np.__version__, sympy=sp.__version__,
                  checks_count=len(CHECKS), checks=CHECKS,
                  files={str(p): sha256(p) for p in (here, document, kalman_source)},
                  scope="Algebra i kontrolowane układy; bez treningu, filmów i nowej formalizacji Lean.")
    output = args.out or here.with_name("RESZTA_KALMAN_WERYFIKACJA.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASSED: {len(CHECKS)} checks; {output}")


if __name__ == "__main__":
    main()
