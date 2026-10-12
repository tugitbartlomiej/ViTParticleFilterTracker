"""Kandydackie kierunki nadzorowane i afiniczny model Koopmana, tylko NumPy.

Python 3.11. Import nie ładuje detektora, generatora ani bibliotek treningowych.
Granice ``time_stop`` i ``fit_stop`` są WYŁĄCZNE, w indeksach checkpointów.
Etykiety zwykle mają kolejność: presence, sinangle, cosangle, logscale,
length_factor, background_brightness. Ich współliniowość zmniejsza wymiar bazy.
Kierunki są kandydacką podprzestrzenią, bez dowodu kompletnej tożsamości obiektu.
Gotowe features[...,1536] dostarcza callback czastki.descriptor_torch:
raw post-LN, foreground mean, background mean oraz centered-RMS first x/y.
Ten moduł nie konstruuje deskryptora i nie zmienia definicji jego momentów.

Przykład (tablice i podział grup przygotowuje wywołujący)::

    basis = fit_directions(features, labels, fit_mask, time_stop=6)
    c = project(features, basis)
    report = validate(c, steps, fit_mask, fit_stop=6, ridge=1e-3,
                      image_groups=operation_ids, horizons=(1, 3))
    model = report["model"]
    target = forecast(model, c[5], horizon=3)
    # c[5] + delta_K jest celem; sama delta_K nie jest pełnym kodem obrazu.
    delta_K = target - c[5]
    # json.dumps(to_jsonable(report), ensure_ascii=False, allow_nan=False)

Walidacja zakłada bazę zamrożoną przed fit_stop. Nie może wykryć przecieku
w już przygotowanym c; dlatego powyżej time_stop nie przekracza fit_stop.
Hiperparametry, horyzonty i grupy należy ustalić przed oceną odłożonej historii.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


__all__ = ["fit_directions", "project", "predict_labels", "predict_labels_from_codes",
           "evaluate_directions", "fit_koopman", "forecast", "validate",
           "safe_cosine", "compare_arms", "to_jsonable"]


def _array(value: Any, name: str, ndim: int | None = None,
           *, finite: bool = True) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name}: wymagane liczby rzeczywiste.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: wymagana tablica liczb.") from exc
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name}: wymagane {ndim} wymiary, otrzymano {result.shape}.")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name}: NaN i nieskończoności nie są dopuszczalne.")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}: wymagana liczba całkowita.")
    if value < minimum:
        raise ValueError(f"{name}: minimum to {minimum}.")
    return int(value)


def _nonnegative(value: float, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name}: wymagana skończona {'dodatnia' if positive else 'nieujemna'} liczba.")
    return result


def _mask(value: Any, n: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.shape != (n,) or not result.any():
        raise ValueError(f"{name}: wymagana niepusta maska bool o kształcie ({n},).")
    return result


def _history(value: Any, mask: Any, stop: int, name: str,
             minimum: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    # Sprawdzamy wartości wyłącznie w wycinku dopasowania. Przyszłość i obrazy
    # odłożone mogą być jeszcze niedostępne (np. wypełnione NaN).
    data = _array(value, name, 3, finite=False)
    if min(data.shape) == 0:
        raise ValueError(f"{name}: osie T, N i D muszą być niepuste.")
    selected = _mask(mask, data.shape[1], "fit_mask")
    stop = _integer(stop, "stop", minimum)
    if stop > len(data):
        raise ValueError("stop przekracza liczbę checkpointów.")
    fit = _array(data[:stop, selected], "historia dopasowania", 3)
    return data, selected, stop, fit


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    tolerance = np.finfo(np.float64).eps * np.maximum(1., np.abs(mean))
    scale = np.where(scale > tolerance, scale, 1.)
    return mean, scale


def _numerical_rank(s: np.ndarray, shape: tuple[int, int]) -> int:
    if not s.size or s[0] == 0:
        return 0
    return int(np.count_nonzero(s > s[0] * max(shape) * np.finfo(float).eps))


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Rozwiązuje mean(||XB-Y||²) + ridge*||B||² bez karania wyrazu wolnego."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    if ridge == 0:
        weights = np.zeros_like(s)
        active = _numerical_rank(s, x.shape)
        weights[:active] = 1. / s[:active]
    else:
        weights = s / (s * s + len(x) * ridge)
    return vt.T @ (weights[:, None] * (u.T @ y))


def fit_directions(features: np.ndarray, labels: np.ndarray, fit_mask: np.ndarray,
                   time_stop: int, rank: int = 8, *, ridge: float = 1e-3) -> dict[str, Any]:
    """Dopasowuje kandydacką bazę do features[T,N,D] i etykiet labels[N,Q].

    Używa tylko features[:time_stop,fit_mask] i labels[fit_mask]. Cechy oraz
    etykiety standaryzuje na dopuszczonych danych. Regresja X->Y ma karę
    mean(||XB-Y||²)+ridge*||B||². Lewa baza SVD macierzy B jest ortonormalna
    w przestrzeni STANDARYZOWANYCH cech. Wymiar <= min(rank,8,rank(Y),rank(B)).
    Stałe etykiety/cechy nie tworzą nowych kierunków; brak kierunków to błąd.
    D=1536 jest układem produkcyjnym, inne D są dozwolone do analiz i testów.
    Słownik zawiera mean[D], scale[D], projection[D,d] i metadane dopasowania.
    """
    data, mask, stop, fit = _history(features, fit_mask, time_stop, "features", 1)
    rank = _integer(rank, "rank", 1)
    ridge = _nonnegative(ridge, "ridge")
    labels = _array(labels, "labels", 2, finite=False)
    if labels.shape[0] != data.shape[1] or labels.shape[1] == 0:
        raise ValueError("labels: wymagany kształt [N,Q], Q>0.")
    label_fit = _array(labels[mask], "etykiety dopasowania", 2)
    label_mean, label_scale = _standardize(label_fit)
    y = (label_fit - label_mean) / label_scale
    label_rank = _numerical_rank(np.linalg.svd(y, compute_uv=False), y.shape)
    if label_rank == 0:
        raise ValueError("Brak zmienności etykiet w części dopasowania.")
    x = fit.reshape(-1, fit.shape[-1])
    mean, scale = _standardize(x)
    coefficients = _ridge_coefficients((x - mean) / scale, np.tile(y, (stop, 1)), ridge)
    u, s, _ = np.linalg.svd(coefficients, full_matrices=False)
    effective_rank = min(rank, 8, label_rank, _numerical_rank(s, coefficients.shape))
    if effective_rank == 0:
        raise ValueError("Cechy nie wyznaczają żadnego nadzorowanego kierunku.")
    projection = u[:, :effective_rank].copy()
    # Stała konwencja znaku; obrót w zdegenerowanej podprzestrzeni nadal nie
    # nadaje osiom jednoznacznego znaczenia semantycznego.
    pivots = np.argmax(np.abs(projection), axis=0)
    projection *= np.where(projection[pivots, np.arange(effective_rank)] < 0, -1., 1.)
    return {"mean": mean, "scale": scale, "projection": projection,
            "label_mean": label_mean, "label_scale": label_scale,
            "coefficients": coefficients,
            "projected_coefficients": projection.T @ coefficients, "singular_values": s,
            "label_rank": label_rank, "rank": effective_rank,
            "time_stop": stop, "fit_indices": np.flatnonzero(mask),
            "ridge": ridge, "status": "candidate"}


def _metric_scale(scale: Any, d: int) -> np.ndarray:
    result = np.ones(d) if scale is None else _array(scale, "scale", 1)
    if result.shape != (d,) or np.any(result <= 0):
        raise ValueError("scale musi zawierać D dodatnich, skończonych liczb.")
    return result


def project(features: np.ndarray, basis: Mapping[str, Any]) -> np.ndarray:
    """Oblicza ((features-mean)/scale) @ projection; zachowuje osie przed D."""
    features = _array(features, "features")
    mean = _array(basis["mean"], "mean", 1)
    scale = _metric_scale(basis["scale"], len(mean))
    projection = _array(basis["projection"], "projection", 2)
    if (features.ndim < 1 or features.shape[-1] != len(mean)
            or projection.shape[0] != len(mean) or projection.shape[1] == 0):
        raise ValueError("Niezgodne wymiary cech i bazy.")
    return ((features - mean) / scale) @ projection


def _label_parameters(basis: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficients = _array(basis["coefficients"], "coefficients", 2)
    mean = _array(basis["label_mean"], "label_mean", 1)
    scale = _metric_scale(basis["label_scale"], len(mean))
    if coefficients.shape[1] != len(mean):
        raise ValueError("Niezgodne wymiary współczynników i etykiet bazy.")
    return coefficients, mean, scale


def predict_labels(features: np.ndarray, basis: Mapping[str, Any]) -> np.ndarray:
    """Odczyt pełnego ridge z features[...,D] do etykiet [...,Q], bez refit.

    Wyniki są w jednostkach oryginalnych etykiet. Odczyt presence jest wynikiem
    regresji, nie skalibrowanym prawdopodobieństwem; nie obcinamy go do [0,1].
    """
    features = _array(features, "features")
    mean = _array(basis["mean"], "mean", 1)
    scale = _metric_scale(basis["scale"], len(mean))
    coefficients, label_mean, label_scale = _label_parameters(basis)
    if (features.ndim < 1 or features.shape[-1] != len(mean)
            or coefficients.shape[0] != len(mean)):
        raise ValueError("Niezgodne wymiary cech i współczynników etykiet.")
    return (((features - mean) / scale) @ coefficients) * label_scale + label_mean


def predict_labels_from_codes(c: np.ndarray, basis: Mapping[str, Any]) -> np.ndarray:
    """Odczyt z c=project(features,basis): c@(projection.T@B), bez nowego fit.

    Przy obcięciu rzędu może być słabszy niż pełny ridge. Wynik [...,Q] jest
    w jednostkach oryginalnych etykiet, z tymi samymi ograniczeniami presence.
    """
    c = _array(c, "c")
    projection = _array(basis["projection"], "projection", 2)
    coefficients, mean, scale = _label_parameters(basis)
    if (c.ndim < 1 or c.shape[-1] != projection.shape[1]
            or projection.shape[0] != coefficients.shape[0]):
        raise ValueError("Niezgodne wymiary kodów, projekcji i etykiet.")
    return (c @ (projection.T @ coefficients)) * scale + mean


def _indices(value: Any, n: int, name: str) -> np.ndarray:
    indices = np.asarray(value)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or np.any(indices < 0) or np.any(indices >= n)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError(f"{name}: wymagane różne indeksy całkowite w [0,{n}).")
    return indices.astype(int, copy=False)


def _weighted_auc(binary: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float | None:
    positives = float(weights[binary == 1].sum())
    negatives = float(weights[binary == 0].sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores, y, w = scores[order], binary[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    positive_mass = np.add.reduceat(w * y, starts)
    negative_mass = np.add.reduceat(w * (1 - y), starts)
    below = np.cumsum(negative_mass) - negative_mass
    auc = np.sum(positive_mass * (below + .5 * negative_mass)) / (positives * negatives)
    return float(np.clip(auc, 0., 1.))


def _label_metrics(truth: np.ndarray, predicted: np.ndarray, weights: np.ndarray,
                   names: list[str], presence_index: int, threshold: float) -> dict[str, Any]:
    weights = weights / weights.sum()
    binary, presence_score = truth[:, presence_index], predicted[:, presence_index]
    r2 = {}
    for q, name in enumerate(names):
        if q == presence_index:
            continue
        y = truth[:, q]
        if np.all(y == y[0]):
            r2[name] = None
        else:
            centered = y - np.sum(weights * y)
            r2[name] = float(1 - np.sum(weights * (y - predicted[:, q]) ** 2)
                             / np.sum(weights * centered ** 2))
    return {"presence": {"auc": _weighted_auc(binary, presence_score, weights),
                         "accuracy": float(np.sum(weights * ((presence_score >= threshold) == binary)))},
            "r2": r2}


def evaluate_directions(features: np.ndarray, labels: np.ndarray, basis: Mapping[str, Any], *,
                        image_groups: Sequence[Any], holdout_mask: np.ndarray | None = None,
                        time_indices: Sequence[int] | None = None, presence_index: int = 0,
                        presence_threshold: float = .5,
                        label_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Raport jakości kandydackiej bazy, niezależny od bramki generacji.

    features[T,N,D], labels[N,Q]; fit_indices odczytuje z zamrożonej basis.
    Domyślnie ocenia wszystkie checkpointy i obrazy spoza fit_indices. Grupy
    fit/holdout muszą być rozłączne. Nie dopasowuje ani nie wybiera parametrów.
    Do raportu konkretnego okresu można przekazać time_indices, np. range(2,21).

    full = pełne współczynniki ridge; projected = odczyt po obciętej projekcji.
    Każda metoda raportuje presence:{auc,accuracy}, r2 według nazw etykiet,
    per_checkpoint i per_group. AUC używa nieobciętego wyniku regresji,
    accuracy progu presence_threshold ustalonego wcześniej, domyślnie .5.
    Etykieta presence musi być binarna 0/1. Jedna klasa: AUC=None; stałe y:
    R²=None. Nie zastępujemy niezdefiniowanych miar zerem ani jedynką.

    W wynikach zbiorczych każda grupa ma równą łączną wagę, a jej obrazy i
    checkpointy równe wagi wewnętrzne. AUC i R² liczymy na tak ważonej puli,
    nie jako średnią AUC/R² grup (które mogą mieć tylko jedną klasę).
    To opis jakości odczytu etykiet; nie dowód kompletnej tożsamości narzędzia.
    Raport nie zawiera accepted i nigdy nie zmienia validate()['accepted'].
    """
    data = _array(features, "features", 3, finite=False)
    labels = _array(labels, "labels", 2, finite=False)
    if min(data.shape) == 0 or labels.shape[0] != data.shape[1] or labels.shape[1] == 0:
        raise ValueError("Niezgodne lub puste wymiary features i labels.")
    fit_indices = _indices(basis["fit_indices"], data.shape[1], "fit_indices")
    fit_mask = np.zeros(data.shape[1], dtype=bool)
    fit_mask[fit_indices] = True
    holdout = _mask(~fit_mask if holdout_mask is None else holdout_mask,
                    data.shape[1], "holdout_mask")
    if np.any(fit_mask & holdout):
        raise ValueError("Obrazy fit i holdoutu muszą być rozłączne.")
    all_groups, all_inverse = _group_layout(image_groups, data.shape[1])
    if set(all_inverse[fit_mask]) & set(all_inverse[holdout]):
        raise ValueError("Grupy fit i holdoutu muszą być rozłączne.")
    held = np.flatnonzero(holdout)
    groups, inverse = _group_layout([all_groups[i] for i in all_inverse[holdout]], len(held))
    times = _indices(np.arange(len(data)) if time_indices is None else time_indices,
                     len(data), "time_indices")
    presence_index = _integer(presence_index, "presence_index")
    if presence_index >= labels.shape[1] or not np.isfinite(presence_threshold):
        raise ValueError("Niepoprawny indeks etykiety presence lub próg accuracy.")
    q = labels.shape[1]
    default_names = (["presence", "sinangle", "cosangle", "logscale", "length_factor",
                      "background_brightness"] if q == 6 else [f"label_{j}" for j in range(q)])
    names = default_names if label_names is None else list(label_names)
    if len(names) != q or any(not isinstance(name, str) for name in names) or len(set(names)) != q:
        raise ValueError("label_names musi zawierać Q różnych nazw tekstowych.")
    truth = _array(labels[holdout], "etykiety holdoutu", 2)
    if not np.isin(truth[:, presence_index], [0., 1.]).all():
        raise ValueError("Etykieta presence musi mieć wartości 0 albo 1.")
    selected = data[times[:, None], held[None, :], :]
    predictions = {"full": predict_labels(selected, basis),
                   "projected": predict_labels_from_codes(project(selected, basis), basis)}
    image_weights = 1. / np.bincount(inverse)[inverse]
    pooled_truth = np.tile(truth, (len(times), 1))
    pooled_weights = np.tile(image_weights, len(times))
    report = {"status": "candidate", "report_type": "direction_quality",
              "affects_generation_gate": False, "holdout_indices": held,
              "time_indices": times, "label_names": names,
              "presence_index": presence_index, "presence_threshold": float(presence_threshold),
              "aggregation": "group_balanced_pooled", "n_groups": len(groups),
              "limitation": "Jakość odczytu etykiet nie dowodzi kompletności ani użycia cech przez detektor."}
    for method, predicted in predictions.items():
        if predicted.shape != (len(times), len(held), q):
            raise ValueError("Wymiar etykiet jest niezgodny z odczytem bazy.")
        result = _label_metrics(pooled_truth, predicted.reshape(-1, q), pooled_weights,
                                names, presence_index, presence_threshold)
        result["per_checkpoint"] = [
            {"index": int(time), **_label_metrics(truth, predicted[t], image_weights,
                                                  names, presence_index, presence_threshold)}
            for t, time in enumerate(times)]
        result["per_group"] = []
        for i, group in enumerate(groups):
            mask = inverse == i
            group_truth = np.tile(truth[mask], (len(times), 1))
            result["per_group"].append({"group": group, **_label_metrics(
                group_truth, predicted[:, mask].reshape(-1, q), np.ones(len(group_truth)),
                names, presence_index, presence_threshold)})
        report[method] = result
    return report


def _uniform_steps(steps: np.ndarray, t: int) -> tuple[np.ndarray, float]:
    steps = _array(steps, "steps", 1)
    if len(steps) != t or t < 2:
        raise ValueError("steps musi zawierać T>=2 checkpointy.")
    intervals = np.diff(steps)
    # Wyłącznie tolerancja arytmetyki float64; nie przybliżamy różnych odstępów.
    equal = np.allclose(intervals, intervals[0], rtol=32 * np.finfo(float).eps, atol=0.)
    if np.any(intervals <= 0) or not equal:
        raise ValueError("steps musi być ściśle rosnące i mieć równe odstępy.")
    return steps, float(intervals[0])


def fit_koopman(c: np.ndarray, steps: np.ndarray, fit_mask: np.ndarray,
                fit_stop: int, ridge: float = 1e-3) -> dict[str, Any]:
    """Dopasowuje wspólne c_next=A@c+b do c[T,N,d], tylko t<fit_stop.

    fit_stop jest wyłącznym indeksem (minimum 2). Ostatnie przejście dopasowania
    to fit_stop-2 -> fit_stop-1. Skale pochodzą wyłącznie z tej historii i
    obrazów fit_mask. Kara ridge dotyczy macierzy w przestrzeni skalowanej;
    wyraz wolny nie jest karany. A[d,d], b[d] są w oryginalnych jednostkach c;
    dla wierszy NumPy: c_next=c@A.T+b. steps sprawdzamy w całej przekazanej osi.
    """
    data, mask, stop, fit = _history(c, fit_mask, fit_stop, "c", 2)
    steps, interval = _uniform_steps(steps, len(data))
    ridge = _nonnegative(ridge, "ridge")
    mean, scale = _standardize(fit.reshape(-1, fit.shape[-1]))
    normalized = (fit - mean) / scale
    x = normalized[:-1].reshape(-1, fit.shape[-1])
    y = normalized[1:].reshape(-1, fit.shape[-1])
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    coefficients = _ridge_coefficients(x - x_mean, y - y_mean, ridge)
    a_scaled = coefficients.T
    b_scaled = y_mean - x_mean @ coefficients
    a = scale[:, None] * a_scaled / scale[None, :]
    b = mean + scale * b_scaled - a @ mean
    return {"A": a, "b": b, "mean": mean, "scale": scale,
            "A_scaled": a_scaled, "b_scaled": b_scaled, "ridge": ridge,
            "fit_stop": stop, "fit_indices": np.flatnonzero(mask),
            "last_fit_step": float(steps[stop - 1]), "step_interval": interval,
            "n_transitions": len(x), "status": "candidate"}


def forecast(model: Mapping[str, Any], c: np.ndarray, horizon: int) -> np.ndarray:
    """Zwraca stan po horizon równych odstępach checkpointów, w kształcie c[...,d].

    Nie zwraca całej ścieżki. horizon=0 zwraca kopię c. Nie dopasowuje żadnych
    parametrów; horyzont jest liczbą odstępów, nie kroków optymalizatora.
    """
    horizon = _integer(horizon, "horizon")
    state = _array(c, "c").copy()
    a, b = _array(model["A"], "A", 2), _array(model["b"], "b", 1)
    if state.ndim == 0 or state.shape[-1] != len(b) or a.shape != (len(b), len(b)):
        raise ValueError("Niezgodne wymiary stanu i operatora.")
    with np.errstate(over="raise", invalid="raise"):
        for _ in range(horizon):
            state = state @ a.T + b
    return state


def _group_layout(groups: Any, n: int) -> tuple[list[Any], np.ndarray]:
    values = np.asarray(groups, dtype=object)
    if values.shape != (n,):
        raise ValueError("image_groups: wymagany jeden identyfikator na obraz.")
    labels, lookup, inverse = [], {}, np.empty(n, dtype=int)
    for i, value in enumerate(values):
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float)) or (
                isinstance(value, float) and not np.isfinite(value)):
            raise ValueError("Identyfikatory grup muszą być tekstem lub skończoną liczbą.")
        if value not in lookup:
            lookup[value] = len(labels)
            labels.append(value)
        inverse[i] = lookup[value]
    return labels, inverse


def _group_means(values: np.ndarray, inverse: np.ndarray, count: int) -> np.ndarray:
    return (np.bincount(inverse, weights=values, minlength=count)
            / np.bincount(inverse, minlength=count))


def _statistics_options(n_bootstrap: int, confidence: float) -> tuple[int, float]:
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 1)
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence musi należeć do (0,1).")
    return n_bootstrap, confidence


def _group_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int,
              confidence: float) -> dict[str, Any]:
    result = {"mean": float(values.mean()), "n_groups": len(values),
              "confidence": confidence, "ci_low": None, "ci_high": None}
    if len(values) < 2:
        return result
    means = np.empty(n_bootstrap)
    # Ograniczenie pamięci bez rozbijania sparowanych obserwacji grupowych.
    batch_size = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        sampled = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sampled].mean(axis=1)
    alpha = (1. - confidence) / 2.
    result["ci_low"], result["ci_high"] = map(float, np.quantile(means, [alpha, 1 - alpha]))
    return result


def _score(errors: np.ndarray, groups: list[Any], inverse: np.ndarray) -> dict[str, Any]:
    per_image = errors.mean(axis=0)
    per_group = _group_means(per_image, inverse, len(groups))
    counts = np.bincount(inverse, minlength=len(groups))
    return {"mean_vector_error": float(per_group.mean()),
            "n_groups": len(groups), "n_images": len(inverse),
            "n_forecasts_per_image": len(errors), "per_image": per_image,
            "group_errors": per_group,
            "per_group": [{"group": group, "mean_vector_error": float(error),
                           "n_images": int(count)}
                          for group, error, count in zip(groups, per_group, counts)]}


def validate(c: np.ndarray, steps: np.ndarray, fit_mask: np.ndarray,
             fit_stop: int, ridge: float = 1e-3, *, image_groups: Sequence[Any],
             holdout_mask: np.ndarray | None = None, horizons: Sequence[int] = (1,),
             n_bootstrap: int = 2000, confidence: float = .95, seed: int = 0,
             n_permutations: int = 0) -> dict[str, Any]:
    """Walidacja rolling origins na odłożonych grupach, bez ponownego dopasowania.

    Operator i skale dopasowujemy raz na c[:fit_stop,fit_mask]. Dla horyzontu h
    początki prognoz to fit_stop-1,...,T-h-1, a cele to origin+h >= fit_stop.
    Przy T=21, fit_stop=15 i horizons=(5,) oceniamy 14->19 oraz 15->20.
    Model nigdy nie korzysta z przyszłych stanów podczas dopasowania. Późniejszy
    stan początkowy jest obserwacją dostępną w swoim czasie, nie próbką fit.

    Domyślny holdout_mask to ~fit_mask. Podziały muszą być rozłączne również
    po image_groups (operacja/film/obraz źródłowy wspólny dla jego wariantów).
    Baselines: constant=c[origin], linear_trend=c[origin]+h*(c[origin]-c[origin-1]).
    Główna miara to norma (pred-truth)/model['scale']. Najpierw uśredniamy
    wszystkie okna/horyzonty per obraz, potem obrazy per grupa, wreszcie grupy
    z równymi wagami. scores_by_horizon ujawnia wyniki każdego horyzontu.

    accepted wymaga niższego błędu niż OBA baselines i dodatniej dolnej granicy
    sparowanego bootstrapu poprawy dla obu, przy >=2 niezależnych grupach.
    Remis (tolerancja numeryczna 1e-12) oznacza odrzucenie. Bootstrap losuje
    całe grupy; nie traktuje okien ani wariantów obrazu jako niezależnych.
    Nie mierzy niepewności pomiędzy treningami ani nowymi okresami czasu.
    Opcjonalna permutacja kolejności czasu dopasowania jest tylko kontrolą
    pomocniczą i nie wpływa na accepted. Wynik to dict; użyj to_jsonable.
    """
    n_bootstrap, confidence = _statistics_options(n_bootstrap, confidence)
    n_permutations = _integer(n_permutations, "n_permutations")
    seed = _integer(seed, "seed")
    data, fit, stop, _ = _history(c, fit_mask, fit_stop, "c", 2)
    model = fit_koopman(data, steps, fit, stop, ridge)
    holdout = _mask(~fit if holdout_mask is None else holdout_mask,
                    data.shape[1], "holdout_mask")
    if np.any(fit & holdout):
        raise ValueError("fit_mask i holdout_mask muszą być rozłączne.")
    all_groups, all_inverse = _group_layout(image_groups, data.shape[1])
    if set(all_inverse[fit]) & set(all_inverse[holdout]):
        raise ValueError("Grupy dopasowania i holdoutu muszą być rozłączne.")
    groups, inverse = _group_layout(
        [all_groups[index] for index in all_inverse[holdout]], int(holdout.sum()))
    horizons = tuple(_integer(h, "horizon", 1) for h in horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons musi być niepustą listą różnych dodatnich horyzontów.")
    if any(stop - 1 + h >= len(data) for h in horizons):
        raise ValueError("Za krótki holdout dla co najmniej jednego horyzontu.")
    # Do pierwszego trendu potrzebna jest obserwacja stop-2. Nie wymagamy
    # istnienia wcześniejszej historii obrazów walidacyjnych.
    _array(data[stop - 2:, holdout], "obserwacje holdoutu", 3)
    folds = []
    errors: dict[str, list[np.ndarray]] = {name: [] for name in
                                          ("koopman", "constant", "linear_trend")}
    scores_by_horizon = {}
    for horizon in horizons:
        start = len(folds)
        for origin in range(stop - 1, len(data) - horizon):
            anchor = data[origin, holdout]
            truth = data[origin + horizon, holdout]
            predictions = {
                "koopman": forecast(model, anchor, horizon),
                "constant": anchor.copy(),
                "linear_trend": anchor + horizon * (anchor - data[origin - 1, holdout]),
            }
            fold_errors = {}
            for name, prediction in predictions.items():
                error = np.linalg.norm((prediction - truth) / model["scale"], axis=-1)
                if not np.isfinite(error).all():
                    raise ValueError("Nieskończony błąd prognozy; sprawdź stabilność modelu.")
                errors[name].append(error)
                fold_errors[name] = error
            folds.append({"origin": origin, "target": origin + horizon,
                          "horizon": horizon,
                          "horizon_updates": horizon * model["step_interval"],
                          "predictions": predictions, "errors": fold_errors})
        scores_by_horizon[str(horizon)] = {
            name: _score(np.stack(values[start:]), groups, inverse)
            for name, values in errors.items()}
    scores = {name: _score(np.stack(values), groups, inverse) for name, values in errors.items()}
    rng = np.random.default_rng(seed)
    comparisons = {}
    for baseline in ("constant", "linear_trend"):
        improvement = scores[baseline]["group_errors"] - scores["koopman"]["group_errors"]
        comparison = _group_ci(improvement, rng, n_bootstrap, confidence)
        comparison["group_improvements"] = improvement
        comparison["mean_improvement"] = comparison.pop("mean")
        tolerance = 1e-12 * max(1., scores[baseline]["mean_vector_error"],
                               scores["koopman"]["mean_vector_error"])
        comparison["better_mean"] = comparison["mean_improvement"] > tolerance
        comparison["supported"] = bool(comparison["better_mean"]
                                        and comparison["ci_low"] is not None
                                        and comparison["ci_low"] > tolerance)
        comparisons[baseline] = comparison
    accepted = all(value["supported"] for value in comparisons.values())
    reason = ("Przewaga nad obiema prognozami bazowymi na odłożonych grupach."
              if accepted else "Brak potwierdzonej przewagi nad obiema prognozami bazowymi.")
    if len(groups) < 2:
        reason = "Mniej niż dwie niezależne grupy holdoutu; brak oceny niepewności między grupami."
    report = {"accepted": accepted, "reason": reason, "model": model,
              "scores": scores, "scores_by_horizon": scores_by_horizon,
              "comparisons": comparisons, "folds": folds,
              "holdout_indices": np.flatnonzero(holdout),
              "protocol": {"fit_stop": stop, "horizons": horizons,
                           "step_interval": model["step_interval"], "rolling_origins": True,
                           "refit_during_validation": False, "seed": seed,
                           "n_bootstrap": n_bootstrap, "confidence": confidence,
                           "metric": "group_macro_mean_scaled_vector_error",
                           "uncertainty": "paired_group_bootstrap",
                           "limitation": "Jedna historia uczenia; okna mogą się nakładać."},
              "permutation": {"supplemental": True, "n_permutations": n_permutations}}
    if n_permutations:
        # Osobny generator: liczba permutacji nie zmienia przedziałów ani bramki.
        perm_rng = np.random.default_rng(seed)
        null_errors = np.empty(n_permutations)
        fit_data = data[:stop, fit]
        for i in range(n_permutations):
            shuffled = fit_data[perm_rng.permutation(stop)]
            null_model = fit_koopman(shuffled, np.asarray(steps)[:stop],
                                     np.ones(int(fit.sum()), dtype=bool), stop, ridge)
            null_fold_errors = []
            for fold in folds:
                prediction = forecast(null_model, data[fold["origin"], holdout], fold["horizon"])
                null_fold_errors.append(np.linalg.norm(
                    (prediction - data[fold["target"], holdout]) / model["scale"], axis=-1))
            null_errors[i] = _score(np.stack(null_fold_errors), groups, inverse)["mean_vector_error"]
        report["permutation"].update({
            "kind": "wspólna permutacja checkpointów historii dopasowania",
            "errors": null_errors,
            "fraction_not_worse": float((1 + np.count_nonzero(
                null_errors <= scores["koopman"]["mean_vector_error"])) / (n_permutations + 1)),
        })
    return report


def safe_cosine(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12
                ) -> tuple[np.ndarray, np.ndarray]:
    """Kosinus w ostatniej osi i maska ważności; dla norm <=eps: (0,False).

    Zero jest technicznym wypełnieniem, nie zgodnością kierunkową. Pomijaj je
    w podsumowaniach według maski ważności, także gdy oba wektory są zerowe.
    """
    a, b = _array(a, "a"), _array(b, "b")
    eps = _nonnegative(eps, "eps", positive=True)
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] == 0:
        raise ValueError("Wektory kosinusa muszą mieć ten sam niepusty wymiar.")
    norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = (norm_a > eps) & (norm_b > eps)
    unit_a = np.divide(a, norm_a[..., None], out=np.zeros_like(a), where=valid[..., None])
    unit_b = np.divide(b, norm_b[..., None], out=np.zeros_like(b), where=valid[..., None])
    return np.clip(np.sum(unit_a * unit_b, axis=-1), -1., 1.), valid


def compare_arms(c_start: np.ndarray, c_control: np.ndarray, c_prim: np.ndarray,
                 delta_k: np.ndarray, *, scale: np.ndarray | None = None,
                 eps: float = 1e-12, image_groups: Sequence[Any] | None = None,
                 n_bootstrap: int = 2000, confidence: float = .95,
                 seed: int = 0) -> dict[str, Any]:
    """Porównuje ramiona w jednej zamrożonej bazie; wszystkie stany mają [N,d].

    delta_k to PRZYROST, nie cel. Cel = c_start+delta_k. c_control oznacza
    kontynuację bez dodatkowej klatki, c_prim kontynuację z dodaniem klatki.
    scale[d] ustal przed rozgałęzieniem (np. model['scale']); None = jednostki c.
    Zwraca per_image: normy, błędy wektorowe do delta_k, kosinusy z maskami,
    additional_projection=<c_prim-c_control,unit(delta_k)> i distance_gain
    = błąd_control-błąd_prim. Wszystkie te miary używają tej samej skali.
    Dodatni rzut nie musi oznaczać dodatniego distance_gain (przestrzelenie).
    Opcjonalne grupy dają równe wagi grup i sparowany CI poprawy distance_gain.
    To wynik zmiany latentu, nie ocena poprawy jakości detekcji.
    """
    start = _array(c_start, "c_start", 2)
    control, prim = _array(c_control, "c_control", 2), _array(c_prim, "c_prim", 2)
    expected = _array(delta_k, "delta_k", 2)
    if min(start.shape) == 0 or any(x.shape != start.shape for x in (control, prim, expected)):
        raise ValueError("Wszystkie ramiona i delta_k muszą mieć ten sam kształt [N,d].")
    eps = _nonnegative(eps, "eps", positive=True)
    metric_scale = _metric_scale(scale, start.shape[-1])
    delta_control = (control - start) / metric_scale
    delta_prim = (prim - start) / metric_scale
    delta_expected = expected / metric_scale
    effect = (prim - control) / metric_scale
    norm_expected = np.linalg.norm(delta_expected, axis=-1)
    projection_valid = norm_expected > eps
    unit_expected = np.divide(delta_expected, norm_expected[:, None],
                              out=np.zeros_like(delta_expected), where=projection_valid[:, None])
    cosine_control, valid_control = safe_cosine(delta_control, delta_expected, eps=eps)
    cosine_prim, valid_prim = safe_cosine(delta_prim, delta_expected, eps=eps)
    error_control = np.linalg.norm(delta_control - delta_expected, axis=-1)
    error_prim = np.linalg.norm(delta_prim - delta_expected, axis=-1)
    per_image = {
        "expected_norm": norm_expected, "control_norm": np.linalg.norm(delta_control, axis=-1),
        "prim_norm": np.linalg.norm(delta_prim, axis=-1), "effect_norm": np.linalg.norm(effect, axis=-1),
        "control_vector_error": error_control, "prim_vector_error": error_prim,
        "control_cosine": cosine_control, "prim_cosine": cosine_prim,
        "control_cosine_valid": valid_control, "prim_cosine_valid": valid_prim,
        "additional_projection": np.sum(effect * unit_expected, axis=-1),
        "projection_valid": projection_valid, "distance_gain": error_control - error_prim,
    }
    valid_for = {"control_cosine": valid_control, "prim_cosine": valid_prim,
                 "additional_projection": projection_valid}
    groups, inverse = _group_layout(np.arange(len(start)) if image_groups is None else image_groups,
                                    len(start))
    summary = {}
    for name, values in per_image.items():
        if values.dtype == np.bool_:
            continue
        valid = valid_for.get(name, np.ones(len(start), dtype=bool))
        group_values = [float(values[(inverse == index) & valid].mean())
                        for index in range(len(groups)) if np.any((inverse == index) & valid)]
        summary[name] = float(np.mean(group_values)) if group_values else None
    result = {"per_image": per_image, "summary": summary, "scale": metric_scale.copy(),
              "n_images": len(start), "aggregation": "group_macro" if image_groups is not None else "image_mean",
              "valid_counts": {name: int(valid.sum()) for name, valid in valid_for.items()},
              "interpretation": "Porównanie reprezentacji; brak wniosku o jakości detekcji."}
    if image_groups is not None:
        n_bootstrap, confidence = _statistics_options(n_bootstrap, confidence)
        group_gains = _group_means(per_image["distance_gain"], inverse, len(groups))
        result["distance_gain_ci"] = _group_ci(group_gains,
            np.random.default_rng(_integer(seed, "seed")), n_bootstrap, confidence)
        result["distance_gain_ci"]["per_group"] = [
            {"group": group, "distance_gain": float(value)} for group, value in zip(groups, group_gains)]
    return result


def to_jsonable(value: Any) -> Any:
    """Rekurencyjnie zamienia NumPy na typy JSON; NaN/Inf na null (None).

    Wynik można przekazać do json.dumps(..., allow_nan=False). Nie zmienia
    wejścia ani nie zapisuje plików. Nieznane obiekty zgłaszają TypeError.
    """
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"Typ {type(value).__name__} nie jest obsługiwany przez raport JSON.")
