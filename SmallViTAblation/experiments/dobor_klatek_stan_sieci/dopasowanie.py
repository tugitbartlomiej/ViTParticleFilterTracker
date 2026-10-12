"""Czy prognoza latentów zmniejsza błąd ramki poza dopasowanymi operacjami?

Wpływ etykiety ocenianej operacji na jej predykcję obala izolację tego pomiaru.
PCA, skale, odczyt L0, EDMD i Langevin są dopasowane od nowa w każdym foldzie.
Etykiety kandydatów spoza L0 wchodzą wyłącznie do końcowych strat. Nie jest to
selekcja przed anotacją: pula kandydatów jest już oznaczona.

Stała receptura: EDMD [z,cos(zR+b),1], 2*d cech cosinusowych; ridge 1e-3
razy średnia przekątna Gram (bez kary wyrazu wolnego). Langevin strict,
AdamW lr=1e-3, weight_decay=1e-4, clip=1, ostatni krok każdego pełnego okna.
Nie używamy walidacji do wyboru epoki, wymiaru, siły regularyzacji ani wagi
mieszanki. Przed odczytem geometrycznym mieszamy dokładnie 0.5*K + 0.5*L.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .dane import target_box, temporal_windows
from .model import ExtendedLangevin, LangevinConfig


RIDGE = 1e-3
LOSS_BETA = .05


def _ridge(x, y):
    gram = x.T @ x
    strength = RIDGE * max(float(np.diag(gram).mean()), 1e-12)
    penalty = np.eye(gram.shape[0], dtype=np.float64) * strength
    penalty[-1, -1] = 0.  # the final column is the unpenalized intercept
    return np.linalg.solve(gram + penalty, x.T @ y)


def _with_bias(x):
    return np.column_stack((x, np.ones(len(x))))


def _smooth_l1(prediction, truth):
    difference = np.abs(prediction - truth)
    return np.where(difference < LOSS_BETA, .5 * difference**2 / LOSS_BETA,
                    difference - .5 * LOSS_BETA).mean(axis=1)


def _fit_pca(features, fit_rows, dimension):
    fitting = np.asarray(features[fit_rows], dtype=np.float64)
    mean = fitting.mean(axis=0)
    centered = fitting - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:dimension]
    components = eigenvectors[:, order]
    # A deterministic sign convention makes the saved coordinates reproducible.
    signs = np.sign(components[np.abs(components).argmax(axis=0), np.arange(dimension)])
    components = components * np.where(signs == 0, 1., signs)
    scale = np.sqrt(np.maximum(eigenvalues[order], 1e-12))
    z = ((np.asarray(features, dtype=np.float64) - mean) @ components) / scale
    return z.astype(np.float32), dict(mean=mean, components=components, scale=scale)


def _lift(z, projection, phase):
    return np.column_stack((z, np.cos(z @ projection + phase), np.ones(len(z))))


def fit_cross_fitted(rows, features, inputs, observed, pool, l0_ids, *, context=8,
                    step=5, dim=24, folds=3, epochs=12, batch_size=256,
                    max_windows=20000, seed=42, device="cpu", artifact_dir=None):
    """Return cross-fitted task losses and boxes in the original order of pool.

    ``rows`` indexes the dense, train-only feature cache; ``pool`` is the
    annotated candidate pool. Missing full windows yield NaN dynamics outputs.
    The fitted detector is a fixed upstream input; this function does not claim
    the detector itself was cross-fitted. Every stateful operation below is.
    """
    features = np.asarray(features)
    inputs = np.asarray(inputs)
    observed = np.asarray(observed)
    if context < 2 or min(dim, epochs, batch_size, max_windows) < 1:
        raise ValueError("positive dimensions/training limits and context>=2 required")
    if features.ndim != 2 or len(features) != len(rows) or features.shape[1] < 1:
        raise ValueError("features must be [len(rows), feature_dim]")
    if inputs.ndim != 2 or len(inputs) != len(rows) or inputs.shape[1] < 1:
        raise ValueError("inputs must be [len(rows), input_dim]")
    if observed.shape != (len(rows),) or observed.dtype != np.bool_:
        raise ValueError("observed must be a boolean vector from detector confidence")
    if not np.isfinite(features).all() or not np.isfinite(inputs).all():
        raise ValueError("features and inputs must be finite")
    if not rows or not pool or any(r.get("split") != "train" for r in pool):
        raise ValueError("nonempty train-only rows and pool required")
    frame_ids = [str(r["frame_id"]) for r in rows]
    row_index = {name: index for index, name in enumerate(frame_ids)}
    pool_ids = [str(r["frame_id"]) for r in pool]
    if len(row_index) != len(rows) or len(set(pool_ids)) != len(pool):
        raise ValueError("duplicate frame IDs")
    if not set(pool_ids) <= set(row_index):
        raise ValueError("pool frame missing from feature rows")
    initial_ids = list(map(str, l0_ids))
    if not initial_ids or len(set(initial_ids)) != len(initial_ids) or not set(initial_ids) <= set(pool_ids):
        raise ValueError("L0 must contain unique IDs present in pool")
    groups = np.asarray([str(r["group_id"]) for r in rows])
    pool_groups = np.asarray([str(r["group_id"]) for r in pool])
    pool_to_row = np.asarray([row_index[name] for name in pool_ids], dtype=np.int64)
    if not np.array_equal(groups[pool_to_row], pool_groups):
        raise ValueError("pool/cache group mismatch")
    group_names = sorted(set(groups.tolist()))
    if not 2 <= folds <= len(group_names):
        raise ValueError("folds must be between 2 and number of independent groups")
    windows = temporal_windows(rows, context=context, step=step)
    target_to_window = {int(window[-1]): index for index, window in enumerate(windows)}
    truths = np.stack([target_box(row) for row in pool])
    pool_index = {name: index for index, name in enumerate(pool_ids)}
    rng = np.random.default_rng(seed)
    held_sets = np.array_split(rng.permutation(group_names), folds)
    destination = Path(artifact_dir) if artifact_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        if any((destination / f"GLE_fold{i}.pt").exists() for i in range(folds)):
            raise FileExistsError("GLE fold artifacts already exist; use a new artifact directory")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    n = len(pool)
    result = {"e_obs": np.full(n, np.nan), "e_dyn": np.full(n, np.nan),
              "pred_obs_box": np.full((n, 4), np.nan), "pred_dyn_box": np.full((n, 4), np.nan)}
    audit = dict(seed=int(seed), context=int(context), frame_step=float(step), epochs=int(epochs),
                 batch_size=int(batch_size), max_windows=int(max_windows), loss_beta=LOSS_BETA,
                 ridge=RIDGE, ensemble_weights=[.5, .5], conditional=False,
                 target_labels_used_for_fitting="L0 only, excluding each held group",
                 detector_cross_fitted=False, folds=[])
    for fold, held in enumerate(held_sets):
        held_names = sorted(map(str, held.tolist()))
        fit_names = sorted(set(group_names) - set(held_names))
        fit_rows = np.flatnonzero(np.isin(groups, fit_names))
        held_pool = np.flatnonzero(np.isin(pool_groups, held_names))
        readout_ids = [name for name in initial_ids if groups[row_index[name]] in fit_names]
        if len(readout_ids) < 3:
            raise ValueError(f"fold {fold}: fewer than 3 L0 readout samples outside held groups")
        dimension = min(int(dim), features.shape[1], len(fit_rows) - 1, len(readout_ids) - 2)
        if dimension < 1:
            raise ValueError(f"fold {fold}: insufficient non-held feature rows")
        z, pca = _fit_pca(features, fit_rows, dimension)
        input_mean = np.asarray(inputs[fit_rows], dtype=np.float64).mean(axis=0)
        input_scale = np.asarray(inputs[fit_rows], dtype=np.float64).std(axis=0).clip(1e-6)
        scaled_inputs = ((inputs - input_mean) / input_scale).astype(np.float32)
        readout_rows = np.asarray([row_index[name] for name in readout_ids])
        readout_targets = truths[[pool_index[name] for name in readout_ids]]
        readout = _ridge(_with_bias(z[readout_rows].astype(np.float64)), readout_targets)
        observed_boxes = np.clip(_with_bias(z[pool_to_row[held_pool]].astype(np.float64)) @ readout, 0., 1.)
        result["pred_obs_box"][held_pool] = observed_boxes
        result["e_obs"][held_pool] = _smooth_l1(observed_boxes, truths[held_pool])
        fitting_windows = windows[np.isin(groups[windows[:, -1]], fit_names)]
        fit_windows_available = len(fitting_windows)
        fold_rng = np.random.default_rng(seed + 1009 * fold)
        if len(fitting_windows) > max_windows:
            chosen = fold_rng.choice(len(fitting_windows), size=max_windows, replace=False)
            fitting_windows = fitting_windows[np.sort(chosen)]
        dynamic_pool = np.asarray([index for index in held_pool if int(pool_to_row[index]) in target_to_window], dtype=np.int64)
        eval_windows = np.asarray([windows[target_to_window[int(pool_to_row[index])]] for index in dynamic_pool],
                                  dtype=np.int64).reshape(-1, context + 1)
        config = LangevinConfig(latent_dim=dimension, input_dim=inputs.shape[1], context=context,
                                hidden_dim=64, heads=2, residual_rank=8, conditional=False)
        loss_history = []
        koopman = None
        component_sums = {name: 0. for name in ("D", "JP", "W", "Z", "N")}
        latent_errors = {name: 0. for name in ("koopman", "langevin", "ensemble", "persistence")}
        score_errors = {name: [] for name in ("koopman", "langevin")}
        # All module initialization is on CPU, before .to(device), so only the
        # CPU RNG is consumed. Restore that state; do not reseed unrelated GPUs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 1009 * fold)
            model = ExtendedLangevin(config).to(torch_device)
            if len(fitting_windows):
                random_features = 2 * dimension
                projection = fold_rng.normal(size=(dimension, random_features)) / np.sqrt(dimension)
                phase = fold_rng.uniform(0., 2. * np.pi, size=random_features)
                x_lift = _lift(z[fitting_windows[:, -2]].astype(np.float64), projection, phase)
                y_lift = _lift(z[fitting_windows[:, -1]].astype(np.float64), projection, phase)
                operator = _ridge(x_lift, y_lift)
                koopman = dict(projection=projection, phase=phase, operator=operator,
                               latent_dim=dimension, random_features=random_features)
                z_tensor = torch.as_tensor(z, device=torch_device)
                input_tensor = torch.as_tensor(scaled_inputs, device=torch_device)
                observed_tensor = torch.as_tensor(observed, device=torch_device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                model.train()
                for _ in range(epochs):
                    order = fold_rng.permutation(len(fitting_windows))
                    loss_sum = 0.
                    for start in range(0, len(order), batch_size):
                        batch_windows = fitting_windows[order[start:start + batch_size]]
                        idx = torch.as_tensor(batch_windows, dtype=torch.long, device=torch_device)
                        z_batch = z_tensor[idx]
                        output = model(z_batch, input_tensor[idx], observed_tensor[idx])
                        loss = (output["prediction"][:, -1] - z_batch[:, -1]).square().mean()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"nonfinite GLE training loss in fold {fold}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        loss_sum += float(loss.detach()) * len(batch_windows)
                    loss_history.append(loss_sum / len(fitting_windows))
                model.eval()
                with torch.no_grad():
                    for start in range(0, len(eval_windows), batch_size):
                        batch_windows = eval_windows[start:start + batch_size]
                        idx = torch.as_tensor(batch_windows, dtype=torch.long, device=torch_device)
                        output = model(z_tensor[idx], input_tensor[idx], observed_tensor[idx])
                        langevin = output["prediction"][:, -1].cpu().numpy().astype(np.float64)
                        lifted = _lift(z[batch_windows[:, -2]].astype(np.float64), projection, phase)
                        koopman_z = (lifted @ operator)[:, :dimension]
                        ensemble = .5 * koopman_z + .5 * langevin
                        if not np.isfinite(ensemble).all():
                            raise FloatingPointError(f"nonfinite latent prediction in fold {fold}")
                        indices = dynamic_pool[start:start + len(batch_windows)]
                        boxes = np.clip(_with_bias(ensemble) @ readout, 0., 1.)
                        result["pred_dyn_box"][indices] = boxes
                        result["e_dyn"][indices] = _smooth_l1(boxes, truths[indices])
                        latent_truth = z[batch_windows[:, -1]].astype(np.float64)
                        for name, prediction in (("koopman", koopman_z), ("langevin", langevin),
                                                  ("ensemble", ensemble), ("persistence", z[batch_windows[:, -2]])):
                            latent_errors[name] += float(((prediction - latent_truth)**2).sum())
                        for name, prediction in (("koopman", koopman_z), ("langevin", langevin)):
                            pred_box = np.clip(_with_bias(prediction) @ readout, 0., 1.)
                            score_errors[name].extend(_smooth_l1(pred_box, truths[indices]).tolist())
                        for name, value in output["terms"].items():
                            component_sums[name] += float(value[:, -1].double().square().sum().cpu())
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        evaluated = len(dynamic_pool) if koopman is not None else 0
        fold_audit = dict(fold=fold, fit_groups=fit_names, held_groups=held_names,
                          fit_rows=len(fit_rows), readout_l0_ids=readout_ids, readout_samples=len(readout_ids),
                          latent_dim=dimension, fit_windows_available=fit_windows_available,
                          fit_windows=len(fitting_windows), held_pool=len(held_pool),
                          held_windows_available=len(dynamic_pool), held_windows=evaluated,
                          missing_context=len(held_pool) - len(dynamic_pool),
                          dynamics_trained=koopman is not None, training_loss=loss_history,
                          information=dict(conditional=False, target_latent=False, target_metadata=False,
                                           residual_min_lag=1, context=context,
                                           events="expected return from detector-confidence gap",
                                           observed="detector score >= 0.5; not physical presence"),
                          mean_e_obs=float(np.mean(result["e_obs"][held_pool])),
                          mean_e_dyn=float(np.mean(result["e_dyn"][dynamic_pool])) if evaluated else None,
                          mean_task_errors={name: float(np.mean(values)) if values else None
                                            for name, values in score_errors.items()},
                          latent_mse={name: total / (evaluated * dimension) if evaluated else None
                                      for name, total in latent_errors.items()},
                          component_rms={name: float(np.sqrt(total / (evaluated * dimension))) if evaluated else None
                                         for name, total in component_sums.items()})
        if destination is not None:
            checkpoint = dict(config=asdict(config), state_dict=state, pca=pca,
                              input_scaling=dict(mean=input_mean, scale=input_scale),
                              readout=dict(coefficients=readout, coordinates="normalized cxcywh", ridge=RIDGE,
                                           clip=[0., 1.], l0_ids=readout_ids), koopman=koopman, audit=fold_audit)
            with (destination / f"GLE_fold{fold}.pt").open("xb") as stream:
                torch.save(checkpoint, stream)
        audit["folds"].append(fold_audit)
    audit["pool_size"] = n
    audit["dense_rows"] = len(rows)
    audit["dynamic_scores_available"] = int(np.isfinite(result["e_dyn"]).sum())
    result["audit"] = audit
    return result
