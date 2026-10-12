"""Czy dynamika pojedynczych patchy zmniejsza poza próbą błąd słabej etykiety narzędzia?

Pooling przestrzenny, fit z etykiet ocenianej grupy lub odczyt latentów celu
unieważnia ten test. Cząstka oznacza stały indeks patcha w filmie (opis Eulera),
nie śledzony punkt ani fizyczny egzemplarz narzędzia. Średnie obliczamy dopiero
z błędów osobnych predykcji patchy, nigdy z ich wektorów reprezentacji.

Tokeny wejściowe pochodzą ze stałej projekcji kanałów ViT. W każdym foldzie
fitowane są wyłącznie na innych grupach: skala tokenów i wejść, odczyt
foreground na L0, EDMD oraz pełny skalowany D+JP+W+Z+N. Odczyt nie dostaje
współrzędnych patcha. Etykieta bbox jest słabym nadzorem obszaru, nie maską
rzeczywistego narzędzia. GT kandydatów służy tylko do oceny i ustalonego wyboru
do 16 dodatnich i 16 ujemnych patchy na klatkę, wspólnego dla predyktorów.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from .dane import target_box, temporal_windows
from .model import LangevinConfig
from .token_model import ScaledTokenLangevin


RIDGE = 1e-3
PROBABILITY_CLIP = .001
PATCHES_PER_CLASS = 16


def token_targets(row, grid_shape=(24, 24)):
    """Binary weak bbox labels in row-major order; nearest center for tiny boxes."""
    height, width = map(int, grid_shape)
    if min(height, width) < 1:
        raise ValueError("positive patch grid dimensions required")
    cx, cy, bw, bh = target_box(row)
    yy, xx = np.meshgrid((np.arange(height) + .5) / height,
                         (np.arange(width) + .5) / width, indexing="ij")
    mask = (xx >= cx - bw / 2) & (xx < cx + bw / 2) & (yy >= cy - bh / 2) & (yy < cy + bh / 2)
    if not mask.any():
        mask.flat[int(np.argmin((xx - cx)**2 + (yy - cy)**2))] = True
    return mask.reshape(-1)


def particle_trajectories(tokens, windows, patch_indices, mean, scale):
    """Gather one fixed patch per trajectory, with no spatial aggregation."""
    windows = np.asarray(windows, dtype=np.int64)
    patches = np.asarray(patch_indices, dtype=np.int64)
    if windows.ndim != 2 or patches.shape != (len(windows),):
        raise ValueError("windows must be [B,T] and patch_indices [B]")
    values = np.asarray(tokens[windows, patches[:, None]], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite sampled token")
    return ((values - mean) / scale).astype(np.float32)


def _frame_rng(seed, frame_id, namespace):
    digest = hashlib.sha256(f"{seed}|{namespace}|{frame_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _sample_labeled_patches(row, grid_shape, seed, namespace):
    labels = token_targets(row, grid_shape)
    rng = _frame_rng(seed, str(row["frame_id"]), namespace)
    chosen, target, weights = [], [], []
    classes = [np.flatnonzero(labels), np.flatnonzero(~labels)]
    present_classes = sum(bool(len(indices)) for indices in classes)
    for value, indices in zip((1., 0.), classes):
        if not len(indices):
            continue
        indices = np.sort(rng.choice(indices, size=min(PATCHES_PER_CLASS, len(indices)), replace=False))
        chosen.extend(indices.tolist())
        target.extend([value] * len(indices))
        weights.extend([1. / (present_classes * len(indices))] * len(indices))
    return np.asarray(chosen, dtype=np.int64), np.asarray(target), np.asarray(weights)


def _bias(values):
    return np.column_stack((values, np.ones(len(values))))


def _ridge(x, y, weights=None):
    if weights is not None:
        root_weight = np.sqrt(weights / np.mean(weights))
        x = x * root_weight[:, None]
        y = y * (root_weight[:, None] if y.ndim == 2 else root_weight)
    gram = x.T @ x
    penalty = np.eye(gram.shape[0]) * RIDGE * max(float(np.diag(gram).mean()), 1e-12)
    penalty[-1, -1] = 0.
    return np.linalg.solve(gram + penalty, x.T @ y)


def _probability(z, coefficients):
    return np.clip(_bias(np.asarray(z, dtype=np.float64)) @ coefficients,
                   PROBABILITY_CLIP, 1. - PROBABILITY_CLIP)


def _bce(probability, target):
    return -(target * np.log(probability) + (1. - target) * np.log1p(-probability))


def _lift(z, projection, phase):
    return np.column_stack((z, np.cos(z @ projection + phase), np.ones(len(z))))


def fit_token_cross_fitted(rows, tokens, inputs, pool, l0_ids, *, excluded_fit_groups=(), grid_shape=(24, 24),
                           context=8, step=5, folds=3, epochs=12, batch_size=256,
                           max_windows=20000, seed=42, device="cuda", artifact_dir=None):
    """Fit independent patch dynamics; return balanced BCE per candidate frame.

    ``max_windows`` counts (temporal-window, patch-index) pairs, sampled uniformly
    without GT. Global W inputs are expanded to these particles; Z observation
    flags are patch-specific ``g(z)>=0.5``. All predictions are strict: no target
    latent or target metadata enters the prediction of that target. Groups in
    ``excluded_fit_groups`` (D) can only be scored: they never supply scaler
    statistics, support readout labels, temporal fitting windows, or operators.
    """
    tokens = np.asanyarray(tokens)  # preserve mmap; never copy the full token cache
    inputs = np.asarray(inputs)
    if tokens.ndim != 3 or tokens.shape[0] != len(rows) or tokens.shape[1] != int(np.prod(grid_shape)):
        raise ValueError("tokens must be [len(rows), grid_height*grid_width, channels]")
    if inputs.shape != (len(rows), 6) or not np.isfinite(inputs).all():
        raise ValueError("inputs must be finite [len(rows),6]")
    if not len(rows) or not len(pool) or context < 2 or min(epochs, batch_size, max_windows, tokens.shape[2]) < 1:
        raise ValueError("nonempty data, context>=2 and positive training limits required")
    if any(row.get("split") != "train" for row in list(rows) + list(pool)):
        raise ValueError("candidate pool and context must be train-only")
    row_ids = [str(row["frame_id"]) for row in rows]
    row_index = {name: index for index, name in enumerate(row_ids)}
    pool_ids = [str(row["frame_id"]) for row in pool]
    if len(row_index) != len(rows) or len(set(pool_ids)) != len(pool):
        raise ValueError("duplicate frame IDs")
    if not set(pool_ids) <= set(row_index):
        raise ValueError("candidate missing from token cache")
    initial_ids = list(map(str, l0_ids))
    if not initial_ids or len(set(initial_ids)) != len(initial_ids) or not set(initial_ids) <= set(pool_ids):
        raise ValueError("L0 must contain unique IDs from pool")
    groups = np.asarray([str(row["group_id"]) for row in rows])
    pool_groups = np.asarray([str(row["group_id"]) for row in pool])
    pool_rows = np.asarray([row_index[name] for name in pool_ids], dtype=np.int64)
    if not np.array_equal(groups[pool_rows], pool_groups):
        raise ValueError("candidate/cache group mismatch")
    names = sorted(set(groups.tolist()))
    excluded_names = set(map(str, excluded_fit_groups))
    if not excluded_names <= set(names):
        raise ValueError("excluded fit groups must exist in token rows")
    eligible_names = sorted(set(names) - excluded_names)
    if not 2 <= folds <= len(eligible_names):
        raise ValueError("at least two independent nonempty FIT folds required")
    if any(groups[row_index[name]] in excluded_names for name in initial_ids):
        raise ValueError("support labels must be FIT-only, never excluded D")
    windows = temporal_windows(rows, context=context, step=step)
    target_window = {int(window[-1]): index for index, window in enumerate(windows)}
    candidate_window = np.asarray([target_window.get(int(index), -1) for index in pool_rows])
    pool_index = {name: index for index, name in enumerate(pool_ids)}
    patch_count, dimension = tokens.shape[1:]
    n = len(pool)
    sample_owner, sample_patch, sample_labels, sample_weights = [], [], [], []
    diagnostics = []
    for index, row in enumerate(pool):
        patches, targets, weights = _sample_labeled_patches(row, grid_shape, seed, "score")
        sample_owner.extend([index] * len(patches))
        sample_patch.extend(patches.tolist())
        sample_labels.extend(targets.tolist())
        sample_weights.extend(weights.tolist())
        diagnostics.append(dict(frame_id=str(row["frame_id"]), group_id=str(row["group_id"]),
                                sampled_patch_ids=patches.tolist(), scored_foreground=int(targets.sum()),
                                scored_background=int((1. - targets).sum()),
                                has_temporal_context=bool(candidate_window[index] >= 0)))
    sample_owner = np.asarray(sample_owner, dtype=np.int64)
    sample_patch = np.asarray(sample_patch, dtype=np.int64)
    sample_labels, sample_weights = np.asarray(sample_labels), np.asarray(sample_weights)
    result = dict(e_obs=np.zeros(n), e_dyn=np.full(n, np.nan), e_persistence=np.full(n, np.nan))
    audit = dict(seed=int(seed), particle="one fixed raw ViT patch index", spatial_pooling=False,
                 temporal_correspondence="Eulerian image-grid cell, not physical object identity",
                 grid_shape=list(map(int, grid_shape)), latent_dim=int(dimension), context=int(context),
                 frame_step=float(step), epochs=int(epochs), batch_size=int(batch_size), max_windows=int(max_windows),
                 conditional=False, detector_cross_fitted=False, ensemble_weights=[.5, .5],
                 excluded_fit_groups=sorted(excluded_names), eligible_fit_groups=eligible_names,
                 all_fits_exclude_d=True, support_ids=initial_ids,
                 target="bbox center-inside weak foreground; nearest patch if no center inside",
                 readout="linear ridge on token channels only; no coordinates; balanced L0 classes per frame",
                 score="balanced binary cross entropy over up to 16 foreground and 16 background patches",
                 probability_clip=[PROBABILITY_CLIP, 1. - PROBABILITY_CLIP],
                 labels_used_for_ranking=True, folds=[])
    destination = Path(artifact_dir) if artifact_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        if any((destination / f"GLE_fold{i}.pt").exists() for i in range(folds)):
            raise FileExistsError("token fold artifacts already exist; use a new directory")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    # Every fold holds independent FIT groups. D is added only to its scored
    # groups; changing D data cannot affect which FIT samples train a model.
    held_sets = [list(map(str, part.tolist())) for part in
                 np.array_split(np.random.default_rng(seed).permutation(eligible_names), folds)]
    for index, name in enumerate(sorted(excluded_names)):
        held_sets[index % folds].append(name)
    for fold, held in enumerate(held_sets):
        held_names = sorted(held)
        fit_names = sorted(set(eligible_names) - set(held_names))
        fit_rows = np.flatnonzero(np.isin(groups, fit_names))
        held_pool = np.flatnonzero(np.isin(pool_groups, held_names))
        readout_ids = [name for name in initial_ids if groups[row_index[name]] in fit_names]
        if not readout_ids:
            raise ValueError(f"fold {fold}: no non-held L0 labels for readout")
        fold_rng = np.random.default_rng(seed + 1009 * fold)
        statistic_pairs = fold_rng.choice(len(fit_rows) * patch_count,
                                          size=min(100000, len(fit_rows) * patch_count), replace=False)
        statistic_rows = fit_rows[statistic_pairs // patch_count]
        statistic_patches = statistic_pairs % patch_count
        statistic_tokens = np.asarray(tokens[statistic_rows, statistic_patches], dtype=np.float64)
        if not np.isfinite(statistic_tokens).all():
            raise ValueError("nonfinite token in FIT statistics")
        token_mean = statistic_tokens.mean(axis=0)
        token_scale = statistic_tokens.std(axis=0).clip(1e-6)
        input_mean = np.asarray(inputs[fit_rows], dtype=np.float64).mean(axis=0)
        input_scale = np.asarray(inputs[fit_rows], dtype=np.float64).std(axis=0).clip(1e-6)
        normalized_inputs = ((inputs - input_mean) / input_scale).astype(np.float32)
        readout_rows, readout_patches, readout_labels, readout_weights = [], [], [], []
        for name in readout_ids:
            patches, labels, weights = _sample_labeled_patches(pool[pool_index[name]], grid_shape, seed, "readout")
            readout_rows.extend([row_index[name]] * len(patches))
            readout_patches.extend(patches.tolist())
            readout_labels.extend(labels.tolist())
            readout_weights.extend(weights.tolist())
        raw_readout = np.asarray(tokens[np.asarray(readout_rows), np.asarray(readout_patches)], dtype=np.float64)
        readout_z = (raw_readout - token_mean) / token_scale
        if not np.isfinite(readout_z).all():
            raise ValueError("nonfinite L0 readout token")
        coefficients = _ridge(_bias(readout_z), np.asarray(readout_labels), np.asarray(readout_weights))
        fitting_windows = windows[np.isin(groups[windows], fit_names).all(axis=1)]
        if set(groups[fit_rows]) & (set(held_names) | excluded_names):
            raise AssertionError("held or D row reached fold fitting")
        if len(fitting_windows) and not np.isin(groups[fitting_windows], fit_names).all():
            raise AssertionError("non-FIT temporal history reached dynamics fitting")
        possible_pairs = len(fitting_windows) * patch_count
        sampled_pairs = fold_rng.choice(possible_pairs, size=min(max_windows, possible_pairs), replace=False)
        train_windows = fitting_windows[sampled_pairs // patch_count]
        train_patches = sampled_pairs % patch_count
        train_z = particle_trajectories(tokens, train_windows, train_patches, token_mean, token_scale)
        step_scale = np.std(train_z[:, -1] - train_z[:, -2], axis=0).clip(1e-4) if len(train_z) else np.ones(dimension, dtype=np.float32)
        acc_scale = np.std(train_z[:, -1] - 2. * train_z[:, -2] + train_z[:, -3], axis=0).clip(1e-4) if len(train_z) else np.ones(dimension, dtype=np.float32)
        config = LangevinConfig(latent_dim=dimension, input_dim=6, context=context,
                                hidden_dim=64, heads=2, residual_rank=8, conditional=False)
        loss_history, koopman = [], None
        component_sums = {name: 0. for name in ("D", "JP", "W", "Z", "N")}
        latent_errors = {name: 0. for name in ("koopman", "langevin", "ensemble", "persistence")}
        separate_task = {name: np.zeros(n) for name in ("koopman", "langevin")}
        prediction_count = 0
        print(f"[token-fit] fold={fold} groups={len(fit_names)}/{len(held_names)} L0={len(readout_ids)} particles={len(train_z)}", flush=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 1009 * fold)
            model = ScaledTokenLangevin(config, torch.as_tensor(step_scale, dtype=torch.float32),
                                       torch.as_tensor(acc_scale, dtype=torch.float32)).to(torch_device)
            if len(train_z):
                projection = fold_rng.normal(size=(dimension, 2 * dimension)) / np.sqrt(dimension)
                phase = fold_rng.uniform(0., 2. * np.pi, size=2 * dimension)
                x_lift = _lift(train_z[:, -2].astype(np.float64), projection, phase)
                y_lift = _lift(train_z[:, -1].astype(np.float64), projection, phase)
                operator = _ridge(x_lift, y_lift)
                koopman = dict(projection=projection, phase=phase, operator=operator, latent_dim=dimension)
                train_inputs = normalized_inputs[train_windows]
                train_observed = (_probability(train_z.reshape(-1, dimension), coefficients) >= .5).reshape(train_z.shape[:2])
                z_tensor = torch.as_tensor(train_z, device=torch_device)
                input_tensor = torch.as_tensor(train_inputs, device=torch_device)
                observed_tensor = torch.as_tensor(train_observed, device=torch_device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                model.train()
                for epoch in range(epochs):
                    order = fold_rng.permutation(len(train_z))
                    loss_sum = 0.
                    for start in range(0, len(order), batch_size):
                        indices = torch.as_tensor(order[start:start + batch_size], device=torch_device)
                        output = model(z_tensor[indices], input_tensor[indices], observed_tensor[indices])
                        loss = (output["prediction"][:, -1] - z_tensor[indices, -1]).square().mean()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"nonfinite token GLE loss, fold {fold}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        loss_sum += float(loss.detach()) * len(indices)
                    loss_history.append(loss_sum / len(train_z))
                    print(f"[token-fit] fold={fold} epoch={epoch+1}/{epochs} mse={loss_history[-1]:.6f}", flush=True)
                model.eval()
            held_samples = np.flatnonzero(np.isin(sample_owner, held_pool))
            # Score the observed token separately, including frames lacking history.
            for start in range(0, len(held_samples), batch_size):
                sample_ids = held_samples[start:start + batch_size]
                owners, patches = sample_owner[sample_ids], sample_patch[sample_ids]
                values = np.asarray(tokens[pool_rows[owners], patches], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite observed scoring token")
                probabilities = _probability((values - token_mean) / token_scale, coefficients)
                np.add.at(result["e_obs"], owners, sample_weights[sample_ids] * _bce(probabilities, sample_labels[sample_ids]))
            if koopman is not None:
                eligible_pool = held_pool[candidate_window[held_pool] >= 0]
                result["e_dyn"][eligible_pool] = 0.
                result["e_persistence"][eligible_pool] = 0.
                dynamic_samples = held_samples[candidate_window[sample_owner[held_samples]] >= 0]
                with torch.no_grad():
                    for start in range(0, len(dynamic_samples), batch_size):
                        sample_ids = dynamic_samples[start:start + batch_size]
                        owners, patches = sample_owner[sample_ids], sample_patch[sample_ids]
                        batch_windows = windows[candidate_window[owners]]
                        values = particle_trajectories(tokens, batch_windows, patches, token_mean, token_scale)
                        observed = (_probability(values.reshape(-1, dimension), coefficients) >= .5).reshape(values.shape[:2])
                        output = model(torch.as_tensor(values, device=torch_device),
                                       torch.as_tensor(normalized_inputs[batch_windows], device=torch_device),
                                       torch.as_tensor(observed, device=torch_device))
                        langevin = output["prediction"][:, -1].cpu().numpy().astype(np.float64)
                        koopman_z = (_lift(values[:, -2].astype(np.float64), projection, phase) @ operator)[:, :dimension]
                        ensemble = .5 * koopman_z + .5 * langevin
                        if not np.isfinite(ensemble).all():
                            raise FloatingPointError(f"nonfinite patch prediction, fold {fold}")
                        label, weight = sample_labels[sample_ids], sample_weights[sample_ids]
                        for name, prediction in (("ensemble", ensemble), ("persistence", values[:, -2]),
                                                  ("koopman", koopman_z), ("langevin", langevin)):
                            loss = _bce(_probability(prediction, coefficients), label)
                            target = result["e_dyn"] if name == "ensemble" else result["e_persistence"] if name == "persistence" else separate_task[name]
                            np.add.at(target, owners, weight * loss)
                            latent_errors[name] += float(((prediction - values[:, -1])**2).sum())
                        for name, value in output["terms"].items():
                            component_sums[name] += float(value[:, -1].double().square().sum().cpu())
                        prediction_count += len(sample_ids)
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        scored_pool = held_pool[np.isfinite(result["e_dyn"][held_pool])]
        scale_pair_hash = hashlib.sha256(np.column_stack((statistic_rows, statistic_patches)).astype(np.int64).tobytes()).hexdigest()
        train_pair_hash = hashlib.sha256(np.column_stack((sampled_pairs // patch_count, train_patches)).astype(np.int64).tobytes()).hexdigest()
        fold_audit = dict(fold=fold, fit_groups=fit_names, held_groups=held_names, fit_rows=len(fit_rows),
                          excluded_fit_groups=sorted(excluded_names),
                          token_scaler_fit_groups=sorted(set(groups[statistic_rows].tolist())),
                          input_scaler_fit_groups=fit_names,
                          readout_fit_groups=sorted(set(groups[np.asarray(readout_rows)].tolist())),
                          gle_fit_groups=sorted(set(groups[train_windows].reshape(-1).tolist())),
                          koopman_fit_groups=sorted(set(groups[train_windows].reshape(-1).tolist())),
                          readout_support_ids=readout_ids,
                          readout_l0_ids=readout_ids, readout_frames=len(readout_ids),
                          readout_tokens=len(readout_labels), readout_foreground=int(np.sum(readout_labels)),
                          readout_background=int(len(readout_labels) - np.sum(readout_labels)),
                          readout_coordinates=False, token_scale_samples=len(statistic_pairs),
                          token_scale_pair_sha256=scale_pair_hash, fit_temporal_windows=len(fitting_windows),
                          possible_particle_windows=possible_pairs, fit_particles=len(train_z),
                          particle_sample_sha256=train_pair_hash, sampling_uses_gt=False,
                          held_frames=len(held_pool), scored_frames=len(scored_pool), scored_particles=prediction_count,
                          missing_temporal_context=int(np.sum(candidate_window[held_pool] < 0)),
                          training_loss=loss_history, dynamics_trained=koopman is not None,
                          model_class="ScaledTokenLangevin", information=dict(conditional=False,
                              target_latent=False, target_metadata=False, spatial_pooling=False,
                              observed="per-patch L0 readout g(z)>=0.5, not physical identity",
                              residual_min_lag=1, context=context),
                          mean_e_obs=float(np.mean(result["e_obs"][held_pool])) if len(held_pool) else None,
                          mean_e_dyn=float(np.mean(result["e_dyn"][scored_pool])) if len(scored_pool) else None,
                          mean_e_persistence=float(np.mean(result["e_persistence"][scored_pool])) if len(scored_pool) else None,
                          mean_task_errors={name: float(np.mean(values[scored_pool])) if len(scored_pool) else None
                                            for name, values in separate_task.items()},
                          latent_mse={name: total / (prediction_count * dimension) if prediction_count else None
                                      for name, total in latent_errors.items()},
                          component_rms={name: float(np.sqrt(total / (prediction_count * dimension))) if prediction_count else None
                                         for name, total in component_sums.items()})
        if destination is not None:
            checkpoint = dict(model_class="ScaledTokenLangevin", config=asdict(config), state_dict=state,
                              token_scaling=dict(mean=token_mean, scale=token_scale),
                              input_scaling=dict(mean=input_mean, scale=input_scale),
                              step_scale=step_scale, acc_scale=acc_scale,
                              readout=dict(coefficients=coefficients, coordinates=False, ridge=RIDGE,
                                           probability_clip=[PROBABILITY_CLIP, 1.-PROBABILITY_CLIP],
                                           target=audit["target"], l0_ids=readout_ids),
                              koopman=koopman, grid_shape=list(grid_shape), audit=fold_audit)
            with (destination / f"GLE_fold{fold}.pt").open("xb") as stream:
                torch.save(checkpoint, stream)
        audit["folds"].append(fold_audit)
    for index, row in enumerate(diagnostics):
        for key in ("e_obs", "e_dyn", "e_persistence"):
            row[key] = float(result[key][index]) if np.isfinite(result[key][index]) else None
    audit.update(pool_size=n, dense_frames=len(rows), patches_per_frame=patch_count,
                 dynamic_scores_available=int(np.isfinite(result["e_dyn"]).sum()))
    result.update(audit=audit, diagnostics=diagnostics)
    return result
