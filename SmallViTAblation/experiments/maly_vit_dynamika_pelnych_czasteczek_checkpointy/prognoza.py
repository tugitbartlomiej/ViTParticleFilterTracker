"""Can causal histories forecast full masked-token vectors beyond persistence?

Failure to beat persistence/velocity on frozen held-out origins falsifies a
pilot claim of improvement. All learned quantities use only completed states;
no future teacher forcing, gradients, minibatches or optimizer moments.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import json
import time
from datetime import datetime, timezone
import numpy as np
import psutil
from scipy.linalg import eigh
from dane import HERE, config, masked_trajectory, sha


class TokenRidge:
    """Independent per-token affine delta predictors, fit by dual ridge.

Each feature block is normalized by its centered training Gram trace/N.
The memory model averages state and prior-delta kernels. The output remains
a full vector in whichever space was supplied (768D or an rD PCA basis).
"""
    def __init__(self, history, alpha, memory=False):
        assert history.ndim == 3 and len(history) == 41
        self.memory = memory
        blocks = [history[1:-1].transpose(1, 0, 2)]
        if memory:
            blocks.append((history[1:-1] - history[:-2]).transpose(1, 0, 2))
        y = (history[2:] - history[1:-1]).transpose(1, 0, 2)
        assert y.shape[1] == 39 and all(x.shape[1] == 39 for x in blocks)
        self.y_mean = y.mean(axis=1)
        self.y_centered = y - self.y_mean[:, None]
        self.blocks = []
        grams = []
        for block in blocks:
            mean = block.mean(axis=1)
            centered = block - mean[:, None]
            gram = centered @ centered.transpose(0, 2, 1)
            scale = np.maximum(np.trace(gram, axis1=1, axis2=2) / gram.shape[1], 1e-30)
            self.blocks.append((centered, mean, scale))
            grams.append(gram / scale[:, None, None])
        gram = sum(grams) / len(grams)
        eye = np.eye(gram.shape[1])[None]
        self.inverse = np.linalg.solve(gram + alpha * eye, np.broadcast_to(eye, gram.shape))
        self.training_pairs = y.shape[1]

    def next(self, state, previous):
        values = [state]
        if self.memory:
            values.append(state - previous)
        kernels = []
        for value, (train, mean, scale) in zip(values, self.blocks):
            kernels.append(np.einsum('pnd,pd->pn', train, value - mean) / scale[:, None])
        kernel = sum(kernels) / len(kernels)
        coefficients = np.einsum('pn,pnm->pm', kernel, self.inverse)
        delta = self.y_mean + np.einsum('pn,pnd->pd', coefficients, self.y_centered)
        return state + delta


def common_pca(history, weights, rank_max=16):
    mean = history.mean(axis=0)
    centered = history - mean[None]
    weighted = centered * np.sqrt(weights[None, :, None] / len(history))
    rows = weighted.reshape(-1, 768)
    covariance = rows.T @ rows
    values, basis = eigh(covariance, subset_by_index=[768-rank_max, 767], check_finite=True)
    values, basis = values[::-1].clip(0), basis[:, ::-1].copy()
    assert np.allclose(basis.T @ basis, np.eye(rank_max), atol=1e-10)
    return mean, basis, values / max(np.trace(covariance), 1e-30)


def rollout(history, horizon_steps, alpha=None, basis=None, memory=False, baseline=None, rho=0.8):
    largest = max(horizon_steps)
    z0 = history[-1]
    if baseline:
        v = history[-1] - history[-2]
        result = {}
        for h in horizon_steps:
            multiplier = 0 if baseline == 'persistence' else h if baseline == 'velocity' else (1-rho**h)/(1-rho)
            result[h] = z0 + multiplier * v
        return result
    if basis is None:
        embedded = history
    else:
        embedded = history @ basis
    model = TokenRidge(embedded, alpha, memory)
    previous, state = embedded[-2].copy(), embedded[-1].copy()
    initial = state.copy()
    result = {}
    for h in range(1, largest+1):
        nxt = model.next(state, previous)
        previous, state = state, nxt
        assert np.isfinite(state).all(), 'Non-finite autonomous rollout'
        if h in horizon_steps:
            result[h] = state.copy() if basis is None else z0 + (state-initial) @ basis.T
    return result


def score(prediction, truth, initial, mi, weights, basis=None):
    error2 = np.square(prediction-truth).sum(axis=-1)
    true_delta = truth-initial
    predicted_delta = prediction-initial
    actual2 = np.square(true_delta).sum(axis=-1)
    pred2 = np.square(predicted_delta).sum(axis=-1)
    rms = float(np.sqrt(weights @ error2))
    persistence = float(np.sqrt(weights @ actual2))
    valid = (actual2 > 1e-20) & (pred2 > 1e-20)
    cosine = np.full(len(mi), np.nan)
    cosine[valid] = np.sum(true_delta[valid]*predicted_delta[valid], axis=-1)/np.sqrt(actual2[valid]*pred2[valid])
    by_image = []
    for i in range(12):
        keep = mi == i
        w = weights[keep]/weights[keep].sum()
        by_image.append(dict(image=i, rms_l2=float(np.sqrt(w @ error2[keep])),
                             persistence_rms_l2=float(np.sqrt(w @ actual2[keep])),
                             mean_l2=float(w @ np.sqrt(error2[keep])),
                             equal_token_rms_l2=float(np.sqrt(error2[keep].mean()))))
    result = dict(rms_l2=rms, persistence_rms_l2=persistence,
                  mean_l2=float(weights @ np.sqrt(error2)),
                  mean_l2_relative_to_persistence=float((weights @ np.sqrt(error2))/max(weights @ np.sqrt(actual2), 1e-30)),
                  relative_to_persistence=rms/max(persistence, 1e-30),
                  relative_to_full_state=rms/max(float(np.sqrt(weights @ np.square(truth).sum(-1))), 1e-30),
                  mean_valid_displacement_cosine=float(weights[valid] @ cosine[valid]/weights[valid].sum()) if valid.any() else None,
                  valid_cosine_token_count=int(valid.sum()),
                  equal_image_equal_token_rms_l2=float(np.sqrt(np.mean([x['equal_token_rms_l2']**2 for x in by_image]))),
                  by_image=by_image,
                  token_error_l2=np.sqrt(error2).tolist(), token_true_displacement_l2=np.sqrt(actual2).tolist())
    if basis is not None:
        outside = true_delta - (true_delta @ basis) @ basis.T
        floor2 = np.square(outside).sum(-1)
        projected_error = (prediction-truth) @ basis
        decomposition = np.square(projected_error).sum(-1) + floor2
        assert np.allclose(error2, decomposition, atol=1e-8, rtol=1e-8)
        result['pca_outside_motion_floor_rms_l2'] = float(np.sqrt(weights @ floor2))
        result['pca_outside_motion_fraction_squared'] = float((weights @ floor2)/max(weights @ actual2, 1e-30))
    return result


def validate_math():
    # Independent geometric property: reconstruction preserves the exact
    # orthogonal component and squared error decomposes by Pythagoras.
    rng = np.random.default_rng(13)
    q, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    z = rng.normal(size=(5, 12)); dz = rng.normal(size=(5, 12)); c = rng.normal(size=(5, 3))
    prediction = z + c @ q.T
    assert np.allclose(prediction @ (np.eye(12)-q@q.T), z @ (np.eye(12)-q@q.T))
    error = prediction-(z+dz)
    assert np.allclose(np.square(error).sum(), np.square(error@q).sum()+np.square(dz-(dz@q)@q.T).sum())
    # A constant full-vector velocity is recovered by the affine intercept;
    # this catches target/index mistakes and recursive teacher forcing.
    v = rng.normal(size=(3, 8))*0.01; z0 = rng.normal(size=(3, 8))
    hist = z0[None] + np.arange(41)[:, None, None]*v[None]
    for memory in [False, True]:
        model = TokenRidge(hist, 0.1, memory)
        prev, cur = hist[-2], hist[-1]
        for _ in range(10): prev, cur = cur, model.next(cur, prev)
        assert np.allclose(cur, hist[-1]+10*v, atol=1e-10)


def main():
    validate_math()
    audit = json.loads((HERE/'AUDYT_ZGODNOSCI.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'passed'
    cfg = config(); started = time.time()
    z, steps, mi, ti, coverage, weights, meta = masked_trajectory()
    horizons = [h//cfg['sample_every_updates'] for h in cfg['horizons_updates']]
    assert all(h*cfg['sample_every_updates'] == H for h,H in zip(horizons,cfg['horizons_updates']))
    families = ['full_ar1', 'pca_ar1', 'pca_ar2']
    validation = []
    choices = {}
    for origin in cfg['validation_origins']:
        t = int(np.flatnonzero(steps == origin)[0])
        hist = z[t-40:t+1].astype(np.float64)
        _, U, energy = common_pca(hist, weights)
        for family in families:
            for rank in ([None] if family == 'full_ar1' else cfg['pca_ranks']):
                Q = None if rank is None else U[:, :rank]
                for alpha in cfg['ridge_alphas']:
                    forecasts = rollout(hist, horizons, alpha=alpha, basis=Q, memory=family=='pca_ar2')
                    for h,pred in forecasts.items():
                        s = score(pred, z[t+h], hist[-1], mi, weights, Q)
                        validation.append(dict(family=family, rank=rank, alpha=alpha, origin=origin,
                                               horizon=h*10, ratio=s['relative_to_persistence']))
        print('VALIDATED', origin, 'elapsed', round(time.time()-started,1), flush=True)
    for family in families:
        candidates = []
        keys = sorted({(r['rank'] or 0,r['alpha']) for r in validation if r['family']==family})
        for rank,alpha in keys:
            selected = [r['ratio'] for r in validation if r['family']==family and (r['rank'] or 0)==rank and r['alpha']==alpha]
            assert len(selected) == len(cfg['validation_origins'])*len(horizons)
            candidates.append((float(np.mean(selected)), rank, alpha))
        loss,rank,alpha = min(candidates)
        choices[family] = dict(rank=rank or None, alpha=alpha, mean_validation_ratio=loss)
    (HERE/'WYBOR_WALIDACJA.json').write_text(json.dumps(dict(frozen_before_test=True, maximum_validation_target=600,
               selection_objective='mean coverage-weighted RMS ratio to persistence over validation origins/horizons',
               selected=choices, grid_scores=validation), indent=2)+'\n')
    print('FROZEN_CHOICES', json.dumps(choices), flush=True)
    records = []
    for origin in cfg['test_origins']:
        t = int(np.flatnonzero(steps==origin)[0])
        hist = z[t-40:t+1].astype(np.float64)
        _, U, energy = common_pca(hist, weights)
        specifications = [('persistence',dict(baseline='persistence'),None),
                          ('velocity',dict(baseline='velocity'),None),
                          ('damped_velocity',dict(baseline='damped_velocity',rho=cfg['damped_velocity_rho']),None)]
        for family,choice in choices.items():
            Q = None if choice['rank'] is None else U[:,:choice['rank']]
            specifications.append((family,dict(alpha=choice['alpha'],basis=Q,memory=family=='pca_ar2'),Q))
        for family,kwargs,Q in specifications:
            forecasts = rollout(hist,horizons,**kwargs)
            for h,pred in forecasts.items():
                r = score(pred,z[t+h],hist[-1],mi,weights,Q)
                r.update(family=family,origin=origin,horizon_updates=h*10,
                         latest_state_used_for_fit=origin,fit_pairs=39 if family in families else 0,
                         pca_past_variance_explained=float(energy[:Q.shape[1]].sum()) if Q is not None else None)
                records.append(r)
        print('TESTED',origin,'elapsed',round(time.time()-started,1),flush=True)
    summary = {}
    for family in sorted({r['family'] for r in records}):
        summary[family] = {}
        for H in cfg['horizons_updates']:
            rows = [r for r in records if r['family']==family and r['horizon_updates']==H]
            summary[family][str(H)] = dict(mean_ratio_to_persistence=float(np.mean([r['relative_to_persistence'] for r in rows])),
                  mean_rms_l2=float(np.mean([r['rms_l2'] for r in rows])),
                  minimum_ratio=float(min(r['relative_to_persistence'] for r in rows)),
                  maximum_ratio=float(max(r['relative_to_persistence'] for r in rows)))
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='complete_pilot',
                  elapsed_seconds=time.time()-started,source_sha256=meta['latent_sha256'],
                  config_sha256=sha(HERE/'CONFIG.json'),one_training_trajectory=True,
                  sampled_states=101,masked_tokens=218,vector_dimension=768,
                  input_array_ram_bytes=z.nbytes,selected=choices,summary=summary,
                  process_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None),
                  ram_available_after_bytes=psutil.virtual_memory().available,
                  particle_indices=[dict(image=int(i),token=int(j),coverage=float(c),weight=float(w)) for i,j,c,w in zip(mi,ti,coverage,weights)],
                  images=[dict(index=i,film=r['film'],original_role=r['role']) for i,r in enumerate(meta['images'])],
                  observations_independent=False,metadata_batch_history_available=False,
                  future_access_in_fit=False,teacher_forcing=False,records=records)
    (HERE/'METRYKI_PILOT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
