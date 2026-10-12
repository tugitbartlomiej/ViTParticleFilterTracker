"""Porównuje Koopmana i GLE na tych samych 128 kanałach; przegrana z kontrolą obala przewagę."""
from dane import HERE, SOURCE, np, read, write_json, sha, load_data, load_bank
from metryki import measure, block_bootstrap
from datetime import datetime
from zoneinfo import ZoneInfo
from threadpoolctl import threadpool_limits
import time

RIDGES = [0.0001, 0.001, 0.01, 0.1, 1, 10]

def fit_koopman(q, w, dt, end, ridge, prior_identity=False):
    """Affine EDMD in the dictionary (1,x); the ridge prior is either zero or identity."""
    x = q - q[500]
    target = np.arange(500 + dt, end + 1, dt)
    X = x[target-dt].reshape(-1, 128)
    Y = x[target].reshape(-1, 128)
    if prior_identity: Y = Y - X
    weights = np.tile(w / len(target), len(target))
    weights = weights / weights.sum()
    mx, my = weights @ X, weights @ Y
    sx = np.maximum(np.sqrt(weights @ ((X-mx)**2)), 1e-10)
    sy = np.maximum(np.sqrt(weights @ ((Y-my)**2)), 1e-10)
    XX, YY = (X-mx)/sx, (Y-my)/sy
    G = XX.T @ (XX*weights[:, None])
    R = XX.T @ (YY*weights[:, None])
    gamma = ridge*np.trace(G)/128
    slope = np.linalg.solve(G + gamma*np.eye(128), R)
    raw = slope*sy[None, :]/sx[:, None]
    b = my - mx @ raw
    K = raw.T + (np.eye(128) if prior_identity else 0)
    fitted = X @ K.T + b
    expected = ((X-mx)/sx @ slope)*sy + my + (X if prior_identity else 0)
    error = float(np.max(np.abs(fitted-expected)))
    assert error < 1e-9
    return dict(kind='koopman', K=K, b=b, dt=dt, ridge=ridge,
                prior='identity' if prior_identity else 'zero', fit_targets=target,
                native_vs_scaled_error=error, spectral_radius=float(np.max(np.abs(np.linalg.eigvals(K)))),
                parameter_count=128*128+128)

def gle_from_bank(bank):
    model = bank['selected']
    assert model['D'] == 128 and not model['coupled'] and model['L'] == 2
    blocks = model['fg']['B']
    return dict(kind='langevin', A=blocks[:128].T, b=model['fg']['b'],
                memory=np.stack([blocks[128*(j+1):128*(j+2)].T for j in range(2)]),
                L=2, dt=model['dt'], ridge=model['ridge'], fit_targets=model['fit_targets'],
                parameter_count=3*128*128+128)

def rollout(q, origins, H, model):
    origins = np.asarray(origins, dtype=int)
    dt = model['dt']
    assert H % dt == 0
    anchor = q[500]
    x = q[origins]-anchor
    kind = model['kind']
    initial = x.copy()
    history = [q[origins-j*dt]-q[origins-(j+1)*dt] for j in range(2)] if kind=='langevin' else []
    velocity = q[origins]-q[origins-dt] if kind=='velocity' else None
    result = {0:q[origins].copy()}
    for h in range(dt, H+1, dt):
        if kind=='koopman': x = x @ model['K'].T + model['b']
        elif kind=='langevin':
            delta = x @ model['A'].T + model['b']
            for j in range(2): delta += history[j] @ model['memory'][j].T
            x = x + delta
            history = [delta, history[0]]
        elif kind=='velocity': x = initial + h/dt*velocity
        elif kind=='persistence': x = initial.copy()
        else: raise ValueError(kind)
        assert np.isfinite(x).all(), 'Nonfinite rollout'
        result[h] = x + anchor
    return result

def metric(pred, q, origins, H, w, base):
    return measure(pred, q[origins+H], q[origins], w, base['epsilon_q'], base['majority_q'])

def save_models(models):
    arrays = {}
    metadata = {}
    for dt, bank in models.items():
        metadata[str(dt)] = {}
        for name, m in bank.items():
            metadata[str(dt)][name] = {}
            for key, value in m.items():
                if isinstance(value, np.ndarray): arrays[f'dt{dt}__{name}__{key}'] = value
                else: metadata[str(dt)][name][key] = value
            if m['kind']=='koopman':
                augmented = np.eye(129)
                augmented[:128, :128] = m['K']
                augmented[:128, 128] = m['b']
                arrays[f'dt{dt}__{name}__augmented_K'] = augmented
    np.savez_compressed(HERE/'MACIERZE.npz', **arrays)
    write_json(HERE/'MODELE.json', metadata)

def main():
    threadpool_limits(4)
    assert not (HERE/'METRYKI.json').exists(), 'Existing result: do not refit using known test'
    start = time.time()
    data, hashes = load_data()
    models, choices, oldbanks, audits = {}, {}, {}, []
    for dt in (1, 10):
        q, w = data['s_2'], data['w_2']
        old = load_bank(dt)
        oldbanks[dt] = old
        origins = np.arange(700, 750, dt)
        hs = [1, 10, 50] if dt==1 else [10, 50]
        grid = []
        for ridge in RIDGES:
            model = fit_koopman(q, w, dt, 700, ridge)
            preds = rollout(q, origins, 50, model)
            vals = {str(H): metric(preds[H], q, origins, H, w, old['local'])[0] for H in hs}
            record = dict(ridge=ridge, mean_skill=float(np.mean([v['skill'] for v in vals.values()])),
                          metrics={H:{k:v[k] for k in ('MSE','skill','sign_accuracy')} for H,v in vals.items()})
            grid.append(record)
            print('Validation Koopman', dt, ridge, record['mean_skill'], flush=True)
        choice = max(grid, key=lambda r:r['mean_skill'])
        choices[str(dt)] = dict(selected=choice, grid=grid, selection_horizons=hs,
                               final_fit_target_max=799 if dt==1 else 790)
        end = 799 if dt==1 else 790
        direct = fit_koopman(q, w, dt, end, choice['ridge'])
        increment = fit_koopman(q, w, dt, end, old['local_L0']['ridge'], prior_identity=True)
        expected_K = np.eye(128) + old['local_L0']['fg']['B'].T
        error = max(float(np.max(np.abs(increment['K']-expected_K))),
                    float(np.max(np.abs(increment['b']-old['local_L0']['fg']['b']))))
        assert error < 1e-9, ('Increment fit disagrees with frozen L0', error)
        audits.append(dict(check='independent_identity_prior_fit_equals_source_L0', dt=dt, max_error=error))
        models[dt] = dict(langevin=gle_from_bank(old), koopman_stan=direct, koopman_przyrost=increment,
                          velocity=dict(kind='velocity',dt=dt), persistence=dict(kind='persistence',dt=dt))
    save_models(models)
    freeze = dict(completed=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),
                  channels0=data['channels0'], channels1=data['channels1'], K=128, PCA=False,
                  choices=choices, sources=hashes, matrices_sha=sha(HERE/'MACIERZE.npz'),
                  test_previously_examined=True, no_test_tuning=True,
                  code_sha={p:sha(HERE/p) for p in ('porownanie.py','dane.py','metryki.py')})
    write_json(HERE/'WYBOR_PRZED_TESTEM.json', freeze)
    print('Freeze before test', freeze['completed'], flush=True)
    records, predictions = {}, {}
    for dt in (1, 10):
        origins = np.arange(800, 951, dt)
        hs = [1,2,5,10,20,50] if dt==1 else [10,20,50]
        records[str(dt)] = {}
        source_metrics = read(SOURCE/f'METRYKI_DT{dt}.json')
        with np.load(SOURCE/f'PROGNOZY_DT{dt}.npz') as original_preds:
            for probe in (2,8):
                q, w = data[f's_{probe}'], data[f'w_{probe}']
                record = dict(origins=origins, horizons=hs, models={}, common_co10={})
                for name, m in models[dt].items():
                    pred = rollout(q, origins, 50, m)
                    scored = {}
                    for H in hs:
                        summary, details = metric(pred[H], q, origins, H, w, oldbanks[dt]['local'])
                        summary['bootstrap'] = block_bootstrap(details, dt, H)
                        scored[str(H)] = summary
                        if name in ('langevin','koopman_przyrost'):
                            old_name = 'selected' if name=='langevin' else 'local_L0'
                            old_score = source_metrics['probes'][str(probe)]['models'][old_name][str(H)]
                            assert abs(summary['MSE']-old_score['MSE']) < 1e-9
                            assert abs(summary['sign_accuracy']-old_score['sign_accuracy']) < 1e-9
                        if name=='langevin' and H in (1,10,50):
                            err = float(np.max(np.abs(pred[H]-original_preds[f'probe{probe}_H{H}'])))
                            assert err < 1e-9
                            audits.append(dict(check='independent_GLE_rollout_equals_source', dt=dt, probe=probe, H=H, max_error=err))
                        if name in ('langevin','koopman_stan','koopman_przyrost') and H in (1,10,50):
                            predictions[f'dt{dt}_probe{probe}_{name}_H{H}'] = pred[H]
                    record['models'][name] = scored
                    if dt==1:
                        ii = np.arange(0,len(origins),10)
                        record['common_co10'][name] = {str(H):metric(pred[H][ii],q,origins[ii],H,w,oldbanks[dt]['local'])[0] for H in (10,20,50)}
                    if name not in ('velocity','persistence'):
                        for origin in (800,950):
                            reference = rollout(q,[origin],50,m)
                            prefix = rollout(q[:origin+1],[origin],50,m)
                            poisoned = q.copy(); poisoned[origin+1:] = np.nan
                            poison = rollout(poisoned,[origin],50,m)
                            assert all(np.array_equal(reference[h],prefix[h]) and np.array_equal(reference[h],poison[h]) for h in reference)
                            audits.append(dict(check='prefix_poison',dt=dt,probe=probe,model=name,origin=origin,all128=True,passed=True))
                records[str(dt)][str(probe)] = record
    assert freeze['matrices_sha'] == sha(HERE/'MACIERZE.npz')
    np.savez_compressed(HERE/'PROGNOZY.npz', **predictions)
    write_json(HERE/'AUDYT.json', dict(all_passed=True,checks=audits))
    write_json(HERE/'METRYKI.json', dict(K=128,channels0=data['channels0'],records=records,
        seconds=time.time()-start,completed=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()))
    print('Done seconds',time.time()-start,flush=True)
    for dt in (1,10):
        print('MAIN',dt,{name:{H:(v['skill'],v['sign_accuracy']) for H,v in mm.items()} for name,mm in records[str(dt)]['2']['models'].items()},flush=True)

if __name__=='__main__': main()
