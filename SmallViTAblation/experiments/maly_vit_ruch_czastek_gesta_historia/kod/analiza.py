"""Czy ruch tokenów maski wyznacza kierunki swoiste dla narzędzia na innych filmach? Wersja dla gęstej historii.

Metoda bez zmian względem ../../maly_vit_ruch_czastek_maska_1px/analiza.py (ten sam podział filmów, korekta dryfu
tła, SVD niecentrowanych zmian, podłogi, KS). Różnice wyłącznie techniczne:
  - wybór stanów: co --co kroków albo jawna lista --kroki (np. 60,120 do porównania z pomiarem lokalnym),
  - ROC AUC liczone rangami (Mann-Whitney, remisy po 0,5) - venv Edenu nie ma sklearn; wynik identyczny,
  - dodatkowo AUC w czasie: średnia po obrazach testowych dla każdego przedziału osobno.
Tezę osłabia brak separacji od tła lub brak przewagi nad normą ruchu i losową podprzestrzenią.

    python analiza.py --dane <wyniki>/dane --wyniki <wyniki>/analiza_co10 --co 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata

KS = [1, 2, 4, 8, 16, 32]
W_CZASIE = ['norma_ruchu', 'norma_po_korekcie_tla', 'kierunek_sredni', 'svd_8', 'losowe_8', 'svd_32', 'losowe_32']


def roc_auc(labels, score):
    labels = np.asarray(labels).astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    assert n1 > 0 and n0 > 0
    r = rankdata(score)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_data(dane, co, kroki, od, do):
    meta = json.loads((dane / 'metadata.json').read_text(encoding='utf-8'))
    assert meta.get('status') == 'complete', meta.get('status')
    Zmm = np.load(dane / 'LATENT_GESTY.npy', mmap_mode='r')
    wszystkie = np.asarray(meta['kroki'], dtype=int)
    assert len(wszystkie) == len(Zmm) and np.all(np.diff(wszystkie) == 1)
    if kroki:
        steps = np.asarray(kroki, dtype=int)
    else:
        steps = wszystkie[(wszystkie % co == 0) & (wszystkie >= od) & (wszystkie <= (do if do >= 0 else wszystkie[-1]))]
    idx = np.searchsorted(wszystkie, steps)
    assert np.array_equal(wszystkie[idx], steps) and len(steps) >= 2 and np.all(np.diff(steps) > 0)
    Z = np.stack([np.asarray(Zmm[i]) for i in idx])                  # stany w RAM, float32
    masks = dict(np.load(dane / 'maski.npz'))
    assert Z.ndim == 4 and Z.shape[2:] == (600, 768), Z.shape
    assert len(meta['images']) == Z.shape[1]
    for k in ('tool', 'background', 'tool25', 'coverage'):
        assert masks[k].shape == Z.shape[1:3], (k, masks[k].shape)
    assert np.array_equal(masks['tool'], masks['coverage'] > 0)
    assert not np.any(masks['tool'] & masks['background'])
    assert np.all(masks['tool'].sum(1) > 0) and np.all(masks['background'].sum(1) > 0)
    return meta, Z, masks, steps


def movement(Z, masks, steps, image, t):
    delta = np.asarray(Z[t+1, image], dtype=np.float64) - Z[t, image]
    assert np.isfinite(delta).all()
    velocity = delta / float(steps[t+1] - steps[t])
    drift = velocity[masks['background'][image]].mean(axis=0)
    return velocity - drift, velocity, drift


def fit_directions(Z, masks, steps, indices, mask_key):
    C = np.zeros((768, 768), dtype=np.float64)
    mean = np.zeros(768, dtype=np.float64)
    n = 0
    for i in indices:
        for t in range(len(steps)-1):
            v, _, _ = movement(Z, masks, steps, i, t)
            x = v[masks[mask_key][i]]
            if not len(x):
                continue
            C += (x.T @ x) / len(x)
            mean += x.mean(axis=0)
            n += 1
    assert n > 0
    C /= n
    mean /= n
    vals, vec = eigh(C, check_finite=True)
    vals, vec = vals[::-1].clip(0), vec[:, ::-1].copy()
    # Konwencja znaku wyłącznie do reprodukowalnego zapisu, bez znaczenia semantycznego.
    for j in range(vec.shape[1]):
        if vec[np.argmax(np.abs(vec[:, j])), j] < 0:
            vec[:, j] *= -1
    unit = mean / max(np.linalg.norm(mean), 1e-30)
    assert np.allclose(vec.T @ vec, np.eye(768), atol=1e-7)
    return vals, vec, mean, unit


def evaluate(Z, masks, steps, indices, directions, mean_direction, random_directions, target_key, drift_reference):
    records = []
    for i in indices:
        positive, negative = masks[target_key][i], masks['background'][i]
        if not positive.any():
            continue
        keep = positive | negative
        labels = positive[keep].astype(int)
        store = {}
        for t in range(len(steps)-1):
            raw = (np.asarray(Z[t+1, i], dtype=np.float64) - Z[t, i]) / float(steps[t+1]-steps[t])
            # Nie wykorzystuj maski obrazu testowego do przekształcenia jego cech.
            v = raw - drift_reference[t]
            x = v[keep]
            scores = {'norma_ruchu': np.linalg.norm(raw[keep], axis=1),
                      'norma_po_korekcie_tla': np.linalg.norm(x, axis=1),
                      'kierunek_sredni': x @ mean_direction}
            p2 = np.cumsum((x @ directions[:, :max(KS)]) ** 2, axis=1)
            r2 = np.cumsum((x @ random_directions) ** 2, axis=1)
            for k in KS:
                scores[f'svd_{k}'] = np.sqrt(p2[:, k-1])
                scores[f'losowe_{k}'] = np.sqrt(r2[:, k-1])
            for name, score in scores.items():
                entry = store.setdefault(name, {'labels': [], 'score': [], 'auc_interval': []})
                entry['labels'].append(labels)
                entry['score'].append(score)
                entry['auc_interval'].append(roc_auc(labels, score))
        stats = {}
        for name, entry in store.items():
            y, s = np.concatenate(entry['labels']), np.concatenate(entry['score'])
            stats[name] = {'auc': roc_auc(y, s),
                           'auc_srednia_po_przedzialach': float(np.mean(entry['auc_interval'])),
                           'sredni_wynik_narzedzie': float(s[y == 1].mean()),
                           'sredni_wynik_tlo': float(s[y == 0].mean())}
        records.append({'image_index': int(i), 'tokens_tool': int(positive.sum()),
                        'tokens_background': int(negative.sum()), 'metrics': stats,
                        'auc_w_czasie': {n: store[n]['auc_interval'] for n in W_CZASIE}})
    summary = {}
    if records:
        for name in records[0]['metrics']:
            vals = [r['metrics'][name]['auc_srednia_po_przedzialach'] for r in records]
            summary[name] = {'mean_auc': float(np.mean(vals)), 'median_auc': float(np.median(vals)),
                             'min_auc': float(np.min(vals)), 'max_auc': float(np.max(vals))}
    w_czasie = {n: np.mean([r['auc_w_czasie'][n] for r in records], axis=0).tolist() for n in W_CZASIE} if records else {}
    for r in records:
        del r['auc_w_czasie']
    return {'by_image': records, 'summary': summary, 'auc_w_czasie_srednia_obrazow': w_czasie}


def plots(out, Z, masks, steps, basis, vals, test_ids, results):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(np.arange(1, 65), np.cumsum(vals)[:64] / max(vals.sum(), 1e-30))
    ax[0, 0].set(xlabel='Liczba osi SVD', ylabel='Udział energii ruchu', ylim=(0, 1.02),
                 title='Dopasowanie tylko na filmach uczących')
    summ = results['mask_1px']['summary']
    for pref, lab in [('svd_', 'Kierunki ruchu narzędzia'), ('losowe_', 'Losowe osie')]:
        ax[0, 1].plot(KS, [summ[f'{pref}{k}']['mean_auc'] for k in KS], 'o-', label=lab)
    ax[0, 1].axhline(summ['norma_ruchu']['mean_auc'], ls='--', color='gray', label='Norma całego ruchu')
    ax[0, 1].axhline(summ['kierunek_sredni']['mean_auc'], ls='-.', color='tab:red', label='Średni kierunek (ze znakiem)')
    ax[0, 1].axhline(.5, ls=':', color='black')
    ax[0, 1].set(xlabel='Liczba kierunków', ylabel='ROC AUC', title='Ruch narzędzie / tło: odłożone filmy')
    ax[0, 1].legend(fontsize=8)
    w = results['mask_1px']['auc_w_czasie_srednia_obrazow']
    srodki = (steps[1:] + steps[:-1]) / 2
    okno = max(1, (len(srodki) // 50))
    xs = np.convolve(srodki, np.ones(okno) / okno, mode='valid')      # 'valid': bez sztucznych spadków na brzegach
    for n, styl in [('kierunek_sredni', '-'), ('svd_8', '-'), ('losowe_8', '--'), ('norma_ruchu', ':')]:
        ax[1, 0].plot(xs, np.convolve(w[n], np.ones(okno) / okno, mode='valid'), styl, label=n)
    ax[1, 0].axhline(.5, ls=':', color='black')
    ax[1, 0].set(xlabel='Krok uczenia (środek przedziału)', ylabel='ROC AUC (średnia 4 obrazów testowych)',
                 title=f'AUC w czasie' + (f', średnia ruchoma {okno} przedziałów' if okno > 1 else ''))
    ax[1, 0].legend(fontsize=8)
    i = int(test_ids[0])
    tokens = np.flatnonzero(masks['tool'][i])
    # Wybór po masce, nie po wielkości efektu. Żadnego uśredniania trajektorii.
    selected = tokens[np.linspace(0, len(tokens)-1, min(6, len(tokens))).astype(int)]
    for j in selected:
        path = (np.asarray(Z[:, i, j], dtype=float)-Z[0, i, j]) @ basis[:, :2]
        ax[1, 1].plot(path[:, 0], path[:, 1], '-', lw=.8, label=f'token {j}')
        ax[1, 1].scatter(path[-1, 0], path[-1, 1], marker='x')
    ax[1, 1].set(xlabel='Zmiana rzutu na oś 1', ylabel='Zmiana rzutu na oś 2',
                 title=f'Osobne trajektorie, obraz {i}, {len(steps)} stanów')
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'RUCH_TOKENOW.png', dpi=150)
    plt.close(fig)


def write_summary(out, report):
    s = report['results']['mask_1px']['summary']
    st = report['steps']
    lines = [f"# Analiza: {len(st)} stanów, kroki {st[0]}..{st[-1]}, odstępy {sorted(set(np.diff(st).tolist()))}", '',
             f"Dopasowanie {len(report['fit_films'])} filmów, ocena {len(report['heldout_films'])} odłożonych filmów.",
             f"Do 90 procent energii skorygowanego ruchu potrzeba {report['directions_for_90pct_motion_energy']} osi SVD.", '',
             f"Norma całego ruchu: AUC {s['norma_ruchu']['mean_auc']:.4f}.",
             f"Norma po odjęciu referencyjnego dryfu: AUC {s['norma_po_korekcie_tla']['mean_auc']:.4f}.",
             f"Podpisany rzut na średni kierunek: AUC {s['kierunek_sredni']['mean_auc']:.4f}."]
    for k in KS:
        lines.append(f"{k} kierunków ruchu: AUC {s[f'svd_{k}']['mean_auc']:.4f}; losowe: {s[f'losowe_{k}']['mean_auc']:.4f}.")
    for key in ('mask_25pct_same_directions', 'mask_1px_directions_fitted_25pct'):
        s2 = report['results'][key]['summary']
        lines.append(f"{key}: średni kierunek {s2['kierunek_sredni']['mean_auc']:.4f}, svd_8 {s2['svd_8']['mean_auc']:.4f}, "
                     f"losowe_8 {s2['losowe_8']['mean_auc']:.4f}.")
    (out / 'PODSUMOWANIE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dane', required=True)
    ap.add_argument('--wyniki', required=True)
    ap.add_argument('--co', type=int, default=10)
    ap.add_argument('--kroki', default='', help='jawna lista kroków, np. 60,120 (zastępuje --co)')
    ap.add_argument('--od', type=int, default=0)
    ap.add_argument('--do', type=int, default=-1)
    a = ap.parse_args()
    dane, out = Path(a.dane), Path(a.wyniki)
    out.mkdir(parents=True, exist_ok=True)
    kroki = [int(x) for x in a.kroki.split(',') if x.strip()]
    meta, Z, masks, steps = load_data(dane, a.co, kroki, a.od, a.do)
    films = [m.get('film') or m['frame_id'].split('/')[-2] for m in meta['images']]
    unique = sorted(set(films))
    assert len(unique) >= 4, 'Za mało filmów do podziału poza dopasowaniem'
    order = np.random.default_rng(20260930).permutation(unique).tolist()   # ten sam podział co pomiar lokalny
    cut = max(2, int(len(order) * 2/3))
    train_films, test_films = order[:cut], order[cut:]
    train = [i for i, f in enumerate(films) if f in train_films]
    test = [i for i, f in enumerate(films) if f in test_films]
    assert not set(train_films) & set(test_films)
    print('kroki', steps[0], '..', steps[-1], f'({len(steps)})', 'train', train, 'test', test, flush=True)
    vals, basis, mean, unit = fit_directions(Z, masks, steps, train, 'tool')
    vals25, basis25, mean25, unit25 = fit_directions(Z, masks, steps, train, 'tool25')
    drift_reference = np.stack([np.mean([movement(Z, masks, steps, i, t)[2] for i in train], axis=0)
                                for t in range(len(steps)-1)])
    random_basis = np.linalg.qr(np.random.default_rng(20260930).normal(size=(768, max(KS))))[0]
    results = {'mask_1px': evaluate(Z, masks, steps, test, basis, unit, random_basis, 'tool', drift_reference),
               'mask_25pct_same_directions': evaluate(Z, masks, steps, test, basis, unit, random_basis, 'tool25', drift_reference),
               'mask_1px_directions_fitted_25pct': evaluate(Z, masks, steps, test, basis25, unit25, random_basis, 'tool', drift_reference)}
    cum = np.cumsum(vals) / max(vals.sum(), 1e-30)
    counts = {k: masks[k].sum(1).astype(int).tolist() for k in ('tool', 'tool25', 'background')}
    report = {'status': 'pomiar_eksploracyjny', 'seed': meta.get('seed'), 'slurm_job': meta.get('slurm_job'),
              'steps': steps.tolist(), 'shape': list(Z.shape),
              'fit_films': train_films, 'heldout_films': test_films, 'fit_indices': train, 'heldout_indices': test,
              'counts': counts, 'directions_for_90pct_motion_energy': int(np.searchsorted(cum, .9)+1),
              'energy_first_32': (vals[:32]/max(vals.sum(), 1e-30)).tolist(), 'results': results,
              'interpretation': 'Kierunki ruchu po odjęciu wspólnego dryfu tła. Nie dowód tożsamości ani przyczynowości.',
              'forecast_performed': False,
              'source_sha256': {'metadata.json': digest(dane / 'metadata.json'), 'maski.npz': digest(dane / 'maski.npz'),
                                'LATENT_GESTY.npy': meta.get('latent_sha256')}}
    np.savez_compressed(out / 'KIERUNKI_RUCHU.npz', directions=basis, eigenvalues=vals,
                        mean_velocity_contrast=mean, mean_direction=unit,
                        directions_25pct=basis25, eigenvalues_25pct=vals25, mean_direction_25pct=unit25,
                        random_directions=random_basis, train_indices=train, test_indices=test, steps=steps,
                        background_velocity_reference=drift_reference)
    (out / 'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    plots(out, Z, masks, steps, basis, vals, test, results)
    write_summary(out, report)
    print((out / 'PODSUMOWANIE.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
