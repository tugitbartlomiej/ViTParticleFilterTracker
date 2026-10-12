"""Czy dynamika czastek wskazuje bledy na innych operacjach ponad statyke i historie?

Brak poprawy wzgledem R i placebo P podwaza zasadnosc kolejnego etapu treningu.
Ten skrypt NIE trenuje detektora. Sam import niczego nie uruchamia.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dane import sha256
from .modele import (circular_placebo, descriptors, dynamic_features, fit_direction,
                     history_indices, iou_xywh)


def metrics(y, score, fraction, probability=True):
    if not len(y):
        raise ValueError('Pusta ocena')
    top = np.argsort(-score, kind='stable')[:max(1, int(np.ceil(len(y) * fraction)))]
    both = len(np.unique(y)) == 2
    return dict(n=len(y), miss_rate=float(y.mean()),
                auc=float(roc_auc_score(y, score)) if both else None,
                ap=float(average_precision_score(y, score)) if both else None,
                brier=float(brier_score_loss(y, score)) if probability else None,
                top_precision=float(y[top].mean()))


def paired_summary(records, cfg):
    """Najpierw srednia po ziarnach dla operacji, bootstrap par po operacjach."""
    summaries = {}
    rng = np.random.default_rng(12091)
    for target in ('detection', 'localization'):
        for comparator in ('R', 'P', 'K', 'L'):
            for measure in ('auc', 'ap', 'top_precision'):
                by_operation = {}
                for r in records:
                    a = r['targets'][target]['F'][measure]
                    b = r['targets'][target][comparator][measure]
                    if a is not None and b is not None:
                        by_operation.setdefault(r['operation'], []).append(a - b)
                values = np.asarray([np.mean(v) for v in by_operation.values()])
                key = f'{target}:F-{comparator}:{measure}'
                if len(values) < 2:
                    summaries[key] = dict(operations=len(values), mean=None, ci95=None)
                    continue
                boot = rng.choice(values, (cfg['bootstrap'], len(values)), replace=True).mean(1)
                summaries[key] = dict(operations=len(values), mean=float(values.mean()),
                                      ci95=np.quantile(boot, [.025, .975]).tolist(),
                                      caveat='Eksploracyjny przedzial; zalezne dopasowania CV, wiele porownan bez korekty.')
    return summaries


def run(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['folds'] < 3 or cfg['observation_variance'] <= 0 or not 0 < cfg['top_fraction'] < 1:
        raise ValueError('Nieprawidlowa konfiguracja')
    if args.out.exists():
        raise FileExistsError('Nowy bieg wymaga nowego katalogu; nie nadpisujemy wynikow')
    audit_path = args.data.with_suffix('.audit.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['split'] not in ('train', 'val'):
        raise ValueError('Zakaz doboru i eksploracji na zbiorze testowym')
    with np.load(args.data, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if len(set(data['frame_id'])) != len(data['frame_id']):
        raise ValueError('Powtorzone frame_id')
    for name in ('tokens', 'mask', 'gt_box', 'pred_box', 'confidence', 'tip', 'size'):
        if not np.isfinite(data[name]).all():
            raise ValueError(f'Niefinitywne dane {name}')
    if ((data['confidence'] < 0) | (data['confidence'] > 1)).any():
        raise ValueError('Pewnosc detektora poza [0,1]')
    past = history_indices(data, cfg['history'], cfg['frame_step'])
    valid = np.flatnonzero((past >= 0).all(1))
    operations = np.unique(data['operation'][valid])
    if len(operations) < max(8, 2 * cfg['folds']):
        raise ValueError('Za malo operacji z pelna historia dla podzialu A/B/C')
    overlap = iou_xywh(data['pred_box'], data['gt_box'])
    targets = dict(localization=(overlap < cfg['iou_threshold']).astype(int))
    targets['detection'] = (targets['localization'].astype(bool) |
                            (data['confidence'] < cfg['confidence_threshold'])).astype(int)
    reference = np.load(args.reference_direction, allow_pickle=False) if args.reference_direction else None
    metrics_by_operation, model_audits, predictions = [], [], []
    for seed in cfg['seeds']:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(operations)
        chunks = np.array_split(shuffled, cfg['folds'])
        for fold, c_ops in enumerate(chunks):
            remaining = np.array([op for op in shuffled if op not in c_ops])
            a_ops, b_ops = np.array_split(remaining, 2)
            if set(a_ops) & set(b_ops) or set(a_ops) & set(c_ops) or set(b_ops) & set(c_ops):
                raise AssertionError('Przeciek podzialu operacji')
            a_all = np.flatnonzero(np.isin(data['operation'], a_ops))
            b = valid[np.isin(data['operation'][valid], b_ops)]
            c = valid[np.isin(data['operation'][valid], c_ops)]
            if min(len(b), len(c)) < cfg['min_frames_per_partition']:
                raise ValueError('Za malo klatek w B/C')
            direction = fit_direction(data['tokens'], data['mask'], a_all, cfg['ridge'])
            signal, residual, geom = descriptors(data, direction)
            k, l, f, spectrum = dynamic_features(signal, residual, past, a_all, cfg)
            g = np.c_[data['confidence'], geom]
            identity = np.c_[g, signal]
            static = np.c_[identity, residual]
            # Wszystkie dodatkowe warianty widza te same opoznienia i roznice.
            history = np.full((len(static), static.shape[1] * cfg['history']), np.nan)
            history[valid] = static[past[valid]].reshape(len(valid), -1)
            difference = np.full_like(signal, np.nan)
            difference[valid] = signal[valid] - signal[past[valid, 0]]
            r = np.c_[static, history, difference]
            dynamic = np.c_[k, l, f]
            placebo = circular_placebo(dynamic, data, np.r_[b, c], seed + fold, cfg['history'])
            arms = dict(G=g, I=identity, R=r, K=np.c_[r, k], L=np.c_[r, k, l],
                        F=np.c_[r, dynamic], P=np.c_[r, placebo])
            if not all(np.isfinite(x[np.r_[b, c]]).all() for x in arms.values()):
                raise ValueError('Niefinitywne cechy; sprawdz stabilnosc modeli, nie usuwaj pojedynczych wynikow')
            model_audit = dict(seed=seed, fold=fold, A=a_ops.tolist(), B=b_ops.tolist(), C=c_ops.tolist(),
                               direction=direction.tolist(), spectrum=spectrum,
                               feature_dimensions={name: x.shape[1] for name, x in arms.items()},
                               reference_cosine=None,
                               nominal_nis_C_mean=float(f[c, 2].mean()),
                               nominal_nis_note='Wartosc okolo 1 nie dowodzi poprawnej kalibracji; R zalozone, Q dopasowane.')
            if reference is not None:
                if reference.shape != direction.shape or np.linalg.norm(reference) == 0:
                    raise ValueError('Wektor referencyjny niezgodny z baza tokenow')
                model_audit['reference_cosine'] = float(direction @ reference / np.linalg.norm(reference))
            # Prog sygnalu tylko z A. Dalsze zestawienie nie jest interwencja przyczynowa.
            strong_threshold = float(np.median(signal[a_all, 4]))
            model_audit['strong_signal_threshold_A'] = strong_threshold
            result_scores = {}
            for target_name, y in targets.items():
                if len(np.unique(y[b])) != 2:
                    raise ValueError(f'Jedna klasa bledow w B: {target_name}, seed={seed}, fold={fold}')
                scores = {'U': 1 - data['confidence'][c]}
                # Jednostka naturalna to operacja: kazda ma taka sama laczna wage.
                weights = np.array([1 / np.sum(data['operation'][b] == op) for op in data['operation'][b]])
                weights *= len(weights) / weights.sum()
                for name, x in arms.items():
                    classifier = make_pipeline(StandardScaler(), LogisticRegression(
                        C=cfg['logistic_c'], max_iter=3000, random_state=seed))
                    classifier.fit(x[b], y[b], logisticregression__sample_weight=weights)
                    if classifier[-1].n_iter_.max() >= 3000:
                        raise RuntimeError('Regresja logistyczna nie osiagnela zbieznosci')
                    scores[name] = classifier.predict_proba(x[c])[:, 1]
                result_scores[target_name] = scores
            for operation in c_ops:
                mask = data['operation'][c] == operation
                if not mask.any():
                    continue
                metrics_by_operation.append(dict(seed=seed, fold=fold, operation=str(operation),
                    targets={target: {name: metrics(targets[target][c][mask], score[mask],
                              cfg['top_fraction'], name != 'U') for name, score in scores.items()}
                             for target, scores in result_scores.items()}))
            for j, i in enumerate(c):
                row = dict(seed=seed, fold=fold, frame_id=str(data['frame_id'][i]),
                           operation=str(data['operation'][i]), number=int(data['number'][i]),
                           iou=float(overlap[i]), confidence=float(data['confidence'][i]),
                           contrast=float(signal[i, 4]), strong_signal=bool(signal[i, 4] >= strong_threshold),
                           miss_detection=int(targets['detection'][i]), miss_localization=int(targets['localization'][i]))
                for target, scores in result_scores.items():
                    for name, score in scores.items():
                        row[f'{target}_{name}'] = float(score[j])
                predictions.append(row)
            model_audits.append(model_audit)
    # Katalog tworzymy dopiero po pomyslnym zakonczeniu wszystkich podzialow.
    args.out.mkdir(parents=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    code_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    report = dict(status='completed_diagnostic_only', config=cfg, source_commit=commit,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  sha256=dict(data=sha256(args.data), audit=sha256(audit_path), config=sha256(args.config), code=code_hashes),
                  frames_loaded=len(data['frame_id']), frames_with_history=len(valid),
                  input_audit=audit, operations=len(operations), paired=paired_summary(metrics_by_operation, cfg),
                  per_operation=metrics_by_operation,
                  limitations=['Nie zmierzono korzysci treningu.', 'Nominalny Kalman; brak pomiaru wymuszenia kamery.',
                               'Stara baza PCA i guard rejestratora; patrz input_audit.',
                               'Maski i geometria GT: diagnostyka oznaczonej puli, nie selekcja bez etykiet.'])
    (args.out / 'REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (args.out / 'MODELE.json').write_text(json.dumps(model_audits, indent=2, allow_nan=False), encoding='utf-8')
    with (args.out / 'PREDYKCJE.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--reference-direction', type=Path, help='Tylko porownanie cosinusa, bez uzycia do oceniania')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
