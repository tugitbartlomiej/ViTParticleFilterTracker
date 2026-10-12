"""Czy ruch tokenów maski wyznacza kierunki swoiste dla narzędzia na innych filmach?

Tezę osłabia brak separacji od tła lub brak przewagi nad normą ruchu i losową
podprzestrzenią. Nie utożsamiamy osi największego ruchu z przyczynową tożsamością.
Wejście: nieuśredniony latent (czas, obraz, 600, 768) i maski z dane.py/pomiar.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / 'wyniki' / 'dane'
OUT = HERE / 'wyniki'
KS = [1, 2, 4, 8, 16, 32]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_data():
    meta = json.loads((DATA / 'metadata.json').read_text(encoding='utf-8'))
    Z = np.load(DATA / 'latent.npy', mmap_mode='r')
    masks = dict(np.load(DATA / 'maski.npz'))
    steps = np.asarray(meta['steps'], dtype=int)
    assert Z.ndim == 4 and Z.shape[2:] == (600, 768), Z.shape
    assert len(steps) == len(Z) and np.all(np.diff(steps) > 0), steps
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
                entry['labels'].extend(labels.tolist())
                entry['score'].extend(score.tolist())
                entry['auc_interval'].append(float(roc_auc_score(labels, score)))
        stats = {}
        for name, entry in store.items():
            y, s = np.asarray(entry['labels']), np.asarray(entry['score'])
            stats[name] = {'auc': float(roc_auc_score(y, s)),
                           'auc_srednia_po_przedzialach': float(np.mean(entry['auc_interval'])),
                           'sredni_wynik_narzedzie': float(s[y == 1].mean()),
                           'sredni_wynik_tlo': float(s[y == 0].mean())}
        records.append({'image_index': int(i), 'tokens_tool': int(positive.sum()),
                        'tokens_background': int(negative.sum()), 'metrics': stats})
    summary = {}
    if records:
        for name in records[0]['metrics']:
            vals = [r['metrics'][name]['auc_srednia_po_przedzialach'] for r in records]
            summary[name] = {'mean_auc': float(np.mean(vals)), 'median_auc': float(np.median(vals)),
                             'min_auc': float(np.min(vals)), 'max_auc': float(np.max(vals))}
    return {'by_image': records, 'summary': summary}


def plots(Z, masks, steps, basis, vals, test_ids, results):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ii = np.arange(Z.shape[1])
    ax[0, 0].bar(ii-.18, masks['tool'].sum(1), .36, label='choć 1 piksel')
    ax[0, 0].bar(ii+.18, masks['tool25'].sum(1), .36, label='co najmniej 25%')
    ax[0, 0].set(xlabel='Indeks obrazu', ylabel='Liczba tokenów', title='Maska w pełnej klatce: 600 tokenów')
    ax[0, 0].legend()
    ax[0, 1].plot(np.arange(1, 65), np.cumsum(vals)[:64] / max(vals.sum(), 1e-30))
    ax[0, 1].set(xlabel='Liczba osi SVD', ylabel='Udział energii ruchu', ylim=(0, 1.02), title='Dopasowanie tylko na filmach uczących')
    summ = results['mask_1px']['summary']
    for pref, lab in [('svd_', 'Kierunki ruchu narzędzia'), ('losowe_', 'Losowe osie')]:
        ax[1, 0].plot(KS, [summ[f'{pref}{k}']['mean_auc'] for k in KS], 'o-', label=lab)
    ax[1, 0].axhline(summ['norma_ruchu']['mean_auc'], ls='--', color='gray', label='Norma całego ruchu')
    ax[1, 0].axhline(.5, ls=':', color='black')
    ax[1, 0].set(xlabel='Liczba kierunków', ylabel='ROC AUC', title='Ruch narzędzie / tło: odłożone filmy')
    ax[1, 0].legend(fontsize=8)
    i = int(test_ids[0])
    tokens = np.flatnonzero(masks['tool'][i])
    # Wybór po masce, nie po wielkości efektu. Żadnego uśredniania trajektorii.
    selected = tokens[np.linspace(0, len(tokens)-1, min(6, len(tokens))).astype(int)]
    for j in selected:
        path = (np.asarray(Z[:, i, j], dtype=float)-Z[0, i, j]) @ basis[:, :2]
        ax[1, 1].plot(path[:, 0], path[:, 1], '.-', label=f'token {j}')
        ax[1, 1].scatter(path[-1, 0], path[-1, 1], marker='x')
    ax[1, 1].set(xlabel='Zmiana rzutu na oś 1', ylabel='Zmiana rzutu na oś 2', title=f'Osobne trajektorie, obraz {i}, {len(steps)} stanów')
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'RUCH_TOKENOW.png', dpi=170)
    plt.close(fig)


def write_report(report):
    s = report['results']['mask_1px']['summary']
    counts = report['counts']
    steps = report['steps']
    paragraphs = [
        '# Wynik pomiaru ruchu tokenów narzędzia',
        '',
        'Data: 2026-09-30. Analiza lokalna istniejących wag. Nie uruchamiano treningu.',
        f"UWAGA: źródłem jest {report['source_run']}. To próbny, wczesny bieg. Nie jest to pełna historia rozgrzewki do 1000 aktualizacji ani pomiar 100 stanów sieci.",
        '',
        '## Pytanie i aparat',
        '',
        'Czy maska od jednego piksela i osobne trajektorie tokenów ujawniają kierunki ruchu odróżniające narzędzie od tła?',
        f"Odczytano {len(steps)} stanów sieci, kroki: {steps}. Pełny zapis ma kształt {report['shape']}: czas, obraz, token, cecha.",
        '600 tokenów na pełną klatkę 480 × 320, 768 cech na token. Nie uśredniano tokenów ani punktów czasu.',
        f"Dopasowanie: {len(report['fit_films'])} filmów. Ocena: {len(report['heldout_films'])} innych filmów; te same przedziały uczenia, bez testu przyszłego czasu.",
        '',
        '## Maska i kierunki',
        '',
        f"Próg choć jednego piksela: {min(counts['tool'])}–{max(counts['tool'])} tokenów na obraz, mediana {np.median(counts['tool']):g}.",
        f"Próg 25 procent: {min(counts['tool25'])}–{max(counts['tool25'])} tokenów, mediana {np.median(counts['tool25']):g}.",
        f"Do 90 procent energii skorygowanego ruchu na obrazach uczących potrzeba {report['directions_for_90pct_motion_energy']} osi SVD. Nie jest to wymiar tożsamości narzędzia.",
        'Zapisano pełną bazę 768 osi w wyniki/KIERUNKI_RUCHU.npz. Kolumny directions są kierunkami w surowym układzie 768 aktywacji tej sieci. Zapis mean_direction zawiera kierunek średniego ruchu względem tła.',
        'Z ruchu odejmowano wspólny dryf. Na testowych obrazach referencja tła pochodzi wyłącznie z obrazów uczących, bez masek testowych w obliczaniu wyników tokenów.',
        '',
        '## Ocena poza dopasowaniem i podłogi',
        '',
        'ROC AUC mierzy rozdzielenie ruchów tokenów maski i odległego tła. Wartość 0,5 odpowiada brakowi uporządkowania. Wyniki są średnią po przedziałach i obrazach, nie odsetkiem poprawnie wykrytych narzędzi.',
        f"Norma całego ruchu: AUC {s['norma_ruchu']['mean_auc']:.4f}.",
        f"Norma po odjęciu referencyjnego dryfu: AUC {s['norma_po_korekcie_tla']['mean_auc']:.4f}.",
        f"Podpisany rzut na średni kierunek: AUC {s['kierunek_sredni']['mean_auc']:.4f}.",
    ]
    for k in KS:
        a, b = s[f'svd_{k}']['mean_auc'], s[f'losowe_{k}']['mean_auc']
        paragraphs.append(f'{k} kierunków ruchu: AUC {a:.4f}; losowa podprzestrzeń tej samej wielkości: {b:.4f}.')
    paragraphs += [
        '', '## Zastrzeżenia i czego wynik nie dowodzi', '',
        'Nie wybieramy najlepszego wymiaru na teście jako potwierdzonego hiperparametru. Pokazujemy wszystkie zaplanowane wymiary.',
        'To jeden przebieg sieci i niewielki zestaw filmów. Tokeny tego samego obrazu i kolejne przedziały nie są niezależnymi próbami.',
        'Próg jednego piksela obejmuje tokeny z bardzo małym udziałem narzędzia; nie czyni ich czystymi reprezentacjami narzędzia. Tło kontrolne jest oddalone od maski, więc zadanie nie obejmuje wszystkich trudnych tokenów brzegowych.',
        'SVD znajduje kierunki energii zmian. Wynik nie wykazuje przyczynowego wykorzystania ich przez detektor, odróżniania cystotomu od innych narzędzi ani poprawy uczenia.',
        'Nie odtworzono brakujących checkpointów i nie rekonstruowano cząstek ze średnich. Źródła wag i odstępy należy odczytywać z wyniki/dane/metadata.json.',
        'Prognoza Koopmana i Langevina nie była przedmiotem tego biegu: kierunki ruchu nie są jeszcze modelem przyszłej dynamiki. Dostępność gęstej historii rozstrzyga manifest pomiaru.',
        'Dla tych danych średni kierunek ruchu zawiera wstępny sygnał narzędzie/tło, lecz same dominujące osie ruchu nie stanowią przekonującego dowodu tożsamości. Porównanie z jedną losową bazą jest kontrolą orientacyjną, nie testem istotności.',
        '', '## Weryfikacja', '',
        'Sprawdzono kształty danych, rosnące kroki, rozdział filmów, rozłączność maski i tła, ortonormalność kierunków. Syntetyczna kontrola potwierdza dzielenie przez rzeczywiste odstępy kroków i usuwanie wspólnego dryfu bez mieszania tokenów.',
        'Sumy SHA256 źródłowych danych znajdują się w wyniki/REPORT.json. Wykres: wyniki/RUCH_TOKENOW.png.',
    ]
    (HERE/'WYNIKI.md').write_text('\n\n'.join(paragraphs).replace('\n\n\n\n', '\n\n'), encoding='utf-8')


def plot_masks(meta, masks):
    path = DATA / 'obrazy.npy'
    if not path.exists():
        return
    images = np.load(path)
    rows = (len(images)+2)//3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows*3.5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i >= len(images):
            continue
        ax.imshow(images[i])
        for j in np.flatnonzero(masks['tool'][i]):
            color = '#00ee99' if masks['tool25'][i, j] else '#ff9900'
            ax.add_patch(Rectangle(((j%30)*16, (j//30)*16), 16, 16,
                                   edgecolor=color, facecolor='none', linewidth=.85))
        ax.set_title(f"{i}: {meta['images'][i]['film']} | >0: {masks['tool'][i].sum()}, ≥25%: {masks['tool25'][i].sum()}", fontsize=9)
    fig.suptitle('600 tokenów na klatkę. Zielone: ≥25% maski. Pomarańczowe: dodatkowe tokeny od 1 piksela.', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'MASKI_600_TOKENOW.png', dpi=150)
    plt.close(fig)


def main():
    meta, Z, masks, steps = load_data()
    if len(steps) < 2:
        raise RuntimeError('Brak dwóch stanów tej samej sieci: nie można mierzyć ruchu.')
    films = [m.get('film') or m['frame_id'].split('/')[-2] for m in meta['images']]
    unique = sorted(set(films))
    assert len(unique) >= 4, 'Za mało filmów do podziału poza dopasowaniem'
    order = np.random.default_rng(20260930).permutation(unique).tolist()
    cut = max(2, int(len(order) * 2/3))
    train_films, test_films = order[:cut], order[cut:]
    train = [i for i, f in enumerate(films) if f in train_films]
    test = [i for i, f in enumerate(films) if f in test_films]
    assert not set(train_films) & set(test_films)
    print('fit', steps.tolist(), 'train', train, 'test', test, flush=True)
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
    report = {'status': 'pomiar_eksploracyjny', 'source_run': meta.get('run'), 'steps': steps.tolist(), 'shape': list(Z.shape),
              'fit_films': train_films, 'heldout_films': test_films, 'fit_indices': train, 'heldout_indices': test,
              'counts': counts, 'directions_for_90pct_motion_energy': int(np.searchsorted(cum, .9)+1),
              'energy_first_32': (vals[:32]/max(vals.sum(),1e-30)).tolist(), 'results': results,
              'interpretation': 'Kierunki ruchu po odjęciu wspólnego dryfu tła. Nie dowód tożsamości ani przyczynowości.',
              'forecast_performed': False,
              'forecast_reason': 'Ten raport dotyczy kierunków ruchu; prognoza wymaga osobnego podziału po czasie i dostatecznie gęstych wag.',
              'source_sha256': {p.name: digest(p) for p in [DATA/'metadata.json', DATA/'maski.npz', DATA/'latent.npy']}}
    np.savez_compressed(OUT / 'KIERUNKI_RUCHU.npz', directions=basis, eigenvalues=vals,
                        mean_velocity_contrast=mean, mean_direction=unit,
                        directions_25pct=basis25, eigenvalues_25pct=vals25, mean_direction_25pct=unit25,
                        random_directions=random_basis, train_indices=train, test_indices=test,
                        background_velocity_reference=drift_reference)
    (OUT/'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    plots(Z, masks, steps, basis, vals, test, results)
    plot_masks(meta, masks)
    write_report(report)
    print(json.dumps({'shape':list(Z.shape), 'steps':steps.tolist(), 'counts':counts,
                      'rank90':report['directions_for_90pct_motion_energy'],
                      'test':results['mask_1px']['summary']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
