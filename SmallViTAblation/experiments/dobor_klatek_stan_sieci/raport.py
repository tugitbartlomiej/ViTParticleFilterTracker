#!/usr/bin/env python3
"""Build a Polish report from completed pilot artifacts; never run an experiment."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


ARMS = ('random', 'readout_loss', 'koopman_langevin', 'placebo', 'random_large')
LABELS = dict(random='Losowy 1500', readout_loss='Błąd odczytu',
              koopman_langevin='Koopman + Langevin', placebo='Placebo',
              random_large='Losowy 3000')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')
STYLES = ('-', '--', '-', ':', '-.')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, lo=None, hi=None):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), f'{name}: expected a finite number')
    _require((lo is None or value >= lo) and (hi is None or value <= hi),
             f'{name}: number outside allowed range')
    return float(value)


def _near(actual, expected, name):
    _require(math.isclose(_number(actual, name), expected, rel_tol=1e-9, abs_tol=1e-10),
             f'{name}: inconsistent with source values')


def _exposure(curves):
    """Integrate only within shared observed exposure, without extrapolation."""
    lo = max(c[0]['image_presentations_proxy'] for c in curves.values())
    hi = min(c[-1]['image_presentations_proxy'] for c in curves.values())
    _require(hi > lo, 'training curves have no common exposure range')
    areas = {}
    for arm, curve in curves.items():
        xs = [p['image_presentations_proxy'] for p in curve]
        ys = [p['ap'] for p in curve]

        def interpolate(x):
            index = min(max(bisect.bisect_right(xs, x) - 1, 0), len(xs) - 2)
            return ys[index] + (ys[index + 1] - ys[index]) * ((x - xs[index]) / (xs[index + 1] - xs[index]))

        grid = sorted({lo, hi, *(x for x in xs if lo < x < hi)})
        area = sum((b - a) * (interpolate(a) + interpolate(b)) / 2 for a, b in zip(grid, grid[1:]))
        areas[arm] = area / (hi - lo)
    return dict(common_range=[lo, hi], normalized_area=areas)


def validate_inputs(pilot, selection, diagnostics=None):
    """Reject contradictory summaries before producing any user-facing result."""
    _require(pilot.get('status') == 'complete' and selection.get('status') == 'complete',
             'pilot and selection must be complete')
    _require(pilot.get('checkpoint_rule') == 'last_epoch'
             and pilot.get('metric') == 'mAP_0.5:0.95' and pilot.get('evaluation_set') == 'val',
             'primary endpoint must be last-epoch AP on val')
    _require(pilot.get('test_evaluated') is False and selection.get('test_used') is False,
             'test must not be used')
    _require(pilot.get('statistical_significance_tested') is False,
             'this report supports the exploratory single-seed pilot only')
    _require(pilot.get('epochs') == 10 and pilot.get('training_seed') == 42
             and pilot.get('selection_seed') == 42, 'expected ten epochs and fixed seeds 42')
    rows = pilot.get('rows', [])
    _require(len(rows) == 5 and {r.get('strategy') for r in rows} == set(ARMS),
             'expected one completed row for each of five arms')
    by_arm = {r['strategy']: r for r in rows}
    _require(len({r.get('run_id') for r in rows}) == 5
             and all(isinstance(r.get('run_id'), str) and r['run_id'] for r in rows),
             'run ids must be unique and nonempty')
    sizes = set()
    curves = pilot.get('training_curves', {})
    _require(set(curves) == set(ARMS), 'missing training curves')
    selected = selection.get('selection', {}).get('arms', {})
    _require(set(selected) == set(ARMS), 'selection must contain five arms')
    feature = selection.get('feature_meta', {})
    _require(feature.get('feature') == 'last_raw_vit_patch'
             and feature.get('spatial_pooling') is False and feature.get('labels_used') is False,
             'expected raw patch tokens extracted without labels')
    _require(selection.get('labels_used_for_ranking') is True,
             'expected the already-labelled-pool selection protocol')
    for arm in ARMS:
        row = by_arm[arm]
        _number(row.get('ap'), f'{arm}.ap', 0, 1)
        if row.get('ap50') is not None:
            _number(row['ap50'], f'{arm}.ap50', 0, 1)
        budget = 3000 if arm == 'random_large' else 1500
        _require(row.get('budget') == budget and selected[arm].get('selected') == budget,
                 f'{arm}: selection/training budget mismatch')
        sizes.add(_number(row.get('n_images'), f'{arm}.n_images', 1))
        curve = curves[arm]
        _require(len(curve) == 10, f'{arm}: expected ten ordered curve points')
        for index, point in enumerate(curve):
            _require(point.get('epoch_index') == index and point.get('completed_epochs') == index + 1
                     and point.get('image_presentations_proxy') == (index + 1) * budget,
                     f'{arm}: wrong epoch order or exposure proxy')
            _number(point.get('ap'), f'{arm}.curve.ap', 0, 1)
        meta = selected[arm].get('metadata', {})
        _require(meta.get('seed') == 42 and meta.get('selected_count') == budget - 500
                 and meta.get('excluded_initial_count') == 500,
                 f'{arm}: incompatible initial sample or added count')
        for key in ('selected_by_score', 'selected_by_random', 'random_fill_count'):
            _number(meta.get(key), f'{arm}.{key}', 0, budget - 500)
        _require(meta['selected_by_score'] + meta['selected_by_random'] == budget - 500,
                 f'{arm}: selection counts do not sum to budget')
        _require(type(meta.get('spacing_relaxed')) is bool, f'{arm}: missing spacing flag')
        for key in ('requested_min_gap_frames', 'final_min_gap_frames'):
            _number(meta.get(key), f'{arm}.{key}', 0)
        overlap = selected[arm].get('overlap_added', {})
        _require(set(overlap) == set(ARMS), f'{arm}: incomplete overlap table')
        for other, fraction in overlap.items():
            _number(fraction, f'{arm}.overlap.{other}', 0, 1)
        _near(overlap[arm], 1., f'{arm}.self_overlap')
        if row.get('training_seconds') is not None:
            _number(row['training_seconds'], f'{arm}.training_seconds', 0)
    _require(len(sizes) == 1, 'validation image counts differ between runs')
    deltas = {f'koopman_langevin_minus_{arm}': by_arm['koopman_langevin']['ap'] - by_arm[arm]['ap']
              for arm in ARMS if arm != 'koopman_langevin'}
    _require(set(pilot.get('deltas', {})) == set(deltas), 'missing primary contrasts')
    for key, value in deltas.items():
        _near(pilot['deltas'][key], value, key)
    _near(pilot.get('practical_threshold_vs_readout_loss'), .004, 'practical threshold')
    gate = deltas['koopman_langevin_minus_readout_loss'] >= .004 and deltas['koopman_langevin_minus_placebo'] > 0
    _require(type(pilot.get('practical_gate_met')) is bool and pilot['practical_gate_met'] == gate,
             'practical gate must be a boolean consistent with AP contrasts')
    exposure = _exposure(curves)
    saved_exposure = pilot.get('exposure_comparison', {})
    _require(saved_exposure.get('status') == 'computed'
             and saved_exposure.get('common_range') == exposure['common_range'],
             'inconsistent common exposure range')
    for arm in ARMS:
        _near(saved_exposure.get('normalized_area', {}).get(arm),
              exposure['normalized_area'][arm], f'{arm}.normalized_area')
    for value, name in ((selection.get('elapsed_seconds'), 'selection seconds'),
                        (feature.get('elapsed_seconds'), 'extraction seconds')):
        if value is not None:
            _number(value, name, 0)
    if diagnostics is not None:
        _require(diagnostics.get('status') == 'complete', 'diagnostics must be complete')
        for key in ('readout_fit_uses_val', 'used_for_ranking', 'extraction_uses_labels',
                    'langevin_fitted_on_training_clock', 'statistical_significance_tested'):
            _require(diagnostics.get(key) is False, f'diagnostics: incompatible {key}')
        _require(diagnostics.get('cka_only_measures_drift') is True,
                 'CKA must measure representation drift only')
        probe = diagnostics.get('probe', {})
        _require(probe.get('train_frames') == 500 and probe.get('val_frames') == 192,
                 'expected common L0=500 and fixed val probe=192')
        wanted = {(a, e) for a in ARMS for e in (2, 4, 6, 8, 10)}
        diag_rows = diagnostics.get('rows', [])
        _require(len(diag_rows) == 25
                 and {(r.get('arm'), r.get('completed_epochs')) for r in diag_rows} == wanted,
                 'expected five checkpoints from every arm')
        for row in diag_rows:
            _require(row.get('run_id') == by_arm[row['arm']]['run_id'],
                     'diagnostics must reference the same pilot runs')
            for key in ('auroc', 'average_precision', 'balanced_bce'):
                value = row.get('metrics', {}).get(key)
                if value is not None:
                    _number(value, f'diagnostics.{key}', 0, None if key == 'balanced_bce' else 1)
            if row.get('cka_to_previous_checkpoint') is not None:
                _number(row['cka_to_previous_checkpoint'], 'CKA', 0, 1 + 1e-8)
    return dict(rows=by_arm, deltas=deltas, gate=gate, exposure=exposure)


def _plots(pilot, checked, diagnostics, directory):
    # Imported only for artifact generation; validation and CLI --help need no plotting stack.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for arm, color, style in zip(ARMS, COLORS, STYLES):
            curve = pilot['training_curves'][arm]
            for axis, key in zip(axes, ('completed_epochs', 'image_presentations_proxy')):
                axis.plot([p[key] for p in curve], [p['ap'] for p in curve],
                          color=color, linestyle=style, marker='o', markersize=3, label=LABELS[arm])
                axis.scatter(curve[-1][key], checked['rows'][arm]['ap'], color=color,
                             marker='D', s=35, edgecolor='black', linewidth=.5, zorder=5)
        lo, hi = checked['exposure']['common_range']
        axes[1].axvspan(lo, hi, color='grey', alpha=.09, zorder=-1)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.set_ylim(0, 1)
            axis.set_ylabel('AP (IoU 0.50:0.95), val')
        axes[0].set(title='Uczenie względem epok', xlabel='Ukończone epoki', xticks=range(1, 11))
        axes[1].set(title='Uczenie względem nominalnych ekspozycji',
                    xlabel='Ukończone epoki × liczba klatek treningowych')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
        curve_label = 'walidacja EMA' if pilot.get('training_curve_weights') == 'ema' else 'walidacja z treningu'
        fig.suptitle(f'Linie: {curve_label}; romby: niezależna ocena ostatnich wag', fontsize=12)
        fig.tight_layout(rect=(0, .08, 1, .95))
        fig.savefig(directory / 'WYKRES_UCZENIA.png', dpi=180)
        plt.close(fig)
        if diagnostics is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            observed = []
            for arm, color, style in zip(ARMS, COLORS, STYLES):
                rows = sorted((r for r in diagnostics['rows'] if r['arm'] == arm),
                              key=lambda r: r['completed_epochs'])
                ys = [r['metrics'].get('auroc') for r in rows]
                observed.extend(value for value in ys if value is not None)
                for axis in axes:
                    axis.plot([r['completed_epochs'] for r in rows],
                              [math.nan if value is None else value for value in ys],
                              color=color, linestyle=style, marker='o', label=LABELS[arm])
            axes[0].axhline(.5, color='grey', linestyle=':', linewidth=1)
            for axis in axes:
                axis.set(xlabel='Ukończone epoki detektora',
                         ylabel='AUROC: słaba etykieta prostokąta / tło', xticks=(2, 4, 6, 8, 10))
                axis.grid(alpha=.2)
            axes[0].set(title='Pełna skala AUROC', ylim=(0, 1))
            zoom = (max(0, min(observed) - .004), min(1, max(observed) + .004)) if observed else (0, 1)
            axes[1].set(title='Zbliżenie — te same pomiary', ylim=zoom)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
            fig.suptitle('Odczyt patchy na stałej próbie val (192 obrazy)', fontsize=12)
            fig.tight_layout(rect=(0, .08, 1, .95))
            fig.savefig(directory / 'WYKRES_PATCHY.png', dpi=180)
            plt.close(fig)


def _time(value):
    return 'brak pomiaru' if value is None else f'{value:.2f} s'


def _metric(value):
    return 'niedostępne' if value is None else f'{value:.4f}'


def _link(path, output):
    try:
        target = Path(os.path.relpath(path, output)).as_posix()
    except ValueError:  # Windows cannot make a relative link between different drives.
        target = Path(path).as_posix()
    return '<' + target + '>'


def _markdown(pilot, selection, checked, diagnostics, sources, output):
    passed = checked['gate']
    text = ['# Pilot doboru klatek: wynik detektora', '',
            'Eksploracyjny pilot na jednym seedzie treningu (42), z selekcją seed 42. '
            'Pięć ramion trenowano przez 10 epok ze wspólnej inicjalizacji COCO. '
            'Każdy zbiór zawiera wspólne L0: 500 obrazów.', '',
            f'**Ustalona bramka praktyczna: {str(passed).lower()} '
            f'({"spełniona" if passed else "niespełniona"}).** '
            'Warunek: AP Koopman + Langevin minus AP błędu odczytu ≥ 0.004 '
            'oraz AP Koopman + Langevin minus AP placebo > 0. '
            'Nie wykonano testu istotności statystycznej. '
            + ('Wynik uzasadnia replikację z kolejnymi seedami.' if passed else
               'Ten pilot nie spełnił ustalonego kryterium korzyści z selekcji.'), '',
            '## Niezależna ocena końcowa', '',
            'Metryka główna: AP dla IoU 0.50:0.95, skala 0–1. '
            'Oceniono zwykłe wagi ostatniej epoki na pełnym val; '
            f'każde ramię: {checked["rows"][ARMS[0]]["n_images"]} obrazów. '
            'Zbioru test nie użyto. Wartości wczytano z ukończonego raportu pilota; '
            'generator nie uruchamia ponownej ewaluacji modeli.', '',
            '| Ramię | Klatki treningowe | AP ostatniej epoki | AP50 | Czas treningu |',
            '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        row = checked['rows'][arm]
        text.append(f'| {LABELS[arm]} | {row["budget"]} | {row["ap"]:.4f} | '
                    f'{_metric(row.get("ap50"))} | {_time(row.get("training_seconds"))} |')
    text += ['', '| Kontrast końcowego AP | Różnica |', '|---|---:|']
    for arm in ARMS:
        if arm != 'koopman_langevin':
            text.append(f'| Koopman + Langevin minus {LABELS[arm]} | '
                        f'{checked["deltas"][f"koopman_langevin_minus_{arm}"]:+.4f} |')
    text += ['', '## Krzywe uczenia i ekspozycje', '',
             '![Uczenie detektora](WYKRES_UCZENIA.png)', '',
             ('Linie przedstawiają walidację wag EMA zapisaną podczas treningu '
              '(use_ema=true, eval_base_model=false). ' if pilot.get('training_curve_weights') == 'ema' else
              'Linie przedstawiają walidację zapisaną podczas treningu; wybór wag wynika z konfiguracji trenera. ')
             + 'Romby pokazują niezależny wynik '
             'ostatnich zwykłych wag z tabeli powyżej. Te wartości mogą się różnić.', '',
             'Ekspozycje to przybliżenie: ukończone epoki × liczba unikalnych klatek '
             'treningowych. Nie jest to pomiar kroków optymalizatora, czasu ani FLOP. '
             'Losowy 3000 ma dwukrotnie więcej nominalnych ekspozycji na epokę '
             'niż każde ramię 1500. Receptura ma rozgrzewkę learning rate przez dwie epoki, '
             'więc dla różnych budżetów obejmuje ona różną liczbę aktualizacji. '
             'Porównanie po ekspozycjach nie izoluje zatem w pełni wpływu liczby kroków '
             'i harmonogramu learning rate; kontrola 1500 wobec 1500 zachowuje oba.', '',
             f'Wspólny obserwowany zakres ekspozycji: {checked["exposure"]["common_range"][0]}–'
             f'{checked["exposure"]["common_range"][1]} (szare pole). '
             'Średnie AP poniżej to pole pod krzywą podzielone przez szerokość tego zakresu; '
             'interpolacja liniowa między obserwacjami, bez ekstrapolacji.', '',
             '| Ramię | Średnie AP we wspólnym zakresie ekspozycji |', '|---|---:|']
    for arm in ARMS:
        text.append(f'| {LABELS[arm]} | {checked["exposure"]["normalized_area"][arm]:.4f} |')
    text += ['', '## Co rzeczywiście wybrał selektor', '',
             'Dobór jest statyczny: historię filmu wykorzystano przed treningiem detektora '
             'do zamrożenia zbioru. Selektor nie reagował na kolejne checkpointy ani kroki SGD. '
             'Cząstka to pojedynczy patch na stałej pozycji siatki obrazu; nie jest to '
             'potwierdzona trajektoria punktu materialnego narzędzia. Etykiety prostokątów '
             'uczestniczyły w rankingu z już oznaczonej puli.', '',
             f'Brakujące okna historii: {selection["selection"].get("missing_temporal_windows", "brak danych")}. '
             f'Kandydaci z dodatnim zyskiem: {selection["selection"].get("positive_gain", "brak danych")}.', '',
             '| Ramię | Dodane klatki | Według oceny | Losowo | Losowe uzupełnienie rankingu | Odstęp żądany → końcowy | Poluzowanie |',
             '|---|---:|---:|---:|---:|---:|---|']
    arms = selection['selection']['arms']
    for arm in ARMS:
        meta = arms[arm]['metadata']
        text.append(f'| {LABELS[arm]} | {meta["selected_count"]} | {meta["selected_by_score"]} | '
                    f'{meta["selected_by_random"]} | {meta["random_fill_count"]} | '
                    f'{meta["requested_min_gap_frames"]} → {meta["final_min_gap_frames"]} | '
                    f'{"tak" if meta["spacing_relaxed"] else "nie"} |')
    text += ['', 'Losowe uzupełnienie oznacza brak wystarczającej liczby dodatnich ocen '
             'przy ograniczeniu odstępu czasowego. Losowanie jako podstawowa strategia '
             'ramion losowych jest wykazane osobno i nie jest takim uzupełnieniem. '
             'Odstęp podano w klatkach filmu, z uwzględnieniem L0.', '',
             'Udział wspólnych **dodanych** klatek: mianownikiem jest liczba klatek '
             'dodanych w ramieniu wiersza. Nie jest to indeks Jaccarda; tabela może '
             'być niesymetryczna, zwłaszcza przy 1500 wobec 3000.', '',
             '| Ramię wiersza / ramię kolumny | ' + ' | '.join(LABELS[a] for a in ARMS) + ' |',
             '|---|' + '---:|' * len(ARMS)]
    for arm in ARMS:
        text.append('| ' + LABELS[arm] + ' | ' + ' | '.join(
            f'{100 * arms[arm]["overlap_added"][other]:.1f}%' for other in ARMS) + ' |')
    text += ['', '## Koszt procedury', '',
             f'- Ekstrakcja tokenów wspólnego selektora: {_time(selection["feature_meta"].get("elapsed_seconds"))}.',
             f'- Dopasowanie i selekcja: {_time(selection.get("elapsed_seconds"))}.',
             '- Czasy treningów: tabela wyników; brak pola oznacza brak pomiaru, nie zero.', '',
             'Koszt wspólnego detektora L0, transferu danych i pełnej procedury nie jest '
             'tu kompletnie zmierzony. Krótszy trening na 1500 klatkach sam w sobie '
             'nie dowodzi przyspieszenia całej procedury. Czasów równoległych biegów '
             'nie należy sumować jako czasu zegarowego.', '', '## Diagnostyka patchy', '']
    if diagnostics is None:
        text += ['Nie dostarczono DIAGNOSTYKA.json. Raport nie wyciąga wniosku o '
                 'zmianie informacji zawartej w patchach między checkpointami.']
    else:
        text += ['![Odczyt patchy](WYKRES_PATCHY.png)', '',
                 'Odczyt ridge i standaryzację dopasowano osobno przy każdym checkpoincie '
                 'wyłącznie na wspólnym L0 (500 obrazów). Ocena dotyczy stałej próby '
                 '192 obrazów val. Etykieta patcha pochodzi z prostokąta narzędzia; '
                 'nie jest segmentacją ani etykietą tożsamości konkretnego narzędzia. '
                 'Wzrost AUROC oznacza skuteczniejszy dopasowany odczyt liniowy tej słabej etykiety '
                 'w użytej projekcji kanałów. Nie mierzy całej informacji reprezentacji '
                 'ani nie dowodzi wzrostu jakości śledzenia.', '',
                 '| Ramię | Epoka | AUROC | AP patchy | Balanced BCE | CKA do poprzedniego checkpointu |',
                 '|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            for row in sorted((r for r in diagnostics['rows'] if r['arm'] == arm), key=lambda r: r['completed_epochs']):
                metrics = row['metrics']
                text.append(f'| {LABELS[arm]} | {row["completed_epochs"]} | {_metric(metrics.get("auroc"))} | '
                            f'{_metric(metrics.get("average_precision"))} | {_metric(metrics.get("balanced_bce"))} | '
                            f'{_metric(row.get("cka_to_previous_checkpoint"))} |')
        text += ['', 'Balanced BCE używa sigmoidy wyniku ridge; nie jest oceną '
                 'skalibrowanego prawdopodobieństwa. CKA opisuje wyłącznie zmianę '
                 'reprezentacji tych samych patchy, bez odczytu znaczenia. Diagnostyka '
                 'nie uczestniczyła w rankingu. Pięć checkpointów nie służy do dopasowania '
                 'Langevina w osi postępu treningu.']
    text += ['', '## Źródła i powtarzalność', '',
             'Opis procedury: [README.md](dokumentacja/README.md). '
             'Uzasadnienie teoretyczne: [LITERATURA.md](dokumentacja/LITERATURA.md). '
             'Są to lokalne kopie dokumentacji z chwili generowania raportu.', '',
             'Generator sprawdza zgodność końcowych AP, kontrastów, bramki i całek '
             'z krzywych oraz zgodność liczności selekcji. Nie zastępuje audytu '
             'pierwotnych checkpointów i predykcji. Poniższe hashe identyfikują '
             'dokładne wejścia raportu.', '']
    for name, source in sources.items():
        text.append(f'- [{name}]({_link(source["path"], output)}): SHA-256 `{source["sha256"]}`.')
    text.append('')
    return '\n'.join(text)


def build(pilot_path, selection_path, output, *, diagnostics_path=None):
    """Publish an immutable report directory after all input validation succeeds."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'use a new output directory: {output}')
    paths = dict(pilot=Path(pilot_path).resolve(), selection=Path(selection_path).resolve())
    if diagnostics_path is not None:
        paths['diagnostics'] = Path(diagnostics_path).resolve()
    values, sources = {}, {}
    for name, path in paths.items():
        raw = path.read_bytes()
        values[name] = json.loads(raw.decode('utf-8-sig'))
        sources[name] = dict(path=path, sha256=hashlib.sha256(raw).hexdigest())
    checked = validate_inputs(values['pilot'], values['selection'], values.get('diagnostics'))
    sources['raport.py'] = dict(path=Path(__file__).resolve(),
                               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    documents = {}
    for name in ('README.md', 'LITERATURA.md'):
        path = Path(__file__).resolve().parent / name
        documents[name] = path.read_bytes()
        sources[name] = dict(path=path, sha256=hashlib.sha256(documents[name]).hexdigest())
    prose = _markdown(values['pilot'], values['selection'], checked,
                      values.get('diagnostics'), sources, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.raport-', dir=output.parent) as temporary:
        directory = Path(temporary) / 'artifacts'
        directory.mkdir()
        (directory / 'dokumentacja').mkdir()
        for name, raw in documents.items():
            (directory / 'dokumentacja' / name).write_bytes(raw)
        _plots(values['pilot'], checked, values.get('diagnostics'), directory)
        (directory / 'WYNIKI_PILOT.md').write_text(prose, encoding='utf-8')
        if output.exists():
            raise FileExistsError(f'output appeared while rendering: {output}')
        directory.rename(output)
    return dict(status='complete', out=str(output), practical_gate_met=checked['gate'],
                files=sorted(p.name for p in output.iterdir()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-report', type=Path, required=True)
    parser.add_argument('--selection-report', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.pilot_report, args.selection_report, args.out,
                   diagnostics_path=args.diagnostics)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
