"""Plot frozen comparable Langevin results, without fitting or selecting models."""
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'maly_vit_langevin_kierunki_tozsamosci'
OUT = HERE / 'porownanie_wszystkich_modeli'
OUT.mkdir(exist_ok=True)
SOURCES = [p / f'METRYKI_DT{dt}.json' for p in (OLD, HERE) for dt in (1, 10)]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
old = {dt: read(OLD / f'METRYKI_DT{dt}.json') for dt in (1, 10)}
new = {dt: read(HERE / f'METRYKI_DT{dt}.json') for dt in (1, 10)}
COLORS = ['#147D92', '#E19425', '#5D53A0', '#B54470', '#237949', '#745846']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
LOCAL = ['localU8', 'localPCA4', 'localPCA8', 'localPLS4', 'localPLS8', 'best_short']
COUPLED = ['coupledU8', 'coupledPCA4', 'coupledPCA8', 'coupledPLS4', 'coupledPLS8']
LEGACY = ['L0', 'L1', 'L2', 'L4', 'L8', 'best_H1']
GROUPS = [('Wcześniejsze równania', 'old', LEGACY),
          ('Nowe modele lokalne', 'new', LOCAL),
          ('Nowe modele ze wspólnym ruchem', 'new', COUPLED)]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def label(key, dt):
    if key.startswith('L') and key[1:].isdigit():
        return 'Bez pamięci (L0)' if key == 'L0' else f'Pamięć L{key[1:]}'
    if key in ('best_H1', 'best_short'):
        return f'Wybór pod H{dt}'
    if key == 'oldGLE':
        return 'Poprzedni główny model'
    if key.endswith('U8'):
        return '8 kierunków tożsamości'
    method = 'PCA' if 'PCA' in key else 'PLS'
    return f'8 tożsamości + {key[-1]} {method}'


def metrics(source, key, dt, probe='2'):
    return old[dt]['models'][key] if source == 'old' else new[dt]['probes'][probe]['models'][key]


def horizons(dt):
    return [1, 2, 5, 10, 20, 50] if dt == 1 else [10, 20, 50]


def values(source, key, dt, metric, probe='2'):
    m = metrics(source, key, dt, probe)
    return np.array([m[str(h)][metric] for h in horizons(dt)]) * 100


def decorate(ax, dt, metric, limits, xlabel=True):
    hs = horizons(dt)
    ax.set_xscale('log')
    ax.set_xticks(hs)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(min(hs) / 1.18, max(hs) * 1.18)
    ax.set_ylim(*limits)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis='y', color='#DCE2E8', linewidth=.7)
    ax.set_axisbelow(True)
    if metric == 'sign_accuracy':
        ax.axhline(80, color='#9AA5B1', ls=':', lw=1)
        ax.set_ylabel('Poprawny znak zmiany składowych')
    else:
        ax.axhline(0, color='#84909C', lw=1.2)
        ax.set_ylabel('Redukcja MSE względem braku zmiany')
    if xlabel:
        ax.set_xlabel('Horyzont prognozy [aktualizacje uczenia]')


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=170)
    fig.savefig(OUT / f'{name}.svg')
    plt.close(fig)


def family_panels(metric, zoom=False):
    is_sign = metric == 'sign_accuracy'
    outside = zoom or not is_sign
    fig, axes = plt.subplots(3, 2, figsize=(17, 17 if outside else 14.5))
    if is_sign:
        limits = (40, 85) if zoom else (0, 100)
        title = 'Langevin: ile zmian przewidziano we właściwą stronę?'
        subtitle = 'Odsetek poprawnych wzrostów i spadków wszystkich 8 składowych tożsamości; bez odsiewu małych zmian'
    else:
        all_y = np.concatenate([values(src, key, dt, metric) for _, src, keys in GROUPS for key in keys for dt in (1, 10)])
        limits = (math.floor(min(all_y.min(), 0) / 10) * 10 - 5,
                  math.ceil(all_y.max() / 10) * 10 + 5)
        title = 'Langevin: o ile zmniejsza błąd wielkości zmiany?'
        subtitle = 'Redukcja MSE względem „stan się nie zmieni”; wynik ujemny oznacza pogorszenie, nie ujemną trafność'
    fig.suptitle(title, fontsize=23, fontweight='bold', x=.065, ha='left', y=.98)
    fig.text(.065, .948, subtitle, fontsize=12, color='#465363')
    fig.text(.065, .925, 'Jeden obraz • 31 cząsteczek maski • jeden bieg uczenia • parametry wybrane przed testem'
             + (' • powiększona oś Y: 40–85%' if zoom else ''), fontsize=10.5, color='#465363')
    for row, (group, source, keys) in enumerate(GROUPS):
        for col, dt in enumerate((1, 10)):
            ax = axes[row, col]
            for i, key in enumerate(keys):
                ax.plot(horizons(dt), values(source, key, dt, metric), color=COLORS[i],
                        marker=MARKERS[i], ms=5, lw=1.8, ls='--' if i == 5 else '-',
                        label=label(key, dt), zorder=3)
            if source == 'new':
                ax.plot(horizons(dt), values('new', 'oldGLE', dt, metric), color='#263344',
                        lw=2, ls=(0, (5, 3)), label='Poprzedni główny model', zorder=4)
            decorate(ax, dt, metric, limits)
            n = len(new[dt]['probes']['2']['origin_updates'])
            ax.set_title(f'{group} | krok co {dt}\n{n} momentów rozpoczęcia prognozy', loc='left')
            if zoom or not is_sign:
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.28), ncol=2,
                          fontsize=9.2, frameon=False, columnspacing=1.2)
            else:
                ax.legend(loc='lower left', ncol=2, fontsize=9.3, frameon=False,
                          columnspacing=1.3)
    # Zoom and MSE plots reserve more space beneath panels for legends.
    fig.subplots_adjust(left=.065, right=.985, top=.875, bottom=.15 if outside else .08,
                        wspace=.2, hspace=.95 if outside else .44)
    fig.text(.065, .037,
             'L = liczba zapamiętanych przyrostów. PCA = dominujące zmiany; PLS = kierunki pomocne w prognozie.\n'
             'Oś X logarytmiczna. W obrębie kolumny te same cele i momenty prognoz; kolumny różnią się próbkowaniem.\n'
             'Wyniki punktowe; opisowe przedziały niepewności są w CSV. Źródło: METRYKI_DT1/10.json, 2 eksperymenty SmallViT.',
             fontsize=9.4, color='#526172', va='top')
    stem = 'TRAFNOSC_WSZYSTKIE' + ('_ZOOM' if zoom else '') if is_sign else 'REDUKCJA_BLEDU_WSZYSTKIE'
    save(fig, stem)


def heatmap():
    rows = [(src, key) for _, src, keys in GROUPS for key in keys]
    titles = [f'Stary: {label(key, 1)}' if src == 'old' else
              ('Sprzężony: ' if key in COUPLED else 'Lokalny: ') +
              ('wybór krótkiego H' if key == 'best_short' else label(key, 1)) for src, key in rows]
    titles[5] = 'Stary: wybór krótkiego H'
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 11.5), gridspec_kw={'width_ratios': [6, 3]})
    for col, dt in enumerate((1, 10)):
        ax = axes[col]
        matrix = np.array([values(src, key, dt, 'sign_accuracy') for src, key in rows])
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(len(horizons(dt))), labels=horizons(dt))
        ax.set_yticks(range(len(rows)), labels=titles if col == 0 else [''] * len(rows))
        ax.tick_params(length=0)
        ax.set_title(f'Krok modelu co {dt}', pad=14)
        ax.set_xlabel('Horyzont [aktualizacje uczenia]')
        for i in range(len(rows)):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f'{matrix[i,j]:.1f}%'.replace('.', ','), ha='center', va='center',
                        color='white' if matrix[i, j] >= 64 else '#182738', fontsize=11)
        for edge in (5.5, 11.5):
            ax.axhline(edge, color='white', lw=3)
    fig.suptitle('Dokładne wartości trafności: wszystkie porównywalne warianty',
                 x=.055, ha='left', fontsize=20, fontweight='bold', y=.98)
    fig.text(.055, .94, 'Poprawny znak wszystkich składowych tożsamości • sonda 2 • wyniki testowe', fontsize=12)
    fig.subplots_adjust(left=.29, right=.89, top=.88, bottom=.12, wspace=.08)
    cax = fig.add_axes([.92, .18, .017, .64])
    fig.colorbar(im, cax=cax, format=PercentFormatter(100)).set_label('Trafność znaku')
    fig.text(.055, .065,
             'To 15 rodzin i 2 osobne wybory pod najkrótszy horyzont; część wyborów powtarza tę samą konfigurację.\n'
             'PCA4 w kroku co1 = nowy wybór krótkiego H. Stary L1 w kroku co10 = stary wybór krótkiego H.\n'
             'Nie jest to procent poprawnie przewidzianych całych wektorów ani procentowa dokładność ich długości.',
             fontsize=10, color='#526172', va='top')
    save(fig, 'TABELA_TRAFNOŚCI')


def selected_uncertainty():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, dt in enumerate((1, 10)):
        choice = new[dt]['selection']
        for row, metric in enumerate(('sign_accuracy', 'skill')):
            ax = axes[row, col]
            for key, color, text in [('oldGLE', '#3A485B', 'Poprzedni główny'),
                                     (choice, '#147D92', 'Nowy wybrany na walidacji')]:
                m = metrics('new', key, dt)
                y = values('new', key, dt, metric)
                low, high = np.array([m[str(h)]['bootstrap']['CI95'][metric] for h in horizons(dt)]).T * 100
                ax.plot(horizons(dt), y, marker='o', color=color, lw=2, label=text)
                ax.fill_between(horizons(dt), low, high, color=color, alpha=.14)
            decorate(ax, dt, metric, (40, 85) if row == 0 else (-40, 80))
            ax.set_title(f'Krok co {dt} | {choice}', loc='left')
            ax.legend(fontsize=9, frameon=False, loc='lower left')
    fig.suptitle('Niepewność porównania modeli wybranych przed testem', fontsize=19, fontweight='bold', y=.98)
    fig.subplots_adjust(top=.90, bottom=.14, hspace=.4, wspace=.2)
    fig.text(.07, .06, 'Pasma: opisowy bootstrap 95%, bloki 50 aktualizacji, 300 powtórzeń, tylko około 3 efektywne bloki.\n'
             'Jeden bieg i jeden główny obraz. To nie są przedziały z niezależnych ziaren. Oś trafności powiększona: 40–85%.', fontsize=10)
    save(fig, 'WYBRANE_MODELE_NIEPEWNOSC')


def transfer():
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for row, (group, _, keys) in enumerate(GROUPS[1:]):
        for col, dt in enumerate((1, 10)):
            ax = axes[row, col]
            for i, key in enumerate(keys):
                ax.plot(horizons(dt), values('new', key, dt, 'sign_accuracy', '8'),
                        color=COLORS[i], marker=MARKERS[i], lw=1.7, label=label(key, dt))
            ax.plot(horizons(dt), values('new', 'oldGLE', dt, 'sign_accuracy', '8'),
                    color='#263344', lw=2, ls='--', label='Poprzedni główny model')
            decorate(ax, dt, 'sign_accuracy', (0, 100))
            ax.set_title(f'{group} | krok co {dt}', loc='left')
            ax.legend(loc='lower left', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Drugi obraz: trafność bez ponownego dopasowania', fontsize=21, fontweight='bold', y=.97)
    fig.text(.07, .925, 'Sonda 8 • 7 cząsteczek narzędzia • te same bazy i równania • własna znana historia obrazu', fontsize=11)
    fig.subplots_adjust(top=.855, bottom=.12, hspace=.38, wspace=.19)
    fig.text(.07, .045, 'Wszystkie dostępne nowe warianty i dawna kontrola. Poprawny znak wszystkich 8 składowych, bez odsiewu.\n'
             'To drugi obraz tego samego biegu uczenia; nie nowe ziarno. Stare warianty L0/L2/L4/L8 nie miały zapisanego tego transferu.', fontsize=10)
    save(fig, 'TRAFNOSC_DRUGI_OBRAZ')


def export_data():
    records = []
    for dt in (1, 10):
        assert old[dt]['origin_updates'] == new[dt]['probes']['2']['origin_updates']
        for h in horizons(dt):
            for field in ('MSE', 'sign_accuracy', 'persistence_MSE'):
                assert np.isclose(old[dt]['models']['selected'][str(h)][field],
                                  new[dt]['probes']['2']['models']['oldGLE'][str(h)][field], atol=1e-12)
        for probe in ('2', '8'):
            groups = GROUPS if probe == '2' else GROUPS[1:] + [('Wcześniejsza kontrola', 'new', ['oldGLE'])]
            for group, src, keys in groups:
                for key in keys:
                    for h in horizons(dt):
                        m = metrics(src, key, dt, probe)[str(h)]
                        assert 0 <= m['sign_accuracy'] <= 1
                        assert np.isfinite(m['skill'])
                        r = dict(probe=probe, dt=dt, family=group, source=src, model=key,
                                 horizon_updates=h, origins=len(new[dt]['probes'][probe]['origin_updates']),
                                 sign_percent=100*m['sign_accuracy'], skill_percent=100*m['skill'],
                                 MSE=m['MSE'], persistence_MSE=m['persistence_MSE'])
                        for metric in ('sign_accuracy', 'skill'):
                            bounds=m['bootstrap']['CI95'][metric]
                            r[f'{metric}_CI95_low_percent']=bounds[0]*100
                            r[f'{metric}_CI95_high_percent']=bounds[1]*100
                        records.append(r)
    with (OUT/'DANE_WYKRESOW.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    aliases = {'dt1': {'oldGLE': 'old:L4', 'new:best_short': 'new:localPCA4'},
               'dt10': {'oldGLE': 'old:L1', 'old:best_H1': 'old:L1'}}
    for dt, source, left, right in [(1,'old','selected','L4'), (10,'old','selected','L1'),
                                   (10,'old','best_H1','L1'), (1,'new','best_short','localPCA4')]:
        assert np.array_equal(values(source,left,dt,'sign_accuracy'),values(source,right,dt,'sign_accuracy'))
    manifest = dict(source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    aliases=aliases, rows=len(records), metric='coverage-weighted sign accuracy across ALL 8 identity coordinates',
                    no_refit=True, no_new_model_selection=True,
                    excludes='Earlier scalar-LDA tasks, full-768D tasks, H100 with different origins, and validation grid configurations without frozen test evaluation')
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    export_data()
    family_panels('sign_accuracy')
    family_panels('sign_accuracy',zoom=True)
    family_panels('skill')
    heatmap()
    selected_uncertainty()
    transfer()
    print(json.dumps({'output':str(OUT), 'png':len(list(OUT.glob('*.png'))),
                      'svg':len(list(OUT.glob('*.svg'))), 'status':'verified'},ensure_ascii=False))
