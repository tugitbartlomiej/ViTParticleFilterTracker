"""Reports frozen model comparison; never selects using test performance."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
H_TABLE = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
LABELS = {
    'koopman': 'Koopman', 'finite_gle': 'Langevin: skończona pamięć',
    'exp_gle': 'Langevin: wykładnicza pamięć',
    'stable_koopman': 'Koopman: przycięte widmo',
    'persistence': 'Brak ruchu', 'velocity': 'Ostatnia prędkość',
    'damped_velocity': 'Wygaszana prędkość', 'ols_trend': 'Trend OLS',
    'reversion': 'Powrót do historycznej średniej',
    'kalman_velocity': 'Obserwator Kalmana',
    'coupled_observer': 'Obserwator narzędzie + tło'}


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8-sig'))


def label(config):
    name = config['family'].split('__')[0]
    text = LABELS.get(name, name)
    if 'dt' in config:
        text += ' co ' + str(config['dt'])
    if config.get('kind') == 'reversion':
        text += f" (okno {config['window']}, czas {config['tau']})"
    return text


def value(x, percentage=True):
    if x is None or not np.isfinite(x):
        return 'brak'
    return f'{100*x:.2f}%' if percentage else f'{x:.3f}'


def metric_curve(record, group, metric):
    curve = record['groups'][group]['by_horizon']
    return np.array([np.nan if curve[str(h)].get(metric) is None else curve[str(h)][metric]
                     for h in range(1, 501)])


def save(fig, name):
    fig.savefig(HERE / (name + '.png'), dpi=170, bbox_inches='tight', facecolor='white')
    fig.savefig(HERE / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style(ax):
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlim(1, 500)
    ax.set_xticks([1, 10, 50, 100, 200, 500])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(alpha=.23)
    ax.spines[['top', 'right']].set_visible(False)


def plots(choice, result):
    families = choice['families']
    family_ids = list(dict.fromkeys([r['id'] for r in families.values()]
                                   + [choice['best_mean_mse_skill']]))
    records = result['records']; winner = choice['global_winner']
    colors = {i: plt.get_cmap('tab20')(k % 20) for k, i in enumerate(family_ids)}
    colors[choice['best_mean_mse_skill']] = '#111827'
    colors[winner] = '#875c46'
    h = np.arange(1, 501)
    for group, group_label in [('fit', '8 obrazów dopasowania'),
                               ('transfer', '4 obrazy transferu')]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for ax, islinear, title in zip(axes, [True, False],
                                      ['Koopman i Langevin', 'Obserwatory i proste modele ruchu']):
            for ident in family_ids:
                if ident.startswith('linear_') != islinear:
                    continue
                rec = records[ident]; y = metric_curve(rec, group, 'sign_accuracy')
                tag = label(rec['config']) + (' • wybór z walidacji' if ident == winner else
                      ' • wybór według MSE' if ident == choice['best_mean_mse_skill'] else '')
                ax.plot(h, y, color=colors[ident], lw=2.7 if ident == winner else 1.45,
                        ls='--' if rec['config'].get('dt') == 10 else '-', label=tag)
            ax.axhline(.85, color='#b91c1c', ls=':', lw=1.9, label='Cel 85%')
            ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel('Trafność znaku łącznej zmiany')
            ax.set_title(title, loc='left', fontsize=12); style(ax)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8.8,
                      frameon=False, borderaxespad=0)
        axes[-1].set_xlabel('Horyzont: liczba aktualizacji sieci po starcie w kroku 500')
        fig.suptitle(f'128 oryginalnych kanałów, bez PCA — {group_label}',
                     x=.08, ha='left', fontsize=15, fontweight='bold')
        fig.text(.08, .015, 'Wybrano po jednym wariancie każdej rodziny na wcześniejszej walidacji. '
                 'Wszystkie kanały; wagi pokrycia maską; równe wagi obrazów.\n'
                 'Jeden start H500. Co10: interpolacja prognoz bez przyszłych pomiarów. Oś czasu: liniowa do10, dalej logarytmiczna.',
                 fontsize=9, color='#444444')
        fig.subplots_adjust(top=.92, bottom=.10, right=.69, hspace=.22)
        save(fig, 'ZNAK_' + group.upper())

    shown = list(dict.fromkeys([winner, choice['best_mean_mse_skill'],
                               families['kalman_velocity']['id'],
                               families['coupled_observer']['id'],
                               families['ols_trend']['id'],
                               families['damped_velocity']['id']]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for col, group in enumerate(['fit', 'transfer']):
        for ident in shown:
            rec = records[ident]; color = colors.get(ident, '#222222')
            tag = label(rec['config']) + (' • główny' if ident == winner else '')
            axes[0, col].plot(h, metric_curve(rec, group, 'skill'), lw=1.7, color=color, label=tag)
            axes[1, col].plot(h, metric_curve(rec, group, 'norm_ratio'), lw=1.7, color=color, label=tag)
        axes[0, col].set_title('Dopasowanie: 8 obrazów' if group == 'fit' else 'Transfer: 4 obrazy', loc='left')
        axes[0, col].axhline(0, color='#333333', lw=.8)
        axes[0, col].set_yscale('symlog', linthresh=1)
        axes[0, col].set_ylim(top=1)
        axes[0, col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, col].set_ylabel('Redukcja MSE względem braku ruchu')
        axes[1, col].axhline(1, color='#333333', ls=':', lw=1)
        axes[1, col].set_yscale('symlog', linthresh=2)
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].set_ylabel('Norma prognozy / norma rzeczywista')
        axes[1, col].set_xlabel('Horyzont w aktualizacjach sieci')
        for ax in axes[:, col]:
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('Sam znak nie wystarcza: wielkość błędu i amplituda ruchu',
                 fontsize=15, x=.07, ha='left', fontweight='bold')
    fig.subplots_adjust(top=.91, bottom=.19, hspace=.25, wspace=.30)
    save(fig, 'BLAD_I_AMPLITUDA')


def trajectory(choice, result):
    # The illustrated coordinates are selected solely by variability before500.
    ident = choice['global_winner']
    with np.load(HERE / 'DANE_J128.npz') as d:
        q = d['q_2'].astype(float); w = d['w_2']; channels = d['channels1']
    token = int(np.argmax(w))
    selected = np.argsort(q[:501, token].var(0))[::-1][:3]
    with np.load(HERE / f'PROGNOZY_{ident}.npz') as p:
        pred = p['q_2'].astype(float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, ch in zip(axes, selected):
        ax.plot(np.arange(350, 501), q[350:501, token, ch], color='#777777', lw=1,
                label='Historia dostępna do startu')
        ax.plot(np.arange(500, 1001), q[500:, token, ch], color='#111827', lw=1.4,
                label='Rzeczywisty dalszy przebieg')
        ax.plot(np.arange(500, 1001), pred[:, token, ch], color='#dc2626', lw=1.6,
                label='Prognoza z kroku 500')
        ax.axvline(500, color='#777777', ls=':', lw=1)
        ax.set_ylabel(f'Kanał {int(channels[ch])}\nwartość natywna')
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(ncol=3, fontsize=9, frameon=False)
    axes[-1].set_xlabel('Aktualizacja sieci')
    fig.suptitle(label(result['records'][ident]['config']) + ': rzeczywista trajektoria i prognoza',
                 fontsize=14, x=.08, ha='left', fontweight='bold')
    fig.text(.08, .015, f'Obraz 2, token o największym pokryciu maską (lokalny indeks {token}). '
             'Kanały wybrane według wariancji wyłącznie do kroku 500.\n'
             'To ilustracja; wynik liczbowy obejmuje wszystkie 128 kanałów i wszystkie wybrane cząsteczki.',
             fontsize=9)
    fig.subplots_adjust(top=.92, bottom=.11, hspace=.20)
    save(fig, 'TRAJEKTORIE')


def report(choice, result):
    winner = choice['global_winner']; rec = result['records'][winner]
    lines = ['# Prognozowanie 128 natywnych kanałów tożsamości do 500 aktualizacji', '',
             'Data: 2026-10-02. Status: obliczono; wynik eksploracyjny z jednego biegu sieci.', '',
             f"Przeszukano **{len(choice['rows'])} konfiguracji**. Wybrany przed testem model: "
             f"**{label(rec['config'])}**, `{winner}`. Bramkę użyteczności na walidacji "
             + ('spełniono.' if choice['gate_passed'] else '**nie spełniono**; jest to wybór diagnostyczny.'), '',
             '## Główny wynik: model wybrany przed testem', '',
             '| Horyzont | Znak: dopasowanie | Znak: transfer | MSEskill: transfer | Norma prognozy / prawdziwa: transfer |',
             '|---:|---:|---:|---:|---:|']
    for h in H_TABLE:
        a = rec['groups']['fit']['by_horizon'][str(h)]
        b = rec['groups']['transfer']['by_horizon'][str(h)]
        lines.append(f"| {h} | {value(a['sign_accuracy'])} | {value(b['sign_accuracy'])} | "
                     f"{value(b['skill'])} | {value(b['norm_ratio'], False)} |")
    lines += ['', 'Próg ponad85 procent dotyczy znaku skumulowanego przemieszczenia każdego kanału. '
              'Nie jest to trafność rozpoznawania narzędzia ani procent dokładnie odtworzonych wektorów. '
              'Dodatni MSEskill oznacza mniejszy błąd niż założenie braku ruchu; ujemny oznacza pogorszenie.', '']
    for group in ['fit', 'transfer']:
        g = rec['groups'][group]
        lines.append(f"- {group}: ciągły zakres ≥85% od H1 kończy się na **{g['continuous_prefix_h85']}**; "
                     f"najdalsze pojedyncze przekroczenie: **{g['maximum_pointwise_h85'] or 'brak'}**.")
    bestmse = result['records'][choice['best_mean_mse_skill']]
    best500 = bestmse['groups']['transfer']['by_horizon']['500']
    lines += ['', f"Osobno przed testem wybrano model według najmniejszego błędu: "
              f"**{label(bestmse['config'])}** (`{choice['best_mean_mse_skill']}`). "
              f"Na H500 w transferze uzyskał **{value(best500['sign_accuracy'])}** znaku i "
              f"**{value(best500['skill'])}** redukcji MSE; norma prognozowanego ruchu wynosiła "
              f"{value(best500['norm_ratio'], False)} normy rzeczywistej. "
              'Nie zastępujemy nim po teście głównego zwycięzcy ustalonego kryterium.']
    lines += ['', '## Najlepszy wariant każdej rodziny, wybrany wyłącznie na walidacji', '',
              'Tabela przedstawia końcowy transfer na cztery obrazy. Jej kolejność nie jest rankingiem na teście.', '',
              '| Rodzina | Znak H1 | Znak H50 | Znak H500 | MSEskill H500 | Norma H500 | Konfiguracja |',
              '|---|---:|---:|---:|---:|---:|---|']
    for fam, item in sorted(choice['families'].items()):
        r = result['records'][item['id']]; g = r['groups']['transfer']['by_horizon']
        lines.append(f"| {label(r['config'])} | {value(g['1']['sign_accuracy'])} | "
                     f"{value(g['50']['sign_accuracy'])} | {value(g['500']['sign_accuracy'])} | "
                     f"{value(g['500']['skill'])} | {value(g['500']['norm_ratio'], False)} | `{item['id']}` |")
    lines += ['', '## Jak policzono', '',
              '- Stan: 128 ustalonych oryginalnych kanałów, bez PCA, rotacji, autoenkodera i wybierania łatwych współrzędnych.',
              '- Dane: 1001 stanów od0 do1000, 218 cząsteczek narzędzia z12 stałych obrazów. Każdy obraz ma równą wagę; wewnątrz obrazu użyto pokrycia maską.',
              '- Walidacja: modele dopasowane na obrazach0–5 do250 lub350; obrazy6–7 oceniane od300 lub400. Wszystkie cele wyboru kończą się do500.',
              '- Refit: obrazy0–7 do500. Test: start500, pełny autonomiczny przebieg do1000. Transfer: obrazy8–11 nie używane do dopasowania równań.',
              '- Wszystkie obrazy pochodzą ze zbioru treningowego detektora. Transfer dotyczy modelu dynamiki, nie nowego testu rozpoznawania narzędzia.',
              '- Modele co10 mają50 iteracji doH500; pośrednie stany interpolowano wyłącznie z prognoz. Ich H1 nie jest niezależną identyfikacją dynamiki jednego kroku.',
              '- Kalman i obserwatory otrzymują rzeczywiste dane wyłącznie do startu. Dalej nie ma pomiarów korygujących ani przyszłych obrazów czy gradientów.',
              '- MSEskill raportujemy jako średnią wyników per obraz. Nie jest to iloraz MSE po połączeniu wszystkich tokenów.',
              '- Wybrane modele są średnimi predyktorami. Model GLE z pamięcią nie stanowi tu dowodu dokładnego Mori–Zwanziga ani białego szumu reszt.', '',
              '## Co ogranicza wniosek', '',
              'Kanały ustalono na500, więc wcześniejsze foldy są retrospektywną walidacją w tej bazie. '
              'Dla H500 po ustaleniu kanałów istnieje tylko jeden moment startu. Obrazy i tokeny obserwują '
              'ten sam trening, więc nie traktujemy ich jako niezależnych biegów i nie podajemy pozornego '
              'przedziału ufności po tysiącach współrzędnych. Ta historia była już wcześniej analizowana; '
              'potwierdzenie wyniku wymagałoby niezależnego biegu lub dalszego odcinka.', '',
              'Sam wysoki znak nie potwierdza wielkości ruchu. Sprawdzać trzeba jednocześnie MSEskill, normę '
              'i przewagę nad prostym trendem. Norma bliska0 oznacza niemal brak prognozowanego ruchu; '
              'bardzo duża norma ujawnia rozbieżność mimo trafienia części znaków.', '',
              '## Pliki', '',
              '- `PLAN.md`: siatka99 modeli, foldy i reguła wyboru przed fitami.',
              '- `WYBOR_PRZED_TESTEM.json`: cała walidacja i zamrożony wybór.',
              '- `OCENA_H500.json` i `POROWNANIE.csv`: metryki każdego H1–500.',
              '- `MODELE_PRZED_TESTEM.json` i `MACIERZE_PRZED_TESTEM.npz`: odtwarzalne parametry.',
              '- `PROGNOZY_*.npz`: pełne trajektorie wybranych wariantów.',
              '- `CO_OPISUJE_MODEL.md`: równania i interpretacja.',
              '- `LITERATURA_I_AUDYT.md`: źródła pierwotne i niezależny audyt.', '',
              '![Trafność transferu](ZNAK_TRANSFER.png)', '',
              '![Błąd i amplituda](BLAD_I_AMPLITUDA.png)', '',
              '![Przykładowe trajektorie](TRAJEKTORIE.png)', '']
    content = '\n'.join(lines)
    for old, new in [('ponad85', 'ponad 85'), ('od0 do1000', 'od 0 do 1000'),
                     ('z12 ', 'z 12 '), ('obrazach0–5', 'obrazach 0–5'),
                     ('do250', 'do 250'), ('lub350', 'lub 350'),
                     ('obrazy6–7', 'obrazy 6–7'), ('od300', 'od 300'),
                     ('lub400', 'lub 400'), ('do500', 'do 500'),
                     ('obrazach0–7', 'obrazach 0–7'), ('obrazy0–7', 'obrazy 0–7'),
                     ('start500', 'start 500'), ('do1000', 'do 1000'),
                     ('Transfer: obrazy8–11', 'Transfer: obrazy 8–11'),
                     ('co10 mają50', 'co 10 mają 50'), ('na500', 'na 500'),
                     ('bliska0', 'bliska 0'), ('siatka99', 'siatka 99')]:
        content = content.replace(old, new)
    (HERE / 'WYNIKI.md').write_text(content, encoding='utf-8')
    (HERE / 'README.md').write_text('# Prognoza J128 do500\n\n'
        'Wyniki: [WYNIKI.md](WYNIKI.md). Protokół: [PLAN.md](PLAN.md).\n\n'
        'Uruchomienie: `F:/Python/python.exe badanie.py --stage validation`, audyt wyboru, '
        '`F:/Python/python.exe badanie.py --stage test`, `F:/Python/python.exe raport.py`. '
        'Istniejących wyborów/testów nie nadpisuje się automatycznie.\n', encoding='utf-8')
    summary = dict(status='computed_exploratory', candidates=len(choice['rows']),
                   global_winner=winner, config=rec['config'], validation_gate=choice['gate_passed'],
                   groups=rec['groups'], basis='128 unchanged native channels; no PCA',
                   forecast_origin=500, maximum_horizon=500,
                   limitation='one temporal origin, one network run, previously seen dataset')
    (HERE / 'WYNIKI.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    keys = ['sign_accuracy', 'balanced_sign_accuracy', 'skill', 'mse', 'persistence_mse',
            'cosine', 'norm_ratio', 'actual_norm_mean', 'predicted_norm_mean', 'finite']
    with (HERE / 'POROWNANIE.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'family', 'group', 'H'] + keys)
        writer.writeheader()
        for ident, r in result['records'].items():
            for group in ['fit', 'transfer', 'all']:
                for h, metric in r['groups'][group]['by_horizon'].items():
                    writer.writerow(dict(config=ident, family=r['config']['family'], group=group, H=h,
                                         **{k:metric.get(k) for k in keys}))


def sign_diagnostic():
    """Post-evaluation description, not a new candidate or changed selection."""
    with np.load(HERE / 'DANE_J128.npz') as d:
        qs = [d[f'q_{i}'].astype(float) for i in range(12)]
        ws = [d[f'w_{i}'] for i in range(12)]
    rows = []
    for h in H_TABLE:
        images = []
        for i, (q, w) in enumerate(zip(qs, ws)):
            actual = np.sign(q[500+h] - q[500])
            past = np.sign(q[h:501] - q[:501-h])
            votes = (past > 0).sum(0) - (past < 0).sum(0)
            predicted_sign = np.sign(votes)
            images.append(dict(probe=i,
                positive=float(w @ (actual > 0).mean(-1)),
                negative=float(w @ (actual < 0).mean(-1)),
                zero=float(w @ (actual == 0).mean(-1)),
                past_majority_sign_accuracy=float(w @ (predicted_sign == actual).mean(-1)),
                past_pairs=int(len(past))))
        groups = {}
        for group, indices in [('fit', range(8)), ('transfer', range(8,12)), ('all', range(12))]:
            groups[group] = {key:float(np.mean([images[i][key] for i in indices]))
                             for key in ['positive','negative','zero','past_majority_sign_accuracy']}
        rows.append(dict(H=h, images=images, groups=groups))
    result = dict(status='post_evaluation_diagnostic_not_model_selection', rows=rows,
                  rule='Per token/channel: majority positive vs negative H-step changes with targets<=500; tie predicts zero',
                  limitation='H500 has one historical pair; no magnitude forecast is claimed')
    (HERE / 'DIAGNOSTYKA_ZNAKOW.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['', '## Dodatkowa diagnostyka znaków', '',
             'Poniższa kontrola nie zmienia wyboru modelu. Znak większości oszacowano dla każdej '
             'cząsteczki i kanału wyłącznie z historycznych zmian danego H kończących się do500. '
             'Remis daje znak0. Nie jest to prognoza wielkości; przy H500 dostępna jest tylko jedna historyczna para.', '',
             '| H | Dodatnie zmiany w transferze | Ujemne zmiany w transferze | Trafność historycznego znaku większości |',
             '|---:|---:|---:|---:|']
    for row in rows:
        g = row['groups']['transfer']
        lines.append(f"| {row['H']} | {value(g['positive'])} | {value(g['negative'])} | {value(g['past_majority_sign_accuracy'])} |")
    with (HERE / 'WYNIKI.md').open('a', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    choice = read('WYBOR_PRZED_TESTEM.json'); result = read('OCENA_H500.json')
    assert choice['global_winner'] == result['global_winner']
    report(choice, result); plots(choice, result); trajectory(choice, result); sign_diagnostic()
    print('Report, CSV, JSON and four PNG/SVG figures generated')
