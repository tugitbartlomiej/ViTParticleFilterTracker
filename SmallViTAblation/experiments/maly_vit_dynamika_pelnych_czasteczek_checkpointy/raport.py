"""Describe the completed pilot, including failures relative to simple baselines."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LABELS = {'persistence':'Persystencja','velocity':'Stała prędkość','damped_velocity':'Wygaszana prędkość',
          'full_ar1':'Pełny ridge 768D','pca_ar1':'PCA + ridge AR(1)','pca_ar2':'PCA + ridge z pamięcią'}
COLORS = dict(zip(LABELS, ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']))


def main():
    m = json.loads((HERE/'METRYKI_PILOT.json').read_text(encoding='utf-8'))
    audit = json.loads((HERE/'AUDYT_ZGODNOSCI.json').read_text(encoding='utf-8'))
    cfg = json.loads((HERE/'CONFIG.json').read_text(encoding='utf-8'))
    assert m['status']=='complete_pilot' and audit['status']=='passed'
    hs = cfg['horizons_updates']
    improvements=[100*(1-m['summary']['pca_ar1'][str(h)]['mean_ratio_to_persistence']) for h in hs]
    improvement_text=' / '.join(f'{v:.2f}' for v in improvements)
    full_h100=100*(m['summary']['full_ar1']['100']['mean_ratio_to_persistence']-1)
    pca_rows=[r for r in m['records'] if r['family']=='pca_ar1']
    h100_rows=[r for r in pca_rows if r['horizon_updates']==100]
    image_ratios=np.array([[x['rms_l2']/x['persistence_rms_l2'] for x in r['by_image']] for r in h100_rows])
    winning_origins=sum(r['relative_to_persistence']<1 for r in pca_rows)
    rank=m['selected']['pca_ar1']['rank']
    fig,axes = plt.subplots(1,3,figsize=(18,4.8),layout='constrained')
    for family,label in LABELS.items():
        rows = m['summary'][family]
        values = [rows[str(h)]['mean_ratio_to_persistence'] for h in hs]
        axes[0].plot(hs,values,marker='o',label=label,color=COLORS[family])
    axes[0].set(xlabel='Horyzont w aktualizacjach wag',ylabel='RMS błędu / RMS persystencji',
                title='Wszystkie warianty: pełne wektory 768D',yscale='log')
    axes[0].set_xticks(hs);axes[0].grid(alpha=.25);axes[0].legend(fontsize=8)
    for family in ['persistence','full_ar1','pca_ar1','pca_ar2']:
        values=[m['summary'][family][str(h)]['mean_ratio_to_persistence'] for h in hs]
        axes[1].plot(hs,values,marker='o',label=LABELS[family],color=COLORS[family])
    axes[1].set(xlabel='Horyzont w aktualizacjach wag',ylabel='RMS błędu / RMS persystencji',
                title='Powiększenie: ridge i persystencja')
    axes[1].set_xticks(hs);axes[1].grid(alpha=.25);axes[1].legend(fontsize=8)
    seen_ranks=set()
    for family in ['pca_ar1','pca_ar2']:
        plot_rank=m['selected'][family]['rank']
        if plot_rank in seen_ranks:continue
        seen_ranks.add(plot_rank)
        values = [np.mean([r['pca_outside_motion_fraction_squared'] for r in m['records']
                   if r['family']==family and r['horizon_updates']==h]) for h in hs]
        axes[2].plot(hs,values,marker='o',label=f'Wspólna baza PCA, r={plot_rank}')
    axes[2].set(xlabel='Horyzont w aktualizacjach wag',ylabel='Udział energii ruchu poza PCA',ylim=(0,1),
                title='Podłoga reprezentacyjna: prawdziwy ruch poza bazą')
    axes[2].set_xticks(hs);axes[2].grid(alpha=.25);axes[2].legend(fontsize=8)
    fig.savefig(HERE/'PROGNOZA_PILOT.png',dpi=160)
    plt.close(fig)
    lines=['# Pilot prognozy pełnych cząsteczek', '',
           'Data: 2026-10-01. Status: lokalny pilot zakończony. Jeden istniejący przebieg uczenia; bez nowego treningu ViT.', '',
           f'**Wynik:** PCA z {rank} kierunkami i ridge AR(1), wybrane wyłącznie na wcześniejszej walidacji, zmniejszyły średni RMS błędu pełnych wektorów względem persystencji o **{improvement_text} procent** dla horyzontów 10 / 20 / 50 / 100 aktualizacji. Pełny ridge 768D był przy H=100 gorszy o {full_h100:.2f} procent. Dodanie jednego opóźnienia w testowanym wariancie nie poprawiło średniej względem PCA AR(1).', '',
           'To mały dodatni wynik predykcyjny jednego przebiegu. Nie potwierdza tożsamości narzędzia ani ogólnej przewagi modeli z pamięcią. Wykres i tabela pokazują także wyniki negatywne.', '',
           '## Aparat i poprawność danych', '',
           'Analizowano 101 stanów co 10 aktualizacji, 12 stałych obrazów i 218 osobnych tokenów oryginalnych masek. Każdy token zachował wszystkie 768 współrzędnych. Gęsty plik 1001 stanów czytano przez mmap; wybrana tablica w RAM zajmuje '+f"{m['input_array_ram_bytes']/1024**2:.2f} MiB.", '',
           'Lokalne forwardy checkpointów 0 i 1000 porównano z zapisami Edenu. Model, obrazy i maski mają sprawdzone pochodzenie. Pomiar: ostatnia MLP po GELU, eval, float32, bez autocast i TF32. Wyniki kontroli:', '',
           '| Krok | Względny błąd L2 lokalnie/Eden | Maks. błąd powtórzenia | BN bez zmian |',
           '|---:|---:|---:|---|']
    for r in audit['results']:
        lines.append(f"| {r['step']} | {r['reference_relative_l2_error']:.3g} | {r['repeated_forward_max_abs_error']:.3g} | {r['buffers_unchanged']} |")
    lines += ['', 'Szczegóły sprzętu i wersji: [AUDYT_ZGODNOSCI.json](AUDYT_ZGODNOSCI.json).', '',
              '## Ocena poza oknem dopasowania', '',
              'Ustawienia wybrano na początkach 400 i 500, z celami najdalej do kroku 600. Test obejmuje początki 600, 700, 800 i 900 oraz horyzonty 10, 20, 50, 100. Operator i PCA używają wyłącznie przeszłości danego początku; wszystkie roll-outy są autonomiczne. Po kroku 1000 nie ma danych.', '',
              'Liczby poniżej to średni stosunek RMS błędu pełnego wektora do persystencji. **Mniej niż 1 oznacza mniejszy błąd niż persystencja; więcej niż 1 oznacza gorszy wynik.** Każdy obraz ma tę samą wagę, a tokeny w obrazie są ważone pokryciem maską.', '',
              '| Model | H=10 | H=20 | H=50 | H=100 |', '|---|---:|---:|---:|---:|']
    for family,label in LABELS.items():
        row=[m['summary'][family][str(h)]['mean_ratio_to_persistence'] for h in hs]
        lines.append('| '+label+' | '+' | '.join(f'{x:.4f}' for x in row)+' |')
    lines += ['', '![Błędy pełnych wektorów i podłoga PCA](PROGNOZA_PILOT.png)', '',
              f'PCA AR(1) było lepsze od persystencji w {winning_origins} z {len(pca_rows)} kombinacji początek/horyzont po zagregowaniu obrazów. Przy H=100 wynik był lepszy w {int((image_ratios<1).sum())} z {image_ratios.size} kombinacji obraz/początek prognozy; dla {int((image_ratios.mean(0)>=1).sum())} obrazów średnia po początkach nie była lepsza. Nie jest to poprawa każdej cząsteczki ani każdej próby.', '',
              '## Co pozostaje poza redukcją', '',
              f"Wybrane {rank} osi opisywało średnio {100*np.mean([r['pca_past_variance_explained'] for r in pca_rows]):.2f} procent ważonej przeszłej wariancji. Znaczna część przyszłego przemieszczenia pozostawała poza tą bazą; w rekonstrukcji stosowano dla niej persystencję, a błąd liczono nadal w pełnych 768D.", '',
              '| Horyzont | Energia przyszłego ruchu poza PCA | Stosunek zwykłej średniej L2 do persystencji |', '|---:|---:|---:|']
    for H in hs:
        rs=[r for r in m['records'] if r['family']=='pca_ar1' and r['horizon_updates']==H]
        lines.append(f"| {H} | {100*np.mean([r['pca_outside_motion_fraction_squared'] for r in rs]):.2f}% | {np.mean([r['mean_l2_relative_to_persistence'] for r in rs]):.4f} |")
    lines += ['', 'Podłoga poza PCA jest oceną reprezentacji przy użyciu prawdziwych celów testowych; predyktor nie miał do niej dostępu. Nie dowodzi ona, że pozostały ruch jest fundamentalnie losowy lub nieprzewidywalny przez bogatszy model.', '',
              '## Ustawienia wybrane przed testem', '',
              '| Rodzina | Ranga PCA | Alpha ridge | Średni stosunek błędu na walidacji |', '|---|---:|---:|---:|']
    for family,r in m['selected'].items():
        lines.append(f"| {LABELS[family]} | {r['rank'] or 'pełne 768D'} | {r['alpha']} | {r['mean_validation_ratio']:.4f} |")
    lines += ['', 'Pełny model korzysta z dual ridge, bez identyfikowania dowolnej macierzy 768×768 z kilkudziesięciu próbek. PCA jest wspólną bazą 768×r dopasowaną z przeszłych odchyleń poszczególnych tokenów. Każdy token ma własny predyktor. To nie jest model jawnych oddziaływań między tokenami.', '',
              '## Granice wyniku', '',
              '- Test dotyczy prognozy ruchu w jednym biegu. Tokeny, obrazy i sąsiednie okna nie są niezależnymi replikacjami uczenia.',
              '- Nie potwierdza to tożsamości konceptu narzędzia. PCA opisuje przeszłą zmienność; osobne badanie par z/bez narzędzia jest potrzebne do semantycznej interpretacji.',
              '- Mniejszy błąd reprezentacji nie jest pomiarem poprawy mAP detektora. Takiego testu tutaj nie wykonywano.',
              '- Brak zapisanego dziennika wykonanych minibatchy tego biegu. Nie użyto historii wejść z innych biegów, gradientów ani momentów AdamW.',
              '- Zmiana tokenów obejmuje zmiany wag i buforów BN. Nie oddzielono tutaj ich przyczynowych wkładów.',
              '- Historyczny podział obrazów fit/holdout nie jest w tym pilocie testem na nieznanych obrazach: każdy predyktor korzysta z dostępnej przeszłości swojego obrazu.',
              '- Nie przeprowadzono testów istotności opartych na fałszywej niezależności 218 tokenów. Liczby i zakresy błędów są opisowe.', '',
              '## Odtwarzanie i artefakty', '',
              'Po zakończonym, sprawdzonym pobraniu uruchomić kolejno `pilot_zgodnosci.py`, `prognoza.py`, `raport.py` środowiskiem z torch, timm, NumPy, SciPy, psutil i matplotlib. Nie wymaga to treningu ani serwera Eden.', '',
              '- [Plan](PLAN.md), [konfiguracja](CONFIG.json), [audyt metody](AUDYT_METODY.md).',
              '- [Pełne metryki i indeksy wszystkich cząsteczek](METRYKI_PILOT.json).',
              '- [Zamrożony wybór ustawień i siatka walidacyjna](WYBOR_WALIDACJA.json).', '',
              f"Czas prognozowania i dopasowań: {m['elapsed_seconds']:.1f} s. Szczyt pamięci procesu: {m['process_peak_working_set_bytes']} bajtów. Hash źródłowego pliku: `{m['source_sha256']}`."]
    (HERE/'WYNIKI_PILOT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status=json.loads((HERE/'WYNIKI.json').read_text(encoding='utf-8'))
    status.update(status='pilot_zakonczony',code_or_results_executed=True,
                  metryki='METRYKI_PILOT.json',raport='WYNIKI_PILOT.md',figure='PROGNOZA_PILOT.png')
    (HERE/'WYNIKI.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('report_ready',HERE/'WYNIKI_PILOT.md')


if __name__=='__main__':main()
