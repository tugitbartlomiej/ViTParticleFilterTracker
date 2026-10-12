"""Generate numerical appendix from verified output files; interpretation is a separate report."""
from pathlib import Path
import json,numpy as np,hashlib,platform,sys
from analyze import OUT,ROOT
def rd(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def f(x,d=3):return 'nieosiągnięty' if x is None else f'{x:.{d}f}'
text=['# Audyt szkła: pełne wyniki liczbowe','', 'Data: 2026-10-10. Wszystkie miary obejmują 768 współrzędnych. MSD jest średnią kwadratu normy przemieszczenia po tokenach i obrazach, bez dzielenia przez 768. Przedziały bootstrapowe resamplują 7 filmów; opisują tę próbę obrazów, nie zmienność między treningami.','']
for mode,label in [('dense','Gęsty bieg 21 epok'),('long','Osobny bieg 60 epok')]:
    r=rd(f'{mode}_metrics.json')
    text += [f'## {label}','','### MSD w równych oknach pomiarowych','','| Układ | Okno kroków | Opóźnienia do dopasowania | Wykładnik MSD | Przedział po filmach |','|---|---:|---:|---:|---:|']
    for name in ['raw','image_relative','cage']:
        for wn,w in r[name]['windows'].items():
            if wn=='warmup':continue
            for slope in w['slopes']:
                if (slope['low'],slope['high']) not in [(1,5),(10,100),(35,80),(100,1000),(500,5000)]:continue
                text.append(f"| {name} | {w['start']}–{w['end']} | {slope['low']}–{slope['high']} | {f(slope['mean'])} | {f(slope['ci'][0])}–{f(slope['ci'][1])} |")
    text += ['','### Starzenie: te same opóźnienia i szerokości okien początkowych','','Każdy wiek reprezentuje początki w oknie szerokości 500 kroków. Między punktami nie interpolowano wektorów. Okna przekraczające wznowienie w kroku 22974 są pomijane.','', '| Układ | Wiek początku | MSD po 1000 kroków | Czas do połowy MSD po 2000 kroków |','|---|---:|---:|---:|']
    for name in ['raw','image_relative','cage']:
        shape={a['age']:a for a in r[name]['shape_times'] if a['fraction']==.5}
        for a in r[name]['aging']:
            q=shape[a['age']];c='≤' if q['left_censored'] else ''
            text.append(f"| {name} | {a['age']} | {f(a['mean'][a['lag'].index(1000)])} | {c}{f(q['time'],1)} |")
    text+=['','Ten czas opisuje kształt w **ustalonym horyzoncie 2000 kroków**, nie zakłada istnienia asymptotycznego plateau.','', '### Czasy do wspólnego progu MSD','','Progi 0,5×, 1×, 2× wartości MSD pierwszego wieku przy opóźnieniu 1000. Są umowne; brak osiągnięcia nie oznacza nieskończonego czasu.','', '| Układ | Mnożnik progu | Próg | Wiek | Czas do progu |','|---|---:|---:|---:|---:|']
    for name in ['raw','image_relative','cage']:
        for th in r[name]['thresholds']:
            for a in th['times']:
                tau=f(a['time'],1)
                if a['left_censored']:tau='≤'+tau
                if a['right_censored']:tau='>'+str(a['max_lag'])+' (cenzurowany)'
                text.append(f"| {name} | {th['factor']} | {f(th['threshold'])} | {a['age']} | {tau} |")
    text+=['','### Kontrola prostym modelem dynamiki','','Dopasowanie na naprzemiennych opóźnieniach i sprawdzenie na pozostałych opóźnieniach **tej samej krzywej**. Nie jest to walidacja na niezależnym treningu. Błąd jest RMSE logarytmu MSD. Model colored OU + diffusion opisuje liniowe ograniczenie ruchu z pamięcią szumu i wolną składową dyfuzyjną; nie ma oddziaływań wielociałowych.','', '| Układ | Okno | Zwykły OU | OU + dyfuzja | OU z pamięcią + dyfuzja |','|---|---|---:|---:|---:|']
    for name in ['raw','image_relative','cage']:
        for wn,w in r[name]['windows'].items():
            if wn=='warmup':continue
            fits=w['fits'];text.append(f"| {name} | {wn} | {f(fits['OU']['heldout_log_rmse'])} | {f(fits['OU_plus_diffusion']['heldout_log_rmse'])} | {f(fits['colored_OU_plus_diffusion']['heldout_log_rmse'])} |")
    text+=['',f'![Diagnostyka {label}]({mode}_diagnostics.png)','']
mv=rd('DENSE_MULTISCALE_VACF.json')
text+=['## Zawracanie: długość przyrostu to osobna zmienna','','Okno 300–1094, sąsiednie niepokrywające się przyrosty. VACF = średni iloczyn skalarny przyrostów / średnia ich kwadratowej normy. Ujemna wartość oznacza przeciwną składową ruchu, nie pełny obrót każdego tokenu.','', '| Układ | Przyrost [kroki] | Korelacja kolejnych przyrostów | Przedział po filmach |','|---|---:|---:|---:|']
for name in ['raw','image_relative','cage']:
    for dt,v in mv[name].items():text.append(f"| {name} | {dt} | {f(v['mean'][1])} | {f(v['ci'][0][1])}–{f(v['ci'][1][1])} |")
dc=rd('DIRECT_CONTROLS.json')
text+=['','## Sąsiedztwo odświeżane na każdym początku pomiaru','','Sześciu najbliższych sąsiadów według pełnej odległości 768D, wyłącznie wewnątrz obrazu. Pięć początków dla każdego wieku; w głównym pomiarze sąsiedzi byli zamrożeni w kroku 300.','', '| Bieg | Wiek | MSD względem odświeżanych sąsiadów po 1000 kroków |','|---|---:|---:|']
for mode in ['dense','long']:
    for a in dc[mode]['dynamic_cage']:
        if 1000 in a['lag']:text.append(f"| {mode} | {a['age']} | {f(a['mean'][a['lag'].index(1000)])} |")
text+=['','## FP32, BF16 i zamrożone statystyki BatchNorm','','Ta kontrola dotyczy osobnego biegu 60 epok. BatchNorm zamrożony na statystykach kroku 300 jest kontrolą wrażliwości; nie jest alternatywnym treningiem.','', '| Początek–koniec | MSD FP32 | MSD BF16 / FP32 | MSD zamrożone BN | MSD FP32 po normowaniu tokenu do długości 1 |','|---|---:|---:|---:|---:|']
for a in dc['precision_bn_comparisons']:
    fp=np.mean(a['fp32']);text.append(f"| {a['start']}–{a['end']} | {f(fp)} | {f(np.mean(a['bf16'])/fp)} | {f(np.mean(a['frozen_bn']))} | {f(np.mean(a['fp32_unit_vectors']),5)} |")
ec=rd('EPOCH_SCALE_BN_CONTROLS.json')
text+=['','## Stałe opóźnienie 5 epok po kontroli długości wektorów','','Początki w trzech sąsiednich epokach. Przejścia przez wznowienie są wykluczone. Normalizacja każdego tokenu do długości 1 zachowuje informację o kierunku, usuwa informację o długości; to kontrola wrażliwości, a nie jedyna właściwa metryka.','', '| Wariant | Wiek [epoki] | MSD po 5 epokach |','|---|---:|---:|']
for name,curves in ec.items():
    for a in curves:
        if 5 in a['lag_epochs']:text.append(f"| {name} | {a['age_epoch']} | {f(a['mean'][a['lag_epochs'].index(5)],5)} |")
text+=['','![Kontrola skali](epoch_scale_controls.png)','']
text+=['','## Odtworzenie','','Kod: `experiments/audyt_szkla_768d_20261010/`. Kolejność:','', '```powershell','python compute_grams.py dense','python analyze.py dense','python multiscale.py','python direct_controls.py','python compute_grams.py long','python analyze.py long','python summarize_controls.py','python epoch_controls.py','python validate_math.py','python final_figure.py','python write_report.py','```','', 'Dane historyczne pozostały bez zmian. Pliki macierzy Grama zachowują informację wystarczającą do odtworzenia średnich MSD i VACF w dowolnych dostępnych parach czasu, ale nie całych wektorów. Tymczasowy zapis latentów długiego biegu jest usuwany po utworzeniu wyników.','']
(OUT/'WYNIKI_LICZBOWE.md').write_text('\n'.join(text),encoding='utf-8')
scripts=ROOT/'experiments/audyt_szkla_768d_20261010'
manifest=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts.glob('*.py')},validation=rd('VALIDATION.json'))
(OUT/'AUDIT_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Numerical report written',flush=True)
