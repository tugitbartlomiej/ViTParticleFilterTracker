"""Raportuje także wynik ujemny i wybór walidacyjny; obala go rozbieżność z JSON.
Nie wybiera wariantów na teście i nie zmienia żadnego modelu.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((HERE/'wyniki/METRYKI.json').read_text(encoding='utf-8'))
    r=data['results']['500']
    names=list(r['validation'])
    lines=['# Wyniki identyfikacji Langevina i Koopmana', '',
        'Aktualizacja: '+data['created']+' (Europe/Warsaw).', '',
        'Nie wykazano lepszego równania prognozy na 500 aktualizacji z historycznych paczek. '
        'Walidacja nowych wariantów wybrała samo jądro GLE bez dodatkowego wejścia. '
        'Strojenie ridge zmieniło wynik transferu s48 z 22,64 do 22,65 procent redukcji MSE, '
        'co jest praktycznie tą samą prognozą. Wynik zależy mocno od biegu; '
        'w głównym punkcie późnego czasu s47 ten GLE jest gorszy od braku zmiany.', '',
        '## Aparat i podział', '',
        'Dwa istniejące nagrania s47/s48, po 3001 stanów i 180 sond. '
        'Kalibracja LDA z 960..1000; wykonane historyczne paczki, ich reprezentacje i wpływy '
        'włączane z opóźnieniem jednego kroku. Odczyt dużych plików przez memmap; '
        'nowy trening ani Eden nie były uruchamiane. Trening s47: etykiety do1800; '
        'walidacja s47: starty1800..2000 co10; refit po wyborze: etykiety do2500. '
        'Transfer s48: starty2100..2500 co10 i cele do3000. Późny test s47: tylko2500->3000.', '',
        'Pierwszy pilot (16 sekund CPU) zachowano w `wyniki/pilot_v1/`. '
        'W nim najlepszy nowy model z paczkami miał 15,49 procent na s48 wobec 22,64 procent GLE. '
        'Po obejrzeniu tego testu przeprowadzono jawnie eksploracyjny etap drugi: '
        'rozszerzono oryginalną bazę jądra i siatkę ridge, zachowując wybór tylko na walidacji. '
        'Etap drugi trwał około13 sekund. Te dane nie są już świeżym testem potwierdzającym.', '',
        'Każde porównanie dotyczy tego samego celu końcowego s(t+H)-s(t). '
        'Stare rekurencyjne rodziny dopasowują średnie blokowe, a ich prognozy absolutne '
        'oceniamy tutaj na końcowym stanie. Nie odtwarzamy tym zmiany w średniej przyszłego bloku. '
        'Wynik jest oceną tych predyktorów dla wspólnego zadania końcowego.', '',
        '## H=500: wszystkie ramiona', '',
        'Umiejętność to 100*(1-MSE_model/MSE_brak_zmiany), a nie procent trafnych znaków. '
        'Kolumna s48 obejmuje 41 zależnych, nakładających się okien diagnostycznych. '
        'Główny punkt bez nakładania to t2500 dla każdego biegu.', '',
        '| Ramię | Walidacja s47 [%] | Test s48 okna [%] | Główny s48 t2500 [%] | Główny s47 t2500 [%] | Znak s48 [%] |',
        '|---|---:|---:|---:|---:|---:|']
    for n in names:
        a=r['test']['47'][n];b=r['test']['48'][n];v=r['validation'][n]
        lines.append(f"| {n} | {v['skill_percent']:.2f} | {b['skill_percent']:.2f} | "
                     f"{b['primary_t2500']['skill_percent']:.2f} | {a['skill_percent']:.2f} | {100*b['sign']:.1f} |")
    chosen=r['selected_new_validation']
    select=r['selected_validation']
    lines += ['',
        f'Wybór spośród nowych rodzin na walidacji: `{chosen}`; '
        f'wybór spośród wszystkich kontroli i modeli: `{select}`. '
        'Nie wybrano modelu o najlepszym wyniku na s48. '
        'Zmiana GLE ridge0.001 na0.0001 nie daje istotnej użytkowo poprawy. '
        'Sam wpływ paczki dodany do oryginalnej bazy ma na s48 18,91 procent wobec '
        '22,65 procent tej samej bazy bez wejścia. Deskryptory z oboma wpływami '
        'mają 0,24 procent. Współczynniki poszczególnych cech nie są interpretowane przyczynowo.', '',
        '## Horyzonty i zamrożony wybór', '',
        '| H [aktualizacje] | Wybrany nowy wariant | Umiejętność s48 [%] | Oryginalny GLE [%] | Znak wybranego [%] |',
        '|---:|---|---:|---:|---:|']
    for h,hh in data['results'].items():
        n=hh['selected_new_validation']; a=hh['test']['48'][n];b=hh['test']['48']['legacy_gle']
        lines.append(f"| {h} | {n} | {a['skill_percent']:.2f} | {b['skill_percent']:.2f} | {100*a['sign']:.1f} |")
    lines += ['',
        'Na H10 walidacja wybrała jądro z historycznym wpływem paczki. '
        'Nie jest to dowód przydatności tego wejścia na H500. '
        'RMSE, MAE, znak wszystkich sond i znak połowy największych rzeczywistych zmian '
        'są w METRYKI.json; maska największych zmian służy tylko do oceny.', '',
        '![Porównanie prognoz](wyniki/WYKRES.png)', '',
        '## Reszty, ranga i kalibracja', '',
        'Pełna baza samego jądra ma rangę21/21 i kondycję około8; baza jądra '
        'z opisem i oboma wpływami ma rangę52/57 i kondycję około1,36e19. '
        'Część deskryptorów jest stała lub zależna. Ridge stabilizuje predykcję, '
        'ale nie dowodzi jednoznacznej identyfikacji dynamiki ani wpływu sterowania. '
        'Brak interwencji na polityce i tylko jeden bieg uczący ograniczają wzbudzenie.', '']
    for h in ('10','500'):
        rr=data['results'][h];n=rr['selected_new_validation'];a=rr['test']['48'][n]
        dg=a['residual_diagnostics']
        lines.append(f"Dla H{h}, `{n}`, s48: skośność {dg['skew']:.2f}, "
            f"kurtoza nadmiarowa {dg['excess_kurtosis']:.2f}, ACF lag1 {dg['residual_lag1']:.2f}, "
            f"korelacja reszty z historycznym stanem {dg['corr_past_state']:.2f}, "
            f"z wpływem paczki {dg['corr_past_batch_effect']:.2f}. "
            f"Wariancje pierwszej/drugiej połowy: {dg['variance_first_half']:.3g}/{dg['variance_second_half']:.3g}. "
            f"Orientacyjny przedział90 pokrywa {100*a['interval90_coverage']:.1f} procent "
            f"testowych wyników, szerokość {a['interval90_width']:.3g} jednostek s.")
        lines.append('')
    lines += [
        'ACF dla H500 nie jest testem białości jednokrokowego szumu: okna silnie się nakładają. '
        'H10 ma rozstawienie startów10, więc cele nie nakładają się, lecz sondy i stany '
        'wciąż są zależne. Kształt rozkładu, zależności i wariancję oceniamy osobno; '
        'nie stwierdzamy gaussowskości ani białości. Przedziały wyznaczono z błędów '
        'historycznej walidacji, bez testowych reszt w wejściu. Pokrycie dalekie od90 '
        'pokazuje ograniczoną kalibrację przy przeniesieniu między biegami.', '',
        '## Sprawdzenia matematyczne i przyczynowość', '',
        'Wszystkie nowe predyktory przeszły test równości prognozy z prefiksu '
        'i z pełnego pliku oraz po podmianie sufiksu stanów/wejść. '
        'To obejmuje odczyt tylko ukończonego wpływu t-1, nie wpływu aktualizacji t. '
        'Dodatkowy test dojrzałych etykiet i niezmienności dopasowania po zatruciu '
        'sufiksu znajduje się w `wyniki/TESTY.json`.', '',
        'Lean: cztery twierdzenia o skalarnej pamięci nad Int, kompilacja exit0 bez '
        'sorry/admit i nowych aksjomatów. SymPy:28 par dokładnych kontroli pamięci, '
        'Gaussian AR z niezerową kowariancją i dwa przykłady rang. '
        'To weryfikacja algebraiczna, a nie formalny dowód modelu ViT lub jego trafności. '
        'Pełny macierzowy wzór ma wyprowadzenie indukcyjne i kontrole SymPy; '
        'zakres Lean jest węższy. Szczegóły w matematyka/MATEMATYKA.md.', '',
        '## Czego pilot nie dowodzi i co dalej', '',
        'Nie potwierdza poprawy H500, domknięcia latentu, obserwowalności/sterowalności '
        'pełnej sieci, identyfikacji skutku dowolnej przyszłej paczki ani relacji FDT dla AdamW. '
        'Nie ustala nieosiągalnej granicy prognozy. Negatywny wynik tej siatki nie '
        'wyklucza innych reprezentacji paczek i modeli ich przyszłej dystrybucji.', '',
        'Biegi mają stały zbiór i tylko3000 aktualizacji. Dwa ziarna są jedynymi '
        'replikacjami;180 sond i41 okien nie zwiększają ich liczby. '
        'Sondy obejmują wspólne obrazy i warianty; nie jest to test nowych filmów '
        'ani ogólnego rozpoznawania narzędzia. LDA każdego biegu jest osobną '
        'kalibracją z jego rozgrzewki. Historia s47 z tej samej fazy już nie '
        'jest odłożonym źródłem przy dowolnym kolejnym strojeniu.', '',
        'Rozsądny następny etap to dłuższe wieloziarnowe nagrania z różnymi, '
        'z góry zadanymi politykami paczek oraz jawny model ich warunkowego '
        'przyszłego wpływu i reszt. Najpierw potrzebne są kilka niezależnych '
        'okien H500 po treningu/walidacji i rejestracja tych samych wejść. '
        'Nie uruchomiono takiej kampanii, nie wdrożono nowego równania do pętli SD '
        'i nie wykonano commitów.', '',
        '## Artefakty', '',
        '`MODEL.md`, `CONFIG.json`, `kod/dane.py`, `kod/pilot.py`, `kod/raport.py`, '
        '`kod/testy.py`, `wyniki/AUDYT.json`, `wyniki/METRYKI.json`, '
        '`wyniki/PROGNOZY.npz`, `wyniki/MODELE.npz`, `wyniki/TESTY.json`, '
        '`wyniki/WYKRES.png`, `wyniki/pilot_v1/`, `zrodla/` i `matematyka/`.']
    (HERE/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hs=[int(h) for h in data['results']]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,label in [('legacy_gle','Original GLE'),('gle_kernel_ridge','GLE tuned ridge'),
                       ('gle_kernel_effect','GLE + past batch effect'),
                       ('gle_kernel_descriptor_effect_ped','GLE + past descriptors/effects')]:
        ys=[(data['results'][str(h)]['test']['48'][name]['mse']/
             data['results'][str(h)]['test']['48'][name]['baseline_mse'])**.5 for h in hs]
        ax.plot(hs,ys,'o-',label=label)
    ax.axhline(1,color='black',linestyle='--',linewidth=1,label='No change')
    ax.set_xscale('log');ax.set_xlabel('Forecast horizon (updates)')
    ax.set_ylabel('RMSE / RMSE of no change')
    ax.set_title('Frozen s47 validation choices, transfer to s48 (exploratory)')
    ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(HERE/'wyniki/WYKRES.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
