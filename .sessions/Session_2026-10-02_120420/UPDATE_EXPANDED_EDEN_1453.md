# Uzupełnienie sesji: rozszerzony stan NNDMD i Eden

Zapis: 2026-10-02 14:53, Europe/Warsaw. Jest to dalsze rzeczowe podsumowanie sesji, nie transkrypcja. Zachowuje poprzednie raporty i ich wyniki.

## Nowe zlecenie użytkownika

Po wcześniejszym NNDMD na 34 obrazach użytkownik zlecił trening na dokładnie 1500 obrazach źródłowej bazy, na wszystkich 600 cząsteczkach każdego obrazu. Następnie jawnie polecił zatrzymać dotychczasowe lokalne przebiegi i benchmarki oraz przygotować nowy trening na Edenie.

Aktualny wariant rozszerza wejście o historię przyrostów pełnego z768 i historyczne charakterystyki zmian parametrów oraz dostępnych stanów AdamW. Learned state ma pozostać 768D, A ma rozmiar 768×768, decoder ma być nieliniowy. Nie stosujemy PCA. J128 pozostaje podzbiorem oryginalnych kanałów do dodatkowej oceny po dekodowaniu; najnowsza intencja treningu obejmuje pełne 768, bez dodatkowego ważenia J128.

Batch32 oznacza 32 obrazy, nie 32 tokeny. Mikrofragmenty służą ograniczeniu pamięci; gradient ma reprezentować wszystkie 600 tokenów każdego obrazu z równą wagą obrazów. Jeden wspólny operator per token nie stanowi automatycznie modelu oddziaływań między cząsteczkami.

## Co już sprawdzono

- Manifest dokładnie 1500 obrazów zachowuje źródłową kolejność i hashe; generator lokalny potwierdził zgodność pikseli 1500/1500. Baza ma bbox, bez masek segmentacji.
- Dostępnych jest 101 checkpointów wag od k0000 do k1000 co 10 aktualizacji. Pełny dump wszystkich stanów 768 dla 1500 obrazów zajmowałby około 279,245 GB, więc przygotowano streaming i cache o ograniczonej liczbie slotów.
- Wersja rozszerzona używa 13 slotów: dwa stany historyczne oraz origin i dziesięć przyszłych celów okna. Źródłowy ViT jest zamrożony w eval; kontrolowane są hashe, kompletność i brak zmian BN.
- Dane źródłowego biegu zawierają także 11 pełnych zapisów AdamW co 100 aktualizacji. Wykonawca danych potwierdził na Edenie klucze net/opt/krok oraz rzeczywiste exp_avg/exp_avg_sq w k0100. Dostępne RMS momentów i zmian wag są charakterystykami amplitudy; nie odtwarzają kierunków całych gradientów.
- Niezależna kontrola CPU wcześniejszego rdzenia straty wykazała zgodność gradientów pełnej paczki i nierównych mikrofragmentów, z największą różnicą około 3,73e−9. Ten test dotyczył rdzenia z-only, nie stanowi zatwierdzenia nowego rozszerzonego modelu.

## Granica czasowa i praca pozostała

Planowana selekcja ma używać inner fit do checkpointu 80 i walidacji 80→90. Po wyborze następuje refit do 90, a dopiero potem autonomiczny test 90→100. Każdy krok checkpointowy odpowiada 10 aktualizacjom ViT. Przyszłe prawdziwe wagi, momenty, gradienty ani reprezentacje nie mogą zasilać prognozy. Przyszłe aux muszą być prognozowane lub aktualizowane z własnych prognoz i jawnie znanych zegarów.

W chwili zapisu trwa przygotowanie telemetrii CPU na Edenie i kodu nowego modelu. Nie potwierdzono jeszcze kompletnego audytu modelu, wysłania nowego joba ani wyników treningu. Użytkownik autoryzował zgłoszenie zadania Eden po weryfikacji; wykonawca danych odpowiada za job. Audytor nie używa GPU. Indeksowanie sesji wykonywane jest oddzielnie na CPU.

Brak wyniku nowego wariantu nie zmienia negatywnego wyniku wcześniejszego eksperymentu opisanego w [poprzednim uzupełnieniu](UPDATE_NNDMD_FINAL_1325.md). Wcześniejsze osiągnięcie 92,83 procent rozpoznania ROI pozostaje inną metryką niż prognoza dynamiki i nie daje obietnicy 85 procent znaku przyszłych zmian.

## Artefakty

- [Nowy eksperyment i status](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden/status.json).
- [Manifest 1500](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden/MANIFEST_1500.json).
- [Przygotowanie źródeł na Edenie](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden/przygotuj_eden.py).
- [Streaming danych](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden/dane.py).
- [Telemetria historyczna](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden/telemetria.py).
- [Bieżący audyt przed uruchomieniem](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_expandedstate_eden/AUDYT.md).
- [Poprzedni lokalny wariant 1500 i kontrola rdzenia straty](../../SmallViTAblation/experiments/maly_vit_nndmd_1500_obrazow_768/AUDYT.md).

