# Krystalizacja latentu w trakcie uczenia

Pytanie autora (2026-09-22): jak zmienia sie latent detektora w trakcie uczenia, czy sie
krystalizuje, czy da sie po tym poznac, ktore klatki niosly informacje poprawiajaca model,
i czy takie klatki da sie wskazac z analizy koopmanowskiej.

Eksperyment nie uruchamia nowego doboru klatek. Korzysta z materialu, ktory juz jest:
jedenastu checkpointow jednego biegu, dwudziestu siedmiu biegow etapu 1 oraz dwoch nagran
czastek (pula uczaca i klipy sondujace) z tego samego checkpointu.

## Skrypty

| plik | co liczy |
|---|---|
| `krystalizacja.py` | zegar treningu: uczestnictwo kierunkow, udzial statyki, trwalosc najwolniejszych modow, przesuniecie i CKA miedzy checkpointami, odczyt stanu narzedzia |
| `wklad_klatek.py` | ktore klatki model zyskuje, a ktore traci po szczycie jakosci, i czy widac to w latencie |
| `pokrycie_a_bledy.py` | czy film walidacyjny gorzej pokryty przez wybrane klatki uczace ma wiecej pudel |
| `co_przewiduje_wynik.py` | czy po samym zestawie wybranych klatek widac, ktore ramie wyjdzie lepiej |
| `telemetria_treningu.py` | kiedy siec jest wytrenowana: szczyt jakosci i granica tolerancji we wszystkich biegach |

Zaleznosci miedzy skryptami rozwiazuje `PYTHONPATH=experiments/krystalizacja_latentu`.

## Wyniki (2026-09-22)

1. Latent ustawia sie po pierwszej epoce, a potem juz tylko dryfuje. Przesuniecie klatki miedzy
   checkpointami spada z 1,05 do 0,26, a podobienstwo geometrii (CKA) trzyma sie okolo 0,86
   i nigdy nie dochodzi do jednosci. Pelnej krystalizacji nie ma.
2. Po szczycie jakosci latent sie rozlewa: liczba faktycznie uzywanych kierunkow rosnie z 8,6
   do 16,4 (blok `query`), a jakosc na walidacji spada.
3. Trwalosc najwolniejszych modow, czyli srednia czesc rzeczywista wartosci wlasnych operatora
   kroku klatki, ma szczyt w epoce 5. Szczyt jakosci to epoka 5 albo 6. Zgodnosc utrzymuje sie
   w 32 z 36 sprawdzonych ustawien (cztery bloki latentu, trzy wymiary, trzy liczby modow).
   To jest kandydat na sygnal konca treningu liczony bez etykiet.
4. Model staje sie przy tym pewniejszy siebie mimo gorszych wynikow: srednia pewnosc detekcji
   rosnie z 0,83 do 0,93.
5. Wynik negatywny: trafnosc na klatce ani jej zmiana po szczycie nie daja sie przewidziec
   z latentu miedzy filmami (R2 nie przekracza 7 procent).
6. Wynik negatywny: odleglosc filmu walidacyjnego od zbioru uczacego nie porzadkuje pudel
   (korelacja rangowa 0,0 na dwunastu filmach); calosc surowej korelacji robi jeden film odstajacy.
7. Wynik slaby: po samym zestawie wybranych klatek widac tylko ramie wyraznie zle (B0, kolejne
   klatki filmu). Wsrod zdrowych ramion przestrzen Koopmana nie bije kierunkow losowych.
8. Telemetria 28 biegow: szczyt jakosci wypada w epoce 8 (mediana, zakres od 4 do 16), a juz
   w epoce 6 jakosc jest w granicy pol procenta od szczytu. Trening trwa 30 epok, wiec ostatnie
   dwadziescia kilka epok tylko szkodzi: jakosc spada o okolo 0,03.
9. Kampania NIGDY nie zapisywala straty na walidacji. RF-DETR 1.10 ma `compute_val_loss: "auto"`,
   co oznacza pominiecie, gdy zadne narzedzie jej nie pilnuje. Dopisany przelacznik `--val-loss`
   w `pipeline/train/train_rfdetr.py` wlacza ja bez zmiany przepisu optymalizacji.

10. Trzy biegi z pelna telemetria (ziarna 42, 1337, 2024) potwierdzaja to niezaleznie.
    Strata na walidacji ma dolek w epoce 6, 6 i 7, a jakosc szczyt w epoce 6, 8 i 7.
    Strata na klatkach uczacych spada do konca z 6,2 do 1,07, a walidacyjna zawraca w epoce
    szostej i rosnie z 3,22 do 3,65. Uczenie do konca kosztuje srednio 0,030 mAP.
11. Sygnal bez etykiet (szczyt trwalosci, bloki `query` i `features`) wskazuje epoke 5, 5 i 7.
    Pojscie za nim kosztuje srednio 0,004 mAP zamiast 0,030, czyli okolo osiem razy mniej
    niz trening do konca. Blok `global_pool` jest zawodny, potrafi wskazac epoke 3.
    Zastrzezenie: wagi zapisywano co dwie epoki, wiec rozdzielczosc sygnalu to dwie epoki.

## Dane

- `dane/TELEMETRIA_stage1.json` - zrzut wszystkich skalarow TensorBoard z 28 biegow kampanii.
- `dane/biegi/<bieg>/TELEMETRIA.json` - to samo dla trzech biegow z zapisana strata walidacyjna.
- `dane/TRWALOSC_telemetria.json` - krzywe trwalosci modow dla tych biegow, policzone na klastrze
  (latenty zostaja na Edenie, bo waza okolo gigabajta).
- `wyniki/REPORT_*.json` - wyniki poszczegolnych skryptow.

## Biegi na Edenie

`eden/run_telemetria_epok.slurm` powtarza bieg `B2_uniform_time_b1500_sel42` na trzech ziarnach
z pelna telemetria: strata na walidacji co epoke, wagi co dwie epoki, latent kazdego zachowanego
checkpointu, zrzut skalarow do `TELEMETRIA.json`. Wyniki lada w
`~/smallvit_runs/2026-09-21_koopman_langevin/results_telemetria`.

Pulapka: `sbatch --export=ALL,ZMIENNA=...` konczy sie na tym klastrze anulowaniem zadania przez
scheduler po sekundzie (`CANCELLED by 0`). Zmienne trzeba wyeksportowac w powloce przed `sbatch`.
