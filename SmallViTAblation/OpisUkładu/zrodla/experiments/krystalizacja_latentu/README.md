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
| `kiedy_przestac.py` | czy szczyt trwalosci modow wskazuje epoke konca treningu bez etykiet |
| `zajetosc_tozsamosci.py` | puste kwadraty przestrzeni tozsamosci wzgledem zbioru oznaczonego |
| `nasycenie_tozsamosci.py` | ile klatek trzeba, zeby obstawic kwadraty tozsamosci, cztery sposoby wyboru |
| `buduj_ramiona_250.py` | trzy zestawy po 250 klatek (po kwadratach, rownomiernie, losowo) do biegow na Edenie |
| `wzor_domkniecia.py` | sprawdzenie wzoru z dokumentu 44: algebra na ukladzie syntetycznym, wyciek i pamiec na puli |

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

## Wyniki (2026-09-23)

12. Dziewiec biegow na 250 klatkach (sufit 60 epok, patience 15, `wyniki/REPORT_maly_budzet.json`).
    mAP 0,5:0,95 na walidacji: po kwadratach 0,6296 +- 0,0011, losowo 0,6319 +- 0,0023,
    rownomiernie w czasie 0,6143 +- 0,0096; odniesienie 1500 klatek 0,6866. Kwadraty remisuja
    z losowym (-0,0023) i sa o 0,057 ponizej 1500 klatek. Wygrana nad rownomiernym (+0,015) jest
    obciazona: przy dwoch klatkach na film `linspace` bierze pierwsza i ostatnia klatke filmu,
    tak jest ze 155 z 250 klatek. Kwadraty maja najmniejszy rozrzut miedzy ziarnami, najmniejszy
    spadek do ostatniej epoki i najpozniejszy szczyt (epoki 13 do 22 wobec 6 do 11).
13. Blad w osiach kwadratow: osie to mody 5, 6 i 7, a 6 i 7 to para sprzezona o tej samej czesci
    rzeczywistej (korelacja 1). Kratka 16^3 byla w praktyce 16^2 (252 zajete komorki z 256).
    Poprawka na przyszlosc: z pary sprzezonej brac czesc rzeczywista i urojona jednego modu.
14. Wzor domkniecia (dokument 44, `wyniki/REPORT_wzor_domkniecia.json`). Na ukladzie syntetycznym
    jadro pamieci z rekurencji Lin21b rowna sie B D^(l-1) C co do zera, zwiazek fluktuacji
    i dyssypacji zgadza sie dokladnie, a przyklad C = 0 pokazuje pamiec zero przy niezerowym
    wycieku (poprawka do dokumentu 43). Na puli: wyciek tozsamosci w slowniku 96 wynosi 0,153
    (scena 0,096); pamiec z korelacji 0,108 lezy ponizej podlogi szumu 0,142, ale pamiec z szumu
    0,080 lezy ponad swoja podloga 0,049 i oba jadra sie nie zgadzaja. Najprostsze wyjasnienie:
    stale przesuniecie filmu (okolo polowy zmiennosci osi tozsamosci jest miedzy filmami).

## Dane

- `dane/TELEMETRIA_stage1.json` - zrzut wszystkich skalarow TensorBoard z 28 biegow kampanii.
- `dane/biegi/<bieg>/TELEMETRIA.json` - to samo dla trzech biegow z zapisana strata walidacyjna.
- `dane/biegi/maly_budzet/<bieg>/` - telemetria, raport walidacji, mAP po epokach i manifest
  dziewieciu biegow na 250 klatkach.
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
