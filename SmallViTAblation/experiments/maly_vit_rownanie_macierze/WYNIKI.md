# Wynik: uczone macierze jak w TS nie pomagają; prognoza bez przyszłych klatek stoi na 57–67 procentach (2026-09-29)

## Pytanie

Czy równanie o budowie z TS (dryf D, jądro pamięci JP, wymuszenie W, szum kolorowy N; wszystkie człony jako sieci
uczone razem błędem kroku) przewiduje jeden krok latentu sond i siły wektora tożsamości w uczeniu małego ViT, gdy
do równania nie wchodzą przyszłe klatki (paczka przewidywanego kroku też jest przyszła, reguła użytkownika). PLAN.md.

## Aparat

Nagrania lokalne ziaren 47 i 48 (`../maly_vit_rownanie_latentu/nagranie_s4{7,8}_sily`). Każde ziarno uczone
i oceniane osobno, bo to różne sieci: wektory tożsamości ziaren 47 i 48 mają cosinus 0,085, czyli układ neuronów
jest inny. Uczenie na krokach 1000–1800 (120 sond z wycinków „fit”), wybór epoki 1800–2000, test 2000–2999 na tych
samych sondach i na 60 sondach „holdout” (nowe narzędzia). Model 11 mln parametrów (D 4,2 mln, JP 0,7 mln, W 5,9 mln,
N 0,2 mln), AdamW 1e-3, EMA 0,995, kara rozdzielności 0,05, jak w TS. `uczenie.py`, `liniowe.py`; liczby
w `wyniki/BIEG_s*.json`, `wyniki/ODNIESIENIA.json`, `wyniki/LINIOWE.json`, krzywe w `wyniki/TRENING_s*.csv`.

## Wynik (umiejętność jednego kroku na teście, sondy „fit”; w nawiasie siła wektora tożsamości)

| równanie | co widzi | ziarno 47 | ziarno 48 |
|---|---|---|---|
| duże sieci D + JP + W + N (`pelne`) | stany do t, paczki t−1..t−16 z sygnałem błędu | 39,5 (17,0) | 40,8 (33,0) |
| to samo bez W (`bez_W`) | stany do t | 39,6 (14,9) | 40,9 (33,7) |
| placebo: paczki sprzed 250 kroków | — | 39,3 (15,4) | 40,3 (31,2) |
| W bez sygnału błędu (`bez_gradientu`) | latenty obrazów paczek t−1..t−16 | 39,6 (16,3) | 40,5 (34,6) |
| pełna macierz A Δz_t (ridge) | stany do t | 52,0 (43,0) | 50,8 (47,5) |
| pełne macierze, 3 ostatnie przyrosty | stany do t | 53,4 (42,5) | 53,6 (50,7) |
| pełna macierz A Δz_t + P S_pęd | + pęd AdamW liczony siecią | 60,9 (53,6) | — |
| **jedna liczba: 0,76 · Δz_t** | stany do t | **56,8 (56,6)** | **56,3 (56,8)** |
| **pęd: 0,94 · S_pęd** | pęd AdamW liczony siecią | **66,8 (65,7)** | **64,1 (63,8)** |
| tylko porównanie: 0,84 Δz_t + 0,99 S_t | paczka kroku t (przyszłe klatki, niedozwolone) | 85,3 (85,7) | 87,8 (89,1) |

Sondy „holdout” dają to samo z dokładnością do 1–2 punktów (sieci: 38,4–39,1; bezwładność 55,3–55,9; pęd 64,9–65,7).
Dla ziarna 48 współczynniki odniesień dopasowane na ziarnie 47 (to skalary, więc przenoszą się między sieciami).

## Werdykt

- **Duże uczone sieci przegrywają z jedną liczbą bezwładności o 16–17 punktów.** Na krokach uczenia trafiają około
  89 procent, na wyborze epoki 50, na teście 40: wkuwają okres uczenia. To samo, co w TS (94 na uczących cząstkach,
  62 na nowych), tylko ostrzej, bo tu cały zbiór to jeden przebieg: 800 kroków × 120 sond na 11 mln parametrów.
- **Wymuszenie z przeszłych paczek nic nie wnosi**: `pelne` = `bez_W` = `placebo` = `bez_gradientu` (różnice poniżej
  0,5 punktu), a wyłączenie W w ocenie nic nie zmienia. Wpływ dawnych paczek niesie już pęd, a ten widać w prędkości.
- **Pełne macierze liniowe też przegrywają ze skalarem** (52–54 wobec 57). Sprzężenie między 768 współrzędnymi
  dopasowuje szum okresu uczenia. Ruch latentu jest w każdej współrzędnej ten sam: 76 procent poprzedniego kroku.
- **Najlepsza uczciwa prognoza to pęd optymalizatora przepuszczony przez sieć: 64–67 procent.** Współczynnik 0,94
  przy sile pędu, a bezwładność dostaje wtedy −0,04: pęd zastępuje ją w całości. Bez przeliczania sieci zostaje
  bezwładność, 56–57 procent (równanie T1 z jądrami na każdej współrzędnej: 59 w medianie 10 ziaren Edenu).
- Cel 80 procent jest osiągalny tylko z paczką przewidywanego kroku, czyli z przyszłymi klatkami.

## Uwaga o kodzie TS (sprawdzone przy przenoszeniu)

`Projekt/experiments/tft_langevin_latent_56/model.py`, `SzumKolorowy`: splot ma dopełnienie OKNO − 1 przy jądrze
długości OKNO − 1 i potem jeszcze przesunięcie o jeden, więc N widzi reszty od dnia t−2, a nie t−1 (impuls reszty
w dniu 5 daje pierwsze wyjście w dniu 7; sprawdzone liczbowo). Komentarz w kodzie opisuje opóźnienie od t−1.
W `tft_langevin_czlon_zdarzen` resztę z t−1 dodają osobno odczepy `a1` (po przekątnej), w L1–L3 jej brakuje.
Tutaj splot poprawiony (pierwsze wyjście w dniu 6).

## Czego to NIE dowodzi

- Dwa ziarna, jedno ziarno równania na ramię, jedna budowa sieci (z TS). Mniejsze sieci albo uczenie na wielu
  przebiegach naraz mogłyby wkuwać mniej; na tych danych kierunek jest jednak jasny.
- Tylko jeden krok naprzód i tylko ramię bez dokładania klatek.
- Siła pędu ma w mianowniku AdamW nowy gradient z wagą 0,001 (przeciek pomijalny, nazwany w `dane.py`).
