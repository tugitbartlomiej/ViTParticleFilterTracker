# Sesja 2026-10-06 22:08:07: aplikacja latentu ViT (PCA, położenie, czubek, przebudowa GUI, ustawienia, odpinanie)

## Cel sesji

Ciąg dalszy sesji `Session_2026-10-06_171603` (ta sama rozmowa po kompresji kontekstu). Rozbudowa
i przebudowa aplikacji `SmallViTAblation/app`: widoki 3D (kula kierunków, PCA, liczba cząsteczek,
półprzezroczysty obraz), przestrzeń makrowektorów (wspólny środek, odjęcie położenia patcha), sekwencje
z folderu zdjęć, śledzenie czubka igły, przebudowa GUI w krokach, wygodne pola, plik ustawień, odpinanie
widoków 3D, auto-szukanie w wybranych patchach. Do tego plan małego ViT z 8 tokenami (TokenLearner).

## Typ

experiment

## Workerzy użyci

Cztery podagenty (general-purpose) równolegle przy przebudowie GUI, każdy z osobnymi plikami: Edytor SD
(`app.py`), Detekcja (`zakladka_detekcji.py`), lewy panel filmu (`zakladka_filmu.py`), widoki cząsteczek
i atencji (`czasteczki3d.py`, `przestrzen_makro.py`, `opcje_atencji.py`, `czasteczki.py`). Bez commitów
po ich stronie; każdą zmianę sprawdzono zrzutem i testem odcisku i zacommitowano osobno.

## Co zrobiono (skrót)

- Widoki 3D: pełna kula kierunków ze środkami na płaszczyźnie obrazu i półprzezroczystym zdjęciem; PCA
  wektorów do 3, 4, 8, 16, 32, 128 składowych (TR_0001: 58 / 64 / 76 / 86 / 93 / 98 % wariancji); wybór
  liczby cząsteczek co n-ty patch (600 do 6).
- Przestrzeń makrowektorów: mikrowektory ze wspólnego środka z wypadkową; odjęcie położenia patcha
  (położenie to 39,2 % wariancji latentu K1000, 13 kierunków = 95 % tego wkładu).
- Plan ViT z 8 tokenami (TokenLearner, praca przeczytana w Undermind); RTX 3070 wystarcza (42 ms na krok).
- Folder `app/narzedzie_czarne_tlo`: 400 klatek TR_0001 z samym narzędziem; wczytywanie folderu zdjęć.
- Śledzenie czubka igły jako automatycznie wybranego patcha.
- GUI w 7 krokach, wygoda wprowadzania, `ustawienia.json`, Odepnij / Przypnij dla widoków 3D, exe.
- Auto-szukanie w wybranych patchach: jeden patch środka daje to samo co 600 (latent za ostatnią atencją).
- 13 commitów w gałęzi `codex/eden-nano-training-20260914`, od `433b1e75` do `bbf93145`.

Szczegóły: `SESSION_SUMMARY.md`.
