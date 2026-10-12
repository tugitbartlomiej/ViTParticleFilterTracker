# Pobranie pełnej historii tokenów z Edenu

Data: 2026-10-01. **Status: ukończone i zweryfikowane.**

Pobrano jeden istniejący plik `LATENT_GESTY.npy`: **22 140 518 528 bajtów** (22.140519 GB; 20.619965 GiB).

Plik zawiera **1001 × 12 × 600 × 768 liczb float32**: pełne położenia tokenów dwunastu stałych obrazów przed uczeniem i po każdej z tysiąca aktualizacji. Nie uśredniono tokenów. Ruch wylicza się jako różnicę kolejnych stanów. [Dokładny opis danych](OPIS_DANYCH.md).

- Źródło: istniejący alias SSH `eden-cluster`, plik w `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/`.
- Cel: [../dane/LATENT_GESTY.npy](../dane/LATENT_GESTY.npy).
- Początek lokalnego transferu: 2026-10-01T17:19:59.783638+02:00.
- Zakończenie wraz z weryfikacją: 2026-10-01T17:42:21.165721+02:00.
- Wolne miejsce po ukończeniu: 1 354 088 448 bajtów (1.354 GB).

## Weryfikacja

Zdalnie odczytano nagłówek oraz obliczono świeży SHA256 całego pliku. Rozmiar i mtime nie zmieniły się podczas tej kontroli. Hash był również zgodny z metadanymi zapisanymi po eksperymencie. Po transferze obliczono SHA256 całego pliku lokalnego; wszystkie trzy wartości są identyczne:

`d148b9649154b1ac4957da2c7d39184921ad852b195fdefe0125e28b284c5576`

Sprawdzono nagłówek NumPy, shape, float32, kolejność C oraz 24 małe próbki pełnych wektorów w krokach 0, 1, 500 i 1000. Próbki są skończone. Odczyt wykonano przez mmap, bez ładowania całej tablicy do RAM. Próbkowa kontrola wartości nie jest deklaracją ponownego pełnego skanowania wszystkich wartości.

Transport: OpenSSH SFTP `reget` do pojedynczego `.part`. Co 10 sekund kontrolowano prognozowany końcowy zapas miejsca, z minimum 1 GiB. Po poprawnym hashu wykonano zmianę nazwy w tym samym katalogu. Nie utworzono drugiej pełnej kopii. Skrypt zachowuje nieukończony plik na wypadek potrzeby wznowienia.

Małe obrazy i maski istniały już lokalnie. Ich sumy porównano ze źródłem Eden i umieszczono identyczne pliki obok historii, łącznie około 5,63 MB. Zachowano właściwe metadane biegu gęstego. Nie pobierano stanów AdamW i nie zmieniano ani usuwano niczego na Edenie.

Samo pobranie nie uruchamia forwardów, treningu ani analizy. Odrębny, autoryzowany pilot pełnych tokenów ma własny katalog `experiments/maly_vit_dynamika_pelnych_czasteczek_checkpointy` i nie tworzy drugiego dużego cache.

## Dowody i wznowienie

- [Manifest zdalny](MANIFEST_ZDALNY_2026-10-01.json).
- [Manifest lokalny i próbki](MANIFEST_LOKALNY_2026-10-01.json).
- [Audyt miejsca przed transferem](AUDYT_MIEJSCA_2026-10-01.json).
- [Sumy małych plików pomocniczych](METADANE_ZDALNE_2026-10-01.json).
- [Skrypt pobrania i weryfikacji](pobierz_latent.py).

Wznowienie w tym samym środowisku Python: `pobierz_latent.py download`. Poprawny finalny plik zostanie zachowany, nie nadpisany. Przed przenoszeniem lub kasowaniem czegokolwiek potrzebna byłaby oddzielna decyzja; ten skrypt nie usuwa danych.
