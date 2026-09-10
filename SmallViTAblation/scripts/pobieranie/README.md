Pobieranie Cataract-LMM — po jednym skrypcie na zadanie
=======================================================

Zbiór: https://huggingface.co/datasets/mjahmadi/Cataract-LMM
Całość waży **772 GB**, więc pobieramy wyłącznie to, co potrzebne.

Docelowy katalog ustawia `_config.bat` (domyślnie `F:\Cataract-LMM`).
Zmień tam jedną linię, jeśli chcesz gdzie indziej.

---

## Kolejność

| Skrypt | Rozmiar | Co robi |
|---|---:|---|
| `0_instalacja_i_login.bat` | — | instaluje narzędzie, loguje; **uruchom raz** |
| `1_pobierz_ramki_coco.bat` | **10 MB** | same ramki narzędzi, 6 094 klatki, 12 klas |
| `5_sprawdz_pobrane.bat` | — | podgląd: klasy, rozdzielczości, rozmiar ramek |
| `2_pobierz_klatki.bat` | 448 MB | obrazy do tych ramek, rozpakowuje automatycznie |
| `3_pobierz_yolo.bat` | mały | te same adnotacje jako pliki tekstowe YOLO |
| `4_pobierz_sledzenie_probka.bat` | 100 MB | dwa najmniejsze klipy ze śledzeniem narzędzi |

**Zacznij od zera, potem jedynka, potem piątka.** Dziesięć megabajtów i od razu
wiadomo, czy ich sposób oznaczania pasuje do naszego.

## Zanim uruchomisz cokolwiek

Zbiór jest **za bramką**. Wejdź na stronę zbioru, zaloguj się i zaakceptuj
warunki dostępu, inaczej pobieranie zwróci błąd 401. Token typu READ
wygenerujesz na https://huggingface.co/settings/tokens

## Czego NIE robić

Nie pobieraj całych katalogów:

- `3_Object_Tracking/*` to **170 archiwów i 45 GB**,
- `5_Raw_Videos/*` to trzy tysiące nagrań i większość z 772 GB,
- `2_Instance_Segmentation/videos/*` to cztery archiwa po około 1,9 GB.

Żeby dobrać kolejne klipy śledzenia, dopisz nowe `--include` z nazwą pliku
w skrypcie numer cztery.

## Co dalej z tymi danymi

Katalog segmentacji daje **ramki narzędzi na 6 094 klatkach**, z czego
2 162 są w pełnym HD, czyli w naszej rozdzielczości. To materiał na drugi
zbiór oceny, z innego szpitala.

Katalog śledzenia daje coś innego: **współrzędne końcówki narzędzia klatka po
klatce**. To jest wzorzec do sprawdzenia, jak dobrze działa nasza propagacja
ramek — puszczamy własne śledzenie na ich klipach i porównujemy.

## Licencja

Karta zbioru podaje **CC BY-NC-ND 4.0**, a tekst publikacji przy adnotacjach
mówi CC BY 4.0. **Te dwie informacje są sprzeczne.** Do czasu wyjaśnienia
u autorów trzymamy się ostrzejszej wersji: trenować i publikować wyniki wolno,
ale **przetworzonych klatek ani przerobionych plików adnotacji nie
publikujemy**. Publikujemy listy identyfikatorów i kod, który je odtwarza.
