# Jak tańczyły cząsteczki w latencie K1000 · wersja GPU

> **Zastąpione 2026-10-09.** Użytkownik odrzucił wersję przeglądarkową (WebGPU): animacja ma działać natywnie pod Windows w aplikacji `app`. Natywny odpowiednik to zakładka „Ruch zbioru: cząsteczki i optymalizacja” z compute shaderami OpenGL. Ten folder zostaje tylko jako zapis i źródło kerneli przeniesionych do GLSL; serwer nie jest już uruchamiany.


Interaktywny podgląd tych samych rzeczywistych zapisów co `outputs/vit_k1000_taniec_czasteczek_20261009`, ale cały ruch i wszystkie statystyki liczą compute shadery WGSL w WebGPU. To ruch aktywacji przy zamrożonych wagach K1000, a nie ruch wywołany dalszym uczeniem wag.

## Otwieranie

Uruchom `URUCHOM_WIZUALIZACJE.ps1` z tego folderu. Skrypt startuje lokalny serwer na porcie 8775 i otwiera http://127.0.0.1:8775/ . Wymagana jest przeglądarka z WebGPU, na przykład Chrome lub Edge w wersji 113 lub nowszej. Stary podgląd na porcie 8774 działa niezależnie.

## Podział pracy

Serwer `experiments/vit_k1000_taniec_czasteczek_gpu_20261009/serve.py` sprawdza SHA-256 pliku `projection.npz` i wysyła surowe tablice bez przeliczania. Przeglądarka sprawdza SHA-256 każdej tablicy i kopiuje około 50 MB do buforów GPU. Od tej chwili każda klatka to jeden przebieg kernela ruchu i jeden przebieg rysowania.

| Kernel WGSL | Co liczy |
|---|---|
| `kin_milestone` | położenie 900 000 cząsteczek w czasie, przesunięcie od startu, wielkość koloru |
| `kin_paths` | odcinki 0 → 100 → 500 dla jednego obrazu |
| `kin_control`, `kin_trail` | 128 trajektorii panelu w każdym kroku i ślad ostatnich 40 kroków |
| `stats_*`, `finalize_stats` | granice osi po wszystkich stanach widoku, maksima, środek i promień kamery |
| `heat` | mapa 20 × 30 średniej normy 768D po 1500 obrazach |
| `hist_hi`, `hist_lo` | dokładna mediana i maksimum normy, bez sortowania |
| `curves` | średnia pewność, IoU i liczba sukcesów dla 501 kroków |
| `bars` | średni ruch 8 śledzonych patchy panelu |

## Trzy widoki

1. **Cały zbiór:** wszystkie 900 000 cząsteczek. W krokach 0, 100 i 500 położenia są bitowo równe zapisowi. Pomiędzy nimi GPU prowadzi każdą cząsteczkę po prostym odcinku przemieszczenia, a plakietka nad chmurą informuje, że to stan niezapisany. Przełącznik ruchu między zapisami przywraca skoki między trzema stanami.
2. **Jeden obraz:** wszystkie 600 cząsteczek, opcjonalnie z odcinkami 0 → 100 → 500.
3. **Ruch co krok:** 128 zapisanych trajektorii panelu kontrolnego, bez interpolacji, ze śladem 40 kroków.

Kolor może pokazywać normę ruchu w 768D z zapisu, prędkość na odcinku w rzucie PCA albo przesunięcie w rzucie PCA. W panelu dochodzi tożsamość patcha oraz zapisana norma pojedynczego kroku w 768D. Pewność i IoU są zapisane w każdym kroku, więc krzywe detekcji są rzeczywiste również między stanami chmury.

Przeciągnięcie obraca, kółko myszy przybliża, kliknięcie odczytuje obraz i patch przez teksturę identyfikatorów. Przycisk zapisu klatki renderuje widok poza ekranem i zapisuje PNG w folderze `klatki`.

## Weryfikacja

`WERYFIKACJA_GPU.json` zawiera wynik 34 przypadków porównanych przez numpy na CPU. Wszystkie przeszły na NVIDIA RTX 3070 z architekturą Ampere.

- **Bitowo:** kinematyka 900 000 cząsteczek w trzech stanach z przesunięciami i bez, jeden obraz, panel, ślady, granice osi, mediany i maksima, wybór punktu.
- **Z tolerancją:** odcinki między stanami, długości wektorów i średnie redukcji f32. Największy błąd względny średnich jest rzędu 1e-7.
- **Zgodność ze starą wersją:** hashe kernela dla stanów 0, 100 i 500 bez przesunięć są identyczne z hashami w `WERYFIKACJA_SHADERA.json` starego renderera WebGL.

Weryfikację można powtórzyć przyciskiem na stronie albo otwierając http://127.0.0.1:8775/?selftest .

## Ograniczenia

- Rzut 3D pomija pozostałe wymiary. Mała odległość na ekranie może ukrywać duży ruch w 768D.
- Norma 768D pochodzi z eksportu. Surowe latenty 768D są tylko na Edenie, więc GPU nie przelicza rzutu PCA.
- Odcinki między stanami 0, 100 i 500 nie są zmierzoną trasą.
- Oracle korzysta z etykiet tych samych 1500 obrazów i nie mierzy generalizacji.
