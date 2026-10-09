# Ruch cząsteczek K1000 liczony na GPU

> **Zastąpione 2026-10-09.** Użytkownik odrzucił wersję przeglądarkową (WebGPU): animacja ma działać natywnie pod Windows w aplikacji `app`. Natywny odpowiednik to zakładka „Ruch zbioru: cząsteczki i optymalizacja” z compute shaderami OpenGL. Ten folder zostaje tylko jako zapis i źródło kerneli przeniesionych do GLSL; serwer nie jest już uruchamiany.


Refaktor `vit_k1000_taniec_czasteczek_20261009`. Te same dane, ta sama wspólna PCA3, nowe miejsce obliczeń. Stara rodzina zostaje nietknięta, a jej `projection.npz` czytamy tylko do odczytu po sprawdzeniu SHA-256.

## Co się zmieniło

W starej wersji Python liczył każdą klatkę widoku: wybór stanu, przesunięcia, granice osi, mapy, mediany i krzywe. WebGL tylko obracał i kolorował trzy gotowe bufory chmury, a panel kontrolny pobierał JSON z serwera dla każdego kroku.

Teraz serwer wysyła wyłącznie surowe tablice NPZ w ich oryginalnym układzie C. Compute shadery WGSL w WebGPU liczą:

| Kernel | Co liczy |
|---|---|
| `kin_milestone` | położenie 900 000 cząsteczek w czasie, przesunięcie od startu, kolor |
| `kin_paths` | odcinki 0 → 100 → 500 jednego obrazu |
| `kin_control`, `kin_trail` | 128 trajektorii panelu w każdym kroku i ślad 40 kroków |
| `stats_milestone`, `stats_control`, `finalize_stats` | granice osi po wszystkich stanach widoku, maksima normy i koloru, środek i promień kamery |
| `heat` | mapa 20 × 30 średniej normy 768D po 1500 obrazach dla trzech stanów |
| `hist_hi`, `hist_lo` | dokładna mediana i maksimum normy, wybór pozycyjny po bitach bez sortowania |
| `curves` | średnia pewność, IoU i liczba sukcesów dla 501 kroków |
| `bars` | średni ruch 8 śledzonych patchy panelu |

Kamera, wybór punktu przez teksturę identyfikatorów i ramka osi czytają wyniki bezpośrednio z buforów GPU.

## Kontrakt prawdziwości

- W zapisanych krokach 0, 100 i 500 kernel zwraca wartości bitowo równe zapisowi. Hashe bez przesunięć są identyczne z `WERYFIKACJA_SHADERA.json` starej wersji.
- Między zapisanymi stanami chmury GPU prowadzi każdą cząsteczkę po prostym odcinku przemieszczenia. To pomoc wzrokowa. Plakietka „stan niezapisany” jest widoczna przez cały czas trwania odcinka, a mediana i mapa pokazują wtedy ostatni zapisany stan. Odcinki można wyłączyć; suwak wraca wtedy do trzech stanów.
- Pewność i IoU są zapisane w każdym kroku, więc krzywe i chipy detekcji są rzeczywiste również między stanami chmury.
- Panel kontrolny używa tylko zapisanych kroków; nic nie jest interpolowane.
- Norma 768D pochodzi z eksportu. Surowe latenty 768D leżą na Edenie, więc GPU nie przelicza rzutu PCA.

## Weryfikacja

Przycisk „Zweryfikuj obliczenia GPU względem numpy” uruchamia 34 przypadki. Przeglądarka odczytuje bufory GPU i wysyła je do serwera. Numpy liczy referencję niezależnie i zapisuje `outputs/vit_k1000_taniec_czasteczek_gpu_20261009/WERYFIKACJA_GPU.json`.

Porównania bitowe: kinematyka w zapisanych stanach, panel, ślady, granice, statystyki pozycyjne, wybór punktu. Porównania z tolerancją: odcinki między stanami, długości wektorów, średnie z redukcji w f32.
