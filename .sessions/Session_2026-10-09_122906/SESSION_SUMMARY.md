# Session Summary: Animacje cząsteczek na GPU (OpenGL), szybki start, pomiar naturalny na CUDA i zapis 16-bitowy

## Metadata
- **Date:** 2026-10-09
- **Time:** około 01:40–12:30 (czas lokalny)
- **Status:** Completed
- **Type:** Mixed (refaktor GUI + GPU, benchmark, obliczenia)
- **Gałąź:** `codex/eden-nano-training-20260914`

## Objective
1. Policzyć cząsteczki i ich ruch na GPU („shaderem”), wzorując się na eksperymencie
   `vit_k1000_taniec_czasteczek_20261009`.
2. Po korekcie użytkownika: nie w przeglądarce, tylko natywnie pod Windows (OpenGL albo DirectX) w aplikacji
   `SmallViTAblation/app`, w szczególności animacja „Przestrzeń makrowektorów” w zakładce „Uczenie ViT” i ostatnia
   zakładka z dużą liczbą cząsteczek („Ruch zbioru”).
3. Głęboki refaktor „wszystko na GPU”, optymalizacja GUI, start bez wczytywania sekwencji, dymki przy każdej
   kontrolce; podział na maksymalnie 5 agentów (nie Fable) i audyt końcowy.
4. Pomiar naturalny („Naturalny pomiar obrazów”) szybciej na GPU, dokończenie brakujących pomiarów na GPU, zapis na
   dysk, lepszy (mniejszy, szybszy) format zapisu cząsteczek.

## Context
Eksperyment `vit_k1000_taniec_czasteczek_20261009` miał podgląd WebGL, który tylko rysował trzy gotowe chmury, a
liczył Python. Aplikacja PyQt6 (pyqtgraph.opengl) liczyła animacje na CPU w każdej klatce. Równolegle Codex
(niezatwierdzone pliki) budował zakładkę „Ruch zbioru: cząsteczki i optymalizacja” (`zakladka_ruchu.py`, `ruch_gl.py`).

## Actions Taken
1. Wersja przeglądarkowa WebGPU (`experiments/vit_k1000_taniec_czasteczek_gpu_20261009`, kernele WGSL, sędzia numpy
   34/34) — odrzucona przez użytkownika („nie miało być w przeglądarce”); serwer zatrzymany, folder oznaczony
   „Zastąpione”, kernele posłużyły jako spis do GLSL.
2. `app/makro_gpu.py` (nowy): przestrzeń makrowektorów na compute shaderach OpenGL 4.3 (przelot, mikrowektory, łańcuch,
   wypadkowa, ślady, zdjęcie na siatce cząstek, brzeg, strzałki atencji); `przestrzen_makro.py` przepięty;
   `test_makro_gpu.py` (odczyt buforów GPU vs wzory CPU).
3. Trzech agentów równolegle (rozłączne pliki):
   - Opus — `ruch_gpu.py` (nowy, zastąpił `ruch_gl.py`), `zakladka_ruchu.py`: 900 000 cząsteczek na GPU, magazyn
     stanów osi czasu w VRAM / pierścień, statystyki, mediana, wybór punktu; bez autowczytywania.
   - Opus — `czasteczki3d_gpu.py` (nowy), `czasteczki3d.py`, `widok_gl.py` (usunięte `LinieGrube`).
   - Sonnet — start bez sekwencji (przycisk „Wczytaj ostatni” zamiast „przy starcie”), leniwe zakładki
     (`ZAKLADKI_LENIWE`), model SD przy pierwszym GENERUJ, blitting wykresów w „Uczenie ViT”, `sprawdz_dymki.py`.
4. Audyt: pełne testy, `conftest.py` (wspólne konteksty OpenGL jak w `start.py`), test integracyjny całej aplikacji na
   prawdziwych danych (6 zakładek, historia uczenia 1001 kroków, oś 900 000 cząsteczek, 0 błędów GL), prawdziwy start
   z ustawieniami użytkownika (3,5 s, 0 CPU w tle), README.
5. Pomiar naturalny na GPU (`ruch_naturalny.py`): obrazy dekodowane raz i trzymane na karcie, stan odniesienia w VRAM
   zamiast memmapu 2,8 GB czytanego w każdej paczce, rzut PCA / normy 768D (float64) i metryki detekcji wsadowo na CUDA;
   domyślna paczka 32.
6. `uzupelnij()` + `niekompletne()` + przycisk „Uzupełnij brakujące kroki”: dolicza brakujące checkpointy w tym samym
   folderze, z kontrolą SHA256, bazą PCA z pliku i bitowym odtworzeniem kroku bazowego.
7. Badanie formatu zapisu (21 prawdziwych kroków): bezstratnie najwyżej −24 % (zstd z tasowaniem bajtów, wolniejszy
   odczyt), float16 −50 % (błąd 2e-3), uint16 w granicach kroku −50 % (błąd 6e-5). Użytkownik wybrał 16 bitów.
8. Format `q16` (`ruch_zbioru.kwantyzuj/odkwantyzuj/zapisz_klatke/konwertuj_q16`, `Timeline.read_raw/quant`), domyślny
   dla nowych pomiarów; shadery `ruch_gpu.py` w wariancie f32/q16 (`stan()`, `unpackUnorm2x16`, bufor QP); przycisk
   „Zapisz kopię 16-bitową”.
9. Uzupełniono na GPU pomiary `20261009T094824_natural_41cd2ac1904b` (8 -> 101, 275 s) i
   `20261009T081936_natural_07e68c7cdb6f` (1 -> 101, 384 s); kopie `_q16` trzech pełnych pomiarów.

## Results

### Key Findings
| Pomiar | Przed | Po |
|---|---|---|
| Przestrzeń makro, CPU na klatkę przelotu | 2,67 ms | 0,27 ms |
| Przestrzeń makro, zmiana kroku (CPU) | 7,3 ms | 0,74 ms |
| Widok 3D, klatka filmu | 95 ms | 15 ms |
| Ruch zbioru, odtwarzanie | 21–22 kl./s | 45–56 kl./s |
| Uczenie ViT, krok z przerysowaniem | 823 ms | 55 ms |
| Start do „Gotowe” | 8,3 s | 2,9–3,5 s |
| Pomiar naturalny | ok. 64 obr./s | ok. 500 obr./s |
| Pełny pomiar na dysku / w VRAM | 1,45 GB / 1480 MB | 0,73 GB / 752 MB (q16) |

- Przestrzeń makro: w spoczynku bitowo równo z CPU, w przelocie błąd ≤ 4e-6; obrazy różnią się w < 0,005 % pikseli.
- Pomiar naturalny GPU: klatki bitowo równe starej ścieżce CPU, metryki różnica ≤ 6e-14.
- Uzupełnienie przerwanego pomiaru: bitowo równe pomiarowi bez przerwy (f32 i q16).
- Szybkość pomiaru prawie nie zależy od paczki (16/32/64): ograniczeniem jest sam model.

### Issues Encountered
- Python 3.11 (`F:\Python\python.exe`): `OpenGL_accelerate` niezgodny z numpy 2 — aplikacja i testy GL na 3.13
  (zapis w pamięci projektu `aplikacja-python-313`).
- `QOpenGLWidget.repaint()` nie rysuje synchronicznie — testy wymuszają klatkę `grabFramebuffer()`.
- Pozostawione `GL_UNPACK_ALIGNMENT=1` psuło napisy QPainter — przywracanie stanu GL po rysowaniu.
- Bez wspólnych kontekstów w pytest programy pyqtgraph były nieważne (GL 1281) — `conftest.py`.
- `ZstdDecompressor` współdzielony między wątkami powodował segfault w badaniu formatu — osobny na wywołanie.
- Długie heredoki w Bash Git przerywały się — łatki zapisywane plikami w scratchpadzie.

## Files Generated/Modified
**`SmallViTAblation/app`:** nowe `makro_gpu.py`, `czasteczki3d_gpu.py`, `ruch_gpu.py`, `conftest.py`,
`sprawdz_dymki.py`, `test_makro_gpu.py`, `test_czasteczki3d_gpu.py`, `test_ruch_gpu.py`; zmienione
`przestrzen_makro.py`, `czasteczki3d.py`, `widok_gl.py`, `app.py`, `start.py`, `ustawienia.py`, `wygoda.py`,
`zakladka_uczenia.py`, `zakladka_filmu.py`, `zakladka_detekcji.py`, `zakladka_korekty.py`, `zakladka_pchniecia.py`,
`panel_rownania.py`, `okno_zrodel_sily.py`, `README.md`; z sesji Codexa (przepisane/rozszerzone) `zakladka_ruchu.py`,
`ruch_zbioru.py`, `ruch_naturalny.py`, `test_ruch_zbioru.py`, `test_ruch_naturalny.py`, `test_zakladka_ruchu.py`;
usunięty `ruch_gl.py`.

**Eksperyment (zastąpiony):** `experiments/vit_k1000_taniec_czasteczek_gpu_20261009/`,
`outputs/vit_k1000_taniec_czasteczek_gpu_20261009/`.

**Dane:** `outputs/ruch_czasteczek_gui/20261009T094824_natural_41cd2ac1904b` i `..._07e68c7cdb6f` (uzupełnione),
kopie `*_q16` trzech pełnych pomiarów.

**Pamięć projektu:** `wizualizacje-natywne-gpu.md`, `aplikacja-python-313.md`.

## Commands Used
```
F:\Python\3.13\python.exe -B -m pytest -p no:cacheprovider -q test_*.py
F:\Python\3.13\python.exe sprawdz_dymki.py --podsumowanie
F:\Python\3.13\python.exe test_makro_gpu.py
F:\Python\3.13\python.exe start.py
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

## Next Steps
- [ ] Zdecydować o usunięciu oryginałów float32 trzech pomiarów (kopie `_q16` zostały, około 4,3 GB do odzyskania);
      trzy pomiary to ten sam wynik (różnią się tylko paczką)
- [ ] Ewentualnie zapis q16 także dla przebiegów optymalizacji (`Engine` w `ruch_zbioru.py` pisze dalej float32)
- [ ] Ustawienia zakładek Uczenie, Pchnięcie i Korekta nie są zapisywane (brak w `MODULY` w `ustawienia.py`)
- [ ] Odtwarzanie w zakładce Film nie było mierzone ani optymalizowane
