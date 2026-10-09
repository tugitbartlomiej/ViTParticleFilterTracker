# Session: Animacje cząsteczek na GPU (OpenGL), szybki start, pomiar naturalny na CUDA i zapis 16-bitowy

**Created:** 2026-10-09 12:29:06
**Type:** Mixed (refaktor GUI + GPU, benchmark, obliczenia)
**Status:** Completed

## Quick Summary
Animacje aplikacji `SmallViTAblation/app` przeniesione na compute shadery OpenGL 4.3 (przestrzeń makrowektorów,
widok 3D nad obrazem, ruch zbioru 900 000 cząsteczek), z pracą rozdzieloną na trzech agentów (Opus, Opus, Sonnet)
i audytem końcowym. Aplikacja startuje pusta w około 3 s. Pomiar ruchu naturalnego liczy się na GPU (około 8 razy
szybciej), brakujące kroki przerwanych pomiarów są dokładane na GPU, a nowy zapis 16-bitowy zajmuje połowę miejsca.

## Key Results
- Przestrzeń makro na GPU: CPU klatki przelotu 2,67 -> 0,27 ms, zmiana kroku 7,3 -> 0,74 ms; w spoczynku bitowo równo
- Widok 3D: klatka filmu 95 -> 15 ms; ruch zbioru: 21–22 -> 45–56 kl./s; start 8,3 -> 2,9–3,5 s, bez autowczytywania
- Pomiar naturalny: ok. 64 -> 500 obrazów/s (101 × 1500 w ok. 5 min), klatki bitowo równe dawnej ścieżce CPU
- Uzupełniono dwa przerwane pomiary (8/101 i 1/101 -> 101/101, baza odtworzona bitowo)
- Zapis q16: 1,45 -> 0,73 GB na dysku i 1480 -> 752 MB VRAM; trzy kopie `_q16` (oryginały zostały)
- Testy: 120 zaliczonych, 1 pominięty; 1112 kontrolek, 0 bez dymka

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
