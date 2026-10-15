# Session: Okienko 11 — pole w komórce tokenu, podkroki przez 6 bloków, zapis przejść

**Created:** 2026-10-15 04:37
**Type:** Mixed (aplikacja GUI, analiza)
**Status:** Paused

## Quick Summary
Kontynuacja sesji z 2026-10-14 (pułapka optyczna jako ognisko wiązek). Okienko 11 aplikacji dostało: reflektory w skali sfery
(film 12), obrys ogniska 3D, zapis przejść przez sieć na dysk, pole tylko w kuli komórki tokenu (połowa odległości do sąsiada),
kadr ze środkiem w koloidzie albo w pułapce i stałą skalą, zieloną strzałkę ruchu w nadchodzącym kroku, płynące strzałki pola,
stały widok w podkrokach z zapalonymi reflektorami w uwadze i MLP, przyciemnione tło pozostałych cząsteczek, podgląd ogniskowej
dopiero po kliknięciu. Napisany (nieprzetestowany) moduł pola w podkrokach wszystkich 6 bloków (`app/pole_podkrokow.py`).

## Key Results
- Cząsteczka 226: komórka tokenu 0,003–0,355 r99 (mediana 0,15 r99); 0 innych tokenów w sześcianach komórek w 806 checkpointach;
  zero pola w komórce w 0 z 806 checkpointów (liniowe oszacowanie pułapki 9–270 r_kom dalej).
- Ruch koloidu między checkpointami (uczenie) jest prawie prostopadły do skoku wiązek bloku 6 (głąb sieci): kąt mediana 85°,
  kwartyle 55–117°, ruch ≈ 0,5 długości skoku.
- Obrys ogniska (połowa jasności sumy wiązek gaussowskich): blok 4 wydłużenie ×2,1, bloki 1 i 6 ×1,2 (krok 65640).
- Zapis przejść: 44 MB/ckpt float32 (22 MB float16), wczytanie 0,07 s zamiast 0,1–0,7 s; 6 przejść zapisanych w cache.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
