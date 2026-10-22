# Session: LaserLatentSpaceApp — wydajność GPU, kilka zdjęć, szybsza pułapka, okienka ustawień

**Created:** 2026-10-22 17:26
**Type:** Mixed (aplikacja GUI: wydajność, nowe funkcje, testy)
**Status:** Completed

## Quick Summary
Ciąg dalszy sesji `Session_2026-10-22_162024` (wydzielenie aplikacji „Lasery i koloidy”). Aplikacja liczy i rysuje tylko
widoczne okienka. Start przywraca ostatni układ, jest menu Plik, a sfera pokazuje kilka zdjęć naraz (domyślnie limit 5,
najwyżej 32). Pułapka optyczna liczy się szybciej dzięki analitycznemu jakobianowi i może liczyć kilka cząsteczek naraz.
Okienka ustawień z prawego kliku są osobnymi oknami ze zwijanymi grupami. Naprawiono nachodzenie okienek F2/F3. Powstały
testy GUI (38) i dymków (14).

## Key Results
- Odtwarzanie w układzie domyślnym: 16–18 → 50–55 kl./s; `ustaw_czas` 75 ms → ~1 ms (liczone i rysowane tylko widoczne okienka).
- Pułapka: jakobian analityczny zamiast autogradu, 100 → 68 ms na checkpoint (siatka 17); 3 cząsteczki razem 114 ms
  (38 ms na cząsteczkę). Wynik identyczny ze starą wersją (różnice < 1e-3), jakobian zgodny z autogradem do 1e-10.
- F2/F3: pasek podkroków wymagał 1146 px szerokości i rozpychał okno; teraz przewija się (`okna.Przewijane`).
- Kilka zdjęć na sferze: 3 zdjęcia = 1800 koloidów, Z 2,1 GB na karcie, ~32 kl./s.
- Testy: 147 w pełnym przebiegu i 38 GUI osobno na ekranie, wszystkie przechodzą.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- `tools/` - skrypty pomiarów i prób GUI użyte w sesji
