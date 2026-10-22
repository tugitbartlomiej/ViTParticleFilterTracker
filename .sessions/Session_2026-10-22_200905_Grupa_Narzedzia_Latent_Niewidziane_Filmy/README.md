# Session: Grupa narzędzia w latencie, filmy niewidziane i 24 seedy

**Created:** 2026-10-22 20:09
**Type:** Analysis
**Status:** Completed

## Quick Summary
Etykieta narzędzia z obrysu (udział pikseli w łacie) zamiast ramki GT i zero przestrzenne (ta sama maska przesunięta w tło) w R9 i w eksperymentach. Metoda wykrywania grupy tokenów narzędzia bez etykiet (TokenCut na aktywacjach MLP bloku 6), dobrana na filmach DEV i sprawdzona na TEST. Ocena na 25 filmach TEST, których trening nie widział, i replikacja na 24 seedach.

## Key Results
- Filmy TEST, seed 3047, ep. 400: Dice grupy bez etykiet 0,40 [0,34–0,45] (0,47 na filmach treningu), precyzja 0,55, 87 % tokenów w narzędziu lub ≤ 1 łatę; rozdział 192D i LDA ponad zero przestrzenne w 90 % klatek; IoU detektora 0,65 (0,86).
- Kolejność w uczeniu (seed 47): rozdział średnich 192D od kroku ~100 (57 %), grupa wykrywalna bez etykiet między 1 000 a 10 000, potem stała do ep. 400.
- 24 seedy, k65640: Dice 0,376 (0,329–0,426), 192D 88 % (84–93 %).

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
