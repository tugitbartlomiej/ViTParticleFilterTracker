# Session: Pułapka bloku MLP — matematyka, Jakobian, filmy 16–19 (szklo3_3D, część 2)

**Created:** 2026-10-12 17:08:47
**Type:** Analysis (wizualizacje + literatura + pomiar)
**Status:** Completed

## Quick Summary
Po zapisie sesji 16:06 (filmy 8–12) ugruntowano matematycznie obraz „reflektory–sfera–wiązki–cząsteczka” (research Undermind, dokument LaTeX ze słownikiem i wzorami), zmierzono „pułapkę” jako Jakobian pola sił bloku MLP 6 i zrobiono cztery filmy: 16 (siodło z wirem), 17 (ognisko + 599 prawdziwych tokenów), 18 (dwa widoki, dokładny Jakobian) i 19 (dwa widoki zsynchronizowane krokiem uczenia). Wykryto, że filmy 8–16 używały wag z innego biegu niż zapisane aktywacje.

## Key Results
- Właściwe wagi do aktywacji: `assets/MalyViT_35000_od_zera_s47/wagi` (806 checkpointów, R² latentu = 1,0); `01_bieg_35000_gesty_zapis/.../wagi` to inny bieg (R² spada do −0,03).
- Pułapka (dokładny J tokenu narzędzia 316, blok 6): siodło (≈51 % kierunków cofa) z wirem prawie jak w macierzy losowej (‖A‖/‖S‖ 0,995 → 0,958, symetryczność 50 → 52 %), ślad/192 ≈ −0,007, siła pola ×3,0. Film 16 (złe wagi) przeszacował kontrakcję i wzrost (×14).
- W 60. epoce blok odpycha od tokenu 316 wszystkie tokeny narzędzia (mediana ×1,098) mocniej niż tło (×1,020), po LayerNorm.
- Wypadkowa 768 sił w ognisku to 6–17 % sumy ich długości (siły się znoszą — „przeciwlasery”).
- Najlepsze filmy: 12 (reflektory) i 19 (pułapka w dwóch zsynchronizowanych widokach).

## Files
- `SESSION_SUMMARY.md` - pełna dokumentacja
