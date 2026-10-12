# Session: Lasery na sferze i źródło drgań cząsteczek

**Created:** 2026-10-12 04:29
**Type:** Analysis
**Status:** Completed

## Quick Summary
Kontynuacja badania „tokeny ViT jak kryształ”. Powstały dwa nowe filmy 3D renderowane na GPU (moderngl, OpenGL 4.3): 768 laserów na sferze (neurony fc1) świecących 600 wiązkami do cząsteczek i jedną do środka, oraz film o tym, co trzęsie cząsteczkami (generator = wagi θ, paczki, długość kroku jako temperatura). Dopisano wynik naped_v2 z Edena do dokumentu werdyktu.

## Key Results
- Film 4 `lasery_4_sfera.mp4` (79 s): ogniska przesuwają się średnio o 3,1 odstępu po 60 epokach (narzędzie 5,1); wiązki rzadsze i mocniejsze (natężeń > 0,1: 40% → 16%, max 2,3 → 5,1).
- Film 5 `lasery_5_drgania.mp4` (84 s): skok tokenów rośnie z długością kroku wag (Spearman 0,35, krok po kroku); rozgrzewka lr podgrzewa (klatka 0,62 → 0,98 odstępu); energia skoku: 48% wspólna dla zdjęcia, 36% wspólna z sąsiadami, 9% własna.
- naped_v2: przyrosty wag nie prognozują ruchu tokenów poza oknem (R² ≈ 0); werdykt bez zmian.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
