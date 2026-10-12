# Session: Werdykt — tokeny ViT tworzą pełzający kryształ na siatce patchy

**Created:** 2026-10-11 21:02
**Type:** Analysis
**Status:** Completed

## Quick Summary
Sesja zamknęła badanie „ruch tokenów jak w szkle”. Na Edenie policzono testy kryształu (pozycja patchy, kontrole bez kodowania pozycji, CNN, wysokie lr, ziarna 48 i 49, fale na siatce, Lindemann, podprzestrzenie, podatność). Lokalnie wyrenderowano sześć filmów. Werdykt zapisano jako główny dokument `WERDYKT_KRYSZTAL_2026-10-11.md`.

## Key Results
- Układ jest krystaliczny w około dwóch trzecich: sąsiedztwo zgodne z siatką w 70% (bez kodowania pozycji 34%, losowo 1,3%), ruch zbiorowy 60–73%, adres z pozycji R² 0,44–0,56.
- Typ: dwuwymiarowy, sprężyście skorelowany kryształ z nieporządkiem podstawieniowym, pełzający pod wpływem uczenia (wykładnik γ_L ok. 0,27). Nie jest to szkło ani ciecz.
- Korekty: 0,60 to cos kierunku wiązania, a nie odsetek zachowanych wiązań. `wspolny_naped_wagi.py` ma niewyrównany czasowo przyrost wag.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- Główny dokument: `SmallViTAblation/outputs/porownanie_szklo_20261010/WERDYKT_KRYSZTAL_2026-10-11.md`
