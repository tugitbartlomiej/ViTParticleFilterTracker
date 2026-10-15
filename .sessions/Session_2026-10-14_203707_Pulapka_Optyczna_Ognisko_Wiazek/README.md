# Session: Pułapka optyczna jako ognisko wiązek, zasady układu, pomiary jak w pęsecie

**Created:** 2026-10-14 20:37
**Type:** Mixed (Analysis + aplikacja + literatura + SSH)
**Status:** Completed

## Quick Summary
Sesja zaczęła się od walidacji ViT-S/16 z Model Zoo na Imagenette i porządków na Edenie, a skończyła na nowym obrazie
„pułapki”: od 2026-10-14 pułapka = pułapka optyczna z laserów (ognisko 768 wiązek bloku wokół jednej cząsteczki), a dawny
„środek drgań” to „średnia skoków koloidu”. W aplikacji powstało okienko 11 (zbliżenie na ognisko jednej cząsteczki w osi uczenia
i w podkrokach, z kontrolą „suma wkładów = ruch” ~10⁻⁶), zasady układu przeniesiono do `app/` (Z1–Z29), napisano mosty ML↔fizyka,
plan 8 eksperymentów na Edenie (pomiary jak w pęsecie) i przegląd literatury o ogniskach.

## Key Results
- ViT-S/16 (Model Zoo, seed 42) na Imagenette: top-1/1000 76,3 %, top-1/10 98,8 % po 90. epoce; późne bloki skrajnie zimne (T 0,03–0,12).
- Ognisko 768 wiązek naprawdę przesuwa cząsteczkę: błąd ≤ 1,3·10⁻⁶ w 806 checkpointach (blok 6) i 3–8·10⁻⁷ w podkrokach MLP;
  w podkrokach uwagi przesuwają ją inne cząsteczki (600 wkładów, błąd 2·10⁻⁶).
- Zero pola wiązek (ognisko w sensie pułapki) istnieje w zasięgu 3 r99 tylko w 7,7 % checkpointów; zwykle cząsteczka jest niesiona
  prądem wiązek z siodłem i wirem wokół. Skok to 5–68 % sumy długości wkładów zależnie od bloku i etapu (reszta się znosi).
- Literatura (2 deep searche, 213 + 97 prac): nikt nie mierzy na jednym tokenie, jak wkłady neuronów sumują się i znoszą
  (sprawność ogniska, spójność, apertura) ani jak to zmienia się w uczeniu — luka publikacyjna, zwłaszcza dla ViT.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
