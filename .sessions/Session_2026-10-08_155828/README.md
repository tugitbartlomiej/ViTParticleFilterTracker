# Session: Session_2026-10-08_155828

**Created:** 2026-10-08 15:58:28 Europe/Warsaw
**Type:** Training / Benchmark / SSH
**Status:** Completed

## Quick Summary

Sprawdzono, jak podmiana jednego obrazu w puli 256 obrazów wpływa na dalsze 500 aktualizacji detektora SmallViT K8000. Porównano kontrolę z ośmioma unikalnymi naturalnymi obrazami wyszukanymi dla 11 obrazów interpolacji latentu m0011.

## Key Results

- Sama kontynuacja treningu odpowiadała za większość poprawy: best IoU sondy wzrosło z 0.3980 do 0.8559, a guard AP50 z 0.6645 do 0.7192.
- Najlepszy kompromis dał obraz `TR_0007_S1_P03/1165`: efekt ponad kontrolę wyniósł +0.0081 IoU i +0.0077 guard AP50.
- `TR_0007_S1_P03/1397` wyraźnie zaszkodził sondzie: efekt confidence -0.0791 i IoU -0.0312.
- To pojedynczy seed; małe efekty wymagają replikacji.
- Dodatkowy test z wszystkimi 8 podmianami jednocześnie podniósł IoU sondy o 0.91 punktu ponad kontrolę, lecz obniżył confidence o 2.77 punktu i guard AP50 o 0.52 punktu.

## Files

- `SESSION_SUMMARY.md` - pełny opis metody, wyników, zadań Eden i artefaktów.
- `SmallViTAblation/outputs/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008/results/ANALIZA.md` - raport wynikowy.
- `SmallViTAblation/outputs/vit_k8000_podmiana_8z256_dino_m0011_eden_20261008/results/ANALIZA.md` - wspólny test ośmiu podmian.
