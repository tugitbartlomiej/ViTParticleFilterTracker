# Session: szklo1_2D — PDF o ViT, filmy 13–16 (GPU), atencja 6 bloków vs przesunięcie centrów

**Created:** 2026-10-12 17:05
**Type:** Mixed (dokumentacja, animacje, analiza)
**Status:** Completed

## Quick Summary
Powstał 11-stronicowy PDF do nauki standardowego ViT od obrazu do treningu, z porównaniem z MalyViT. Film 2 („lasery_2_ruch”) rozbudowano o mapę atencji, centra na zdjęciu oka, IoU i pewność. Renderowanie przeniesiono na GPU (moderngl + NVENC), a film skrócono do 54 s. Atencję 6 bloków i rollout policzono dla 806 checkpointów i zbadano jej związek z przesunięciem centrów studni.

## Key Results
- PDF `SmallViTAblation/docs/vit_od_poczatku_do_konca/vit_od_poczatku_do_konca.pdf`: liczba parametrów ViT-B/16 = 86 567 656, zweryfikowana kodem.
- Film 16 `lasery_16_iou_pewnosc_osobno.mp4`, 54 s, render GPU w ~30 s: mozaika atencji 6 bloków, osobne wykresy IoU i pewności, bez słupków zmian.
- Związek atencji z odpływem centrów jest słaby i idzie przez narzędzie. Bloki 4–6 mają ρ ≈ +0,15…0,3 (blok 6 w epokach 1–5: 0,32), poza narzędziem ≈ 0. Średnia bloków i rollout ≈ 0. Zmiany z checkpointu na checkpoint nie są skorelowane, a różnice per patch mieszczą się w szumie permutacji.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- `analysis/ATENCJA_CENTRA.json`, `analysis/atencja_centra.png` - kopia wyników analizy
