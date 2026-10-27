# Session: Mosty do fizyki MałegoViT — ocena 24 seedów, kompensacja, wzmocnienie przejściowe, balanced amplification

**Created:** 2026-10-27 01:42
**Type:** Mixed (Analysis + SSH Eden + Literature)
**Status:** Active (ostatnie zadanie Eden 1933161_4 w toku)

## Quick Summary
Sesja domknęła eksperyment `mosty_fizyki_24seedy_ablacje` (ocena predykcji, WNIOSEK, RAPORT, commit 50ceb2e1) i uruchomiła
trzy eksperymenty pochodne na Edenie: `kompensacja_lodyga_i_cifar` (kontrola frozen_both + CIFAR DeiT/MałyViT; commit efd2b6a8),
`wzmocnienie_przejsciowe_pseudospektrum` (Kreiss, abscysa numeryczna, najgorszy kierunek, odporność IoU) oraz
`balanced_amplification_schur_odczyt` (forma Schura, zawór, wyrównanie z odczytem, pochodzenie nienormalności, żywe reflektory;
wersja 2 z BA3c/BA3d w toku). Dwa wyszukiwania Undermind: nienormalność w fizyce/korze/fotonice i audyt nowości.
Rozmowy o ścieżce publikacji (TMLR ~75 %, ICML ~45 %, NeurIPS plan B).

## Key Results
- Teza A/B (nienormalność pola MLP rośnie w treningu 0,76 → 0,92) replikuje się na 24 seedach, w DeiT i MałymViT na CIFAR,
  także przy losowych etykietach; teza C odrzucona; kompensacja funkcjonalna potwierdzona (frozen_both AP 0,46–0,62).
- Nienormalność ma skutek dynamiczny: stała Kreissa i luka abscysy rosną 5/5; najgorszy kierunek wzmacniany 2,2–4,5×
  (zgodnie z σ_max co do 1 %); wyjście odporne (ΔIoU ≤ 0,013).
- Balanced amplification: łańcuch Schura niskiej rangi (c5 0,15 → 0,63–0,74; PR_N 62 → 3–8) i zawór (0,43 → 0,18–0,36)
  potwierdzone; część funkcjonalna odwrócona (v₁ wyrównane z odczytem 30–50× losowe, u₁ nie). Nienormalność siedzi w wagach
  W₂W₁ (0,73 → 0,85–0,90); MLP niemal wyłączony na tokenach narzędzia (GELU′ 0,01–0,04).
- Audyt nowości: brak w literaturze pomiaru jakobianu podwarstwy MLP; najbliżsi Li26ad (uwaga), Fer26d (cały blok), Aub24.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- `analysis/` - (puste; dane w folderach eksperymentów `SmallViTAblation/experiments/`)

## Transkrypt
`W:\Chaty\Claude\CodeCLI\config\projects\W--Studia-PhD-projekt-VIT-ViTParticleFilterTracker\7a3f5b79-c883-4cdf-9559-5f8815a97330.jsonl`
