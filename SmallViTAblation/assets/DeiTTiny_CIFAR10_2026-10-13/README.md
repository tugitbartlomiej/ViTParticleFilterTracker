# DeiT-Tiny / MalyViT na CIFAR-10 (Eden, bieg 2026-10-13), pobrane 2026-10-16

Pochodzenie: Eden `~/smallvit_runs/2026-10-13_deit_cifar10/` (zadania Slurm 1827646 s47_obciecie, 1827647 s47_bez_obciecia,
1827690/1827779 maly_s47_obciecie, 1827696/1827780 maly_s48_obciecie, 1827697/1827781 maly_s47_wd0, 1827698/1827782 maly_s47_losowe_etykiety),
kod: `kod/` (kopia z Edena) i `experiments/deit_cifar10_20261013/`. Ziarna 47 i 48, przepis jak s47 (AdamW 5e-4, wd 0,05, paczka 32, 60 epok = 93 780 kroków).
Opis eksperymentu: `experiments/deit_cifar10_20261013/README.md`. Struktura jak na Edenie: `wyniki/<bieg>/{dane,gesty,pelny,wagi}`.

Pobrano (SHA-256 z Edena zgodne dla KAZDEGO pliku, patrz `SHA256SUMS_EDEN_2026-10-16_*.txt`):
- trzy biegi glowne `s47_obciecie`, `s47_bez_obciecia`, `maly_s47_obciecie`: dane, gesty (pca32_*, pelne768_*), pelny (stany AdamW), wagi ZREDUKOWANE
  (kroki k<=1000 co 10/100 oraz co 2000: 148 plikow na bieg zamiast ~1088);
- kontrole `maly_s48_obciecie`, `maly_s47_wd0`, `maly_s47_losowe_etykiety`: dane, gesty, pelny, wagi tylko k<=100 (11 plikow);
- `kod/`, `logi/`, `dane/` (CIFAR-10 uint8: zgodne z `assets/Dane_CIFAR10`).
NIE pobrano (zostaje na Edenie): pozostale wagi k*.pt (≈940 plikow na bieg glowny, ≈265 na kontrole), `wyniki/proba`.
