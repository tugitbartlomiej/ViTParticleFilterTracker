# Ablation Study - zbiorcze wyniki (lokalna ewaluacja)

Protokol: external = scalone 2333 obrazy (FULL) / minus 48 leaked (CLEAN); same-dist = video-level split, identyczny dla wszystkich wariantow. Progi TP/FP: conf=0.3, IoU=0.5; mAP z pycocotools.

| Wariant | Opis | Ext FULL mAP@0.5 | Ext CLEAN mAP@0.5 | Ext FULL mAP@[.5:.95] | Same-dist test mAP@0.5 | Same-dist val mAP@0.5 | d Ext FULL vs V1 | d SD test vs V1 |
|---|---|---|---|---|---|---|---|---|
| V1 | FULL pipeline (baseline, ep270) | 21.40% | 21.40% | 5.54% | - | - | +0.00pp | - |
| V2 | NO_FOURIER | 2.19% | 2.19% | 0.41% | - | - | -19.21pp | - |
| V3 | NO_EL2N | 1.51% | 1.51% | 0.37% | - | - | -19.89pp | - |
| V4 | QUALITY_ONLY | 1.54% | 1.54% | 0.34% | - | - | -19.86pp | - |
| V5 | RANDOM | 1.37% | 1.37% | 0.29% | - | - | -20.03pp | - |
| V6 | DIVERSITY_ONLY (DINO+K-Center) | 1.80% | 1.80% | 0.47% | - | - | -19.60pp | - |
| V7 | EL2N_ONLY (random+EL2N rerank) | 3.63% | 3.63% | 0.67% | - | - | -17.77pp | - |

Sanity-check: V1 Ext FULL powinno wynosic ~85.74% (headline artykulu, LEGACY benchmark 20260112). Istotna rozbieznosc = inny protokol, wyjasnic przed wpisaniem do tekstu.
