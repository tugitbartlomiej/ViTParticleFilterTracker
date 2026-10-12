# Stan wykonania

## Zakończone: produkcja i synteza 2026-10-09

Joby1825171/1825172/1825187 COMPLETED0:0. Collector51580 zakończył transfer wszystkich trzech folderów; ponownie zweryfikowano1539 plików. Obie migawki frozen: wszystkie18 hashów nadal zgodne ze zgłoszeniem. Bez rerunów i bez zmiany estymatorów.

Raport końcowy: [WNIOSKI_WZORCE.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/WNIOSKI_WZORCE.md). Dane opisowe, kontrole pokrycia i bootstrap po filmach: [SYNTEZA_DIAGNOSTYKA.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json). Kontrola raportu: [VALIDATION_SYNTHEZY.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/VALIDATION_SYNTHEZY.json).

Wspólny ruch istnieje; DMD prędkości pokonuje last-increment o3,61/5,65% RMSE768 h5 w fit/min_norm. Scalar DMDc poprawia fit, pogarsza min_norm i targety; ogółem ratio do persistence0,99223,95% bootstrap[0,97830;1,01213]. PCA32 zatrzymuje tylko31–33% energii ruchu targetów. Nie wykonano aplikacji korektora ani oceny poprawy IoU/mAP.

## Historia przygotowania

- 1825145 COMPLETED: PyDMD DMDc + deeptime DMD/VAMP, stary panel128, 10 testów.
- 1825150 COMPLETED: panel128 po usunięciu algebraicznego nakładania cech VAMP, 11 testów; DMD/DMDc i ich wyniki bez zmian.
- Odbiornik docelowego panelu1024 gotowy; rzeczywiste pola producenta zgodne.
- Odbiornik całej populacji gotowy: PCA32, streaming assignment VAMP/kmeans wszystkich900k, pełne768 normy/kąty, train-only fit, autonomous deeptime DMD przyrostu co5. Testy 13 PASS na prywatnym środowisku.
- Docelowe joby zgłasza main jednym wywołaniem `submit_after_merge.py` po znanym ID scalenia Hubble. Ten wykonawca nie zgłasza równolegle konsumentów. Skrypty gotowe do dependency afterok; immutable copy tworzona przez wrapper po gate testów.
- Wynik test-fit learnedB RMSE1.6482962 wobec persistence0.02479245 zachowany. Nie retunowano testu. Nowy panel ma raportować OOD/rank/stabilność na własnym rozłącznym splicie filmów.
- Model wykonawcy: gpt-6-astra, effort xhigh, potwierdzone w metadanych sesji.
- Ostatnia poprawka operacyjna wdrożona na Edenie: manifest panelu jako argument `$1`, bez listy `--export`. Estymatory/hiperparametry niezmienione. Po wdrożeniu13testów PASS (7.19s), oba skrypty SLURM przechodzą kontrolę składni. Hashe lokalne/Eden zgodne; szczegóły `READY_FOR_MAIN.json`.
- Aktualizacja: main zgłosił konsumenty1825171(panel64)/1825172(population900k), oba afterok1825166.
- Osobne uzupełnienie zlecone później:1825187(population900k_dmdc_supplement), afterok1825166, zgłoszone02:10:31UTC. Nowy immutable snapshot, istniejący snapshot/marker bez zmian. Gate7testówPASS, w tym rzeczywiste PyDMD API i kontrola pełnego błędu768/liczności całej próbki. Szczegóły `SUPPLEMENT_DMDC_SUBMITTED.json` i `DODATEK_DMDC_POPULACJA.md`.
