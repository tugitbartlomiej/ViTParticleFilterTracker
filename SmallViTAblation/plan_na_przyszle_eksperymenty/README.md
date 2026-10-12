# Plan na przyszłe eksperymenty

Folder na datowane plany przyszłych eksperymentów SmallViTAblation. Każdy plan ma datę, godzinę i opisowy tytuł. Zrealizowane eksperymenty mają własne katalogi w experiments/.

## Zapisane plany

- [2026-10-01 14:06:21 — poprawa równań Langevina i Koopmana](2026-10-01_14-06-21_poprawa_rownan_langevina_i_koopmana.md): prognoza zmiany siły tożsamości na 500 aktualizacji ze stanu oraz historii paczek i ich wpływu. Plan doprecyzowany po przeglądzie kodu, następnie uzupełniony lokalnym pilotem identyfikacji. Pilot nie potwierdził poprawy H=500; dodano etap bliższego zapisu i przyszłej podprzestrzeni tożsamości. Eksploracyjny H1 osiąga80,98procent znaków, ale nie80procent redukcjiMSE; bez nowego treningu.
- [2026-10-01 14:35:31 — wyszukiwanie pełnej sceny SD w pełnych klatkach](2026-10-01_14-35-31_wyszukiwanie_pelnej_sceny_sd_w_pelnych_klatkach.md): cechy całego obrazu, całość do całości, z układem sceny, narzędziem i tłem. DETR-E wskazane przez użytkownika do późniejszej weryfikacji; opcjonalne pełnoobrazowe DINO + SSIM. Crop retrieval jako kontrola; bez implementacji i nowych testów.

Ostatnia modyfikacja indeksu: 2026-10-01 15:00:30 (Europe/Warsaw).

- [2026-10-01 16:28:13 - ruch cząsteczek a uczenie detekcji na stałych klatkach](2026-10-01_16-28-13_ruch_czasteczek_a_uczenie_detekcji_na_stalych_klatkach.md): rozdzielenie zmiany sieci i zmiany obrazu, pełne tokeny i wyniki detekcji, kontrole tła, sekwencje i prognozy. Plan rozszerzenia; pomiary jeszcze niewykonane.

- [2026-10-01 16:46:21 - szukanie podprzestrzeni tożsamości narzędzia w latencie](2026-10-01_16-46-21_szukanie_podprzestrzeni_tozsamosci_narzedzia_w_latencie.md):400par narzędzieobecne/usunięte, pełne cząsteczki, podprzestrzeń odporna na kąty i test wpływu na detekcję. Rozwinięcie wcześniejszych planów; wyników jeszcze nie ma.

- [2026-10-02 12:49:00 — SD z czystego szumu i zgodność prognozowanego latentu768](2026-10-02_12-49-00_sd_z_czystego_szumu_prowadzenie_prognozowanym_latentem_768.md): start bez img2img, pełne768 per token z naciskiem J128, końcowe kryterium pokrycia do operacjonalizacji, budżet/rejection oraz ranking pełnych klatek niewykorzystanej puli train: obecny DINO pełnoobrazowy, cosine>0,85, adaptacyjnie 0–100 bez dopełniania; FAISS/precompute i dodatkowe cechy opcjonalne. Plan, bez nowych testów.

Aktualizacja indeksu: 2026-10-02T12:49:00+02:00 (Europe/Warsaw).

Doprecyzowanie kroku5 (baza cech i adaptacyjny ranking): 2026-10-02T12:53:58+02:00 (Europe/Warsaw).

Aktualna wersja kroku5: DINO/cosine i opcjonalne rozszerzenia, 2026-10-02T12:55:28+02:00 (Europe/Warsaw).

- [2026-10-02 12:59:31 — telemetria użytych klatek, uczenie i walidacja rund](2026-10-02_12-59-31_telemetria_uzytych_klatek_uczenie_i_walidacja_rund.md): dokładny frame-ID/film/hash, moment added vs actual sampled, batch/augment/RNG/checkpoint, old/new presentations oraz porównania pipeline/random/sequential/direct real retrieval. Plan, bez implementacji.

Aktualizacja indeksu: 2026-10-02T12:59:31+02:00 (Europe/Warsaw).

- [2026-10-02 13:02:43 — test zwykłego ViT i pipeline](2026-10-02_13-02-43_test_standardowego_vit_i_pipeline_doboru_klatek.md): A/B/F jako etap minimalny, D/E/C jako ablacje, dopasowane liczebności i aktualizacje, mAP learning-AUC/CI po seed-run oraz koszt end-to-end. 10 seedów zaplanowane, bez nowego treningu.

- [2026-10-02 13:06:44 — luki i poprawki do obecnego pipeline](2026-10-02_13-06-44_luki_i_poprawki_do_obecnego_pipeline.md): osobna notatka o artefaktach LaMa, układzie celu, prognozie, SD, różnorodności retrieval i uczciwym teście ViT; hipotezy/minimalne kontrole/opcjonalne rozszerzenia, bez nowego kroku 8 i bez implementacji.
