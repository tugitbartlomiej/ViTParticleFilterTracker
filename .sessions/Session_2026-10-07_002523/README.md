# Sesja: analiza treningu SD z pełnym latentem ViT K8000

- Zapis: 2026-10-07 00:25:23 +02:00 (Europe/Warsaw).
- Typ: analiza i dokumentacja.
- Status: porównanie **zakończone**, job1823544 COMPLETED0:0. Wybrano A_control best10500, pobrano i sprawdzono model lokalnie, zakończono sprzątanie nowej rodziny. Brak zgodnego sztucznego celuK8000 opisano jako ograniczenie osobnego testu.

Przeanalizowano trening generatora dyfuzyjnego uczonego od losowych wag na Edenie, warunkowanego pełnym naturalnym latentem zamrożonego ViT K8000. VAE uzyskał dobre wyniki rekonstrukcji, a generator reaguje na warunek, ale nie osiągnął wymaganej zgodności latentu i wierności narzędzia. Najlepszy checkpoint pochodzi z kroku 7500; trening zakończył się plateau w kroku 19000.

Pełny opis wyników, ograniczeń i rekomendacji: [SESSION_SUMMARY.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/.sessions/Session_2026-10-07_002523/SESSION_SUMMARY.md).

Powiązane materiały:
- [Wcześniejsza sesja: latent, SD i Langevin](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/.sessions/2026-10-05-latent-vit-sd-langevin.md).
- [Lokalny test najlepszego modelu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/TEST_LOKALNY.json).
- [Historia walidacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_od_zera_k8000_filmy_20261006/eden_kopia/trening/EVAL_HISTORY.jsonl).

Pierwotna analiza nie zmieniała eksperymentu. Późniejsze polecenie „No rób to” uruchomiło osobne wdrożenie `vit_sd_k8000_ulepszenia_20261007`; poprzedni zakończony bieg pozostaje niezmieniony. Nowy kod porównuje cztery warianty: kontrolny, mieszanie filmów, mniejszy learning rate z EMA oraz dodatkową stratę obszaru narzędzia. Szczegóły i aktualizacja stanu są na końcu SESSION_SUMMARY.md.

Końcowy raport nowego porównania: [RAPORT_KONCOWY.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007/RAPORT_KONCOWY.md). Najlepszy względny MSE0,519988, spadek16,63% względem modelu bazowego; proponowane warianty B/C/D nie pokonały kontroli. Lokalne generacje nadal mają artefakty i niewierne narzędzie. Zwolniono17,82GB naEdenie.

Monitoring tego zakończonego porównania został usunięty. Pozostał lokalny podgląd obrazów w RAM.

8 października 2026 przeniesiono wybrany model A_control K10500 do [assets/SD_ViT_K8000_A_control_K10500](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/SD_ViT_K8000_A_control_K10500). [Karta modelu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/SD_ViT_K8000_A_control_K10500/README.md) opisuje architekturę, pochodzenie, metryki i ograniczenia. Wagi i SHA256 pozostają identyczne z lokalnie przetestowaną kopią.
