# Dostępne mocniej wytrenowane modele ViT — 6 października 2026

## Ten sam mały ViT co w eksperymentach latentowych

Na Edenie istnieją checkpointy po 8000 aktualizacjach w `smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s*/.../wagi_koncowe.pth`. Wśród sprawdzonych zapisów końcowych najwyższy historyczny mAP50 ma `petla43b_s47/losowe`: 0.6112320513 przy kroku 8000. Wynik pochodzi z OCENY.json; nie wykonano nowej walidacji i nie należy porównywać go bezpośrednio z innym protokołem RF-DETR.

Źródło: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s47/losowe/wagi_koncowe.pth`.

Pobrana kopia została przeniesiona na polecenie użytkownika do `assets/MalyViT_K8000_s47_losowe/MALY_VIT_K8000_s47_losowe.pth`. Opis i metadane: README.md oraz PROVENANCE.json w tym samym folderze. SHA256 lokalny i zdalny: `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`. Wczytanie strict do MalyViT z experiments/vit_latent128_jedno_zdjecie_90/model_snapshot.py zakończyło się zgodnością wszystkich kluczy. To ta sama architektura, lecz inne wagi i reprezentacja niż checkpoint K1000 użyty do treningu adaptera SD. Nie podmieniono modelu w żadnym eksperymencie.

Nie mylić tego z MODEL_best.pt po 6402 krokach z vit_wyuczenie_latentu_jedno_zdjecie_do85: tam uczenie było dopasowaniem jednego obrazu do sztucznego latentu.

## RF-DETR Nano z backbone'em ViT

Lokalna kolekcja `results/eden_static36_2026-09-15` zawiera 72 checkpointy best/last, 36 biegów po 30 epok. Najwyższy zapisany wynik best_val mAP w tej kolekcji:

`tracking_dino_temporal_hybrid_b2500_s42/best.pth`: AP50 0.9120451286, mAP@[0.5:0.95] 0.6991022979, 12043 obrazy walidacyjne. Trening używał podzbioru 2500 obrazów. Wyniki są zapisane w validation_report.json, zakres treningu w run_manifest.json. Nie przeprowadzono nowego testu.

Ta rodzina nie jest zamiennikiem MalyViT: przykładowa warstwa MLP ma fc1 [1536,384], podczas gdy obecny latent po GELU ma 768 współrzędnych na token. Potrzebny byłby osobny adapter i odpowiednie miejsce pomiaru aktywacji.

## Znaleziona epoka 21

Lokalnie istnieją trzy pliki epoch_021.pth dla seedów 42, 1337 i 2024 w:

`F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/EdenBackup/=/smallvit_runs/2026-09-21_koopman_langevin/results_telemetria/telemetria_uniform_time_b1500_s42/epoch_checkpoints/epoch_021.pth`

Sprawdzono zawartość pliku s42: model_name RFDETRNano, epoch=21, global_step=2068, checkpoint_role=periodic_regular_weights_only. Historyczny regularny mAP@[0.5:0.95] dla epoki 21 wynosi 0.6558498144 według epoch_val_map.json. Nie jest to ten sam mały ViT użyty do SD. Pod odpowiadającą ścieżką na Edenie plik obecnie nie istnieje; dostępna jest lokalna kopia archiwalna.

## Zakres audytu

Przeszukano lokalne drzewa projektu VIT, w tym ignorowane pliki i archiwum EdenBackup, oraz katalog smallvit_runs na Edenie. Odróżniono modele dynamiki, SD, DINO pretraining, YOLO i klasyczny DETR od rozpatrywanych detektorów z ViT. Kolekcje z innymi architekturami i protokołami nie tworzą jednego rankingu jakości.

LOKALNE.json: metadane 72 checkpointów RF-DETR. EDEN.json: ścieżki i historyczne oceny znalezionych końcowych MalyViT. Skrypty audytu: tools/audyt_checkpointow_vit_20261006.py oraz tools/audyt_eden_vit_20261006.py.
