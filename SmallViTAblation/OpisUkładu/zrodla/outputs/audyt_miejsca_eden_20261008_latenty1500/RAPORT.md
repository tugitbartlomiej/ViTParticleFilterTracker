# Audyt miejsca Eden - 8 października 2026, tylko odczyt

Na wybranym systemie plików pozostało 21.030 GB. Bez naruszania modeli, źródeł, pełnych stanów AdamW i istniejących pomiarów można wskazać 0.620 GB odtwarzalnego cache/bytecode i zweryfikowanych pakietów deploy jako kandydatów o niskim ryzyku. Dalsze 2.870 GB to warunkowe smoke i trzy zweryfikowane lokalne kopie historycznego oracle, wymagające decyzji retencji i wykluczenia zależności. Nie ma podstaw do obiecania dziesiątek GB bez archiwizacji ważnych wyników lub zmiany retencji.

Nic nie usunięto, nie zatrzymano ani nie zmieniono na Edenie. Nie zgłoszono jobów. Zapis lokalny ograniczono do tego nowego katalogu raportu. SSH używał BatchMode i StrictHostKeyChecking=yes. Python zdalny działał z wejścia standardowego, bez plików roboczych. Odczyt może aktualizować atime zgodnie z ustawieniami systemu plików; audyt nie wykonywał jawnych operacji zapisu.

## Miejsce, quota i zakres

- `smallvit_runs` ma 86.932 GB przydzielonych bloków wg jednego `du -x` bez podążania za symlinkami.
- Realpath: `/home2/faculty/bpiotrowski/smallvit_runs` = `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs`. Nie są to dwie kopie.
- `df -B1`: pojemność 214748364800 B, wolne 21029957632 B. To wynik systemu plików dla tej ścieżki, nie gwarancja rezerwacji dla eksperymentu.
- `quota` nie jest zainstalowane. `lfs quota -u bpiotrowski /mnt/evafs`: użycie 189178136 KiB, limity bloków i plików 0 (brak ustawionego limitu w tym odczycie), 116077 plików. Quota nie pokazuje dodatkowej dostępnej przestrzeni. Ogranicza obecny `df`, z uwzględnieniem równoległych zmian.
- Własny `.cache` zajmuje 1.034 GB, z czego pip 0.504 GB, HF 0.436 GB, Torch 0.094 GB. Modele HF/Torch zachowano.
- Metadane odczytywano wyłącznie w swoich `smallvit_runs` i wskazanych cache. Katalogi obrazów, wejść, filmów, etykiet i datasets były pomijane podczas inwentaryzacji plikowej. Agregat `du` w swoim katalogu nie analizował zawartości źródeł. Pełna lista pominięć jest w `inventory.json`. Nie badano cudzych katalogów, innych mountów, scratch węzłów ani całych datasets.
- Odczyt własnego home ograniczono do jego pierwszego poziomu. Nie znaleziono własnego `~/tmp`; środowiska `venvs`, `.conda`, `smallvit_data` i `grant_tft` nie były skanowane ani kwalifikowane. Audyt nie obejmuje całego użycia wykazanego przez quota.
- Kopie lokalne sprawdzano w artefaktach `assets`, `outputs`, `experiments`, `results` i archiwum `Eden/EdenBackup/=/smallvit_runs`, pomijając źródła danych oraz symlinki. Brak trafienia oznacza brak potwierdzenia w tym zakresie, nie dowód nieistnienia kopii gdzie indziej.

## Zadania i recovery

`squeue -u bpiotrowski` na początku i końcu audytu było puste. Końcowy ponowny odczyt kolejki, df i quota: `final_state.txt`, 2026-10-08T21:08:16+02:00. `sacct` obejmuje 2026-09-14 do chwili odczytu; surowe dowody w `system_jobs.json`, rozpisanie w `jobs.json`. Nie utożsamiam COMPLETED z naukowo udanym ani zbędnym biegiem.

- K1000/1500: brak nowego biegu w odczytanej kolejce i brak odpowiadającego katalogu w `smallvit_runs` w chwili inwentaryzacji. To nie wyklucza zgłoszenia przez drugiego agenta po odczycie.
- `2026-10-08_vit_k8000_podmiana_1z256`: joby 1824697 i agregator 1824702 COMPLETED; FINAL wskazuje complete/500. STATUS zachował training/500, więc sam STATUS nie jest dowodem aktywnego joba.
- `2026-10-08_vit_k8000_podmiana_8z256`: 1824755 COMPLETED, zachować wyniki i checkpointy.
- `2026-10-06_vit_sd_pelny_latent_filmy`: 1823172 COMPLETED, STATUS complete 20000/20000 po wcześniejszych FAILED. Zachować best i ostatni pełny stan.
- `2026-10-06_vit_sd_od_zera_k8000_filmy`: 1823302 FAILED, późniejszy 1823430 COMPLETED; FINAL opisuje plateau i quality=false przy 19000. `latest_checkpoint.pt` zawiera optimizer, exp_avg, exp_avg_sq i RNG; nie usuwać jako rzekomo starego nieudanego zadania.
- `2026-10-02_nndmd_1500_expandedstate_multigpu`: 1819574 CANCELLED. Zachować latest, final_refit, inner_train, snapshot, cache potrzebny wznowieniu, kod i logi. Brak aktywnego joba nie znosi potrzeby recovery.
- `2026-10-07_maly_vit_sily_czastek`: 1823539 COMPLETED, dopasowanie 1823541 COMPLETED. Pełne Z i siły są źródłem bieżącego DMDc, nie zbędnym temp.

## Ochrona i sprawdzenie kopii

Zachować filmy, ZIP-y filmów, źródłowe obrazy i etykiety, `tracking_static36.tar`, `context_train.tar`, wszystkie K1000/K8000, SD best/base/EMA i provenance. Wszystkie pełne stany AdamW produkcji i historii są poza rekomendacją usuwania.

Bezpiecznie sprawdzono nagłówki ZIP i instrukcje pickle w wybranych checkpointach, bez deserializacji lub wykonania kodu: `details.json`. `ostatni.pt` z 30 września (13.754 GB), pełne `pelny/k1000.pt` i `latest_checkpoint.pt` mają momenty optymalizatora. Smoke również mogą mieć optimizer; oddzielenie ich od produkcji wymaga sprawdzenia celu testu.

Aktualne SHA-256 obu stron sprawdzono dla 16 wybranych zdalnych plików; zgodne kopie znaleziono dla 12. Cztery kandydatury zbieżne rozmiarem mają inne SHA (B_mixed, KONWERTER smoke_v2, checkpoint 19750 i 20000), więc nie są duplikatami sprawdzonych lokalnych modeli. Stabilność rozmiaru/mtime/inode kontrolowano przed i po zdalnym hashowaniu. SHA lokalny pochodzi z odczytu w tym audycie; przed późniejszym usunięciem ponownie sprawdzić istnienie i niezmienność kopii lokalnej. Szczegóły, lokalne ścieżki i SHA są w `verified_copies.json`. Dopasowanie nazwy lub rozmiaru bez SHA nie jest uznawane za kopię. Nie uruchamiano ponownie inferencji ani treningu.

Chronione główne modele mają aktualnie potwierdzone SHA: K1000 `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`, K8000 s47 `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`, SD A_control `1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41`. Lokalne PROVENANCE.json wskazują zgodne ścieżki źródeł; zachowano modele mimo potwierdzonych kopii.

## Kandydaci i rekomendacje

1. **Wysoka pewność, niskie ryzyko:** własny cache pip, odtwarzalny `__pycache__/*.pyc` z zachowaniem kodu i dwa zweryfikowane pakiety deploy. Potencjał razem 619900928 B (0.620 GB). Dla `deploy_bundle.tar.gz` sprawdzono 791 regularnych członów, dla `code_bundle_vit8z256.tar.gz` 7; wszystkie SHA zgodne z pozostawionym rozpakowanym drzewem na Edenie, nie z kopią lokalną. W przyszłej operacji najpierw powtórzyć kontrolę kolejki i ścieżek; usunięcie cache oznacza ewentualne ponowne pobranie/kompilację.
2. **Średnia pewność, warunkowo:** dokładne checkpointy `wyniki/smoke`, `wyniki/smoke_v2` oraz `GPU_SMOKE_*` wskazane w CSV, a także trzy `2026-10-04_vit_optymalizacja_latentu128/results/r*/TARGETS_ORACLE.npz` (łącznie około 1.306 GB, identyczne kopie lokalne SHA). Zachować logi, konfiguracje, raporty i dowód testu. Przed kwalifikacją wykluczyć zdalnych konsumentów i potrzebę danego stanu AdamW oraz potwierdzić kopię lub odtwarzalność. Potencjał dodatkowy 2870177792 B (2.870 GB), nie jest częścią potwierdzonego niskiego ryzyka.
3. **Wysoka pewność, zachować:** 17.427 GB pełnych tablic `Z`, `S_nowa`, `S_ped`, `S_zan`, `S_buf`, `S_grupy` z pomiaru sił. Starszy lokalny WYNIKI.md podaje tylko metadane i wyniki dopasowania, a aktualny audyt kopii pokazuje także nowsze trafienia (szczegóły i SHA w CSV/verified_copies.json). Bieżący `experiments/vit_dmdc_paczki_20261008/PLAN.md` odwołuje się do tych sił i ich zgodnego Z. Nawet zgodne kopie potrzebnych tablic nie są automatycznie eligible.
4. **Wysoka pewność, zachować:** 13.754 GB `2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ostatni.pt` i SD/EMA. To pełny stan, nie duplikat checkpointa EMA. Zysk duży, ale ryzyko utraty wznowienia i zasobów SD niedopuszczalne w tym audycie.
5. **Dalsze istotne miejsce wymaga archiwizacji:** kompletna kopia poza Edenem wraz z SHA, kodem, manifestem i optimizerem, a następnie osobna decyzja retencji. Ten audyt nie wykonał kopiowania dużych plików ani usuwania. Nie rekomenduję użycia chronionych źródeł jako budżetu nowego biegu.

## Budżet pełnych trajektorii

`1500 * 600 * 768 * 4 = 2764800000 B = 2.7648 GB = 2.574920654 GiB` na jeden snapshot FP32 całej bazy. Dla 1000 obrazów: 1.8432 GB. Liczba 1000 w K1000 oznacza checkpoint treningu; dokładna baza eksperymentu ma 1500 obrazów.

| Snapshoty 1500 FP32 | Same latenty GB | Ocena |
| --- | --- | --- |
| 1 | 2.7648 | mieści się obecnie |
| 7 | 19.3536 | teoretycznie mieści się, mały zapas |
| 10 | 27.648 | za mało obecnego miejsca |
| 21 | 58.0608 | wymaga znacznego dodatkowego miejsca |
| 101 | 279.2448 | przekracza całą pojemność tego systemu plików |
| 1001 | 2767.5648 | około 2.77 TB, niemożliwe tutaj |

Obecne wolne miejsce wystarcza na 7 pełnych snapshotów bez pozostałych artefaktów. Po samych kandydatach niskiego ryzyka: 7; po wszystkich warunkowych: 8. Zapas operacyjny 5 GB oraz dodatkowe `before/after/delta`, checkpointy, optymalizatory i atomowy zapis obniżą te liczby. Potwierdzone mało ryzykowne porządki nie rozwiązują problemu długiej trajektorii.

Lokalny PLAN oracle zapisuje `before/after/delta` FP32 i historię FP16 co 100 kroków oraz końcowy krok. To odmienny budżet niż pełna trajektoria FP32. Trzy tablice FP32 dla całej bazy to dodatkowe 8.2944 GB; trzy zestawy stanu latentu/AdamW, jeżeli równocześnie zapisane, również mnożą budżet. Nie zakładać kompresji bez pomiaru. Dobór częstotliwości/formatu lub innego miejsca jest osobną decyzją, nie został zmieniony przez audyt.

## Jak liczone są bajty

Rozmiar `sizeGB` to długość logiczna dzielona przez 10^9. `allocated_bytes` i `bytes_reclaimable` bazują na `st_blocks * 512`, a nie na sumie długości plików. Inode liczone po parze device/inode, jednokrotnie. Gdy część hardlinków jest chroniona lub poza zakresem, odzysk jest 0. Dla wszystkich nazw kwalifikowanego inode blok przypisano tylko raz w tabeli. Identyczne bajty na różnych inode zajmują oddzielne bloki, ale nie są automatycznie eligible.

Wykryto 84 grup inode z hardlinkami (348 nazw). `hardlinks.json` pokazuje wszystkie zaobserwowane nazwy i nlink. Linki katalogów `results_view3`, `results_k4_view` i snapshoty HF nie zostały dereferencjonowane; usunięcie symlinka nie odzyskuje bloków celu. `du` przypisuje współdzielone inode pierwszemu odwiedzonemu katalogowi, więc rozmiary pojedynczych rodzin zależą od kolejności i nie są niezależnymi prognozami odzysku.

Nie badano otwartych deskryptorów na węzłach ani serwerowych kopii/snapshotów; nawet policzone bloki mogą zostać zwolnione z opóźnieniem. To analiza kandydatów, nie autoryzacja kasowania. `eligible=TAK_KANDYDAT` oznacza rekomendację do osobnej operacji, `WARUNKOWO` oznacza brak wystarczających dowodów teraz, `NIE` ochronę.

## Zestawienie klas (nie sumować ponownie z tabelą plików)

| category | allocatedGB | eligible | bytes_reclaimable |
| --- | --- | --- | --- |
| archiwum-zrodlowe | 6.856 | NIE | 0 |
| bytecode | 0.038 | TAK_KANDYDAT | 37765120 |
| cache-modeli | 0.529 | NIE | 0 |
| cache-pip | 0.5 | TAK_KANDYDAT | 500383744 |
| deploy-zweryfikowany | 0.082 | TAK_KANDYDAT | 81752064 |
| historia-adamw | 7.433 | NIE | 0 |
| kod-logi-provenance | 2.596 | NIE | 0 |
| latenty-wyniki | 2.763 | NIE | 0 |
| modele-recovery | 42.07 | NIE | 0 |
| oracle-kopia-archiwalna | 1.306 | WARUNKOWO | 1306398720 |
| sily-latenty | 17.856 | NIE | 0 |
| smoke-checkpoint | 1.564 | WARUNKOWO | 1563779072 |

## Większe pliki i pakiety

Pełna wymagana tabela wszystkich 15929 plików: `TABELA_PLIKOW.csv` (kolumny path, sizeGB, statusjob, localkopia, reason, risk, eligible, bytes_reclaimable oraz dodatkowe pola dowodowe). Poniższa tabela jest jej wyborem, nie dodatkowym budżetem. Nie sumować z klasami lub rozmiarami katalogów.

| path | sizeGB | statusjob | localkopia | reason | risk | eligible | bytes_reclaimable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/context_train.tar | 5.27556608 | 1809382 COMPLETED; historia: CANCELLED by 1299,COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Może zawierać unikalne obrazy, filmy, etykiety lub provenance; input/context tar nie jest automatycznie zbędnym deploy. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie/filmy_zip/TR_0003_S1_P03.zip | 0.870138011 | 1818356 COMPLETED; historia: COMPLETED,FAILED | brak potwierdzonej kopii w sprawdzonym zakresie | Może zawierać unikalne obrazy, filmy, etykiety lub provenance; input/context tar nie jest automatycznie zbędnym deploy. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/modele/sd15/unet/diffusion_pytorch_model.fp16.safetensors | 1.719125304 | 1818496 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ema_010000/diffusion_pytorch_model.safetensors | 1.719142584 | 1818496 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ema_015000/diffusion_pytorch_model.safetensors | 1.719142584 | 1818496 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ema_020000/diffusion_pytorch_model.safetensors | 1.719142584 | 1818496 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ostatni.pt | 13.753652069 | 1818496 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/proba/dane/LATENT_GESTY.npy | 0.464486528 | 1818732 COMPLETED; historia: CANCELLED by 1299,COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Gęsta historia i pełne stany AdamW potrzebne badaniu dynamiki; nie kwalifikować po wieku ani zakończeniu joba. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/runtime_nccl232/nvidia/nccl/lib/libnccl.so.2 | 0.473614536 | 1819918 COMPLETED; historia: COMPLETED,FAILED | brak potwierdzonej kopii w sprawdzonym zakresie | Zachować kod, konfiguracje, wyniki, logi i provenance; niska korzyść lub nieustalona rola. | średni | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-03_maly_vit_gesta_paczka/wyniki/pelny_p128/dane/LATENT_GESTY.npy | 2.214051968 | 1820334 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_gesta_historia_paczka\wyniki\pelny_p128\dane\LATENT_GESTY.npy | Gęsta historia i pełne stany AdamW potrzebne badaniu dynamiki; nie kwalifikować po wieku ani zakończeniu joba. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-03_maly_vit_gesta_paczka/wyniki/pelny_p320/dane/LATENT_GESTY.npy | 2.214051968 | 1820334 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_gesta_historia_paczka\wyniki\pelny_p320\dane\LATENT_GESTY.npy | Gęsta historia i pełne stany AdamW potrzebne badaniu dynamiki; nie kwalifikować po wieku ani zakończeniu joba. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128/results/r0.25/TARGETS_ORACLE.npz | 0.428023934 | 1822387 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_optymalizacja_latentu128\eden\r0.25\TARGETS_ORACLE.npz | Pełna lokalna kopia NPZ potwierdzona SHA256; historyczne zadania COMPLETED. Można rozważyć pozostawienie tylko kopii lokalnej po wykluczeniu zależności zdalnych, z zachowaniem metryk/provenance i bez zmiany jedynej kopii. | średni | WARUNKOWO | 428032000 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128/results/r0.5/TARGETS_ORACLE.npz | 0.436007417 | 1822387 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_optymalizacja_latentu128\eden\r0.5\TARGETS_ORACLE.npz | Pełna lokalna kopia NPZ potwierdzona SHA256; historyczne zadania COMPLETED. Można rozważyć pozostawienie tylko kopii lokalnej po wykluczeniu zależności zdalnych, z zachowaniem metryk/provenance i bez zmiany jedynej kopii. | średni | WARUNKOWO | 436015104 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128/results/r1.0/TARGETS_ORACLE.npz | 0.442346139 | 1822387 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_optymalizacja_latentu128\eden\r1.0\TARGETS_ORACLE.npz | Pełna lokalna kopia NPZ potwierdzona SHA256; historyczne zadania COMPLETED. Można rozważyć pozostawienie tylko kopii lokalnej po wykluczeniu zależności zdalnych, z zachowaniem metryk/provenance i bez zmiany jedynej kopii. | średni | WARUNKOWO | 442351616 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy/wyniki/trening/best_sd.pt | 0.546367845 | 1823302 FAILED, 1823430 COMPLETED; FINAL plateau, stan recovery chroniony | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_sd_od_zera_k8000_filmy_20261006\eden_kopia\trening\best_sd.pt | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy/wyniki/trening/best_vae.pt | 0.546367448 | 1823302 FAILED, 1823430 COMPLETED; FINAL plateau, stan recovery chroniony | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy/wyniki/trening/latest_checkpoint.pt | 1.471421639 | 1823302 FAILED, 1823430 COMPLETED; FINAL plateau, stan recovery chroniony | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wyniki/smoke/checkpoint_00000001.pt | 0.542330751 | 1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000 | brak potwierdzonej kopii w sprawdzonym zakresie | Zakończony test techniczny; kandydat po potwierdzeniu zbędności tego stanu, kopii i zachowaniu kodu, metryk oraz logów. Ma momenty AdamW lub stan modelu. | średni | WARUNKOWO | 542334976 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wyniki/smoke/checkpoint_00000002.pt | 0.542330751 | 1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000 | brak potwierdzonej kopii w sprawdzonym zakresie | Zakończony test techniczny; kandydat po potwierdzeniu zbędności tego stanu, kopii i zachowaniu kodu, metryk oraz logów. Ma momenty AdamW lub stan modelu. | średni | WARUNKOWO | 542334976 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wyniki/trening/best_checkpoint.pt | 0.542331007 | 1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000 | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\assets\StableDiffusionModel\best_checkpoint.pt; F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_sd_pelny_latent_filmy_eden_20261006\best_checkpoint.pt | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wyniki/trening/checkpoint_00019750.pt | 0.542331007 | 1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000 | sprawdzone kandydatury lokalne mają inne SHA256; NIE są zgodną kopią (verified_copies.json) | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wyniki/trening/checkpoint_00020000.pt | 0.542331007 | 1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000 | sprawdzone kandydatury lokalne mają inne SHA256; NIE są zgodną kopią (verified_copies.json) | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/S_buf.npy | 2.764800128 | 1823541 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/S_grupy.npy | 3.594240128 | 1823541 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/S_nowa.npy | 2.764800128 | 1823541 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_dmdc_paczki_20261008\source_eden\S_nowa.npy | Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/S_ped.npy | 2.764800128 | 1823541 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/S_zan.npy | 2.764800128 | 1823541 COMPLETED; historia: COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/Z.npy | 2.767564928 | 1823541 COMPLETED; historia: COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_dmdc_paczki_20261008\source_eden\Z.npy | Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/wyniki/porownanie/A_control/best_sd.pt | 0.546432766 | 1823544 COMPLETED | SHA256 zgodny: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\assets\SD_ViT_K8000_A_control_K10500\best_sd.pt | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/wyniki/porownanie/B_mixed/best_sd.pt | 0.546432766 | 1823544 COMPLETED | sprawdzone kandydatury lokalne mają inne SHA256; NIE są zgodną kopią (verified_copies.json) | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/wyniki/porownanie/C_optim/best_sd.pt | 0.54638355 | 1823544 COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/wyniki/porownanie/D_roi/best_sd.pt | 0.546383614 | 1823544 COMPLETED | brak potwierdzonej kopii w sprawdzonym zakresie | Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie. | wysoki | NIE | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256/deploy_bundle.tar.gz | 0.081727638 | 1824702 COMPLETED; historia: COMPLETED | lokalna niepotwierdzona; pełna rozpakowana kopia na Edenie, każdy człon SHA zgodny (deploy_verified.json) | Każdy regularny człon archiwum ma identyczny SHA256 w pozostawionym rozpakowanym drzewie; zweryfikowano stabilność i właściciela. Usunięcie samego archiwum pozostawia kod i wejścia. | niski | TAK_KANDYDAT | 81735680 |
| /home2/faculty/bpiotrowski/smallvit_runs/code_bundle_vit8z256.tar.gz | 1.5372e-05 | brak aktywnych; dokładne powiązanie joba nieustalone | lokalna niepotwierdzona; pełna rozpakowana kopia na Edenie, każdy człon SHA zgodny (deploy_verified.json) | Każdy regularny człon archiwum ma identyczny SHA256 w pozostawionym rozpakowanym drzewie; zweryfikowano stabilność i właściciela. Usunięcie samego archiwum pozostawia kod i wejścia. | niski | TAK_KANDYDAT | 16384 |

## Własne rodziny: agregat du

| path | sizeGB | statusjob | eligible | bytes_reclaimable |
| --- | --- | --- | --- | --- |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy | 21.315 | 1818496 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek | 17.885 | 1823541 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie | 9.023 | 1818736_7 COMPLETED; historia: CANCELLED by 1299,COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci | 5.276 | 1809382 COMPLETED; historia: CANCELLED by 1299,COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-03_maly_vit_gesta_paczka | 5.165 | 1820334 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie | 3.383 | 1818356 COMPLETED; historia: COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera | 2.811 | 1817935_3 COMPLETED; historia: COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy | 2.725 | 1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000 | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy | 2.577 | 1823302 FAILED, 1823430 COMPLETED; FINAL plateau, stan recovery chroniony | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia | 2.274 | 1818732 COMPLETED; historia: CANCELLED by 1299,COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000 | 2.263 | 1823465_6 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia | 2.198 | 1823544 COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36 | 2.109 | brak aktywnych; dokładne powiązanie joba nieustalone | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_maly_vit_sila_it2 | 1.458 | 1818051_19 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128 | 1.413 | 1822387 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper | 1.266 | 1819918 COMPLETED; historia: COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-14_tracking_nano | 1.236 | 1807634 COMPLETED; historia: COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_konwerter_dino_filmy | 0.999 | 1823170/1823174 COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu | 0.438 | 1819574 CANCELLED by 1299; historia: CANCELLED by 1299 | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256 | 0.285 | 1824702 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_wektor | 0.134 | 1817949_9 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_cech_adaptacyjny | 0.119 | brak aktywnych; dokładne powiązanie joba nieustalone | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_2gpu_model_smoke | 0.105 | 1819570 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-23_fastgan | 0.09 | 1809936 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_sila | 0.09 | 1818025_8 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-14_nano_smoke | 0.082 | 1807587 COMPLETED; historia: COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla | 0.058 | brak aktywnych; dokładne powiązanie joba nieustalone | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden | 0.056 | 1819567 COMPLETED; historia: COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_gesta_historia | 0.05 | brak aktywnych; dokładne powiązanie joba nieustalone | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_8z256 | 0.024 | 1824755 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-21_koopman_langevin | 0.02 | 1809878 COMPLETED; historia: CANCELLED by 0,CANCELLED by 1299,COMPLETED,FAILED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-25_kontynuacja_dobor | 0.007 | 1814611_1 COMPLETED; historia: CANCELLED by 1299,COMPLETED | NIE: nie usuwać całej rodziny | 0 |
| /home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_rownanie_latentu | 0.0 | 1818331_8 COMPLETED; historia: COMPLETED | NIE: nie usuwać całej rodziny | 0 |

## Dowody i ograniczenia

`system_jobs.json`, `jobs.json`, `sizes_final.txt`, `inventory.json`, `details.json`, `local_candidates.json`, `remote_hashes.json`, `verified_copies.json`, `deploy_verified.json`, `hardlinks.json`, `classification.json`, `SUMMARY.json`. Błędy SSH odróżniono od banera w plikach stderr. Brak błędów inwentaryzacji: []. Zachowano kod zbierania i generowania raportu wyłącznie w tym katalogu. Odczyt ma charakter punktowy; przed późniejszą decyzją konieczna aktualizacja bieżącego stanu, bo inne zadania i agenci mogą działać równolegle.
