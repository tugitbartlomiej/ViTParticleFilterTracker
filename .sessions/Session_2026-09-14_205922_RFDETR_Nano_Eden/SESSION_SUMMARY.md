# Podsumowanie sesji — 2026-09-14

## Cel i aktualna autoryzacja

Projekt: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`.

Celem jest naukowo poprawna ocena kuracji klatek chirurgicznego video przed treningiem detektora oraz poprawa wcześniejszego artykułu DETR/YOLO zgodnie z uwagami recenzentów. Pierwszy etap wykorzystuje mały, szybki RF-DETR Nano i CATARACT-LMM. Diagnostyka latentu przez Koopmana, Langevina i Mori–Zwanziga stanowi późniejszą, oddzielną warstwę badań.

Użytkownik najpierw polecił zatrzymać treningi po smoke testach i usunąć ich robocze pozostałości. Następnie wyraźnie autoryzował rozpoczęcie treningu: „Ok, to w takim razie możesz przystąpić do treningu”. To zwolniło poprzednie wstrzymanie dla obecnej kampanii. Nie anulować obecnego zadania na podstawie historycznych dokumentów o pauzie.

Ostatnie polecenie użytkownika dotyczy zapisu sesji w `.sessions`. Podczas samego zapisu nie uruchamiano nowych treningów ani nie sprawdzano ponownie SSH.

## Stan klastra — ostatnio zweryfikowany

Połączenie: `ssh eden-cluster`, użytkownik `bpiotrowski`.

SLURM: konto `transformers_vsc`, partycja `hopper`. Nie używać partycji `debug`; ograniczenia projektu znajdują się w `Eden/EDEN_CONSTRAINTS.md`.

Aktualne wdrożenie: `~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation`.

- Zadanie **1807599**: docelowy smoke test RF-DETR Nano na CATARACT-LMM, **COMPLETED**, kod `0:0`, czas `00:02:57`. 64 klatki treningowe, 64 walidacyjne, 5 epok, seed 42, jeden H100. Raport potwierdził trening oraz niezależną walidację wag ostatniej epoki i najlepszego modelu.
- Zadanie **1807600**, `nano_calibration`: ostatnio **RUNNING**. Jeden węzeł Hopper, **3 H100**, 24 CPU, 128 GiB RAM, limit 24 godzin, bez wyłączności na cały węzeł. Trzy niezależne procesy, każdy na jednym GPU; nie jest to DDP.
- Seedy treningowe: **42, 1337, 2024**. Wszystkie trzy procesy ukończyły pierwszą epokę i rozpoczęły drugą podczas ostatniej weryfikacji. Kolejka robocza: 3 running, 0 pending, 0 done, 0 failed; heartbeaty były aktualne.
- Zadanie zgłoszono i rozpoczęto 14 września około **20:46:56 Europe/Warsaw**. Ostatnie potwierdzenie postępu pochodzi z około 20:49–20:50. Nie jest to gwarancja stanu w chwili kolejnego otwarcia sesji.
- Nie ma jeszcze końcowego raportu kalibracji ani dowodu poprawy wynikającej z kuracji danych.
- `Requeue=0` ustawiono dla aktywnego zadania przez `scontrol`; skrypt dla kolejnych zgłoszeń ma `--no-requeue`. Nie utworzono automatycznego łańcucha kolejnych kampanii ani automatyzacji monitorującej.

## Co dokładnie trenujemy teraz

Obecne trzy treningi to **kalibracja przepisu treningowego**, a nie pełne porównanie selektorów.

- RF-DETR Nano, 384 px, BF16, batch 16, gradient accumulation 1.
- 30 epok, AdamW, learning rate `1e-4`, backbone learning rate `1e-5`, weight decay `1e-4`, warmup 2 epoki.
- Bez early stopping i bez augmentacji w tym prowizorycznym przepisie kalibracyjnym.
- **Ten sam deterministyczny podzbiór 500 klatek** wybranych czasowo dla każdego z trzech seedów. Zmieniamy seed treningu, nie losujemy trzech różnych selekcji danych.
- Walidacja na wszystkich **12043 dostępnych klatkach z dostarczonym bbox** w zamrożonym podziale walidacyjnym. Test nie został zmaterializowany ani użyty do oceny modelu.
- Każdy trening startuje niezależnie z tych samych wag COCO. SHA-256 wag: `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`.

Dane stagingowane są raz do `/dev/shm/bpiotrowski_1807600_ablation/ablation_data`. Skrypt sprząta staging po zakończeniu zadania. **Nie usuwać go podczas aktywnego treningu.**

W trakcie treningu metryki są w `checkpoints_raw/metrics.csv`, a nie dopiero w końcowym `training_logs/metrics.csv`. Zwykła walidacja RF-DETR zapisuje AP; nie zakładać, że automatycznie obliczana jest krzywa validation loss.

Katalogi wyników na Edenie:

```text
~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/results/calibration/tracking_uniform_b500_s42_calibration/
~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/results/calibration/tracking_uniform_b500_s1337_calibration/
~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/results/calibration/tracking_uniform_b500_s2024_calibration/
```

Na normalnym zakończeniu wrapper przygotowuje m.in. `last.pth`, `best.pth`, `run_manifest`, `epoch_val_map.json`, `training_performance.json`, `eval_last_epoch/`, `eval_best_val/`, `validation_report.json`. Ocena obu checkpointów odbywa się niezależnie na całej walidacji. Wagi końcowe nie gwarantują wznowienia z pełnym stanem optymalizatora. Wrapper odmawia mieszania nowego przebiegu z istniejącymi wynikami; po awarii należy świadomie obsłużyć istniejący przebieg lub użyć nowego identyfikatora.

## Dane i ograniczenia interpretacji

Źródło: `~/Cataract-LMM/3_Object_Tracking/annotations`, 170 archiwów ZIP z JPEG i `annotation.csv`. Lokalny audyt: `Dataset/Cataract-LMM/_audyt_pacjentow.json`.

Użyto 167 konserwatywnych grup operacji, **nie potwierdzonych identyfikatorów pacjentów**. Element nazwy `P03` oznacza fazę, nie pacjenta. Podział z seedem 42: 117 grup train, 25 val, 25 test, proporcje 0.7/0.15/0.15. Dokładne duplikaty TR70/TR138 i TR88/TR144: wyłączono TR138 i TR144. Podobne, przesunięte TR45/TR49 umieszczono w jednej grupie.

Pula kandydatów to klatki `1, 6, 11, ...`, czyli stride 5, a nie wszystkie klatki surowego video. Bbox pochodzi z kolumny `cystotome_bbox`, kategoria `instrument`; nie jest to sztucznie skonstruowany bbox wokół końcówki.

Puste lub nieznane bbox **nie zostały uznane za negatywne przykłady**. Obecna walidacja jest warunkowa: dostępne klatki z dostarczoną adnotacją obiektu. Nie wolno z tego wyciągać wniosku o false alarms na całym video. Semantyka brakujących adnotacji wymaga wyjaśnienia przed publikacją.

Audyt wykrył 738 brakujących obrazów w ZIP (297 train, 441 val); usunięto je z kandydatów przed selekcją i zapisano raport. Po uwzględnieniu dostępności oraz polityki dostarczonych bbox: **57073 train**, **12043 val**, **15795 test**. Test nie jest częścią aktualnego pakietu treningowego.

Eksport kalibracji zawiera 500 train + 12043 val, poprawne rozmiary obrazów i COCO annotations. Sprawdzono brak identycznych zdekodowanych obrazów między wyeksportowanym train i val. To nie dowodzi rozdzielności pacjentów ani braku duplikatów względem niezmaterializowanego testu.

Pula indeksu: `index/pool_manifest.jsonl`; SHA-256 indeksu: `c78f4b40b6db30023ff1c330a43e26ffd7d2f034d2a628f6dbd375d7dd997857`. Towarzyszą jej `split_manifest.json`, `index_summary.json`, `missing_images_audit.json` i log indeksowania.

Selektor kalibracyjny rozdziela kwoty według pierwiastka z pojemności grup i wybiera środki przedziałów czasowych. Nie należy automatycznie utożsamiać go z innymi implementacjami `uniform` w projekcie.

## Środowiska i sprzątanie

Aktywny interpreter treningu:

```text
~/venvs/rfdetr_nano_clean_1_10_1/bin/python
```

Python 3.11.15, RF-DETR 1.10.1, torch 2.9.1+cu128, torchvision 0.24.1, Lightning 2.6.6, transformers 5.17.0, numpy 2.3.5, pycocotools 2.0.11.

`~/venvs/rfdetr_python311` jest wymaganym prefiksem bazowego interpretera i bibliotek standardowych. **Zachować go.** Środowisko `tsjepa_py311` zostało całkowicie usunięte zgodnie z autoryzacją użytkownika, po przeniesieniu zależności Nano i potwierdzeniu niezależności importów. `pip check`, zgodność `pip freeze` oraz późniejszy smoke na H100 potwierdziły działanie po migracji.

Stary katalog smoke `~/smallvit_runs/2026-09-14_nano_smoke` usunięto po archiwizacji małych logów. Po pomyślnym nowym smoke usunięto także jego robocze checkpointy, kopie obrazów i nieudane katalogi przygotowania. Raporty zachowano m.in. w `~/smallvit_runs/2026-09-14_tracking_nano/audit_smoke/`. Dane oryginalne i bieżąca kalibracja zostały zachowane.

## Kod i dokumentacja

Najważniejsze pliki w `SmallViTAblation`:

- `tools/prepare_tracking_calibration.py`: audyt dostępności ZIP, zamrożony podział, eksport COCO, weryfikacja rozmiarów i pikseli, deterministyczne 500 train, pełna val, osobny wariant smoke-only i SHA archiwum.
- `pipeline/train/run_with_validation.py`: trening Nano i niezależna ocena ostatnich oraz najlepszych wag, sprawdzenie liczby ocenionych obrazów i raport kalibracji.
- `pipeline/train/smoke_rfdetr.py`: jawne wskazanie walidacyjnych annotations.
- `eden/run_nano_calibration.slurm`: jeden Hopper, trzy GPU, trzy procesy i współdzielony staging.
- `eden/submit_nano_calibration.sh`: bramka pomyślnego rzeczywistego smoke, trzy zadania robocze z seedami, zamrożone źródła i konfiguracja, brak automatycznego dalszego łańcucha.
- `configs/protocol_tracking_calibration.yaml`: rzeczywisty podział tracking i przepis obecnej kalibracji.
- `docs/Analiza_ponownej_publikacji/33_START_TRENINGOW_CATARACT_LMM_NANO.md`: kanoniczny opis aktualnej kampanii, danych, zasobów, wyników smoke i ograniczeń.
- `docs/Analiza_ponownej_publikacji/evidence_tracking_training_2026-09-14/`: manifesty, SHA, raport smoke, snapshot rozpoczętych treningów, audyt brakujących obrazów i sprzątania.

Dokument 32 opisuje historyczne wstrzymanie i audyt środowiska; aktualna autoryzacja oraz uruchomiony trening są opisane w dokumencie 33.

Istotna wcześniejsza poprawka: RF-DETR po treningu może podmienić w pamięci wagi na EMA. Ostatnie zwykłe wagi trzeba odzyskiwać z Lightning `last.ckpt`, z kontrolą epoki i prefiksów parametrów; nie utożsamiać ich z modelem znajdującym się w pamięci po `train()`.

## Git

Gałąź: `codex/eden-nano-training-20260914`.

HEAD podczas zapisu: **2de59ad1**. Ostatnie commity:

```text
2de59ad1 docs(eden): record running three-seed calibration and clean successful smoke
ef2d5348 docs(eden): freeze tracking split, calibration recipe and successful target smoke
1aaacf76 feat(data): prepare separate tracking smoke alongside full calibration validation
5ec12af9 feat(eden): gate three-seed calibration submission on successful tracking smoke
c3329adb fix(data): audit missing ZIP images before freezing calibration subsets
4fd46460 feat(eden): prepare audited tracking calibration and three-GPU Nano jobs
```

Źródła aktywnej kalibracji są zamrożone na **ef2d5348**; późniejszy commit dokumentacyjny nie zmienia już działających procesów. Smoke używał źródeł **1aaacf7633c49c561df2f50afc55a79f12181d7b**. Nie podmieniać źródeł aktywnego przebiegu pod pozorem aktualizacji dokumentacji.

Przed zapisaniem tej notatki brak zmian w śledzonych plikach. Nieśledzone, niezwiązane elementy zachowano: `AdvancedDatasetSelection/PLAN_v3_RF-DETR.md` oraz `Dataset/`. Nie wykonano push.

## Kolejne kroki badań

1. Sprawdzić zakończenie 1807600, raporty trzech seedów, faktyczną liczbę epok, mAP i czas/koszt treningu. Wyniki kalibracji służą ustaleniu przepisu, nie potwierdzają przewagi selektora.
2. Przygotować właściwą macierz **4 selektory × 3 budżety × 3 seedy = 36 treningów na jeden zbiór**: random, próbkowanie czasowe, coverage cech i pełny pipeline autora; budżety 500/1500/2500. Pełna macierz jeszcze nie wystartowała. Istniejący profil z trzema metodami daje 27, nie 36 przebiegów.
3. Przed kampanią zamrozić definicje składników i cache cech. Nie nazywać odtworzeniem starego artykułu wariantu z inną wersją DINO, wymiarami S/F czy inną definicją E. Confidence error nie jest automatycznie kanonicznym EL2N.
4. Ablacje D/E/S/F: priorytetowo leave-one-out, następnie wymagane przez recenzentów wszystkie 15 niepustych kombinacji. Przy trzech seedach i jednym budżecie daje to 45 przebiegów; pełny wariant można współdzielić tylko przy identycznych warunkach.
5. Dodać full-pool reference i mocny selektor literaturowy uwzględniający obiekty lub korelację czasową. Sama kombinacja DINO/Fourier/SAM/clustering nie dowodzi nowości.
6. Równe K wymaga identycznego przepisu treningowego. Równa liczba epok przy różnych K nie oznacza równego kosztu: potrzebne porównanie przy stałej liczbie kroków oraz raport kosztu selekcji i treningu. Nie zakładać, że większa pula zawsze pogarsza model albo zawsze daje lepszy wynik.
7. Raportować wyniki seedów i niepewność na poziomie grup operacji, bez traktowania sąsiednich klatek jako niezależnych obserwacji. Wybór konfiguracji i checkpointów tylko na val, końcowy test po zamrożeniu metody.
8. Stary zbiór wymaga odzyskania nieprzeselekcjonowanych oryginałów: `SelectedImages.tar` z 20 tys. obrazów nie jest bezstronną pełną pulą. Nie mieszać tracking i segmentation jako rzekomo niezależnych zbiorów bez audytu wspólnych operacji.
9. Po wykazaniu efektu na Nano sprawdzić transfer na większe detektory przy tych samych podzbiorach. Koopman/Langevin/Mori–Zwanzig i aktywna kuracja pozostają osobnym etapem: zacząć od zamrożonego latentu i pokazać, że diagnostyka przewiduje użyteczne interwencje, zamiast uznawać resztę równania za bezpośredni dowód brakujących informacji.

Użytkownik chce zachować poprawkę artykułu jako główny cel. Rozszerzenie dynamiczne nie powinno blokować pierwszych porównań selekcji.

## Wznowienie sesji

Najpierw przeczytać tę notatkę i dokument 33, potem sprawdzić aktualny stan zamiast zakładać, że snapshot nadal obowiązuje:

```powershell
git status --short
ssh -o BatchMode=yes eden-cluster 'squeue -j 1807600'
ssh -o BatchMode=yes eden-cluster 'sacct -j 1807599,1807600 --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES'
```

Sprawdzić logi SLURM, kolejkę roboczą oraz `checkpoints_raw/metrics.csv` dla trzech katalogów wynikowych; jeśli zakończone, odczytać końcowe `validation_report.json`. Nie uruchamiać ponownie tych samych identyfikatorów przebiegów bez sprawdzenia istniejących wyników.

## Pozostałe ustalenia użytkownika

- Wprowadzać commity po poprawkach kodu i dokumentacji.
- Nie wykonywać własnego OCR; korzystać z dostarczonego programu/skryptu MinerU. Gotowe OCR artykułów jest w `F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja\Ocr\raw`.
- Linki do artykułów i dostępnych PDF zapisywać w `F:\Studia\Articles\Zbiory\2026-09-11_RT-DETR_selekcja_ewaluacja\linki`.
- Analizy ponownej publikacji użytkownik przeniósł do `SmallViTAblation/docs/Analiza_ponownej_publikacji`.
- Projekt grantowy `F:\Studia\MCHTR\Grant\Projekt` jest kontekstem dla późniejszego rozszerzenia dynamiki latentu, nie dowodem skuteczności kuracji dla obrazów chirurgicznych.

Ta notatka jest podsumowaniem umożliwiającym wznowienie, nie pełnym stenogramem rozmowy.
