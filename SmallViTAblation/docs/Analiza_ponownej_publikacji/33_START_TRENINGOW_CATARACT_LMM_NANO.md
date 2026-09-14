# Start treningów RF-DETR Nano na CATARACT-LMM — 2026-09-14

Autor polecił rozpocząć trening. To nowe polecenie znosi pauzę opisaną w
dokumencie 32 dla tej kampanii. Nie oznacza wznowienia starych kolejek ani
uruchamiania eksperymentów Koopmana/Langevina.

## Kolejność rzeczywistych zadań

1. Smoke na docelowych adnotacjach: zadanie SLURM **1807599**, COMPLETED,
   exit 0, czas 00:02:57; raport `passed`. Jedna H100,
   8 CPU, 32 GiB, 5 epok, 64 train / 64 val, seed 42. Pełny trening,
   eksport last/best i niezależna ewaluacja obu. Techniczny smoke używa
   istniejącego profilu `nano_smoke`; rzeczywisty podział danych opisuje
   osobny manifest, nie historyczne liczby filmów w domyślnym protokole.
2. Po sukcesie smoke zgłoszono zadanie **1807600**, RUNNING na Hopperze:
   **trzy treningi kalibracyjne**, po 30 epok na tych
   samych 500 klatkach i seedach 42, 1337, 2024. Jedno zadanie SLURM rezerwuje
   3 H100, 24 CPU, 128 GiB, jeden węzeł Hopper, limit 24 godziny.
   Podział i recepturę opisuje `configs/protocol_tracking_calibration.yaml`.
   Preflight potwierdził trzy H100 PCIe i CUDA 12.8; wszystkie trzy procesy
   zakończyły pierwszą epokę (31 kroków) i rozpoczęły drugą. Kolejka miała
   running=3, pending=0, failed=0. Hash wag pretrained był identyczny w trzech
   biegach. Model może nadal trenować w chwili odczytu dokumentu;
   nie ma jeszcze końcowego wyniku kalibracji.
3. Po kontroli zbieżności, czasu i walidacji zamrażamy recepturę kampanii
   porównawczej. Kalibracja nie jest pełnym badaniem skuteczności kuracji.

Nie zgłaszamy automatycznego łańcucha `afterany`. Pomyślna kontrola smoke jest
warunkiem zgłoszenia kalibracji. Stara macierz D/E/S/F nie jest uruchamiana
przez ten skrypt. Zakończenie kalibracji opróżnia jej trzyzadaniową kolejkę.
Automatyczne ponowne zgłoszenie przez SLURM jest wyłączone (`Requeue=0`);
ta zmiana została zastosowana także do działającego zadania bez jego przerwania.

## Dane i zakres wniosków

- Podzbiór: `3_Object_Tracking`, wyłącznie ramki cystotomu. Nie tworzymy
  sztucznych ramek wokół punktów końcówki ani nie łączymy z segmentacją.
- Siatka kandydatów: klatki 1, 6, 11 itd. (stride 5). Wnioski o późniejszej
  selekcji dotyczą tej jawnej puli, nie wszystkich klatek surowych filmów.
- Stały split seed 42, proporcje 70/15/15. Audyt daje 117 grup train,
  25 val i 25 test. To konserwatywne grupy operacji, bez potwierdzonej
  rozłączności pacjentów. Połączone podobne klipy pozostają w tej samej grupie.
- Z 170 ZIP-ów pominięto dokładne kopie TR_0138 i TR_0144. Przed sprawdzeniem
  dostępności obrazów było 57 367 oznaczonych kandydatów train, 12 447 val,
  15 795 test. Nie wykonano ewaluacji test ani eksportu jego obrazów.
- Audyt ZIP-ów wykrył 738 brakujących obrazów w siatce: 297 train i 441 val.
  Braki wykluczamy przed selekcją i zachowujemy ich rejestr. Nie dopasowujemy
  podziału do wyników modelu. Pełne liczby dostępnych oznaczonych kandydatów
  podaje manifest przygotowanego pakietu.
  Dostępnych oznaczonych kandydatów jest 57 073 train i 12 043 val;
  nie wszystkie brakujące obrazy miały podaną ramkę, stąd różnica względem
  prostego odejmowania wszystkich 738 braków od liczby oznaczonych klatek.
- Puste komórki bbox nie są uznawane za potwierdzone negatywy. Ocena tej
  wersji jest warunkowa dla klatek z podaną ramką cystotomu; nie mierzy jeszcze
  fałszywych alarmów na wszystkich klatkach bez narzędzia. Do publikacyjnej
  oceny całego wideo trzeba ustalić semantykę pustych etykiet i negatywy.

500 klatek kalibracyjnych wybieramy przez przydział kwot proporcjonalny do
pierwiastka pojemności grup i środki równych przedziałów uporządkowanych
kandydatów w każdej grupie. Wybór jest stały dla wszystkich trzech seedów;
zmienia się wyłącznie seed treningu. Nie jest to trzy niezależne losowanie
podzbiorów ani test przewagi nad random.

Kalibracja używa wszystkich dostępnych oznaczonych kandydatów val z siatki
stride 5. Nie wybieramy na val klatek według jakości obrazu, cech czy błędu
detektora. Smoke używa osobnego stałego panelu 64/64.
Pakiet kalibracji zawiera 500 train i 12 043 val. Kontrola dry-run potwierdziła
dostępność wszystkich eksportowanych obrazów. Przygotowanie obrazów i
manifestów trwało 358.64 s, poza późniejszym pakowaniem TAR i hashowaniem.

## Receptura i wykonanie

RF-DETR Nano 1.10.1, torch 2.9.1+cu128, Python 3.11.15, wejście 384,
BF16, batch 16, grad_accum 1, AdamW, lr 1e-4, lr_backbone 1e-5,
weight_decay 1e-4, warmup 2 epoki, bez early stopping. W tej kalibracji
augmentacje są wyłączone; recepturę naukową i ewentualne augmentacje ustalamy
przed kampanią. Każdy model startuje niezależnie z tych samych wag COCO.

Każdy proces roboczy widzi tylko jedną przydzieloną GPU. To równoległe
eksperymenty, bez DDP i bez przekazywania wag między seedami. Dane rozpakowywane
są raz dla całego zadania do dostępnego katalogu lokalnego, z kontrolą miejsca
i sprzątaniem przy zakończeniu. Na login node przygotowujemy wyłącznie metadane
i pakiety obrazów; trening i inferencja CUDA odbywają się w zadaniu SLURM.
RF-DETR domyślnie pomija liczenie `val/loss` w tej recepturze; do kalibracji
zbieżności zachowujemy loss treningowy i AP walidacyjne po epokach. Nie
deklarujemy istnienia krzywej loss walidacyjnego.

Kod uruchamiania:

- `tools/prepare_tracking_calibration.py`: kontrola dostępności obrazów,
  wymiarów, dekodowania i dokładnego overlapu pikseli między eksportowanym
  train i val; jawne dane COCO, manifesty, osobny smoke, pakiet TAR i SHA256.
- `eden/run_nano_calibration.slurm`: rezerwacja trzech GPU oraz kontrola
  końcowego stanu trzech eksperymentów.
- `eden/submit_nano_calibration.sh`: zgłoszenie trzech seedów po sukcesie smoke.
- `pipeline/train/run_with_validation.py`: trening i osobna inferencja last/best
  na całej walidacji, próg operacyjny wybierany na val, standardowe COCO AP.

Środowisko:

```text
~/venvs/rfdetr_nano_clean_1_10_1/bin/python
```

Nie korzystamy już z usuniętego `tsjepa_py311`. Preflight SLURM sprawdza CUDA,
BF16 i liczbę GPU na Hopperze po migracji interpretera i bibliotek.

Wdrożenie:

```text
~/smallvit_runs/2026-09-14_tracking_nano/
```

Wyniki kalibracji:

```text
SmallViTAblation/results/calibration/tracking_uniform_b500_s42_calibration/
SmallViTAblation/results/calibration/tracking_uniform_b500_s1337_calibration/
SmallViTAblation/results/calibration/tracking_uniform_b500_s2024_calibration/
```

W każdym biegu zachowujemy last.pth, best.pth, logi epok, konfigurację,
manifest, koszt pamięci, predykcje i raport osobnej walidacji. Wagi last/best
służą do inferencji; nie obiecujemy wznowienia przerwanego treningu ze stanem
optymalizatora. Błędy nie są podstawą bezwarunkowego uruchomienia całej kampanii.
Po zakończeniu smoke usunięto jego last/best i kopie obrazów; zachowano raporty,
logi i hashe w `audit_smoke/reports.tar.gz`. Nie usunięto danych ani wag
działającej kalibracji.

## Co jest kolejnym etapem

Plan 36 porównań na jeden zbiór pozostaje celem: cztery selektory × trzy budżety
× trzy seedy. Przed jego zgłoszeniem trzeba zamrozić recepturę i przygotować
cache cech, warianty random/temporal/coverage oraz pełny wariant metody autora
w tym samym kontrakcie unikalnych źródeł. Dzisiejszy kod kalibracji nie generuje
36 biegów i nie zastępuje historycznej metody profilem `balanced`.
Osobno pozostają ablacje 15 kombinacji D/E/S/F, kontrola pełnej puli i
porównanie literaturowe. Wynik kalibracji ani wynik smoke nie dowodzi nowości
lub poprawy skuteczności kuracji.

Dowody przygotowania danych:
[indeks](evidence_tracking_training_2026-09-14/index_summary.json),
[stały podział](evidence_tracking_training_2026-09-14/split_manifest.json),
[brakujące obrazy](evidence_tracking_training_2026-09-14/missing_images_audit.json).
Dowody uruchomienia:
[raport smoke](evidence_tracking_training_2026-09-14/smoke_report.json),
[pakiet kalibracji](evidence_tracking_training_2026-09-14/data_manifest.json),
[SHA256 pakietu](evidence_tracking_training_2026-09-14/data_archive_sha256.json).
[Stan startu i pierwsze logi epok](evidence_tracking_training_2026-09-14/training_start_snapshot.json),
[sprzątanie smoke](evidence_tracking_training_2026-09-14/smoke_cleanup.json).
