# Start na Edenie: RF-DETR-Nano, smoke i statyczna kuracja

Data audytu: 2026-09-14. Ten dokument ustala aktualną kolejność startu zgodnie
z decyzją autora. Pierwszym modelem jest RF-DETR-Nano. Dokumenty 24 i 27 nadal
opisują dalsze możliwości; ich domyślny start od YOLO nie jest zleceniem obecnej
kolejki. Langevin, Koopman, GLE i active learning są osobnym, późniejszym etapem.

## Czy można już nacisnąć „start”?

**Aktualizacja:** poniższa tabela opisuje audyt przed wdrożeniem.
Stan rzeczywistego uruchomienia z tej samej daty jest w
[dokumencie 31](31_EDEN_RFDETR_SMOKE_WDROZENIE.md).

Nie w pełni. Kod adapterów i kolejki istnieje, ale stan lokalnego repozytorium
nie oznacza wdrożenia na Edenie. Audyt SSH był wyłącznie odczytowy: nie
instalowano pakietów, nie rozpakowywano zbiorów i nie wysyłano zadań SLURM.

| Element | Stan stwierdzony przez SSH |
|---|---|
| `ssh eden-cluster` | Połączenie działa; host `eden.eden`. |
| SLURM | `sbatch` dostępny; konto `transformers_vsc` istnieje. |
| GPU | Partycja `hopper` udostępnia 8 H100 na węźle; brak własnych zadań w momencie audytu. Dostępność zmienia się w czasie. |
| Nowy projekt | Nie znaleziono `SmallViTAblation` ani `ViTParticleFilterTracker` w przeszukanym katalogu domowym do głębokości 4. |
| Python 3.11 | Jest w `~/.conda/envs/tsjepa_py311/bin/python`, lecz nie w domyślnym PATH loginu. |
| Istniejące środowisko | Python 3.11.15, torch 2.9.1+cu128, torchvision 0.24.1, NumPy 2.0.2, PyYAML 6.0.3. |
| Brakujące zależności | W powyższym środowisku brak `rfdetr`, `pycocotools` oraz `ultralytics`. Do startu Nano potrzebne są dwie pierwsze; YOLO nie jest teraz wymagany. |
| Stare obrazy | Znaleziono JPEG w `~/datasets_20250606_test/datasets/DETR_augmented_dataset_20250218/images/`; sprawdzono przykładowe pliki, nie kompletność całego zbioru. |
| Stary podział | `~/DETR/fullpool_videosplit/` zawiera train/val/test i `video_split.json`. Adnotacje mają odpowiednio 71 316 / 10 780 / 9 240 obrazów. |
| Cataract-LMM tracking | Znaleziono 170 ZIP w `~/Cataract-LMM/3_Object_Tracking/annotations/`. Przykładowe `TR_0001_S1_P03.zip` zawiera 2046 JPEG, CSV i JSON; nie sprawdzono integralności wszystkich ZIP. |
| Cataract-LMM segmentation | Są katalogi anotacji COCO/YOLO; nie znaleziono ZIP do głębokości 3. Kompletu obrazów tej części nie potwierdzono. |

Przed pierwszym treningiem trzeba wdrożyć kod z zakończeniami linii LF,
przygotować osobne środowisko RF-DETR, sprawdzić rzeczywistą wersję API,
zapisać lock zależności i hash wag COCO oraz zbudować małą paczkę danych,
manifest puli, wariant i kolejkę. Nie modyfikujemy środowiska używanego przez grant.

## Co trenujemy najpierw?

### 0. Preflight, bez treningu

- Sprawdzenie wszystkich plików wskazanych przez smoke COCO oraz geometrii bbox.
- Zachowanie starego `video_split.json`; żadnego nowego losowego podziału klatek.
- Jedna klatka źródłowa liczy się raz. Na pierwszym porównaniu dawnych danych
  pomijamy kopie `_aug_N`, żeby nie mylić liczby plików z budżetem źródeł.
- Wariant train nie zawiera obrazów z val/test. Walidacja smoke jest małym,
  z góry zapisanym podzbiorem operacji walidacyjnych, wspólnym dla wariantów.
- Dry-run adaptera Nano i kolejki, bez dostępu do testu zewnętrznego.
- Na przydzielonej GPU: import RF-DETR, CUDA/BF16, ładowanie wag, pojedyncza
  predykcja. To nie jest test wydajności na węźle logowania.

### 1. Jeden trening sanity na dawnym zbiorze

Model: **RF-DETR-Nano**, rozdzielczość 384, BF16, start z wag COCO, 64
zweryfikowane klatki train, seed 42, 5 epok, bez augmentacji online i bez
early stopping. Wybór `uniform_time` wystarczy; nie liczymy jeszcze SAM,
Fouriera ani EL2N. Cel: dataset loader, poprawne klasy, gradienty, loss,
walidacja, zapis `last.pth`/`best.pth` i inferencja z zapisanego checkpointu.

Spadek loss i poprawne bbox są kontrolą nauki. Pięć epok nie gwarantuje overfitu;
jeżeli trzeba sprawdzić zdolność zapamiętania 64 przykładów, wydłużamy wyłącznie
ten osobny bieg diagnostyczny. Jego wynik nie służy do rankingu selektorów.

### 2. Małe porównanie statycznej selekcji na dawnym zbiorze

Stały budżet 500 unikalnych klatek train, ten sam model i recepta, 5 epok.

| Wariant | Po co |
|---|---|
| `random` | Losowy podzbiór źródeł przy tym samym budżecie. |
| `uniform_time` | Silne odniesienie uwzględniające redundancję wideo. |
| `coverage` | Statyczna różnorodność w prawdziwej, zamrożonej reprezentacji. |

Każdy wariant wykonujemy z seedami **42 i 1337**: razem **6 biegów**.
Pula, split, rozdzielczość, inicjalizacja, epoki i walidacja są identyczne.
Wszystkie trzy selektory mają domyślnie odstęp 5 klatek i ten sam składnik
bilansowania operacji. `random` oznacza więc losowy priorytet z tymi ograniczeniami,
a nie niezależne losowanie jednostajne. Raportujemy liczbę rozluźnień odstępu;
czyste losowanie bez tych ograniczeń może stanowić osobną kontrolę w pełnym pilocie.
`coverage` wymaga wcześniej policzonego cache prawdziwych cech train; cache
fixture/fake nie może tworzyć wyników modelu. Jeżeli cache nie jest gotowy,
najpierw uruchamiamy cztery biegi random/uniform, potem dodajemy dwa coverage.

### 3. Ten sam smoke na Cataract-LMM

Najpierw część **Object Tracking**, bo potwierdzono obecność obrazów w ZIP.
Zamrażamy split po operacjach/filmach, materializujemy mały podzbiór oraz
powtarzamy bieg 64-klatkowy i sześć biegów 500-klatkowych. `cystotome_bbox`
opisuje instrument; tip jest punktem. Puste etykiety wymagają QA przed
uznaniem ich za negatywy. Nie łączymy tych wyników z dawnym celem detekcji
ani z wieloklasową instance segmentation w jednej tabeli bez remapu zadania.

Łącznie domyślnie **7 treningów na każdy zbiór**: jeden sanity i sześć
porównań. Rozpoczynamy od jednego treningu na jednej GPU; nie wysyłamy całej
macierzy, zanim nie przejdzie zapis i ponowne załadowanie checkpointu.

## Co mierzymy w smoke?

- poprawność anotacji, liczbę obrazów rzeczywiście ocenionych i liczbę obiektów;
- loss, brak NaN/Inf, ukończone epoki i checkpointy;
- COCO mAP@0.5:0.95, mAP@0.5, AR@100 na ustalonej walidacji;
- precision/recall/F1 przy progu wybranym na val, jeżeli robimy ocenę operacyjną;
- czas epoki, czas całego biegu, VRAM, batch, precyzję i GPU model;
- identyczne hashe splitu/wariantów/recepty w sparowanych biegach.

Warunek awansu: pełna ścieżka działa, checkpoint można ponownie załadować,
metryki są liczone dla zadeklarowanego zestawu obrazów i koszt jest zmierzony.
Nie wymagamy „zwycięstwa kuracji” po pięciu epokach i dwóch seedach. Nie
otwieramy jeszcze testu wewnętrznego ani zewnętrznego do strojenia selektorów.

## Kolejny etap: poprawka do artykułu

Po smoke wykonujemy statyczny pilot Nano: 500/1500/2500 ramek, seedy
42/1337/2024, 30 epok i trzy powyższe selektory. Profil
`nano_static_pilot` ogranicza go do 27 biegów na zbiór. Czas mierzymy na Edenie;
liczba epok pilota nie jest jeszcze potwierdzoną receptą publikacyjną.

Następnie oddzielna, ograniczona kolejka sprawdza **oryginalny M-DESF**,
cztery ablacje leave-one-group-out oraz losowy i czasowy baseline przy tym
samym K. EL2N proxy trenuje się wyłącznie na train. `coverage` nie zastępuje
ablacji oryginalnego połączenia DINO/EL2N/SAM/Fourier. Nie wysyłamy domyślnego
całego Tier A z `build_run_manifest.py` jako smoke.

Po zamrożeniu recepty następuje potwierdzenie wieloma sparowanymi seedami,
ocena na teście oraz transfer do RT-DETR. Dopiero kolejna, osobna warstwa
wprowadza uncertainty/active learning, a po niej diagnostykę pamięci i reszt
GLE. Dotychczasowych checkpointów ep170 nie używamy jako inicjalizacji nowej
ablacji ani jako bezpośrednio porównywalnych rezultatów z innej recepty.

## Przygotowane wejścia wykonawcze

- `configs/profiles.yaml`: `nano_smoke` (5 epok) oraz
  `nano_static_pilot` (30 epok); wyłącznie statyczne strategie.
- `pipeline/active_learning/simulate.py --static`: pierwszy budżet wybiera
  wskazany selector, bez wspólnego L0. Random/uniform nie wymagają embeddings;
  coverage wymaga prawdziwego cache. Bez tej flagi pierwszy budżet jest wspólnym
  L0 i nie stanowi porównania trzech statycznych metod.
- `eden/run_smoke.slurm`: 1 H100, 8 CPU, 32 GB RAM, limit 2 godziny;
  bez rezerwowania całego węzła. Nie uruchamia funkcji latent dynamics.
- `eden/run_node.slurm`: używany wewnętrznie do stagingu paczki i cleanup;
  jego dyrektywy SBATCH są ignorowane, kiedy wywołuje go powyższy skrypt Bash.
- `eden/runner.py`: respektuje listę GPU przydzielonych przez SLURM.
- evaluator RF-DETR: jawne `pretrain_weights` i wejście `predict(images, ...)`,
  zgodnie z [kodem RF-DETR](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/detr.py)
  i [API Nano](https://rfdetr.roboflow.com/reference/nano/).

Przykład po wdrożeniu kodu i przygotowaniu wariantów/danych:

```bash
# SMOKE_DATA_ROOT zawiera przygotowany pool_manifest z właściwego zbioru.
cd "$(dirname "$PROJECT_DIR")"
for seed in 42 1337; do
  for strategy in random uniform_time; do
    "$PY" -m SmallViTAblation.pipeline.active_learning.simulate \
      --static --pool "$SMOKE_DATA_ROOT/pool_manifest.jsonl" \
      --strategy "$strategy" --budgets 500 --seed "$seed" \
      --out "$PROJECT_DIR/variants/nano_smoke_500"
  done
done
# Gdy TRAIN_FEATURES_NPZ jest prawdziwym cache train, dodaj dwa warianty coverage:
if [[ -n "${TRAIN_FEATURES_NPZ:-}" ]]; then
  for seed in 42 1337; do
    "$PY" -m SmallViTAblation.pipeline.active_learning.simulate \
      --static --pool "$SMOKE_DATA_ROOT/pool_manifest.jsonl" \
      --embeddings "$TRAIN_FEATURES_NPZ" --strategy coverage \
      --budgets 500 --seed "$seed" --out "$PROJECT_DIR/variants/nano_smoke_500"
  done
fi

cd "$PROJECT_DIR"
mkdir -p slurm_logs
# PY wskazuje interpreter osobnego, sprawdzonego środowiska RF-DETR.
"$PY" tools/build_active_manifest.py \
  --variants-dir "$PROJECT_DIR/variants/nano_smoke_500" \
  --profile nano_smoke --out "$PROJECT_DIR/runs/nano_smoke_500.jsonl"
"$PY" eden/queue.py init "$PROJECT_DIR/eden/qdir_nano_smoke_500" \
  --queue-file "$PROJECT_DIR/runs/nano_smoke_500.jsonl"
export QUEUE_DIR="$PROJECT_DIR/eden/qdir_nano_smoke_500"
# DATA_ARCHIVE jest zweryfikowaną małą paczką train/val; nie całym Cataract-LMM.
export PROJECT_DIR PY QUEUE_DIR DATA_ARCHIVE
export RESULTS_DIR="$PROJECT_DIR/results/nano_smoke_500"
sbatch eden/run_smoke.slurm
```

Warianty dla sanity 64 i dla porównania 500 trzymamy w osobnych katalogach.
Generator selekcji zapisuje tylko train, lecz paczka DATA_ARCHIVE musi także
zawierać stałe `annotations_val.json` oraz dostępne obrazy walidacyjne.
Ścieżki wariantów generujemy na Edenie. Kolejka obejmuje trening; inferencja
i pełne metryki są osobnym krokiem, który trzeba wykonać po zapisaniu wag,
podając model **rf-detr-nano** i rozdzielczość **384**, a nie domyślne RF-DETR-Small.
Job smoke sprawdza, czy wszystkie zadania kolejki są done; sam status
COMPLETED w SLURM nie zastępuje kontroli logów oraz wyników ewaluacji.

## Zakres sprawdzenia po poprawkach

Lokalnie: testy CPU, regresje API inferencji i przydziału GPU, dry-run Nano,
kompilacja Pythona. Skrypt smoke sprawdzamy składniowo przez Bash na Edenie.
Żaden z tych kroków nie jest dowodem działania treningu CUDA na H100.
Brak rzeczywistego treningu pozostaje jawnym ograniczeniem gotowości.

Wynik kontroli 2026-09-14: 50 testów CPU przeszło; dry-run Nano na 64 obrazach
potwierdził model, 384 piksele, BF16 i receptę 5 epok, przy 0 wykonanych epokach.
Standalone kolejki także przeszedł, a `bash -n` na Edenie zaakceptował oba
skrypty stagingu/smoke. Jedyna uwaga pytest dotyczy lokalnej wersji optree.
