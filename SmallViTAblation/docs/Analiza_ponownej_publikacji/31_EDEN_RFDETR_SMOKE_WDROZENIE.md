# RF-DETR-Nano: wdrożenie smoke na Edenie

Data: 2026-09-14. Pierwszym modelem jest **RF-DETR-Nano**. YOLO może być
późniejszą kontrolą transferu. Koopman, Langevin i aktywne uczenie należą
do kolejnej warstwy badań; nie są składnikami pierwszego treningu.

**Wynik: pierwszy kompletny smoke działa. Zadanie 1807587 — COMPLETED, exit 0,
2 min 30 s.** Trening, eksport zwykłych ostatnich wag i best/EMA, ponowne
wczytanie obu oraz ewaluacja wszystkich 64 obrazów zakończyły się poprawnie.
Nie oznacza to gotowości pełnego pilota ani potwierdzonej przewagi selekcji:
pełna pula źródłowa i smoke CATARACT-LMM są następnymi krokami.

## Środowisko

**Późniejsza aktualizacja z 2026-09-14:** na polecenie autora usunięto
`tsjepa_py311` po uniezależnieniu Nano. Biblioteki GPU przeniesiono do venv,
a interpreter pochodzi teraz z `~/venvs/rfdetr_python311`.
[Aktualny stan i zakres kontroli](32_EDEN_PAUZA_SRODOWISKA_I_PLAN_TRENINGOW.md).
Opis poniżej dokumentuje konfigurację w chwili zakończonego smoke.

Odczyt przez `ssh eden-cluster` potwierdził istniejące środowisko
`~/.conda/envs/tsjepa_py311`: Python 3.11.15, torch 2.9.1+cu128,
torchvision 0.24.1, Lightning 2.5.6, transformers 4.57.6.
Brakowało RF-DETR i pycocotools. To użyteczna baza sprzętowa/Python,
ale nie gotowe środowisko RF-DETR.

RF-DETR 1.10.1 wymaga Lightning >=2.6 i transformers >=5.1.
Próba osobnego venv dziedziczącego pakiety przez `--system-site-packages`
ujawniła konflikty m.in. z neuralforecast, vLLM i sentence-transformers.
Nie zmieniono środowiska JEPA. Środowisko dziedziczące
`~/venvs/rfdetr_nano_1_10_1` było próbą diagnostyczną; usunięto je w tej sesji,
aby zwolnić limit liczby plików. Nie było użyte do treningu.

Docelowe środowisko z oddzielnymi zależnościami RF-DETR:

```bash
~/venvs/rfdetr_nano_clean_1_10_1/bin/python
```

Pełna instalacja jest opisana przez `eden/requirements-rfdetr-smoke.txt` w czystym
venv Python 3.11. Instalacja trafiła na limit quota: dokładny odczyt `df -i`
wykazał 200 000/200 000 plików, mimo około 21 GiB wolnego miejsca.
Usunięto wyłącznie nowo utworzone kopie GPU z tego venv oraz nieużywane
środowisko diagnostyczne; torch,
torchvision, Triton i NVIDIA są dowiązaniami do istniejących bibliotek
`tsjepa_py311`. RF-DETR i jego pozostałe zależności są instalowane oddzielnie;
nie włączono `system-site-packages`. Listę dowiązań zapisano w
`~/venvs/rfdetr_nano_clean_1_10_1/GPU_LINKS.json`.
To rozwiązanie oszczędza miejsce, ale zależy od zachowania wskazanej instalacji
GPU: nie usuwać ani nie aktualizować jej w trakcie macierzy. Pełną niezależną
kopię można zbudować z requirements po zapewnieniu odpowiedniej quota.
Najważniejsze ograniczenia: RF-DETR 1.10.1, torch 2.9.1,
torchvision 0.24.1. Każda kolejka zapisuje pełne `pip freeze` jako
`environment.txt`. Pierwsza instalacja ma również logi w `~/rfdetr_smoke_setup/`.

API zweryfikowano w kodzie zainstalowanego pakietu. `amp` należy do konfiguracji
modelu; `amp_dtype` do treningu. Adapter ustawia COCO maxDets=100, zapisuje
efektywną konfigurację, konfigurację i hash wag początkowych, logi CSV oraz
szczytową pamięć CUDA. Opis użycia producenta:
[RF-DETR — trening](https://rfdetr.roboflow.com/learn/train/).
Szczegóły API należy wiązać z zainstalowaną wersją, ponieważ strona instrukcji
może opisywać inną generację trenera.

## Dane: co faktycznie jest dostępne

`~/DETR/fullpool_videosplit/` zawiera istniejący podział 35/8/7 nagrań.
Pliki train/val obejmują także augmentacje offline. Oryginałów jest odpowiednio
17 832 i 2695; liczby 71 316 i 10 780 obejmowały augmentacje.

Rozpakowany katalog `~/datasets_20250606_test/.../images` zawiera tylko
10 obrazów. Nie jest kompletną pulą do treningu. Na smoke przygotowano
obrazy z istniejącego `~/20kSelectedImages.tar`. Jest to archiwum już
wyselekcjonowanych danych: nie wolno traktować go jako pełnej puli do
udowodnienia przewagi nowej selekcji.

`tools/prepare_eden_smoke.py`:

- pomija `aug_source` i `_aug_`, zachowuje dotychczasowy podział nagrań;
- wybiera oryginały z dostępnego archiwum, z podobną liczbą na nagranie
  i ze środków przedziałów czasowych;
- sprawdza dekodowanie JPEG, rozmiar, bbox, identyfikatory klas, duplikaty SHA256;
- zapisuje COCO, wariant, kopię podziału, manifest pochodzenia i archiwum;
- nie odczytuje anotacji testowych i nie wyciąga obrazów testowych.

Zbudowana paczka ma **64 train / 64 val**, po jednej anotacji na obraz.
Train obejmuje wszystkie 35 nagrań train; val wszystkie 8 nagrań val.
Zachowano klasę `tool`, ID 0, i oryginalne bbox. To kontrola działania
dotychczasowej reprezentacji etykiet; nie nadaje jej automatycznie znaczenia
całego narzędzia ani etykiety końcówki z nowego zbioru.

## Co dokładnie trenujemy

| Etap | Model i dane | Liczba biegów | Cel |
|---|---|---:|---|
| Obecny smoke techniczny | RF-DETR-Nano, 64 train / 64 val, seed 42, 5 epok | 1 | Sprawdzenie pełnej ścieżki trening–checkpoint–predykcje–metryki |
| Następny smoke selekcji | Nano, 500 oryginalnych train, random / uniform_time / coverage, seedy 42 i 1337, 5 epok | 6 | Sprawdzenie mechaniki selekcji i kosztu; bez tezy o przewadze |
| Pilot naukowy | Nano, budżety 500/1500/2500, te same 3 metody, seedy 42/1337/2024, 30 epok | 27 | Ocena efektu kuracji przy ustalonej recepcie |
| CATARACT-LMM | Najpierw osobny smoke Nano na tracking bbox i zatwierdzonym podziale operacji | najpierw 1 | Potwierdzenie nowego formatu i semantyki etykiet |
| Ablacje właściwego pipeline'u | Pełny selektor i usuwanie kolejnych składników, mocne baseline'y | według macierzy | Wkład metody potrzebny do artykułu |
| Transfer i dynamika | Większy detektor, potem osobno AL/Koopman/GLE | później | Generalizacja i dodatkowa hipoteza badawcza |

Sześciu biegów selekcji ani pilota 27 biegów nie uruchamiamy automatycznie
w ramach obecnego smoke. Najpierw należy odtworzyć i zamrozić pulę źródłową,
cache rzeczywistych cech dla coverage i wspólną walidację. Random w istniejącym
selektorze oznacza losowy priorytet z ograniczeniami czasowymi i bilansowaniem
nagrań; szczegóły opisano w dokumencie 30.

Pierwszy trening: **384×384, BF16, batch 16, akumulacja 1, AdamW,
LR 1e-4, LR enkodera 1e-5, weight decay 1e-4, warmup 2 epoki**.
Start z wag COCO; 5 epok. RF-DETR 1.10.1 przy 64 obrazach i batch 16
stosuje losowanie ze zwracaniem do minimum 80 ekspozycji na epokę:
**5 batchy/epokę, 25 kroków łącznie, 400 ekspozycji na 64 unikalne obrazy**.
To sprawdzono w `training/module_data.py` oraz logu rzeczywistego zadania.
Nie oznacza to 400 niezależnych klatek. Większe pule biblioteka wyrównuje do
wielokrotności efektywnego batcha; w macierzy raportujemy też rzeczywiste kroki
i ekspozycje, nie wyłącznie liczbę epok.
Bez augmentacji online, multi-scale i early stopping. Walidacja co epokę.
EMA pozostaje ustawieniem RF-DETR zapisanym w efektywnej konfiguracji;
`best` pochodzi z wyboru według walidacji, `last` z ostatnich wag modelu.

Jedna H100, 8 CPU, 32 GiB RAM, limit 2 godzin. To limit zadania, nie prognoza
czasu pięciu epok. A100 na partycji `short` również może wykonywać tę receptę;
pierwszy bieg na Hopperze zmierzy rzeczywisty koszt. W macierzy wiele niezależnych
biegów może korzystać po jednej GPU; nie potrzebujemy 8 GPU dla jednego Nano.

## Uruchomienie i kontrola

Wdrożenie:

```bash
cd ~/smallvit_runs/2026-09-14_nano_smoke/SmallViTAblation
export PY="$HOME/venvs/rfdetr_nano_clean_1_10_1/bin/python"
export DATA_ARCHIVE="$PWD/data/old64.tar"
# Unikalny RUN_TAG przy każdym nowym biegu. Skrypt odmawia nadpisania wyników.
export RUN_TAG=old64_nano_s42_smoke06
unset EXISTING_RUN_ID
bash eden/submit_nano_smoke.sh
```

Skrypt inicjuje osobną kolejkę, zapisuje wersje bibliotek i hash danych,
następnie wysyła jedno zadanie SLURM. Skrypty powłoki mają wymuszone LF
w `.gitattributes`. Archiwum kodu ma zapisany commit źródłowy w
`../SOURCE_COMMIT` i SHA256 w `../SOURCE_ARCHIVE.sha256`.

```bash
squeue -u "$USER"
cat "eden/queues/$RUN_TAG/slurm_job_id.txt"
"$PY" eden/queue.py status "eden/queues/$RUN_TAG" --json
tail -n 60 eden/queues/"$RUN_TAG"/logs/*.log
```

Warunkiem technicznego sukcesu jest `smoke_report.json` ze statusem `passed`
oraz zakończenie SLURM bez błędu. Samo `Submitted batch job` nie oznacza
udanego treningu. Raport powstaje dopiero po treningu oraz osobnej inferencji
z `last.pth` i `best.pth` na wszystkich 64 obrazach walidacyjnych.

Wyniki: `results/$RUN_TAG/$RUN_TAG/`. Zapisujemy predykcje COCO, mAP50:95,
mAP50, mAP75, AR100, liczniki TP/FP/FN i F1. Próg 0.5 dla TP/FP/F1 jest
wyłącznie stałą kontrolą smoke; w badaniu naukowym wybieramy próg na walidacji.
mAP jest liczone z predykcji od 0.001, niezależnie od progu operacyjnego.

Dane robocze trafiają na dysk/RAM węzła i są sprzątane po zadaniu; wyniki
zostają w katalogu projektu. Pozostają dwa checkpointy oraz logi. Pięć epok
na 64 obrazach nie dowodzi jakości selekcji ani wartości publikacyjnej.

## Stan wykonania

Paczka danych i kod są wdrożone. `pip check`: **No broken requirements found**;
import `rfdetr.training` udany. Preflight na H100 PCIe potwierdził torch
2.9.1+cu128, RF-DETR 1.10.1 i CUDA 12.8.

- Zadanie **1807583**: FAILED przed treningiem, brak prawa zapisu w `/raid`.
  Poprawiono sprawdzanie zapisu i przejście na `/dev/shm`.
- Zadanie **1807584**, `RUN_TAG=old64_nano_s42_smoke02`: wykonało 5 epok,
  lecz zakończyło się FAILED przy odbiorze checkpointów. RF-DETR 1.10.1
  zostawia `last.ckpt`, a nie oczekiwane `last.pth`.
- Zadanie **1807585**, `RUN_TAG=old64_nano_s42_smoke03`, kod `02d024b`:
  wykonało 5 epok i zapisało oba checkpointy. Końcowa ewaluacja wykryła slot
  tła zwracany przez RF-DETR; zadanie zakończyło się FAILED na walidacji klas.
- Zadanie **1807586**, `RUN_TAG=old64_nano_s42_eval04`, kod `77f3982`:
  ponowna inferencja i ewaluacja istniejących wag z smoke03, bez nowego treningu.
  Zakończyło się COMPLETED, ale dalszy audyt wykazał, że oba pliki zawierały
  wagi EMA. Tego raportu nie używamy jako dowodu rozdzielenia last/best.
- Zadanie **1807587**, `RUN_TAG=old64_nano_s42_smoke05`, kod `7be3857`:
  **COMPLETED, exit 0, 00:02:30**. Pełny smoke z poprawioną semantyką eksportu
  ostatnich wag; kolejka ma done=1, failed=0, pending=0, running=0.

Końcowa implementacja pobiera zwykłe wagi ostatniej epoki z `last.ckpt`
zapisanego przez Lightning, sprawdza numer epoki i zapisuje hash źródła oraz
liczbę kroków. Nie używa modelu pozostawionego w pamięci: callback RF-DETR
`RFDETREMACallback.on_train_end` podmienia go na EMA. `best.pth` pozostaje
checkpointem wybranym według walidacji. Oba końcowe pliki są tylko wagami
do inferencji, nie pełnym stanem optymalizatora do wznowienia treningu.

Koszt zakończonego treningu smoke05: 61.69 sekund (obejmuje
inicjalizację, 5 epok, walidacje i zapis; nie obejmuje późniejszej inferencji).
Szczyt GPU: 6.43 GB alokowane / 7.18 GB zarezerwowane, czyli około
5.99/6.69 GiB. Loss po epokach: 11.0941, 8.8955, 6.7420, 5.9131, 5.6328.
Trening i dwie niezależne inferencje trwały łącznie 103.31 s; pozostały czas
SLURM obejmuje preflight, staging i zamknięcie kolejki. Na login node nie
wykonywano treningu ani inferencji GPU.
Seedy są ustawione, ale nie obiecujemy bitowej deterministyczności CUDA/BF16;
powtórne krótkie biegi z seed 42 dały lekko różne wyniki. Pilot wymaga
sparowanych seedów i raportowania rozrzutu, nie jednego wybranego wyniku.

Wynik końcowy: `smoke_report.json` ma status **passed**. `checkpoint_audit.json`
potwierdza epokę 4 (indeks od zera), 25 kroków, źródłowy hash `last.ckpt`,
wybór best z EMA i różnice w 464/466 tensorach pomiędzy last a best.

| Checkpoint | mAP50:95 | mAP50 | AR100 | Obrazy |
|---|---:|---:|---:|---:|
| last, zwykłe ostatnie wagi | 0.06296 | 0.15209 | 0.45156 | 64 |
| best, wybrany na val, EMA | 0.06244 | 0.15542 | 0.46094 | 64 |

To niskie wyniki krótkiego smoke, nie dowód skutecznej kuracji.
Przy kontrolnym progu 0.5 oba modele nie zwróciły detekcji obiektu:
TP=0, FP=0, FN=64. Pięć epok wystarczyło do sprawdzenia infrastruktury i
spadku loss, nie do otrzymania gotowego detektora. Best wybierano przez
metrykę podczas treningu BF16; końcowa standardowa inferencja ma inną
ścieżkę numeryczną, dlatego nie zakładamy identyczności obu pomiarów AP.

Lokalne dowody: [raport](evidence_eden_smoke_2026-09-14/smoke_report.json),
[audyt checkpointów](evidence_eden_smoke_2026-09-14/checkpoint_audit.json),
[manifest treningu](evidence_eden_smoke_2026-09-14/run_manifest.json),
[log metryk](evidence_eden_smoke_2026-09-14/training_metrics.csv),
[pochodzenie danych](evidence_eden_smoke_2026-09-14/data_manifest.json).

Wagi były zapisane na Edenie pod ścieżką poniżej. Po zakończeniu kontroli
2026-09-14 zostały usunięte na polecenie autora wraz z roboczymi danymi i kolejkami.
Zachowano raporty i logi; aktualny stan opisuje
[dokument 32](32_EDEN_PAUZA_SRODOWISKA_I_PLAN_TRENINGOW.md).

```text
~/smallvit_runs/2026-09-14_nano_smoke/SmallViTAblation/results/old64_nano_s42_smoke05/old64_nano_s42_smoke05/
```

Przed rozszerzeniem kolejki: odtworzyć pełną pulę starych danych albo przygotować
zatwierdzony manifest CATARACT-LMM, a następnie wykonać jego osobny smoke.
Nie ma potrzeby wracać do YOLO, aby wykonać kolejny etap na Nano.

Do kolejki przekazujemy `PYTHONDONTWRITEBYTECODE=1`, aby nie mnożyć plików pyc.

- Adapter usuwa z predykcji tylko slot jawnie opisany przez RF-DETR jako
  `__background__`. Nieznane klasy nadal powodują błąd; nie zamieniamy tła
  ani wszystkich wyników na jedyną klasę obiektu.
