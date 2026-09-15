# Kampania statycznej selekcji CATARACT-LMM na Edenie — 2026-09-15

## Cel i granica wniosku

Kampania porównuje cztery statyczne sposoby wyboru klatek przy stałym budżecie
danych. Nie korzysta z Koopmana, Langevina ani Mori–Zwanziga. Jej wyniki mają
najpierw ustalić, czy pokrycie semantyczne i kontrola nadmiarowych sąsiadów
czasowych poprawiają RF-DETR Nano względem prostych kontroli.

Badamy wyłącznie klatki z dostępną ramką cystotomu w części trackingowej
CATARACT-LMM. Split to 117/25/25 konserwatywnych grup operacji; rozłączność
pacjentów nie jest potwierdzona. Test pozostaje niewczytany i niewyeksportowany.

## Zamrożona macierz

- selektory: `random`, `uniform_time`, `dino_coverage`,
  `dino_temporal_hybrid`;
- budżety: 500, 1500 i 2500 unikalnych klatek źródłowych;
- seedy treningu: 42, 1337 i 2024;
- seed selekcji: stałe 42 dla wszystkich treningów;
- razem: 36 niezależnych treningów RF-DETR Nano po 30 epok.

Stały seed selekcji oddziela wariancję optymalizacji od wariancji wylosowanego
podzbioru. Osobne powtórzenia selekcji będą potrzebne dopiero wtedy, gdy
pilotaż pokaże efekt wymagający potwierdzenia.

`dino_coverage` stosuje kwoty per grupa proporcjonalne do pierwiastka liczby
kandydatów oraz farthest-first k-center w cechach DINOv2. Jeden punkt czasowy
inicjuje wybór w każdej aktywnej grupie. `dino_temporal_hybrid` rezerwuje 25
procent kwoty na równomierne kotwice czasowe, a resztę dobiera przez k-center.
To kandydat statyczny, a nie pełna historyczna metoda D/E/S/F.

DINOv2 ViT-S/14 jest przypięty do commita
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Manifest zapisuje hash stanu
modelu, wersję torch, parametry obrazu, hash kodu i tożsamość każdego źródła.
Ranking nie używa GT; etykiety służą jedynie do określenia puli dopuszczonej
do kontrolowanego treningu detektora.

## Receptura i zasoby

Receptura pozostaje zgodna z zakończoną kalibracją: wejście 384, BF16, batch
16, AdamW, 30 epok, ten sam start COCO i pełna walidacja 12 043 klatek po
każdej epoce. Każdy bieg zapisuje `last.pth`, `best.pth` oraz niezależną ocenę
obu checkpointów. Duże pliki predykcji są kompresowane deterministycznie po
obliczeniu metryk; raport przechowuje SHA256 wersji jawnej i archiwum.

Przygotowanie używa jednej H100, 16 CPU i 128 GiB RAM. Trening zależy przez
`afterok` od poprawnego przygotowania i używa maksymalnie czterech H100,
32 CPU i 384 GiB RAM. Cztery workery wykonują po jednym eksperymencie na GPU,
bez DDP. Oba zadania mają `--no-requeue`; błąd przygotowania blokuje trening.
Przyszłe łańcuchy nie mogą przekraczać sześciu zależnych zadań.

## Implementacja

- `tools/prepare_tracking_static_pilot.py` tworzy cache cech, zamrożone
  warianty, wspólną unię obrazów, pełną walidację i pakiet z SHA256;
- `eden/prepare_tracking_static_pilot.slurm` uruchamia ekstrakcję cech;
- `eden/run_tracking_static_pilot.slurm` tworzy kolejkę 36 biegów i uruchamia
  osiem workerów;
- `eden/submit_tracking_static_pilot.sh` zgłasza kontrolowany łańcuch `afterok`;
- `pipeline/train/run_with_validation.py` archiwizuje predykcje bez utraty ich
  hashy i oddziela seed selekcji od seedu treningu.

## Zgłoszenie

Kod kampanii wdrożono z commita `4ec9c8b`. Zadanie przygotowawcze `1807632`
rozpoczęło pracę na Hopperze i poprawnie wczytało przypięty model DINOv2;
pierwsze partie cech zostały zapisane w cache. Pierwotne oczekujące zadanie
ośmiokartowe `1807633` anulowano przed startem po wprowadzeniu limitu czterech
GPU. Zadanie zastępcze `1807634` zawiera tę samą macierz, rezerwuje cztery H100
i ma warunek `afterok:1807632`.
Oba aktywne etapy mają `Requeue=0`.

Artefakty, kolejka i wyniki są zapisywane w:

```text
~/smallvit_runs/2026-09-15_tracking_static36/
```
