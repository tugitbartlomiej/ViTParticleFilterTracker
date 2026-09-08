# 06 — TODO: co napisać w kodzie i jak to puścić na Edenie

---

## Część A. Ile to potrwa

Policzone przez `tools/estimate_time.py` z rzeczywistej listy biegów
(`tools/build_run_manifest.py` daje **144 biegi**).

Jednostka niezależna od sprzętu to **obrazo-epoka** — suma po biegach z
(budżet obrazów × liczba epok). Przy 60 epokach na bieg wychodzi:

| Pozycja | Obrazo-epoki |
|---|---|
| treningi (144 biegi) | 88 110 000 |
| modele proxy dla EL2N (8 × 10 epok) | 5 680 000 |
| **razem** | **93 790 000** |

Czas zależy już tylko od przepustowości, czyli ile obrazów na sekundę przerobi
jedno GPU. Tej liczby **nie zgaduję** — mierzy ją bieg kalibracyjny. Do tego
czasu widełki (z narzutem 15% na ewaluację, zapis i kolejkowanie):

| obrazów/s na GPU | GPU-godziny | doba na 8 GPU |
|---:|---:|---:|
| 30 | 999 | 5,2 |
| 40 | 749 | 3,9 |
| 60 | 499 | 2,6 |
| 80 | 375 | 2,0 |
| 120 | 250 | 1,3 |

**Wniosek:** cały eksperyment mieści się w przedziale **od półtora do pięciu dni**
na jednym węźle `hopper` z ośmioma H100. To jest mniej, niż kosztował jeden
trening DETR w starym podejściu.

**Uwaga na limit czasu:** partycja `hopper` ma MaxTime 5 dni. Przy pesymistycznym
wariancie całość nie zmieści się w jednym zadaniu, więc **i tak trzeba tnąć na
kawałki** — patrz część C. To jest argument za kolejką, nie za jednym wielkim
zadaniem.

Aktualizacja po kalibracji:

```bash
py -3.11 tools/estimate_time.py --throughput <zmierzone> --gpus 8
```

---

## Część B. Kod do napisania

Kolejność ma znaczenie — `T1`–`T3` są darmowe i rozstrzygają resztę.

### Etap 0 — rozpoznanie (bez GPU)

| # | Zadanie | Wyjście |
|---|---|---|
| ~~T1~~ | ~~sprawdzić, czy film = pacjent~~ — **rozstrzygnięte 2026-09-08: jedno nagranie = jeden pacjent, 50 pacjentów, podział 35/8/7 jest podziałem po pacjencie** | nic do napisania |
| **T2** | `tools/dataset_census.py` — policzyć: klatki źródłowe i augmentowane per split, rozkład liczby augmentacji na klatkę, rekoncyliacja 91 336 vs 91 348, dokładne rozmiary train/val/test | `dataset_census.json` + tabela do artykułu |
| **T3** | inwentaryzacja Edenu — czy istnieją już gotowe treningi na podziale po filmach (DETR i YOLO), jakie checkpointy, jaki protokół | wpis w `05_OPEN_QUESTIONS.md` |

### Etap 1 — dane i cechy

| # | Zadanie | Uwagi |
|---|---|---|
| **T4** | `data/build_train_pool.py` — zbudować pulę selekcji **tylko z filmów treningowych**; zapisać `pool_manifest.json` | to naprawia wyciek, o którym recenzenci nawet nie wiedzieli |
| **T5** | `data/prescale_images.py` — przeskalować pulę raz do 512×512 i zapisać jako JPEG | oszczędza dekodowanie w każdej epoce każdego biegu |
| **T6** | `features/build_cache.py` — DINO, Fourier, SAM policzone raz na puli treningowej, zapisane w spójnej kolejności | wzorzec: `Eden/Checkpoints/Ablations/ablation/feature_cache/` |
| **T7** | `features/train_el2n_proxy.py` — proxy RF-DETR-S, 10 epok, tylko klatki treningowe, **jeden na ziarno** | wraca do oryginalnej definicji EL2N |
| **T8** | `features/el2n_from_proxy.py` — policzyć EL2N z proxy, zapisać `el2n_scores_<seed>.pkl` | |

### Etap 2 — selekcja

| # | Zadanie | Uwagi |
|---|---|---|
| **T9** | `selection/select_subset.py` — **jeden** skrypt dla wszystkich 15 kombinacji; grupy i wagi z manifestu, reguła proporcjonalnego przeskalowania | zastępuje pięć prawie identycznych funkcji `variant_*` w starym kodzie |
| **T10** | `selection/baselines.py` — `B-RANDAUG`, `B-RANDSRC`, `B-UNIFORM`, `B-FULLPOOL` | `B-RANDSRC` to żądanie recenzenta 2 |
| **T11** | `selection/materialize.py` — zapisać zbiór jako **manifest + `annotations_train.json`**, nigdy jako kopie obrazów | inaczej setki gigabajtów duplikatów |

### Etap 3 — trening

| # | Zadanie | Uwagi |
|---|---|---|
| **T12** | `train/train_rfdetr.py` — jeden skrypt, wszystkie parametry z `protocol.yaml`, **stała liczba epok, bez early stopping**, zapis `run_manifest.json` z hashami | serce protokołu |
| **T13** | `train/train_transfer.py` — D-FINE-S i bieżąca YOLO, **iso-resource** (ta sama rozdzielczość, epoki, zbiór) | zamyka zarzuty 7 i 8 recenzenta 1 |
| **T14** | ustawienia NMS dla YOLO jako jawny parametr + krótka analiza wrażliwości | zarzut 9 recenzenta 1 |

### Etap 4 — ewaluacja

| # | Zadanie | Uwagi |
|---|---|---|
| **T15** | `eval/evaluate.py` — standardowa inferencja, próg 0,001 do predykcji, mAP z pełnej krzywej PR, zapis predykcji do pliku | bez [Query 81] (kju osiemdziesiąt jeden) |
| **T16** | `eval/operating_point.py` — próg operacyjny wybierany na walidacji, krzywa F1 od progu, test przeniesienia progu na obcą dziedzinę | zarzut 6 recenzenta 1 |
| **T17** | `eval/aggregate.py` — średnia ± odchylenie po ziarnach, tabele w formacie LaTeX gotowe do wklejenia | |
| **T18** | `eval/external_once.py` — jedno przejście po zbiorze zewnętrznym + wpis do `external_access_ledger.jsonl` | zarzut 10 recenzenta 1 |

### Etap 5 — analizy bez GPU

| # | Zadanie |
|---|---|
| **T19** | pokrycie rozkładu **zewnętrznego** przez wybrany zbiór (osadzenia DINO benchmarku) — właściwa miara różnorodności |
| **T20** | skład augmentacji w wybranym zbiorze; korelacja rang EL2N z dwóch rodzin modeli |

---

## Część C. Uruchamianie na Edenie

### Zasada: kolejka, nie sto zadań SLURM

Sto czterdzieści osobnych `sbatch` to sto czterdzieści razy kopiowanie danych
i czekanie w kolejce. Zamiast tego:

> **jedno zadanie SLURM = jeden węzeł = osiem workerów, po jednym na GPU,
> biorących kolejne biegi z jednej listy.**

Dane rozpakowują się **raz** na `/raid`, a workery jadą jeden bieg za drugim,
aż lista się skończy albo zadanie dobije do limitu czasu.

### Pliki do napisania

| # | Plik | Rola |
|---|---|---|
| **T21** | `Eden/Scripts/SmallViTAblation/runner.py` | worker: bierze następny bieg z kolejki, trenuje, ewaluuje, oznacza jako zrobiony |
| **T22** | `Eden/Scripts/SmallViTAblation/queue.py` | kolejka na pliku z blokadą; stan `pending / running / done / failed`; **wznawialna** |
| **T23** | `Eden/Scripts/SmallViTAblation/run_node.slurm` | zadanie na jeden węzeł, osiem workerów w tle, `wait` |
| **T24** | `Eden/Scripts/SmallViTAblation/submit_chain.sh` | łańcuch zadań z `--dependency=afterany`, żeby przejść przez limit 5 dni |

### Szkic zadania SLURM

```bash
#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH -o logs/svit_%j.out

set -euo pipefail

# TMPDIR wg wzorca z Eden/EDEN_CONSTRAINTS.md (NIE /tmp - za male)
if   [[ -n "${SLURM_TMPDIR:-}" ]]; then TMPDIR="$SLURM_TMPDIR"
elif [[ -d /raid ]]; then TMPDIR="/raid/${USER}_${SLURM_JOB_ID}_svit"
else TMPDIR="/dev/shm/${USER}_${SLURM_JOB_ID}_svit"; fi
mkdir -p "$TMPDIR"

cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Dane rozpakowujemy RAZ na wezel, nie raz na bieg
tar -xf ~/data/train_pool_512.tar -C "$TMPDIR"

# Osiem workerow, po jednym na GPU, wszystkie z tej samej kolejki
for GPU in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$GPU python runner.py \
      --queue ~/svit/queue.jsonl \
      --data "$TMPDIR/train_pool_512" \
      --worker "w$GPU" &
done
wait
```

### Wymagania wobec workera (to jest lista kontrolna dla T21/T22)

1. **Wznawialność.** Zadanie ubite limitem czasu wraca i podejmuje kolejkę
   tam, gdzie była. Bieg oznaczony `running` bez żywego procesu wraca do
   `pending` po ustalonym czasie.
2. **Atomowość.** Pobranie biegu z kolejki pod blokadą pliku, żeby dwa workery
   nie wzięły tego samego.
3. **Błąd jednego biegu nie zatrzymuje reszty.** Zapisujemy `failed` z logiem
   i idziemy dalej — tak jak w istniejącym `run_all_full_evals.py`.
4. **Sprzątanie.** Po biegu zostaje `best_model.pth`, `last.pth`, `metrics.json`,
   `run_manifest.json`. Checkpointy pośrednie kasujemy — zasada z
   `EDEN_CONSTRAINTS.md`.
5. **Zero dostępu do zbioru zewnętrznego.** Worker liczy tylko walidację
   i test same-dist. Zbiór zewnętrzny to osobny krok na końcu (T18).

### Kolejność uruchamiania

```bash
# 1. kalibracja - JEDEN bieg, mierzy przepustowosc i liczbe epok
sbatch --gres=gpu:1 --time=04:00:00 run_calibration.slurm

# 2. przeliczyc plan zmierzona liczba
py -3.11 tools/estimate_time.py --throughput <zmierzone>

# 3. proxy EL2N - 3 ziarna (poziom A)
sbatch run_proxy.slurm

# 4. wygenerowac zbiory (bez GPU, na login node)
py -3.11 tools/build_run_manifest.py --tier A --out ~/svit/queue.jsonl
python selection/select_subset.py --queue ~/svit/queue.jsonl

# 5. poziom A - lancuch zadan po jednym wezle
bash submit_chain.sh --queue ~/svit/queue.jsonl --jobs 3

# 6. agregacja, wybor konfiguracji do poziomu B
python eval/aggregate.py

# 7. poziom B + przeniesienie architektur
py -3.11 tools/build_run_manifest.py --tier B --out ~/svit/queue_b.jsonl
bash submit_chain.sh --queue ~/svit/queue_b.jsonl --jobs 2

# 8. JEDNO przejscie po zbiorze zewnetrznym
python eval/external_once.py --checkpoints <wybrane na walidacji>
```

---

## Część D. Zabezpieczenie przed odpowiedzią recenzentów

Trzy rzeczy, których nie da się dorobić po fakcie — muszą być od pierwszego dnia:

1. **`external_access_ledger.jsonl`** — każdy dostęp do zbioru zewnętrznego
   z datą i powodem. Bez tego zdanie „strict zero-look protocol" jest nie do
   obronienia, a recenzent 1 zarzut 10 uderzył dokładnie w to.
2. **`run_manifest.json` przy każdym biegu** — hash commita, hash `protocol.yaml`,
   wersje bibliotek, hash listy wybranych plików. Poprzednio nie dało się
   ustalić, czy wariant V4 to „bez klastrowania" czy „tylko jakość", i sprawa
   wisiała miesiącami.
3. **Surowe wartości per ziarno w załączniku** — nie tylko średnie. Recenzent,
   który chce sprawdzić rozrzut, ma je dostać bez pytania.

Ścieżka do artykułu i recenzji:
`F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\` — recenzje w podkatalogu
`odpowiedziRecenzentów\`.
