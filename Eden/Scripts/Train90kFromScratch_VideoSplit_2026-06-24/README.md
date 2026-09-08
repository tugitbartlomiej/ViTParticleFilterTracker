# Trening DETR + YOLO OD ZERA na PELNYM poolu 91k pod VIDEO-LEVEL splitem

**Utworzono:** 2026-06-24
**Cel:** wytrenowac DETR i YOLO od epoki 0 na CALYM (uncurated) poolu ~91k, ale z
poprawnym podzialem **per-wideo** (35/8/7, seed 42) zamiast losowego frame-level.
To jest **leakage-free odpowiednik UNCURATED baseline z artykulu** (te liczby, ktore
dzis stoja na frame-level splicie: same-dist 74.7% / 64.5%, uncurated external 3.56% /
351 TP). Domyka zarzut recenzenta o video/patient leakage dla galezi "pelny pool".

Para do biezacego eksperymentu: `Train20kFromScratch_2026-06-15` trenuje KUROWANY 20k
pod tym samym video-level splitem (job 1715539, dgx-1, od 2026-06-23). Tu robimy
UNCURATED pelny pool pod tym samym splitem -> czysta para curated-vs-uncurated.

> **AKTUALNY GLOWNY FOCUS (od 2026-06-25): wariant SINGLE-QUERY (`num_queries=1`).**
> Od teraz koncentrujemy sie glownie na treningu DETR z JEDNYM object query (detekcja
> pojedynczego narzedzia), a nie domyslnymi 100. Uzasadnienie danymi: w `annotations_train.json`
> **99.1% klatek ma dokladnie 1 narzedzie** (0.9% ma 2, max 3), wiec `num_queries=1` obcina
> sufit recall tylko o ~0.9 pp. Skrypt: `detr/run_detr_90k_scratch_singlequery_chain.slurm`.
> Pelne szczegoly + status jobow nizej w sekcji "Wariant SINGLE-QUERY".

---

## Co znaczy "90k z video split" (wazne rozroznienie)

Pool augmentowany ma **91 336** klatek z 50 wideo CATARACTS (zweryfikowane na Edenie
2026-06-24). Przy podziale per-wideo:
- **train = WSZYSTKIE klatki z 35 wideo treningowych (71 316), BEZ kuracji do 20k**
- **val   = wszystkie klatki z 8 wideo val (10 780)**
- **test  = wszystkie klatki z 7 wideo test (9 240)**

(val=10 780 i test=9 240 sa bit-identyczne z ablacja v1 -> ten sam split; train to pelen
zbior, ktory ablacja kurowala do 20k.)

Nie da sie miec "91k w train" przy trzymanym tescie — val/test wideo MUSZA wypasc z
train. Czyli "pelny pool pod video split" = train ~71k (uncurated), nie 91k. To wlasnie
odpowiednik uncurated baseline, tylko bez przecieku klatek miedzy splitami.

Roznica vs `Train20kFromScratch_2026-06-15`: tam train byl kurowany do 20k z wideo
treningowych; tu train to PELEN zbior klatek z wideo treningowych (bez selekcji).

---

## JEDEN plik splitu zasila DETR i YOLO

`prepare_fullpool_videosplit.py` buduje z pelnego COCO + istniejacego `video_split.json`
trzy pliki: `annotations_{train,val,test}.json`. Te same JSON-y:
- DETR czyta natywnie (COCO),
- YOLO konwertuje do formatu YOLO **respektujac ten split** (konwersja w yolo slurm
  czyta `annotations_train/val.json`, NIE robi losowego shuffle 90/10 jak stary skrypt).

Split = ten sam `video_split.json` (35/8/7, seed 42), co ablacje V1-V7 i biezacy 20k
-> wszystkie eksperymenty sa na identycznym podziale wideo (spojnosc).

---

## Pliki w tym folderze

```
Train90kFromScratch_VideoSplit_2026-06-24/
├── README.md                                  <- ten plik
├── prepare_fullpool_videosplit.py             <- buduje annotations_{train,val,test}.json (login node, bez GPU)
├── detr/
│   ├── run_detr_90k_scratch_videosplit_chain.slurm   <- DETR od zera, 100 query (chain, partycja long)
│   ├── run_detr_90k_scratch_singlequery_chain.slurm  <- DETR od zera, SINGLE QUERY num_queries=1 [GLOWNY FOCUS]
│   └── submit_detr_chain.sh                           <- zglasza lancuch ogniw (dependency); SLURM_FILE=... wybiera wariant
└── yolo/
    └── run_yolo_90k_from_scratch_videosplit.slurm     <- YOLO od zera (long / dgx, A100)
```

---

## Jak uruchomic (Eden)

### 0. Upload folderu na Eden
```bash
scp -r Train90kFromScratch_VideoSplit_2026-06-24 eden-cluster:~/DETR/Scripts/
# (lokalizacja dowolna; sciezki w slurm sa absolutne)
```

### 1. Zbuduj split JSON (login node, ~1-2 min, bez GPU)
```bash
python3 ~/DETR/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/prepare_fullpool_videosplit.py \
  --annotations ~/datasets_20250606_test/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json \
  --split ~/DETR/ablation/video_split.json \
  --output ~/DETR/fullpool_videosplit
# Oczekiwane: train=71316, val=10780, test=9240 ; overlap wideo = 0
```

### 2. DETR od zera (chain na long / dgx, A100)
```bash
cd ~/DETR/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/detr
./submit_detr_chain.sh 4          # 4 ogniwa, 0->200 (EPOCHS=170 ./submit_detr_chain.sh 4 dla budzetu 170)
```

### 3. YOLO od zera (long / dgx, A100)
```bash
cd ~/Yolo/20k_finetune
sbatch ~/DETR/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/yolo/run_yolo_90k_from_scratch_videosplit.slurm
# EPOCHS=190 sbatch ... dla budzetu 190 (jak base YOLO)
```

Oba na dgx/long (A100, najstabilniej). DETR i YOLO to osobne joby — mozna rownolegle.
Razem z biezacym 20k masz wtedy 3 joby na long (QoS long-part ma zapas: cap grupowy
gpu=34, Ty uzyjesz 4+4+4=12).

---

## Wariant SINGLE-QUERY (`num_queries=1`) — GLOWNY FOCUS od 2026-06-25

DETR trenowany od zera, ale z JEDNYM object query zamiast 100. Cel: model wyspecjalizowany
pod detekcje POJEDYNCZEGO narzedzia na klatce — spojne z protokolem Q81 w artykule, gdzie i
tak wybierane jest pojedyncze query.

**Dlaczego to ma sens dla TEGO zbioru (uzasadnienie danymi):**
`annotations_train.json` (fullpool_videosplit): 71 316 obrazow, srednio 1.01 narz/obraz.
- 99.1% klatek = dokladnie 1 narzedzie
- 0.9% (613) = 2 narzedzia; 2 obrazy = 3 (max 3)

`num_queries=1` ogranicza maks. liczbe wykryc do 1/klatke -> traci sufit recall o **~0.9 pp**
(drugie narzedzie w ~615 klatkach). Dla tego zbioru to dopuszczalny, czysty baseline.
(Ostrzezenie "single query zabije recall" jest prawdziwe tylko dla zbioru z wieloma narzedziami
na klatce — tu NIE zachodzi. Pelna analiza: w repo artykulu
`DETR_IEEE/.todo/Todo_2026-06-25_DETR_query_specialization_single_query_train/`.)

**Co rozni skrypt single-query od wersji 100-query (poza tym IDENTYCZNY):**
- `--num_queries 1` (zamiast 100)
- `--job-name=detr_90k_scratch_q1`, logi `logs/detr_90k_scratch_q1_<jobid>.log`
- osobne katalogi `fullpool_vs_scratch_q1_{checkpoints,output,best}` (nie koliduje ze 100-query)
- te same dane `fullpool_videosplit`, te same EPOCHS=200, init COCO, resume on

**Uruchomienie (ten sam wrapper przez zmienna SLURM_FILE):**
```bash
cd ~/DETR/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/detr
SLURM_FILE=run_detr_90k_scratch_singlequery_chain.slurm ./submit_detr_chain.sh 4
```

**Status (2026-06-25):** zgloszony lancuch 4 ogniw, joby 1720677-1720680 (PENDING, EPOCHS=200).
Wczesniejszy wariant 100-query (joby 1717442-1717445) ANULOWANY na rzecz single-query.
Stop calosci: `scancel 1720677 1720678 1720679 1720680`.

**Status (2026-07-03):** lancuch 1720677-1720680 ANULOWANY przez usera 2026-06-26 po ~19 h
(ogniwo 1 doszlo do ~ep6 z 200; checkpoint_dir PUSTY, bo save co 10 epok -> restart = od ep0;
w `fullpool_vs_scratch_q1_best/` zostal best_model.pth z tego biegu — do nadpisania).
Zmierzony koszt na dgx (log 1720677): **~2.5 h/epoka na A100x4** -> 200 epok = ~21 dni = 5-6 ogniw.

**NOWE warianty HOPPER (2026-07-03):** `run_detr_90k_scratch_singlequery_chain_hopper.slurm`
i `run_detr_90k_scratch_videosplit_chain_hopper.slurm` (partycja hopper, H100x4, 5 d/ogniwo,
te same katalogi co warianty dgx -> ogniwa hopper/dgx wymienne).
Szacunek: ~1.3-1.6 h/epoka -> EPOCHS=170 = ~9-11.5 dnia = 3 ogniwa po 5 dni.
Limity partycji (sinfo 2026-07-03): long/hopper/experimental = 5 d, short/hopper-2 = 1 d
(hopper-2 dodatkowo TmpDisk=0 -> odpada dla 91k pool). QoS long-part: grupowy cap gpu=34.

**ZMIANA DECYZJI (2026-07-03): wracamy do num_queries=100.** Lancuch q1 na hopperze
(1729463-65) zgloszony i ANULOWANY tego samego dnia przed startem. Powody powrotu do 100q:
1) porownywalnosc — retrening ma bronic artykulu przed zarzutem leakage, wiec architektura
   musi byc identyczna z baseline (standard DETR 100 queries + protokol Q81 = wybor
   najlepszego query ze 100; przy num_queries=1 protokol Q81 traci sens),
2) single-query zmienialby dwie zmienne naraz (split + architektura),
3) literatura: queries specjalizuja sie przestrzennie (DETR paper Fig. 7), dokumentacja HF
   zaleca zapas queries ponad max liczbe obiektow; brak opublikowanych dowodow, ze DETR
   z num_queries=1 trenuje sie do dobrej jakosci (nasze 19 h q1 bylo stabilne, ale jakosci
   koncowej nikt nie potwierdzil). Single-query = OSOBNY eksperyment (zob. .todo artykulu).

**AKTYWNE JOBY (2026-07-03):** DETR 100q chain hopper: **1729469 -> 1729470 -> 1729471**
(EPOCHS=170, afterany). YOLO 90k: **1729466** (long/dgx, EPOCHS=190, single 5d).
Stop: `scancel 1729469 1729470 1729471 1729466`.
Uruchomienie od nowa:
```bash
cd ~/DETR/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/detr
EPOCHS=170 SLURM_FILE=run_detr_90k_scratch_videosplit_chain_hopper.slurm ./submit_detr_chain.sh 3
```

**Do pilnowania:** czy ogniwo 1 startuje bez bledu (edge-case matcher/loss przy N=1) —
sprawdzic pierwsze minuty logu po starcie.

**Czystsza alternatywa na pozniej (gdyby liczyl sie ten ~0.9 pp):** wariant A = wymuszone
dopasowanie w Hungarian matcherze (wszystkie GT do jednej query) — zachowuje multi-instance
recall i daje specjalizacje bez utraty pojemnosci. Opisany w TODO artykulu.

---

## Parametry (mirror base-treningu, nie fine-tune)

| | DETR | YOLO |
|---|---|---|
| Entry | `detr_train_ablation.py` | `yolo_train_20k_finetune_v2.py` |
| Init | facebook/detr-resnet-50 (COCO) | yolov8l.pt (COCO) |
| Epochs (default) | 0 -> 200 | 0 -> 200 |
| Partycja | long / dgx, A100 (chain, 4d/ogniwo) | long / dgx, A100 (single, 5d) |
| GPU | 4 (a100) | 4 (a100) |
| Batch | 8/GPU (eff 32) | 32 |
| LR | 1e-4 / backbone 1e-5, cosine, warmup 5 | lr0=0.01, lrf=0.01 (SGD) |
| Patience | 60 | 100 |
| Save co | 10 epok | 10 epok |

LR = wartosci BASE (nie zredukowane fine-tune), bo to trening od zera. Budzet epok
domyslnie 200; punkt porownania z artykulem to ~ep160-170 (base DETR robil 0-170,
base YOLO 0-190). Nadpisz przez `EPOCHS=`.

---

## Po treningu — ewaluacja (do liczb w artykule)

1. **Same-distribution LEAKAGE-FREE:** ewaluacja na `annotations_test.json` (7 wideo,
   ~9240 klatek, w pelni odlozonych per-wideo). To jest czysta liczba same-dist,
   ktora moze zastapic / uzupelnic Tab. 5 (obecnie frame-level, mozliwy optymizm).
2. **Cross-dataset (external):** ten sam protokol Q81 (DETR) / standardowy (YOLO) na
   external benchmark — infra `AdvancedDatasetSelection/ablation/detr/local_eval_2026-06-11`.
   Daje uncurated external pod video splitem (porownanie z 3.56% / 351 TP z artykulu).

### Macierz porownan (jak to wpada w artykul)

| Galaz | Split | Status |
|---|---|---|
| Uncurated 91k (Tab. 5 / 3.56% / 351 TP) | frame-level (LEAKY) | w artykule dzis |
| **Uncurated 91k (TEN eksperyment)** | **video-level (czysty)** | tu — do policzenia |
| Curated 20k (26.04% / 34.17%) | frame-level (produkcja) | w artykule dzis |
| Curated 20k from-scratch | video-level (czysty) | job 1715539, biezacy |

Dwie dolne galezie (video-level) daja czysta pare curated-vs-uncurated; jesli kierunek
wnioskow sie utrzyma pod poprawnym splitem -> mocna obrona artykulu.

---

## Uwagi / pulapki

- **train ~71k jest ~3.5x wiekszy niz 20k** -> epoki ~3.5x dluzsze (~2-2.5 h/epoka na
  A100x4). DETR: chain 5-dniowych ogniw (przechodzi przez limit czasu). YOLO jest
  szybki -> mieści sie w JEDNYM 5-dniowym jobie (szac. ~1-2 dni); jesli nie zdazy,
  wznow recznie Ultralytics `resume=True` z `last.pt` (w train_out_scratch_90k_vs/exp/weights/).
- **dgx / A100:** `/raid` NIE jest zapisywalny -> TMPDIR laduje na `/tmp` (md0 1.8T) lub
  `/dev/shm`. Oba slurmy maja odporny wybor TMPDIR. (Inaczej niz hopper-2, gdzie TmpDisk=0.)
- **Init COCO (nie truly-random)** — taki sam jak base, zeby izolowac tylko zmienna
  "video-level split na pelnym poolu". Truly-random init = osobna decyzja (zob. uwagi w
  `Train20kFromScratch_2026-06-15/README.md`).
- **DETR checkpointy** pruning: kamienie milowe co 50 + 5 najnowszych (home ma ~60G wolne).
- **video_split.json** jest REUZYWANY z `~/DETR/ablation/` (seed 42, 35/8/7) — NIE
  regenerujemy, zeby split byl bit-identyczny z ablacjami i biezacym 20k.
