# NNDMD 1500 obrazów: rozszerzony historyczny stan, Eden

Aktualizacja: 2026-10-02, Europe/Warsaw. Status: przygotowanie; job nie został jeszcze zgłoszony. Nowe polecenie zastępuje lokalny wariant Markowa. Lokalny benchmark został zakończony; lokalnego treningu 1500 nie uruchomiono. Zachowujemy poprzednie artefakty.

## Źródła i granice

Dokładnie 1500 obrazów źródłowego treningu SmallViT, w oryginalnej kolejności COCO. SHA256 COCO `3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75`. Manifest zawiera identyfikatory filmu i klatki, SHA pliku i po preprocessing RGB. Wszystkie lokalne obrazy potwierdzono piksel po pikselu względem źródłowego cache. Eden odczyta istniejące archiwum tar i zweryfikuje te same hashe; bez przesyłania pełnego zbioru lub 279 GB trajektorii.

Reprezentacja każdego obrazu to 600 cząsteczek, każda ma pełne 768 współrzędnych po GELU ostatniej MLP. Bboxy istnieją; segmentacji dla całych 1500 nie ma. Loss obejmuje wszystkie 600 tokenów po równo, a każdy obraz i horyzont ma równą wagę. Bbox nie jest maską segmentacji. Checkpointy 101: aktualizacje 0, 10, ..., 1000. Ruch zależy od wag i statystyk BatchNorm; to nie ruch obrazu w filmie.

## Historyczny stan i telemetria

Model NN otrzymuje pełny latent, dwa historyczne przyrosty, podsumowania aktualizacji parametrów/BN oraz rzeczywistych zapisanych stanów AdamW, flagi dostępności i wiek obserwacji. Wagi co 10 zawierają `net,krok`; 11 pełnych stanów co 100 zawiera `net,opt,krok`. Rzeczywisty `opt` ma `step,exp_avg,exp_avg_sq` dla 89 parametrów. Identyfikatory optymalizatora mapujemy przez zweryfikowaną konstrukcyjną kolejność dwóch grup źródłowego AdamW: parametry ndim>=2, następnie ndim<2. Optymalizator nie zapisał nazw parametrów.

`AUX_HISTORY.npz` ma 101×59 cech, maski wejścia i celu oraz wiek ostatniego pomiaru. RMS zmian wag nie jest gradientem. RMS momentów jest podsumowaniem rzeczywistych AdamW momentów, nie ich pełnym stanem. Wejście między zapisami stosuje wyłącznie ostatni wcześniejszy pomiar; target loss momentów występuje tylko przy rzeczywistym zapisie. Początkowy pusty AdamW state jest brakiem, nie zmierzonym zerem. Brak per-step batch-ID, parametrów augmentacji i gradientów. Nie rekonstruujemy ich fikcyjnie.

Encoder rozszerzonego stanu -> wyuczony stan 768 -> A 768×768 -> nieliniowy decoder pełnego natywnego 768 i podsumowań aux. Używany jest rzeczywisty PyKoopman NNDMD z jawnym adapterem obrazowych paczek/streamingu. Wyuczona baza nie jest oryginalną bazą kanałów. J128 pozostaje dodatkową oceną po dekodowaniu, bez PCA. Nie zakładamy domknięcia całej dynamiki wag/BN/AdamW przez te pomiary.

## Czas i walidacja

Sliding windows 10 przejść checkpointowych. Inner fit cp0..80, origin0..70; walidacja autonomiczna cp80->90. Final refit cp0..90, origin0..80; final test cp90->100 (aktualizacje 900->1000), na tych samych znanych 1500 obrazach. To test przyszłego czasu, nie nowych obrazów. Historia cp origin−2..origin+10 wymaga 13-slotowego cache; ujemne lagowe indeksy clamp cp0 wraz z obowiązkowymi flagami braku. Podczas prognozy nie odczytujemy przyszłych latentów, wag, momentów, klatek ani reszt. Rollout to otwarta ewolucja A^h w wyuczonym stanie, bez ponownego kodowania przyszłej historii. Różnice dekodowanych prognoz są diagnostyką przewidywanego ruchu; przyszły aux jest prognozą modelu, nie telemetrią z checkpointu celu. Cache nie wykonuje przyszłych checkpointów z wyprzedzeniem: inner fit czyta najwyżej cp80, walidacja cp81..90 tylko jako cele, a final test cp91..100 dopiero po freeze. Skalowanie/normalizacja nie używa celów walidacji ani testu.

Batch oznacza 32 OBRAZY: 600 tokenów każdego obrazu, microbatch tokenowy i akumulacja gradientu służą jedynie ograniczeniu pamięci. Ostatnia paczka ma 28 obrazów. Kolejność window-major, pełne przejście wszystkich obrazów dla każdego originu. Przed fitami root ustalił maksymalnie 10 pełnych epok inner fit, patience 2; wybór najlepszej epoki wyłącznie przez full768 RMSE walidacji origin80->90, final refit tyle pełnych epok, ile wybrano wewnętrznie. Wcześniejsza propozycja sztywnych 2 epok została zastąpiona przed uruchomieniem. Bez dodatkowej siatki. Ostateczne LR/config zapisuje właściciel modelu przed jobem. Wybór wyłącznie na walidacji; test po freeze. Baselines persystencja, ostatni przyrost i historyczna średnia/reversion na identycznych współrzędnych i originach. MSE/RMSE pełne 768 i J128, znak skumulowanego ruchu i amplituda osobno. Cel 85% znaku nie jest osiągnięciem ani gwarancją.

## Eden i budżet

SSH `eden-cluster` działa; scheduler Slurm, konto `transformers_vsc`. Istnieją źródłowe checkpointy, tar i COCO. Home ma około 36 GB wolnego, dlatego cache nigdy nie trafia do home. W jobie sprawdzamy rzeczywisty node-local `/raid`; przy >350 GB wybieramy full101-cache około 279 GB, aby uniknąć powtórnych ekstrakcji. Inaczej rolling13 około 35.94 GB na `/raid` lub `/dev/shm` z >50 GB wolnego. Bez subsamplingu. Jeden writer, bez równoległego prefetch/GPU jobów. SourceViT po ekstrakcji offload CPU, statystyki BN zamrożone, eval FP32, TF32 wyłączone.

Proponowany job: 1 GPU A100/H100, 8 CPU, 128 GB RAM, limit 2 dni, partition long/hopper. Rzeczywisty sinfo potwierdził limit 5 dni dla obu partycji, więc 2 dni są dopuszczalne. Nie używamy zakazanej debug. Skript tworzy własny scratch z jobID, sprawdza ścieżkę i sprząta wyłącznie własny scratch po wyjściu. Wyniki, logi i modele pozostają w nowym katalogu eksperymentu. Runtime izolowany: PyKoopman1.2.1, PyDMD0.4.1, Lightning2.0.9.post0, źródłowy Torch2.9.1+cu128; pakiety źródłowego venv pozostają nietknięte.

## Weryfikacja i status

Przed sbatch: audyt kodu i kauzalności, test batch32 obrazów, prefix/poison, wszystkie hashe source/manifest/runtime/config oraz git HEAD i dirty snapshot. Job owner: plan_poprawek. Model owner: pykoopman_obliczenia. Audyt: astra_koopman_wagi. Nie deklarujemy uruchomionego treningu bez numeru sbatch, squeue i pierwszego optimizerstep w logu. Po submit zapis `submit_status.json`; później WYNIKI z rzeczywistymi licznikami i negatywnymi wynikami.


Pełna ocena obejmie wszystkie 1500 obrazów i wszystkie deklarowane horyzonty; zapis prognoz do wizualizacji ograniczony do 12 wcześniej określonych przykładów. Metryki per-image/horizon i parametry modelu/A pozostają kompletne. To ograniczenie zapisu, nie próby oceny.

## Aktualizacja 2026-10-02: dwa GPU w jednym DGX

Obowiązująca decyzja użytkownika zastępuje wcześniejszy pojedynczy GPU: dwa pełne A100 na jednym DGX, bez Pascal i MIG/Hopper-2. Globalna paczka nadal ma 32 obrazy (16 na rank), ostatnia 28 (14 na rank). Sumowanie gradientów uwzględnia globalny mianownik 32×600 albo 28×600; regularizer jest liczony tylko raz. Nie zmieniamy modelu, strat, horyzontów, wyboru epoki ani protokołu czasowego.

Wszyscy rankowie wywołują `ensure_window`. Rank 0 jest jedynym writerem nagłówka NPY, metadanych i końcowego SHA. Każdy rank zapisuje wyłącznie rozłączne obrazy `rank, rank+world_size, ...` we wspólnym mmap. Collective error gates kończą wszystkie ranki przy błędzie dowolnego sharda; timeout process group zabezpiecza śmierć procesu. Status `complete` jest publikowany atomowo dopiero po sukcesie wszystkich 1500 obrazów, zamrożonych bufferów i SHA. Pojemność cache jest wspólna dla node, nie mnożona przez liczbę ranków. SourceViT jest odładowany na CPU przed NN; ekstrakcja i uczenie nie konkurują o GPU.

Zachowane nieudane joby: 1819498 (niewritable `/raid`, zero kroków), 1819551 (po pierwszym rzeczywistym kroku błąd duplicate `origin` w telemetrii). Źródła wykonywane i logi pozostają oddzielnie w `failure_1819551`. Naprawa telemetrii nie zmienia loss/modelu. Nowy pełny run otrzymuje osobny remote katalog i snapshot, bez nadpisywania starego `EXECUTED_SOURCE`.

Job 1819567 zakończył poprawnie test NCCL i pojedynczy serial optimizer step; nie był dowodem równoważności rozproszonego modelu. Job 1819570 na dgx-4 zakończył poprawnie rzeczywisty test 1 vs 2 GPU: globalne paczki 32 i 28, 600 cząsteczek, pełne 768 współrzędnych, zgodność loss/gradientów/Adam/predykcji oraz JSON/checkpoint. Fixture była syntetyczna; `actual_fit_steps=0`. Wyniki i niezmienny snapshot w `run_multigpu/smoke_2gpu`. Czasy CUDA synchronized dwóch wczesnych paczek nie stanowią ETA epoki ani pomiaru ekstrakcji.

Pełny job po audycie wykona dodatkowy guard na finalnym helperze, parallel source window 0..10 i dopiero potem pełny fit 1500. Source preflight oraz NN preflight mają osobne liczniki i nie są krokami fitu. Docelowe zasoby: 2 GPU, 1 node DGX, 8 CPU, 128 GB RAM, 48 godzin. Aktualny launcher `run_multigpu/bieg_eden_2gpu.slurm`; cache poza home, realny write probe i owner marker zgodnie z poprawionym selektorem.
