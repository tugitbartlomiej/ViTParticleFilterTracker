# Status infrastruktury

2026-10-02, Europe/Warsaw. Stary job1819574 jest CANCELLED, bez wznowienia. Nowy infra-only job1819903 potwierdzono RUNNING na Hopper-2. Request11 CUDA urządzeń oznacza4pełneH200+7MIG, nie11pełnychGPU.

Kod infra i launcher przeszedł py_compile, bash-n oraz SHA przed sbatch. Wynik komunikacji CUDA/NCCL jeszcze nieznany przy tym zapisie. Właściwy NNfit w nowym eksperymencie jeszcze nie został zgłoszony ani uruchomiony. Nie ma nowych metryk prognozy.

## Wynik pierwszej próby11

Job1819903: **FAILED, exit1, 18 sekund**, zero kroków fitu. Slurm przydzielił żądane zasoby i ustawił `CUDA_VISIBLE_DEVICES=0,1,...,10`. Jednak każdy zapisany rank zgłosił `torch.cuda.device_count()==8`, więc jawna kontrola11 urządzeń zakończyła próbę przed utworzeniem komunikatora NCCL. Błąd: `AssertionError: CUDA sees8, requested11 devices`. Nie jest to wynik jakości NN ani dowód błędu modelu. Per-rankJSON i stdout/stderr zachowano w `infra11_1819903/`.

Pierwsza ujawniona przeszkoda dotyczy enumeracji/mapowania pełnych GPU i MIG przy integer-valued CUDA_VISIBLE_DEVICES. Ewentualna następna izolowana próba mogłaby wyznaczyć odrębny przydzielony UUID na rank i używać localdevice0; nie została wykonana w tym zapisie. Dodatkowo NCCL2.27.5 nie ma dokumentowanego eksperymentalnego wsparcia MIG dostępnego od2.31, więc naprawienie enumeracji nie gwarantuje udanego all-reduce. Nie zmieniono globalnego/runtime produkcyjnego pakietów. Pełny fit11 ani automatyczny fallback4 nie zostały zgłoszone.


## Potwierdzona komunikacja i model na 11 urządzeniach

Job 1819908 poprawnie wystawił każdemu rankowi jeden przydzielony UUID, lecz NCCL 2.27.5 odmówił komunikacji dwóch MIG na jednym fizycznym GPU (Duplicate GPU). Nie było kroków fitu. Biblioteka NCCL 2.32.3 została zainstalowana osobno w katalogu runu, bez zmiany sharedvenv. Driver 595.71.05 / CUDA 13.2; NCCL_MNNVL_ENABLE=0.

Job 1819911: COMPLETED, exit 0; wszystkie 11 ranków PASS, actual ncclGetVersion=23203, SUM=66. Job 1819912: COMPLETED, exit 0, rzeczywiste kroki GPU na syntetycznym batchu 32 i 28 obrazów, po 600 tokenów, pełne 768 współrzędnych, horyzont 10. Porównanie 1 GPU z 11: max różnica gradientu 1.02e-8, wag po Adam 3.49e-6, decoded forecast 1.31e-6; telemetry/checkpoint PASS. Nie jest to fit 1500 ani wynik prognozy na danych źródłowych.

Czasy CUDA-sync: pierwszy batch 8.130 s / 0.734 s, drugi 0.916 s / 0.225 s (1 GPU / 11 urządzeń). To zimne/wczesne kroki, nie ETA epoki. Szczegóły: [RESULT](gpumodel11_1819912/model/RESULT.json), [runtime proof](PROOF_GPU_RUNTIME.json).


## Uruchomienie pełnego wariantu

Slurm1819918 przyjęty i RUNNING na Hopper-2. Źródłowy model i config zachowują finalne hashe. Scratch wybrany przez realny mkdir/writeprobe: `/dev/shm/bpiotrowski_nndmd_expanded_1819918`, full101, tmpfs; wolne1,050,485,868KB, grant384GB. `/raid` odrzucony jako niewritable. Stare zadanie1819574 pozostaje CANCELLED; nie wznowiono go.

Docelowa powtórka CPU core/driver/Gloo miała16 zaliczonych punktów, następnie1vs4 przeszedł do1vs11; ostatnia redundantna próba została przerwana wyłącznie we własnych PID według decyzji roota z powodu wolnego loginu. Status INTERRUPTED, nie PASS. Zachowano TEST_TARGET_RUNTIME_INTERRUPTED.log. Obowiązujące dowody:19 lokalnych testów CPU PASS, dokładny GPU1vs11 PASS, SHA22/22, audyt PASS. Model nie zmienił się pomiędzy dowodami a pełnym runem.

Bieżący status w [submit_status.json](submit_status.json). Etap source_prepare, zero kroków fitu; źródłowy reader i rzeczywisty batchguard muszą przejść przed fitem. To potwierdzone uruchomienie zadania, nie ukończony eksperyment ani potwierdzona trafność.


## Potwierdzony pierwszy fit

Actual source readerpreflight PASS:11 ranków,1500 obrazów,600×768,48.97s; zgodność ze źródłem maxabs2.62e-6. Mediana publikacji/hash pojedynczego checkpointu2.87s. Actualdataset modelguard1vs11/batch32+28/all600 PASS,2 kroki preflight odrębne; model i seed zresetowane. Następnie właściwy innerfit rzeczywiście wykonał 223 kroków do lokalnego snapshotu 2026-10-02T19:09:31.260043+00:00. Etap inner_LRG_0, epoka 1, origin 54. Mediana warmed optimizerstep z bieżącej próbki: 0.1620s. To timing próbki fitu, nie ETA całego eksperymentu. Wynik walidacji/testu pozostaje nieznany.

Małe dowody w [run1819918](run1819918/status.json): status, TRAINING.jsonl, GPU_TRAINING_SMOKE, EXECUTION_PROVENANCE i NORMALIZATION_PROVENANCE.

## Wynik końcowy: 2 października 2026

Slurm 1819918 zakończony COMPLETED, exit 0, po 1:38:22; zakończenie około 22:43 czasu polskiego. Wykonano 7379 aktualizacji uczenia, w tym 1457 końcowego dopasowania; dwie aktualizacje preflight pozostają osobne. Wybrano learning rate operatora 0.0001 i jedną epokę na podstawie walidacji 80–90. Oba warianty zakończyły inner fit po trzech epokach.

Zamrożony test 90–100 na tych samych 1500 obrazach: zbiorcze RMSE pełnych 768 kanałów 0.159935 wobec persistence 0.139319, czyli 14.80% większy błąd. Dla wybranych 128 kanałów: 0.141612 wobec 0.122625, czyli 15.48% większy błąd. Model przegrywa z persistence w każdym z dziesięciu horyzontów obu grup. Dla samego checkpointu 100: pełne 768 kanałów 0.191773 wobec 0.163563; wybrane 128 kanałów 0.171881 wobec 0.143148. Ten wariant nie wykazał korzyści prognostycznej.

Historia testowa była wcześniej eksplorowana; wynik nie stanowi globalnie nietkniętego testu niezależnego. Nie dobierano kolejnych parametrów po teście. Monitor pozostaje wyłączony, nie zgłoszono nowego joba.

Pobrano końcowy checkpoint, normalizer, konfigurację i raporty. SHA256 checkpointu jest zgodne z kopią na Edenie: `ec0a9f92c8781685c9ad06f4f6326dbef0de24f20ed4bc058cfb8565f7121f2f`. [Raport wszystkich horyzontów i kontroli](run1819918/RAPORT_KONCOWY.md), [metryki źródłowe](run1819918/METRICS_TEST.json), [odbiór i weryfikacja](run1819918/ODBIOR_KONCOWY.json).
