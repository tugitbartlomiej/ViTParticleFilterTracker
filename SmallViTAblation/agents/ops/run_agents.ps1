# Uruchamia agentow 1-5 przez `opencode run` w katalogu SmallViTAblation.
# Uzycie:
#   pwsh agents/ops/run_agents.ps1 -Agent 1          # jeden agent
#   pwsh agents/ops/run_agents.ps1 -Agent 1,2,3      # lista
#   pwsh agents/ops/run_agents.ps1                   # wszyscy, sekwencyjnie
#   pwsh agents/ops/run_agents.ps1 -Parallel         # wszyscy, rownolegle (osobne okna)
param(
    [int[]]$Agent = 1..5,
    [switch]$Parallel
)

$ErrorActionPreference = 'Stop'
$proj = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

$common = @'
ZASADY BEZWZGLĘDNE:
- Piszesz WYŁĄCZNIE kod (plus wymagany README.md twojego katalogu i requirements.txt, jeśli dodajesz zależności). Żadnych notatek, raportów ani podsumowań poza kodem.
- Nie edytujesz plików poza swoim katalogiem. Wyjątek: configs/protocol.yaml i configs/variants.yaml — wolno tylko DOPISYWAĆ pola.
- configs/protocol.yaml to jedyne źródło prawdy. Zero zaszytych hiperparametrów w kodzie.
- Python 3.11. Stała liczba epok, brak early stopping. Bez early stopping znaczy bez early stopping.
- Kontrakty plików: agents/README.md sekcja 4 — wypisuj pliki DOKŁADNIE w tym formacie.
- Selekcja/trening widzą tylko split=="train". Zbiór zewnętrzny: zakazany, jeśli twój plik zadaniowy nie mówi inaczej.
- Jeśli plik wejściowy kontraktu (np. pool_manifest.jsonl) jeszcze nie istnieje, generujesz w swoim katalogu małą syntetyczną fixture do testu dymnego, oznaczoną w kodzie jako fixture — nigdy nie udaje prawdziwych danych w ścieżkach produkcyjnych.
- Test końcowy: end-to-end na RF-DETR-Nano @384, 2500 obrazów, 5 epok (jeśli twoje zadanie obejmuje GPU). Zadanie jest skończone dopiero, gdy test przechodzi.
Kolejność pracy: (1) przeczytaj agents/README.md, (2) przeczytaj swój plik AGENT_*.md, (3) przeczytaj configs/protocol.yaml i configs/variants.yaml, (4) zaplanuj, (5) pisz kod, (6) uruchom test dymny, (7) zapisz README.md w swoim katalogu: jak uruchomić i co wychodzi.
'@

$prompts = @{
    1 = @"
Jesteś Agent 1 (cechy i selekcja). Plik zadaniowy: agents/AGENT_1_CECHY_I_SELEKCJA.md.
Twoje katalogi: pipeline/features/ i pipeline/selection/.
Dostarczasz kod:
- pipeline/features/build_cache.py (DINO, FastSAM, Fourier; wznawialne; spójna kolejność frame_id; feature_cache/*.pkl + manifest.json)
- pipeline/features/train_el2n_proxy.py (RF-DETR-Small, el2n.proxy_epochs z protocol.yaml, tylko train, jeden proxy na ziarno, feature_cache/el2n_proxy_<seed>.pth)
- pipeline/features/el2n_from_proxy.py (inferencja po całej puli train, normalizacja 0-1, feature_cache/el2n_<seed>.pkl)
- pipeline/selection/select_subset.py (jeden skrypt na wszystkie 15 kombinacji; wejście queue.jsonl + cache + protocol.yaml; wyjście variants/<run_id>.json wg README 4.2; standaryzacja per grupa, PCA dino_pca_dim, wagi z queue.jsonl, kmeans z seed, deterministyczna reguła pustego klastra opisana w README)
- pipeline/selection/baselines.py (B-RANDAUG, B-RANDSRC z rozpoznaniem _aug_N vs pule wideo, B-UNIFORM, B-FULLPOOL; log którą ścieżką poszedł)
- pipeline/selection/materialize.py (variants/<run_id>.json -> annotations_train.json COCO; nigdy nie kopiuje obrazów)
Możesz wykorzystać istniejące ekstraktory z AdvancedDatasetSelection/feature_extractors/, ale cache musi mieć nowy jednolity format.
Test końcowy: punkty 1-5 z pliku zadaniowego (build_cache 2500 klatek, M-DESF i M-D po 500, powtarzalność ziarna, 4 baseline'y, COCO wczytywane przez pycocotools).

$common
"@
    2 = @"
Jesteś Agent 2 (trening i kolejka). Plik zadaniowy: agents/AGENT_2_TRENING_I_RUNNER.md.
Twoje katalogi: pipeline/train/ i eden/.
PRZED pisaniem czegokolwiek pod SLURM przeczytaj Eden/EDEN_CONSTRAINTS.md w korzeniu repo (F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker).
Dostarczasz kod:
- pipeline/train/train_rfdetr.py (CLI: --run-id --variant-file --protocol --data-root --out; rfdetr Roboflow; wszystkie hiperparametry z protocol.yaml sekcja training; zero early stopping; start zawsze z wag COCO; zero augmentacji w treningu; zapis last.pth + best.pth wg checkpoint_selection.best_val_criterion; seed w torch/numpy/random; run_manifest.json wg README 4.4 z hashem commita, sha256 protocol.yaml i pliku wariantu, wersjami bibliotek, GPU, czasami; kasowanie checkpointów pośrednich; BEZ metryk i BEZ zbioru zewnętrznego)
- eden/queue.py (kolejka plikowa: pending/running/done/failed; atomowe pobranie pod blokadą pliku; timeout running -> pending; wznawialna; błąd jednego biegu nie zatrzymuje reszty)
- eden/runner.py (--queue --data --worker --protocol; pętla: pobierz, trenuj, oznacz, następny; jeden proces na kartę)
- eden/run_node.slurm (partycja hopper, cały węzeł, 8 pracowników; debug ZABLOKOWANE; dane na /raid lub /dev/shm, nigdy /tmp; rozpakowanie RAZ na węzeł; trap cleanup EXIT INT TERM)
- eden/submit_chain.sh (--dependency=afterany)
tools/build_run_manifest.py już działa — nie pisz go od nowa, używaj.
Test końcowy: 4 punkty z pliku zadaniowego, w tym ubicie pracownika w połowie i bieg celowo failing.

$common
"@
    3 = @"
Jesteś Agent 3 (ewaluacja). Plik zadaniowy: agents/AGENT_3_EWALUACJA.md.
Twój katalog: pipeline/eval/.
Dostarczasz kod:
- pipeline/eval/evaluate.py (checkpoint + COCO + katalog obrazów + protocol.yaml -> metrics.json wg README 4.5 + predykcje COCO na dysku; inferencja standardowa, ŻADNEGO wybierania pojedynczego zapytania; próg generowania predykcji evaluation.map_conf_threshold=0.001; mAP przez pycocotools: mAP@0.5, mAP@[0.5:0.95], mAP@0.75, AR@100; GT przycinane do obrazów faktycznie ocenionych)
- pipeline/eval/operating_point.py (próg max-F1 WYŁĄCZNIE na val; krzywa F1 vs próg per zbiór; test przeniesienia progu na external — wszystko z zapisanych predykcji, bez ponownej inferencji)
- pipeline/eval/aggregate.py (średnia±std po ziarnach; surowe wartości per ziarno osobny plik; dwie kolumny checkpointu last_epoch i best_val OBOK SIEBIE, nigdy zmieszane; eksport LaTeX; bootstrap CI tylko przy >=5 ziarnach)
- pipeline/eval/external_once.py (przyjmuje listę checkpointów już wybranych na val; liczy raz; dopisuje wiersz do external_access_ledger.jsonl: data, kto, checkpoint, cel; ODMAWIA gdy wiele checkpointów jednego biegu = przeszukiwanie epok)
Komplet metryk w każdej tabeli: mAP@0.5, mAP@[0.5:0.95], mAP@0.75, AR@100, TP, FP, FN, precision, recall, F1, n_images, n_annotations + liczba pacjentów PRZED liczbą obrazów.
Test końcowy: 4 punkty z pliku zadaniowego (evaluate 200 obrazów, operating_point z predykcji, aggregate na 3 sztucznych biegach, external_once + odmowa przy 10 checkpointach).

$common
"@
    4 = @"
Jesteś Agent 4 (przeniesienie architektur). Plik zadaniowy: agents/AGENT_4_ARCHITEKTURY.md.
Twój katalog: pipeline/transfer/.
Dostarczasz kod:
- pipeline/transfer/train_transfer.py (D-FINE-S z github.com/Peterande/D-FINE oraz bieżąca YOLO — wersja aktualna na 2026, NIE YOLOv8, wybór i uzasadnienie zapisz w README.md katalogu; iso-resource OBOWIĄZKOWE: ta sama rozdzielczość z protocol.yaml, ten sam plik wariantu co do klatki, ten sam budżet epok bez early stopping, ten sam podział, to samo ziarno, ta sama ścieżka ewaluacji; różnice nie-do-zrównania — parametry, optymalizator, assignment — wypisane w README jako materiał do tabeli w artykule)
- analiza NMS dla YOLO: ustawienia (IoU, conf) zapisane do run_manifest.json; wrażliwość kilku progów IoU -> precyzja/czułość; jeśli NMS nie ma znaczenia — pokazujesz liczby, nie pomijasz
- wyniki w DOKŁADNIE formacie metrics.json z README 4.5; pole arch w run_manifest.json rozróżnia architektury
Test końcowy: 4 punkty z pliku zadaniowego, w tym porównanie run_manifestów: rozdzielczość, epoki, ziarno, plik wariantu identyczne z modelem głównym.

$common
"@
    5 = @"
Jesteś Agent 5 (analizy bez GPU i kontrola). Plik zadaniowy: agents/AGENT_5_ANALIZY_I_KONTROLA.md.
Twój katalog: analysis/.
Dostarczasz kod:
- analysis/coverage_external.py [NAJWAŻNIEJSZE] (osadzenia DINO obrazów zewnętrznych i każdego podzbioru; pokrycie: średnia odległość każdego obrazu zewnętrznego do najbliższego treningowego + rozkład; korelacja miary z wynikiem na external; liczysz OSADZENIA, nie wyniki modelu — dlaczego to nie łamie zero-look, zapisz w README)
- analysis/augmentation_mix.py (udział typów augmentacji w każdym wariancie vs w puli; czysta analiza nazw plików)
- analysis/el2n_rank_correlation.py (EL2N z modelu YOLO wg planu AdvancedDatasetSelection/ablation/yolo/PLAN.md; korelacja kolejności rankingów DETR vs YOLO)
- analysis/audit_pool.py (rozszerzenie tools/audit_leakage.py: pacjent, klatka źródłowa, plik + czwarty poziom: podobieństwo obrazu między zbiorami; każdy nowy zbiór przechodzi audyt PRZED wejściem do puli)
- analysis/checklist.py [KONTROLA KOŃCOWA] (wejście docs/04_REVIEWER_MAP.md; raport zamknięte/w toku/brak per zarzut; konkretne sprawdzenia: >=3 ziarna na wariant, komplet 15 kombinacji, wyniki wszystkich perturbacji S-*, istnienie i sensowność external_access_ledger.jsonl, spójność pól metrics.json; uruchamiane przed każdym zgłoszeniem)
- analysis/dataset_table.py (per split: pacjenci, obrazy, klatki źródłowe, instancje; pacjenci PIERWSI; odtworzyć liczby z tabeli w pliku zadaniowym)
Test końcowy: każdy skrypt działa na obecnych danych i daje sensowny wynik; dla checklist.py część pozycji "brak" jest wynikiem POPRAWNYM.

$common
"@
}

if ($Parallel) {
    foreach ($a in $Agent) {
        $tmp = Join-Path $env:TEMP "opencode\agent$a`_prompt.txt"
        Set-Content -LiteralPath $tmp -Value $prompts[$a] -Encoding utf8
        Start-Process pwsh -ArgumentList '-NoExit','-Command',
            "Set-Location -LiteralPath '$proj'; opencode run (Get-Content -LiteralPath '$tmp' -Raw)"
    }
} else {
    foreach ($a in $Agent) {
        $tmp = Join-Path $env:TEMP "opencode\agent$a`_prompt.txt"
        Set-Content -LiteralPath $tmp -Value $prompts[$a] -Encoding utf8
        Write-Host "=== Agent $a ==="
        opencode run (Get-Content -LiteralPath $tmp -Raw)
    }
}
