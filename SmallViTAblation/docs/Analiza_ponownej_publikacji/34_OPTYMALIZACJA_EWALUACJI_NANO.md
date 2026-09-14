# Optymalizacja końcowej ewaluacji RF-DETR Nano — 2026-09-14

## Problem i zmiana

W zadaniu 1807600 trzy modele ukończyły 30 epok w około 44–45 minut. Końcowa, niezależna ocena ostatniego checkpointu trwała następnie ponad dwie godziny. Procesy ewaluacji były aktywne na CPU, miały aktualne heartbeaty i zapisane predykcje całej walidacji; GPU w chwili sprawdzenia miały zerowe wykorzystanie.

W `pipeline/eval/common.py` funkcja `f1_curve` dla każdego z około 200 progów ponownie uruchamiała greedy matching na rosnącym prefiksie predykcji. Oznaczało to wielokrotne grupowanie, sortowanie i obliczanie IoU tych samych bbox. Koszt tej pętli odtworzono na danych syntetycznych i rzeczywistych predykcjach.

Nowa wersja przypisuje TP/FP raz, w stabilnej kolejności malejącej ufności, a wyniki dla progów odczytuje z sum skumulowanych. Dodanie predykcji o niższym score nie zmienia wcześniejszych decyzji greedy matching. Remisy zachowują kolejność wejściową; wszystkie predykcje z dokładnie tym samym score są uwzględniane przez regułę `score >= threshold`.

IoU obliczamy wektorowo, w blokach maksymalnie 256 predykcji i z ograniczeniem liczby par predykcja–GT w bloku. Po dopasowaniu wszystkich GT pozostałe detekcje są FP, więc kończymy obliczenia dopasowania tego obrazu. Ten sam helper obsługuje pojedynczy próg w `match_detections`. Tablica ujemnych score dla `searchsorted` jest tworzona raz.

W `pipeline/eval/evaluate.py` dodano pomiary etapów:

- wczytanie predykcji;
- standardowe COCO mAP;
- wybór progu operacyjnego;
- końcowe TP/FP/FN.

Czasy trafiają do `metrics_<set>.json` jako `metric_phase_seconds`. Plik `evaluation_progress_<set>.json` pokazuje bieżący etap, zakończone pomiary oraz stan `running`, `failed` lub `completed`. Komunikaty etapów mają `flush=True`, aby były widoczne w przekierowanych logach procesów roboczych.

## Zachowane reguły oceny

COCO AP i AR nadal oblicza ta sama implementacja pycocotools, z domyślnym limitem COCO `maxDets=100`. Zapis predykcji, oceniane identyfikatory obrazów i remap kategorii pozostają zgodne z dotychczasowym kontraktem. Krzywa F1 korzysta z tej samej puli kandydatów progów i tej samej reguły rozstrzygania remisów: najniższy próg przy maksymalnym F1.

Progu nadal można nauczyć tylko na walidacji; dla testu wymagane jest jego walidacyjne pochodzenie. Obrazy bez predykcji pozostają częścią mianownika recall. Zliczanie TP/FP/FN zachowuje wcześniejszy kontrakt wszystkich detekcji powyżej progu, niezależny od limitu `maxDets` stosowanego przez COCO AP.

## Pomiary i kontrola zgodności

Nie dodano plików testów. Wykonano jednorazowe porównanie funkcji starej wersji z Git i nowej implementacji oraz pomiar na istniejących predykcjach, bez treningu i inferencji CUDA.

Na 100 wygenerowanych przypadkach obejmujących różne klasy, puste GT, duplikaty detekcji, remisy score i graniczne IoU uzyskano identyczne całe krzywe F1 oraz rezultaty `match_detections`, w tym listy `tp_scores`. Sprawdzono także ocenę dwóch obrazów z predykcją tylko dla jednego: TP=1, FP=0, FN=1, COCO AP50=51/101. Nowy raport etapów zakończył się statusem `completed`, a dopasowanie progu na test zostało odrzucone zgodnie z kontraktem.

| Pomiar | Stara wersja | Nowa wersja |
|---|---:|---:|
| 1200 syntetycznych detekcji, 202 progi | 3,024 s | 0,00403 s |
| 16 rzeczywistych obrazów, 4041 detekcji, 202 progi | 11,645 s | 0,01093 s |

Dla rzeczywistego wycinka całe krzywe były identyczne; przyspieszenie wyniosło około 1066 razy. To porównanie kosztu krzywej F1 na tym wycinku, a nie całego zadania treningowego.

Pełna walidacja seed 42, checkpoint `last.pth`: **12043 obrazy, 3037819 detekcji**. Pierwszy pomiar izolowanej nowej wersji:

- wczytanie danych i remap kategorii: 17,824 s;
- indeks GT, kandydaci progów i krzywa F1: 10,042 s;
- końcowe zliczenia: 0,645 s;
- standardowe COCO mAP: 53,805 s.

Te etapy łącznie trwały około 82 sekund na jednym CPU w przydziale istniejącego zadania. Pomiar obejmuje obliczenia z gotowego pliku predykcji; czas inferencji i zapisu JSON mierzy się oddzielnie. Końcowe ponowienie F1 po dodaniu cache tablicy score jest zapisane w polu `final_cached_score_version` raportu poniżej.

Izolowana ocena ostatnich zwykłych wag seed 42 dała AP50=0,880463 oraz AP50:95=0,637885. Są to wyniki tego checkpointu, nie najlepszego EMA wybranego podczas treningu. Raport benchmarku jest odrębnym artefaktem pomiarowym; końcowy raport kalibracji powstaje dopiero po ocenie last i best przez jej wrapper.

Dowód: [benchmark_seed42.json](evidence_eval_optimization_2026-09-14/benchmark_seed42.json).

## Wdrożenie i dalszy pomiar

Zmiany są przygotowane lokalnie dla następnego zamrożonego wdrożenia. Aktywna kalibracja 1807600 działa na swoim źródłowym commicie `ef2d5348`; nie podmieniano jej kodu ani nie przerywano procesów. Pomiary wykonano na osobnej kopii `common.py` w `~/smallvit_runs/2026-09-14_eval_optimization`, przez CPU-only krok `srun --jobid=1807600 --overlap --ntasks=1 --cpus-per-task=1 --gres=none`. Nie rezerwowano dodatkowego węzła ani GPU.

Przed kolejnym zgłoszeniem należy wdrożyć nowy commit jako osobny katalog źródeł i zachować jego SHA w manifeście przebiegu. Raport etapów pozwala osobno ocenić koszt inferencji, standardowych metryk COCO i doboru progu. Następny potencjalny obszar optymalizacji to batchowanie inferencji; wymaga osobnego pomiaru zgodności predykcji. W tej zmianie nie włączano AMP ani optymalizowanego dekodowania w niezależnej inferencji.

Bezpośredni podgląd stosu aktywnego procesu przez izolowany profiler był niedostępny z powodu uprawnień `ptrace` klastra. Zależności środowiska treningowego nie były zmieniane; wyniki i czasy powyżej pochodzą z odtworzenia obliczeń na zapisanych predykcjach.
