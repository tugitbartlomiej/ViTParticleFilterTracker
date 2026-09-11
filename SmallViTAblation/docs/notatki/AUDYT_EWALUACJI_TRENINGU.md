# Audyt READ-ONLY ewaluacji i treningu — SmallViTAblation

> Archiwum audytu sprzed refaktoru kodu. Aktualne poprawki i testy: [21_STAN_REFAKTORU.md](../21_STAN_REFAKTORU.md). Numery linii i wyniki reprodukcji odnoszą się do zapisanej wersji źródeł.

Data: 2026-09-11. Zapis materiału z zakończonego audytu, bez zmian repozytorium.

## Zakres i ograniczenia

Korzeń: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`.
Zakres: `pipeline/eval/*.py`, `pipeline/train/train_rfdetr.py`,
`pipeline/transfer/{metrics.py,transfer_common.py,train_transfer.py}` oraz
`configs/protocol.yaml`. Odwołania `plik:linia` poniżej są względem tego korzenia
i dotyczą stanu odczytanego podczas audytu; nie stanowią przypięcia do commita.
Przeczytano główne `AGENTS.md`; w audytowanym poddrzewie nie znaleziono dodatkowych.

Nie edytowano kodu ani istniejących dokumentów, nie uruchamiano treningu, GPU,
inferencji detektorów ani ewaluacji rzeczywistych zbiorów. Nie audytowano literatury
ani metod selekcji danych. Reprodukcje korzystają z syntetycznych danych w pamięci.

**Dowody dotyczące API RF-DETR odnoszą się do upstream 1.2.1. Nie potwierdzono
wersji ani konfiguracji środowiska Eden.** W sprawdzonym lokalnym Pythonie 3.11
RF-DETR i Supervision nie były zainstalowane; dostępny Ultralytics miał wersję
8.3.236. Testy zgodności sygnatur i formatu wyjść były testami atrap, a nie
uruchomieniem RF-DETR. Załączony skrypt nie testuje API bibliotek detektorów.
Nie wykazano, że historyczne wyniki eksperymentów faktycznie zawierają opisane błędy.

P1: blokuje wiarygodną ewaluację lub wykonanie deklarowanego przebiegu.
P2: istotna niespójność albo luka walidacji. „Potwierdzony” oznacza dowód w kodzie
lub reprodukcji; dla ustaleń zależnych od biblioteki podano sprawdzoną wersję.

## Odtwarzalne dowody CPU

- Skrypt: [evidence/reproduce_eval_bugs.py](../pochodne/evidence/reproduce_eval_bugs.py).
- Wyniki: [evidence/eval_repro_results.json](../pochodne/evidence/eval_repro_results.json).
- Pierwsze wykonanie: kod wyjścia 0, **5/5 błędów odtworzonych**.
- Python 3.11; NumPy 2.2.5, PyYAML 6.0.2, pycocotools 2.0.8.
- JSON zawiera dokładną wersję Pythona, fixture każdego testu, `observed`,
  `expected`, uzasadnienie oczekiwania, ograniczenia, logi oraz SHA-256 skryptu
  i czterech źródeł. Hash źródeł przed wykonaniem i po nim jest identyczny.

| Przypadek | Observed | Expected |
|---|---|---|
| mAP missing images | AP≈1, AR=1 | AP≈0,504950495, AR=0,5 |
| Transfer matcher | TP/FP/FN=1/1/1 | TP/FP/FN=2/0/0 |
| Wrong category | Transfer i eval z remapem: 1/0/0 | 0/1/1; brak uprawnionego mapowania klasy 99 na 1 |
| Fallback test threshold | Dobór na teście: próg 0,9, F1=1 | Odmowa doboru bez progu wybranego na walidacji |
| Lexicographic checkpoint sort | `checkpoint_best_total.pth`, epoka 20 | Końcowa epoka 59: `checkpoint.pth` lub `checkpoint0059.pth` |

AP oczekiwane w pierwszym teście pochodzi z niezależnego COCOeval z pełną listą
`imgIds=[1,2]`; wartość 51/101 wynika z próbkowania krzywej PR przez COCO.
Metadane epok w ostatnim teście są jawnie syntetyczne, z indeksowaniem od zera.
Test wykonuje rzeczywiste instrukcje sortowania i wyboru pobrane przez AST
z `run_training`, ale nie ładuje żadnych wag. Test fallbacku wykonuje rzeczywistą
funkcję z atrapą nieistniejącego pliku progu. Pozostałe testy wywołują bezpośrednio
funkcje audytowanych modułów.

Powtórzenie bez zapisu jakiegokolwiek pliku:

```powershell
py -3.11 -B 'F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/evidence/reproduce_eval_bugs.py' --stdout
```

Wymagania: `numpy`, `PyYAML`, `pycocotools`. Nie są potrzebne biblioteki detektorów.
Opcja `--repo-root` pozwala wskazać inne położenie katalogu `SmallViTAblation`.
Bez `--stdout` skrypt tworzy wyłącznie sąsiedni `eval_repro_results.json` i odmawia
nadpisania istniejącego pliku. Kod 0 oznacza odtworzenie wszystkich pięciu błędów,
nie poprawność pipeline; kod 1 oznacza błąd testu lub nieodtworzone ustalenie,
a kod 2 — istniejący plik wynikowy.

## Ustalenia

### 1. P1 — COCO mAP pomija obrazy bez detekcji

**Status: potwierdzony, reprodukcja CPU w załączniku.**

Dowód: `pipeline/eval/common.py:245` ustawia `imgIds` na przecięcie obrazów GT
i predykcji. Obrazy zawierające niewykryte obiekty wypadają z mianownika recall,
nawet jeśli poprawnie wymieniono je w sidecarze ocenionych obrazów.

Repro: dwa obrazy, po jednym GT; jedna idealna detekcja tylko na pierwszym.
Kod: AP≈1 i AR=1; pełny COCOeval: AP≈0,50495 i AR=0,5.
Skutek: zawyżone mAP i AR oraz niespójność z licznikami FN.

Druga ścieżka tego problemu: `pipeline/eval/evaluate.py:177` przy imporcie
predykcji bez `--images-dir`/`--evaluated-images` odtwarza zbiór obrazów wyłącznie
z detekcji. Usuwa obrazy bez predykcji już z GT, psując również FN. Ostrzeżenie
nie zatrzymuje obliczenia niepełnych metryk.

### 2. P1 — transfer źle dopasowuje nakładające się obiekty

**Status: potwierdzony, reprodukcja CPU w załączniku.**

Dowód: `pipeline/transfer/metrics.py:76` wybiera największe IoU przed
wykluczeniem już zajętych GT. Jeśli najlepszy GT jest zajęty, predykcja zostaje FP,
chociaż kolejny wolny GT spełnia próg.

Repro: GT `[0,0,10,10]`, `[3,0,10,10]`; dwie predykcje pierwszego prostokąta,
score 0,9 i 0,8, próg IoU 0,5. Transfer: TP/FP/FN=1/1/1.
Prawidłowe dopasowanie do wolnego GT: 2/0/0, zgodnie z matcherem
`pipeline/eval/common.py`. IoU drugiego dopasowania wynosi 70/130>0,5.
Skutek: zaniżone recall i F1, nierówna ewaluacja architektur.

### 3. P1 — niespójna obsługa klas zalicza błędne klasy jako TP

**Status: potwierdzony, reprodukcja CPU w załączniku.**

Dowód: `pipeline/transfer/metrics.py:144` i `:159` odrzucają `category_id` podczas
parsowania. `pipeline/eval/common.py:211` przepisuje wszystkie klasy detekcji
na jedyną kategorię GT. Repro: idealny bbox, GT klasy 1, predykcja klasy 99.
Oba mechanizmy zaliczają TP zamiast FP+FN.

Skutek: błędna klasyfikacja może zostać ukryta; liczniki transferu mogą być
niespójne z jego COCO mAP. Przesunięcie indeksowania wymaga jawnej mapy klas,
a nie arbitralnego zrównania wszystkich identyfikatorów. To nie jest twierdzenie,
że detektory na Edenie rzeczywiście generowały klasę 99.

Dodatkowo `pipeline/eval/operating_point.py:48` stosuje remap bezwarunkowo,
nawet przy `single_class_remap: false`, podczas gdy `evaluate.py` respektuje flagę.
Dobór progu i raport końcowy mogą zatem opierać się na innych regułach klasyfikacji.

### 4. P1 — fallback dobiera próg na teście; zero-look nie jest egzekwowane

**Status: fallback potwierdzony reprodukcją CPU; brak bramki zero-look potwierdzony analizą kodu.**

Dowód: `pipeline/eval/evaluate.py:285` przy braku `operating_point.json`
maksymalizuje F1 na aktualnym zbiorze. Repro dla `test`: funkcja zwraca
`(0.9, 'max_f1_on_set:test')`. To leakage doboru progu; ostrzeżenie go nie usuwa.
Wymagana reakcja bez uprzednio wybranego progu walidacyjnego: odmowa takiego doboru.

CLI dopuszcza zbiory zewnętrzne (`evaluate.py:378`, `:434`, `:446`), ale nie
sprawdza ledgeru ani ukończenia wyboru checkpointu przed inferencją, wbrew
`configs/protocol.yaml:88`. Nie dowiedziono wykorzystania tej możliwości
w historycznych wynikach; ewentualny zewnętrzny wrapper nie został tu zweryfikowany.

### 5. P1 — artefakty best_val/last_epoch kolidują, składanie wyników może się wywrócić

**Status: potwierdzony analizą kodu; błąd składania odtworzony w pamięci podczas audytu.**

Dowód: `pipeline/eval/evaluate.py:105` nazywa predykcje i sidecary tylko według
zbioru; domyślny katalog w `:418` zależy wyłącznie od `run_id`. Artefakty i próg
nie są związane z hashem checkpointu. Kolejna ewaluacja może nadpisać dane
pierwszej i odziedziczyć jej `operating_point.json`.

Ponadto `evaluate.py:329` dopasowuje `metrics_*.json`, w tym zbiorczy
`metrics__last_epoch.json`, po czym w `:332` oczekuje pola `block`.
Repro z takim plikiem w atrapie katalogu dało `KeyError: 'block'`.
Osobne katalogi ograniczają kolizje, lecz kod ich nie wymusza.

### 6. P1 — niezgodne API RF-DETR: wagi, wejście predict i struktura wyjścia

**Status: potwierdzone względem upstream RF-DETR 1.2.1 i formatu Supervision;
środowisko Eden niepotwierdzone.**

Dowody w projekcie:

- `pipeline/eval/evaluate.py:71` przekazuje `weights` zamiast `pretrain_weights`.
  W sprawdzonym API wskazany checkpoint może zostać zignorowany i pozostają
  domyślne wagi modelu.
- `evaluate.py:85` używa `image_path` zamiast wymaganego `images`.
  Atrapa ze zgodną sygnaturą dała `TypeError` braku argumentu `images`.
- `pipeline/train/train_rfdetr.py:312` interpretuje iterację predykcji jako
  `[x1,y1,x2,y2,score,cls]`; iterator `sv.Detections` zwraca
  `(xyxy, mask, confidence, class_id, tracker_id, data)`. Atrapa tego wyniku
  dała błąd konwersji wieloelementowej tablicy na `float`.
- Po naprawieniu struktury trzeba zweryfikować `cls+1` w `train_rfdetr.py:315`:
  dla etykiet już zgodnych z COCO to błędne przesunięcie.

Skutek: brak możliwości inferencji/walidacji albo ewaluacja innych wag i klas
niż deklarowane. Dowody API: [RF-DETR 1.2.1 — detr.py](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/detr.py),
[config.py](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/config.py),
[iterator Supervision 0.26.1](https://raw.githubusercontent.com/roboflow/supervision/0.26.1/supervision/detection/core.py).

### 7. P1 — last.pth wybierany leksykograficznie, nie według epoki

**Status: potwierdzony, reprodukcja CPU rzeczywistych instrukcji źródłowych w załączniku.**

Dowód: `pipeline/train/train_rfdetr.py:394` sortuje wszystkie `checkpoint*.pth`,
a `:410` bierze ostatni plik. Dla standardowych nazw RF-DETR jest to
`checkpoint_best_total.pth`, nie checkpoint ostatniej epoki.
Repro: zadeklarowany best z epoki 20 wygrywa nad checkpointem końcowym epoki 59.

`per_epoch` w `:405`, `best_epoch` w `:415` i `epochs_completed` w `:421`
odnoszą się do indeksu/liczby plików, mieszając checkpointy okresowe, best i EMA.
Skutek: kolumny best/last oraz metadane epok są niewiarygodne.
RF-DETR rzeczywiście tworzy taki plik best:
[upstream 1.2.1 main.py](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/main.py).
Repro nie sprawdza rzeczywistych plików wag ani ich epok na Edenie.

### 8. P1 — konfiguracja treningu RF-DETR nie egzekwuje deklarowanego protokołu

**Status: braki przekazania parametrów potwierdzone w kodzie; konkretne zachowanie
biblioteki sprawdzone względem RF-DETR 1.2.1, bez uruchomienia treningu.**

Dowód: `pipeline/train/train_rfdetr.py:359` sprawdza deklarację braku augmentacji,
ale `:371–386` nie wyłącza transformacji. Używa nazw
`gradient_accumulation_steps` i `lr_backbone`, zamiast `grad_accum_steps`
i `lr_encoder`; mogą pozostać domyślne wartości biblioteki. `amp=True` w `:380`
nie gwarantuje BF16. `_filter_kwargs` w `:338` przepuszcza wszystko przy `**kwargs`,
więc nie dowodzi skutecznego zastosowania parametrów; log odrzuconych parametrów
jest później usuwany w `:511`.

Dodatkowo `train_rfdetr.py:240` przygotowuje tylko `train/valid`, podczas gdy
RF-DETR 1.2.1 bezwarunkowo buduje również dataset `test`. Brak jego adnotacji
blokuje start; samo deklarowanie protokołu nie zapewnia zgodności layoutu.
Skutek: inne LR, akumulacja, augmentacje lub precyzja niż założono, albo brak startu.
Źródła: [TrainConfig](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/config.py),
[transformacje i layout](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/datasets/coco.py),
[trener](https://raw.githubusercontent.com/roboflow/rf-detr/1.2.1/rfdetr/main.py).

### 9. P1 — identyfikatory eksportu YOLO nie są uzgadniane z GT; brak pliku udaje pusty wynik

**Status: potwierdzony względem lokalnego eksportera Ultralytics 8.3.236;
brak mapowania i fallback potwierdzone w kodzie projektu.**

Dowód: `pipeline/transfer/transfer_common.py:187` nadaje GT kolejne identyfikatory
liczbowe. `pipeline/transfer/train_transfer.py:143` zachowuje stem obrazu przy
materializacji. Sprawdzony eksporter Ultralytics bierze `image_id` z nazwy pliku
(`F:/Python/Lib/site-packages/ultralytics/models/yolo/detect/val.py:404`).
`train_transfer.py:408–416` nie wykorzystuje argumentu `coco` do mapowania.

Repro COCOeval podczas audytu: GT `id=1`, predykcja `image_id='val0'` kończy się
`AssertionError: Results do not correspond to current coco set`. Przypadkowo
pokrywające się numery nazw mogą też przypisać predykcję do niewłaściwego obrazu.

Niezależny błąd: brak katalogu lub `predictions.json` daje `[]` w `:411–416`,
co może zostać zapisane jako poprawne zerowe metryki. Brak eksportu powinien
być odróżniony od poprawnego eksportu bez detekcji. Nie zweryfikowano eksportera
wersji Ultralytics używanej do YOLO26 na Edenie.

### 10. P2 — deklarowany max-F1 pomija dopuszczalne progi

**Status: transfer potwierdzony reprodukcją CPU podczas audytu;
przybliżenie głównego ewaluatora potwierdzone analizą kodu.**

Dowód: `pipeline/transfer/metrics.py:101` sprawdza tylko progi `0.01…0.99`,
pomimo generowania predykcji od `0.001`. Repro: jedyna trafna detekcja o score
`0.005` daje wybrane F1=0, choć przy progu `0.001` F1=1.

Główny ewaluator w `pipeline/eval/common.py:343` ogranicza kandydatów do 200
unikalnych score'ów i również nie gwarantuje dokładnego optimum.
Skutek: różne przybliżenia doboru progu dla architektur; możliwe zaniżenie F1.
Jeśli aproksymacja jest zamierzona, wymaga jawnej decyzji i wspólnej reguły.

### 11. P1 — transfer nie realizuje pełnego deklarowanego protokołu

**Status: potwierdzona niekompletność kodu; integracja D-FINE oceniona statycznie,
bez uruchamiania repozytorium zewnętrznego.**

Dowody:

- `pipeline/transfer/train_transfer.py:183–196` nie egzekwuje braku augmentacji,
  akumulacji ani precyzji YOLO, choć te parametry istnieją w protokole.
- `train_transfer.py:289–304` zapisuje regułę `best_val`; `main` w `:391–394`
  oblicza tylko walidację. Brak pełnego raportu obu checkpointów i testu.
- Gałąź D-FINE w `:395–397` nie generuje tu predykcji ani metryk.
- Generator w `:244` wskazuje `dfine_hgnetv2_s.yml` obok wygenerowanego pliku,
  lecz go nie tworzy. Standardowy plik upstream ma inne położenie i nazwę.
- Epoki trafiają do `train_cfg.epochs` (`:261–262`), zamiast głównego `epochs`;
  `img_folder=''` przy względnych `file_name` nie wskazuje `data_root`.
- Brakuje jawnego podania wag COCO D-FINE; `:399–404` wpisuje planowaną liczbę
  epok i ogłasza checkpointy bez sprawdzenia, czy oba zostały zebrane.

Skutek: pipeline może nie wystartować albo raportować niezweryfikowaną zgodność
iso-resource i niekompletne wyniki. Repro generatora w pamięci podczas audytu
potwierdziło strukturę YAML. Źródła integracji:
[konfiguracja D-FINE](https://raw.githubusercontent.com/Peterande/D-FINE/master/configs/dfine/dfine_hgnetv2_s_coco.yml),
[rozwiązywanie include](https://raw.githubusercontent.com/Peterande/D-FINE/master/src/core/yaml_utils.py).
Źródła D-FINE to gałąź master odczytana podczas audytu, bez przypięcia commita;
wnioski należy skonfrontować z faktycznym checkoutem na Edenie.

### 12. P2 — luki walidacji wejść i pochodzenia predykcji

**Status: luki potwierdzone w kodzie; ich wystąpienie w rzeczywistych danych
wymaga osobnego sprawdzenia.**

Dowód: `pipeline/train/train_rfdetr.py:178` przyjmuje dowolny `annotations_train`.
Ścieżka `frame_ids` w `:192–196` sprawdza obecność w poolu, ale nie `split=train`,
duplikaty ani rozłączność z walidacją. Transfer posiada kontrolę przynależności
do train (`pipeline/transfer/transfer_common.py:148`), więc zabezpieczenia są nierówne.
Nie dowiedziono faktycznego przecieku podziałów; to luka na granicy wejścia treningu.

`pipeline/eval/evaluate.py:137` identyfikuje obrazy po samym basename:
`videoA/0001.jpg` i `videoB/0001.jpg` kolidują. Słownik plików oraz `id_by_name`
w `:153–158` mogą wskazać niewłaściwy obraz lub identyfikator.

Import predykcji w `evaluate.py:168` zapisuje następnie deklarowany próg
`0.001` w sidecarze (`:194`) bez dowodu, z jakim progiem powstał plik wejściowy.
Skutek: niezweryfikowany provenance predykcji i możliwość nieporównywalnych krzywych PR.

## Pozostałe ustalenia i wymagane decyzje

- Podstawowe liczenie FN przy braku predykcji oraz FP na obrazach tła działa
  w obu matcherach. Problem usuwania obrazów dotyczy opisanej ścieżki mAP/importu;
  błędu nie należy automatycznie przypisywać całej obsłudze pustych obrazów.
- Ręczne liczniki nie odwzorowują reguł `iscrowd` COCO. Znaczenie zależy od
  obecności takich anotacji; dane generowane przez audytowane konstruktory
  ustawiają `iscrowd=0`. Nie jest to dowód błędu na faktycznym zbiorze.
- W zadanym zakresie nie znaleziono implementacji pomiaru FPS ani bezpośredniego
  adaptera RT-DETR. Nie da się potwierdzić uczciwości porównania FPS; wymaga ona
  osobnego kodu pomiarowego i ustalenia m.in. urządzenia, precyzji, batch size,
  rozgrzewki, synchronizacji oraz zakresu preprocess/postprocess.
- Identyczność wag inicjalizacyjnych wymaga sprawdzenia artefaktów i ich hashy.
  Sam hash protokołu ani użycie domyślnej nazwy wag nie dowodzą identycznego startu.
- Diagnostyczne optimum F1 na zbiorze testowym lub zewnętrznym nie może zastąpić
  metryki przy progu walidacyjnym. W raporcie należy rozróżnić diagnostykę od
  wykorzystania testu do wyboru parametrów.

## Bilans dowodów

Załączony skrypt odtwarza dokładnie pięć zamówionych przypadków. Pozostałe
reprodukcje opisane w raporcie wykonano podczas zakończonego audytu, a ustalenia
integracyjne wynikają z porównania kodu z podanymi źródłami API. Skrypt nie
potwierdza poprawności całego pipeline, API RF-DETR, środowiska Eden ani
historycznych wyników. Zapis obejmuje wyłącznie ten raport, skrypt i JSON wyników.
