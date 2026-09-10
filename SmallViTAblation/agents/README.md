Instrukcje dla agentów — zasady wspólne
======================================

Ten katalog zawiera zadania dla **pięciu niezależnych agentów**. Każdy plik
`AGENT_*.md` jest **samodzielny**: agent nie zna historii projektu i nie musi
jej znać. Przeczytaj swój plik plus ten i to wystarczy.

Autorzy agentów mogą być różnymi modelami. Dlatego nic tu nie jest domyślne —
każdy kontrakt jest wypisany.

---

## 1. O czym jest projekt, w trzech zdaniach

Wykrywamy **końcówkę narzędzia chirurgicznego** na nagraniach z mikroskopu
operacyjnego (operacje zaćmy). Badamy, czy **inteligentny dobór zdjęć
treningowych** daje lepszy model niż losowy wybór tej samej liczby zdjęć.
Poprzednia wersja pracy została odrzucona przez recenzentów za błędy protokołu
eksperymentu, więc wszystko liczymy od nowa, w rygorze opisanym niżej.

Pełny plan: `../PLAN.md`. Specyfikacja protokołu: `../docs/01_PROTOCOL.md`.

## 2. Zasada nadrzędna — łamanie jej unieważnia wyniki

> **Zmieniamy jedną rzecz na raz.** Wszystko poza badanym składnikiem —
> podział danych, budżet obrazów, inicjalizacja, optymalizator, harmonogram,
> liczba epok, reguła wyboru checkpointu, protokół ewaluacji — jest
> **identyczne dla wszystkich wariantów**.

Praktycznie znaczy to trzy rzeczy:

1. **`configs/protocol.yaml` jest jedynym źródłem prawdy.** Żaden skrypt nie
   ma zaszytych hiperparametrów. Wszystko czytane z pliku. Jeśli czegoś brakuje
   w `protocol.yaml`, dopisz to tam, a nie do kodu.
2. **Brak wcześniejszego zatrzymania treningu.** Stała liczba epok dla
   wszystkich. To decyzja świadoma: wcześniej każdy wariant kończył w innej
   epoce, więc dostawał inny budżet obliczeń.
3. **Każdy bieg zapisuje `run_manifest.json`** z hashem kodu i konfiguracji.

## 3. Podział pracy — kto czego NIE dotyka

| Agent | Katalog roboczy | Buduje |
|---|---|---|
| **Master (człowiek + Claude)** | `pipeline/ingest/`, `pipeline/annotate/` | pobieranie nagrań, wycinanie klatek, normalizacja, propagacja ramek narzędzia |
| **Agent 1** | `pipeline/features/`, `pipeline/selection/` | cechy obrazów i wybór podzbiorów |
| **Agent 2** | `pipeline/train/`, `eden/` | trening jednego biegu i kolejka na klastrze |
| **Agent 3** | `pipeline/eval/` | inferencja, metryki, tabele |
| **Agent 4** | `pipeline/transfer/` | te same eksperymenty na innych architekturach |
| **Agent 5** | `analysis/` | analizy bez GPU i kontrola kompletności |

**Nie edytuj plików spoza swojego katalogu.** Wyjątek: `configs/protocol.yaml`
i `configs/variants.yaml` — tam wolno **dopisywać** pola, nigdy zmieniać
istniejących wartości bez uzgodnienia.

## 4. Kontrakty — pliki, przez które agenci się komunikują

Agenci nie rozmawiają ze sobą. Wymieniają się plikami o ustalonym kształcie.

### 4.1 `pool_manifest.jsonl` — daje Master, czytają wszyscy

Jeden wiersz JSON na klatkę. Format wierszowy, bo klatek są setki tysięcy.

```json
{"frame_id":"cataracts/test01/0000746","path":"images/cataracts/test01/0000746.jpg",
 "source_dataset":"cataracts","patient_id":"test01","video_id":"test01",
 "frame_index":746,"native_width":1920,"native_height":1080,"native_fps":29.17,
 "split":"train","label_origin":"human","seed_distance_s":0.0,
 "bbox":[x,y,w,h],"has_instrument":true}
```

Znaczenie pól, których nie widać na pierwszy rzut oka:

- `split` — `train`, `val` albo `test`. **Przypisany po pacjencie**, nigdy po
  klatce. Nigdy tego nie przeliczaj samodzielnie.
- `label_origin` — `human` (człowiek oznaczył) albo `propagated` (ramka
  przeniesiona automatycznie ze śledzenia). Zbiory `val` i `test` mają
  **wyłącznie** `human`.
- `seed_distance_s` — jak daleko w sekundach od klatki oznaczonej ręcznie.
  Dla `human` zawsze 0.
- `bbox` — `[x, y, szerokość, wysokość]` w pikselach obrazu po normalizacji,
  konwencja COCO. Brak narzędzia w kadrze: `bbox` nieobecny, `has_instrument`
  fałsz.

### 4.2 `variants/<run_id>.json` — daje Agent 1, czyta Agent 2

```json
{"run_id":"M-DESF__s42__b5000","variant":"M-DESF","seed":42,"budget":5000,
 "frame_ids":["cataracts/test01/0000746", "..."],
 "selection_meta":{"weights":{"dino":0.35,"el2n":0.30,"sam":0.20,"fourier":0.15},
                   "pca_dim":32,"algorithm":"kmeans","representative":"centroid"}}
```

Liczba `frame_ids` musi **dokładnie** równać się `budget`. Wszystkie muszą mieć
`split == "train"`.

### 4.3 `queue.jsonl` — generuje istniejące narzędzie, czyta Agent 2

Powstaje z `tools/build_run_manifest.py`. Jeden wiersz na bieg, pola:
`run_id`, `tier`, `kind`, `variant`, `groups`, `weights`, `seed`, `budget`,
`arch`, `note`, opcjonalnie `override`.

### 4.4 `results/<run_id>/run_manifest.json` — daje Agent 2, czyta Agent 5

```json
{"run_id":"...","git_commit":"...","protocol_sha256":"...","variant_file_sha256":"...",
 "started_at":"...","finished_at":"...","hostname":"...","gpu":"...",
 "python":"3.11.x","torch":"...","library_versions":{"...":"..."},
 "epochs_completed":60,"resolution":512,"seed":42}
```

### 4.5 `results/<run_id>/metrics.json` — daje Agent 3, czyta Agent 5

```json
{"run_id":"...","checkpoint_rule":"last_epoch|best_val",
 "results":{"val":{...},"test":{...},"external_full":{...},"external_clean":{...}},
 "operating_threshold":0.37,"threshold_selected_on":"val"}
```

Każdy blok wyników zawiera: `mAP_0.5`, `mAP_0.5:0.95`, `mAP_0.75`, `AR@100`,
`TP`, `FP`, `FN`, `precision`, `recall`, `f1`, `n_images`, `n_annotations`.

## 5. Reguły, które łatwo złamać przez nieuwagę

1. **Zbiór zewnętrzny liczymy raz, na końcu.** Wybór epoki i progu ufności
   odbywa się **wyłącznie na zbiorze walidacyjnym**. Każdy dostęp do zbioru
   zewnętrznego dopisujemy do `external_access_ledger.jsonl`.
2. **Selekcja widzi tylko klatki treningowe.** Cechy, PCA i grupowanie liczymy
   na `split == "train"`. Klatki `val` i `test` nie wchodzą do żadnego kroku
   selekcji.
3. **Model startuje zawsze z tych samych wag COCO.** Nigdy z checkpointu
   wcześniejszego eksperymentu.
4. **Inferencja standardowa.** Zwykłe dekodowanie detektora. Żadnego wybierania
   pojedynczego zapytania modelu — to był zarzut recenzenta.
5. **mAP liczymy z pełnej krzywej**: predykcje generujemy z progiem ufności
   0,001, nie 0,3. Próg operacyjny do TP/FP/F1 wybieramy osobno, na walidacji.
6. **Na klastrze nie dzieje się nic z obrazami** poza wczytaniem ich do modelu.
   Żadnego skalowania, przycinania ani augmentacji.

## 6. Jak testować, zanim cokolwiek pojedzie na klaster

Jest do tego osobny, mały model: **RF-DETR-Nano przy wejściu 384**. Mieści się
w sześciu gigabajtach pamięci karty i jeden bieg trwa minuty.

Każdy agent kończy zadanie **testem od końca do końca na 2 500 obrazach przez
5 epok**. Dopiero gdy on przechodzi, kod jest gotowy.

## 7. Definicja gotowości — wspólna dla wszystkich

Zadanie jest skończone, gdy:

1. kod czyta konfigurację z `protocol.yaml`, nie ma zaszytych wartości,
2. przechodzi test na małym modelu, opisany wyżej,
3. wypisuje pliki dokładnie w formacie z sekcji 4,
4. ma krótki `README.md` w swoim katalogu: jak uruchomić i co wychodzi,
5. działa pod **Pythonem 3.11** (tak jest w całym projekcie),
6. nie modyfikuje plików spoza swojego katalogu.

## 8. Czego nie robić

- Nie „poprawiaj" protokołu, bo wydaje się nieoptymalny. Wiele decyzji wygląda
  dziwnie, dopóki nie zna się zarzutów recenzentów. Wątpliwości zapisz
  w `README.md` swojego katalogu, ale kod zrób zgodnie ze specyfikacją.
- Nie dodawaj wcześniejszego zatrzymywania treningu, nawet gdy „oszczędza czas".
- Nie zmieniaj podziału danych ani nie generuj własnego.
- Nie kopiuj obrazów per wariant. Podzbiory to **listy nazw**, obrazy leżą raz.
- Nie instaluj zależności, których nie ma w projekcie, bez zapisania ich
  w `requirements.txt` swojego katalogu.
