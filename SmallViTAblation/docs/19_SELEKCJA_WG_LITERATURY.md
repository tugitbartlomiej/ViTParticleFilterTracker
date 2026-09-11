# 19 — Selekcja danych według literatury: weryfikacja kodu i protokół dla RF-DETR / RT-DETR

Data: 2026-09-11. Zakres: `pipeline/features/`, `pipeline/selection/` (czyli to, co audyt
z `18_AUDYT_PIPELINE_SELEKCJI.md` uznał za sprawne hydraulicznie), stary pipeline
`AdvancedDatasetSelection/` oraz sekcja Methodology artykułu
`DETR_IEEE_FINAL.tex` (linie 143–270).

Podstawa literaturowa: deep search z 2026-09-10 (workspace ViT, 113 prac) plus **dziewięć
prac przeczytanych dziś w całości**: [Zha26e], [Lee24b], [Kan25], [Yag25], [Cho25], [Pau21],
[Zhe24b], [Zhe22], [Xie26]. Poprzednia propozycja: `AdvancedDatasetSelection/PLAN_v3_RF-DETR.md`
(ten dokument ją zawęża do decyzji wykonywalnych w obecnym kodzie i dodaje liczby).

---

## 0. Rozstrzygnięcie nazewnicze: RF-DETR to nie RT-DETR

To nie jest drobiazg, więc na początku:

| | RF-DETR (Roboflow) | RT-DETR / RT-DETRv2 (Baidu) |
|---|---|---|
| backbone | **DINOv2 ViT** | HGNetv2 / ResNet — **nie ViT** |
| w projekcie | `protocol.yaml: model.name = rf-detr-small`, `pipeline/train/train_rfdetr.py` | nieobecny |
| rola w argumentacji | detektor z backbone'em ViT | detektor DETR-style z backbone'em konwolucyjnym |

Obecny adapter i YAML dotyczą **RF-DETR-Small**. Recenzent z
`Reviewer_Comments_to_Authors.pdf`, punkt 7, prosił o współczesny detektor
DETR-style, wymieniając RT-DETR, D-FINE i RF-DETR; nie wymagał małego backbone'u ViT.
RT-DETRv2 jest więc poprawnym kandydatem zgodnym z kierunkiem wskazanym przez
użytkownika, ale wymaga własnego adaptera. RF-DETR pozostaje istniejącą ścieżką
implementacyjną, nie wymogiem recenzji. [Zha26e] jest bliski technicznie, lecz
dotyczy teledetekcji, nie wideo chirurgicznego. Hungarian dotyczy loss/matchingu
treningowego, a standardowy postprocessor i COCOeval są odrębnymi etapami.

---

## 1. Werdykt

Szkielet jest zgodny z literaturą: klastrowanie cech samonadzorowanych, sygnał trudności
z modelu proxy, jednakowy budżet i harmonogram, ziarna, split po pacjencie. **Sześć rzeczy
w kodzie odbiega od literatury na tyle, że każda osobno potrafi zjeść cały efekt selekcji.**
Cztery z nich są tanie do naprawy przed policzeniem cech na prawdziwej puli.

W przejrzanych benchmarkach często raportowano różnice kilku punktów AP.
Nie ustanawia to górnej granicy możliwej poprawy, zwłaszcza przy baseline bliskim
zera. Obecne 0,23% kontra 26,04% wymaga kontroli protokołu i powtórzeń; jego
wielkość sama nie dowodzi błędu. Udokumentowane różnice treningu już uniemożliwiają
interpretację tego porównania jako czystego efektu selekcji.

---

## 2. Co literatura potwierdza (zostaje bez zmian)

- **Klastry na cechach samonadzorowanych + reprezentant z klastra** to uznana metoda
  selekcji bez etykiet: DINOv2 + k-means w rolnictwie [Gom25], prototypowość [Sor22],
  selekcja bez etykiet [Wan21], koresety bez etykiet z dynamiki proxy [Zhe24b].
- **Sonda na mniejszym modelu** jest metodą, nie skrótem [Col19]. RF-DETR-S jako sonda
  dla 144 biegów jest uzasadniona, o ile pokażemy zgodność rankingów (bieg `P-RES`).
- **Proxy trenowany krótko** — [Pau21] liczy wyniki po 10 epokach, [Cho25] po 30 z 200,
  [Yag25] po 12–17. Nasze `el2n.proxy_epochs: 10` jest w tym przedziale.
- **Jednakowy budżet, jednakowy harmonogram, 10 losowań baseline'u losowego** — losowanie
  to mocny baseline o dużej wariancji [Aye23], [Guo22].
- **Pula selekcji tylko z filmów treningowych** (`common.train_pool`) — audyt potwierdził,
  że reguła P2 jest respektowana.
- **Split po pacjencie** i ocena na zbiorze zewnętrznym — bez tego wyniki na wideo są
  zawyżone przez podobieństwo sąsiednich klatek.

---

## 3. Sześć rozbieżności z literaturą

### R1. EL2N liczone z jednego proxy uczonego na 12% puli

Kod: `configs/protocol.yaml:59` (`proxy_subset_size: 2500`),
`pipeline/features/train_el2n_proxy.py:149-153` (losowy podzbiór),
`pipeline/features/el2n_from_proxy.py` (inferencja po całej puli).

[Pau21] robi to inaczej i **różnica nie jest kosmetyczna**:

1. model scoringowy trenuje się na **całym** zbiorze treningowym, a wynik liczy się na tych
   samych danych — to jest zamierzone, nie błąd. Problemem nie jest „in-sample", lecz
   **nierówna ekspozycja**: proxy widzi 2500 przykładów z zadeklarowanej puli.
   Jej wpływ na ranking należy zmierzyć; proporcja zależy od rzeczywistego manifestu.
2. wynik uśrednia się po **10 niezależnych inicjalizacjach**. Dla jednej inicjalizacji
   korelacja Spearmana z tą średnią wynosi **około 0,75**, a przycinanie 50% CIFAR-10
   takim wynikiem spada z ~94,5% do ~90%, czyli **prawie do poziomu losowania**.

Co zrobić: proxy trenować na **całej puli treningowej** przez 10 epok i uśrednić wynik po
ziarnach, które i tak mamy (3 dla poziomu A, 5 dla B). `proxy_subset_size` usunąć albo
ustawić na pełną pulę; dopisać `el2n.n_inits`. Tanio, bo proxy to 10 epok, a nie 60.

### R2. Klatka bez narzędzia dostaje maksymalną trudność

Kod: `pipeline/features/el2n_from_proxy.py:72` — `scores = np.ones(...)  # brak GT -> 1.0`.

[Yag25] pokazuje wprost, że **brak detekcji nie znaczy „trudne"**: obiekt konsekwentnie
pomijany przez wszystkie epoki ma **zerową wariancję**, czyli wynik niski. Rozkład jest
„księżycowy": i bardzo łatwe, i bardzo trudne przypadki mają małą wariancję, a najwyższą
mają przypadki **pośrednie**. Baseline'y wybierające najtrudniejsze traciły **5–15 pp mAP**.
[Pau21] dodaje, że najwyższe wyniki to zwykle odstające lub zaszumione próbki i przy silnym
przycinaniu trzeba **odciąć górę rozkładu**. [Lee24b] i [Yag25] po prostu **ograniczają zbiór
bazowy do obrazów z adnotacjami** (COCO: 117 264 obrazy z adnotacjami).

Co zrobić: klatki tła **wyprowadzić z rankingu trudności** i dać im osobną kwotę
(np. taki udział, jaki mają w puli), a nie wartość 1,0. Przy 14% klatek bez narzędzia
w Cataract-LMM obecny kod może zmieniać udział tła, ale centroidowa selekcja
nie gwarantuje monotonicznego preferowania wysokiego score. Trzeba policzyć skład wyboru.

### R3. To, co kod nazywa EL2N, nie jest EL2N

Kod: `el2n_from_proxy.py` — `score = 1 - pewność najlepszej detekcji dopasowanej po IoU`.

[Pau21]: EL2N = norma L2 różnicy między rozkładem softmax a wektorem jednostkowym klasy
prawdziwej, **uśredniona po inicjalizacjach**. Jeden seed jest zaszumioną
estymacją tej wartości, nie odmienną definicją. Dla dodatniego targetu jednego
sigmoidu norma wynosi `1-p`, dla dwóch klas softmax `sqrt(2)*(1-p)`.
Thresholding, dopasowanie detekcji i tło wymagają osobnej definicji.

Co zrobić: albo nazwa `detection_confidence_error`, albo — lepiej — zamiana na miarę,
która dla detekcji ma dowód: **VPS** [Yag25] lub **DUAL** [Cho25], patrz sekcja 5 krok 2.

### R4. k = budżet, jeden reprezentant, centroid

Kod: `pipeline/selection/select_subset.py:239-240` (`cluster_kmeans(x, budget, ...)`),
`protocol.yaml: cluster_representative: centroid`.

K-means z reprezentantem najbliższym centroidowi jest poprawnym baseline'em;
nie gwarantuje optymalnego pokrycia. Inne zbadane strategie działają odmiennie:

- [Kan25] (AdaDeDup, detekcja, 28k–100k obrazów): **K = 100 klastrów**, a budżet *wewnątrz*
  klastra: `k_i = round(|c_i| * (1 - γ_i'))` z ograniczeniami `k_i ≤ |c_i|` i `Σ k_i = m`.
  Korekta `γ_i'` bierze się z porównania straty proxy na zachowanych i odrzuconych próbkach
  w tym klastrze (`Δl_i = l_i^s − l_i^p`; `Δl_i > 0` → przycinaj mocniej).
- [Zhe22] (CCS): po odcięciu najtrudniejszego ułamka `β` dzieli zakres wyniku na **k = 50
  warstw o równej szerokości** i rozdziela budżet równo między warstwy, zaczynając od
  najmniej licznej (`m_B = min(|B_min|, floor(m / |B|))`).
- [Yag25]: wybór przypadków **pośrednich**, nie najbardziej typowych ani najtrudniejszych.

Propozycja do porównania: `k = B/3 … B/5`, budżet per klaster i stratyfikacja
trudności. To własna hybryda wymagająca walidacji, nie nakaz literatury ani
gotowa reprodukcja AdaDeDup/CCS. Centroid pozostaje punktem odniesienia.

### R5. Macierz wybiera z puli po augmentacji

Kod: `pipeline/features/common.py: train_pool` bierze wszystkie rekordy `split == "train"`,
a to jest **71 316 obrazów z 17 832 kadrów źródłowych** (`protocol.yaml:
analysis.dataset_table_expected`). Deduplikacji kopii `_aug_N` nie ma nigdzie w ścieżce
macierzy — regex `AUG_RE` istnieje tylko w `selection/baselines.py:33`, czyli w baseline'ie.

[Zha26e] liczy wszystko na **oryginalnych obrazach**, a augmentację stosuje dopiero w
treningu po wyborze. W poprzedniej wersji to dało 23,4% wyboru na duplikatach augmentacji.

Co zrobić: pula macierzy = **kadry źródłowe**, augmentacja po wyborze. To wymusza jedną
decyzję liczbową: **`budgets.tier_b: 20000` jest większe niż 17 832 kadrów źródłowych
treningowych**, więc poziom B trzeba obniżyć (np. 12 000) albo zdefiniować jako „wszystkie
kadry źródłowe + augmentacja do budżetu" (otwarte pytanie O2 z `05_OPEN_QUESTIONS.md`).

### R6. Cechy opisują cały kadr, a nie końcówkę narzędzia

Kod: `pipeline/features/build_cache.py: DinoExtractor` — token **CLS** po przeskalowaniu
kadru do kwadratu `dino_image_size: 224` (`protocol.yaml:191`).

[Lee24b] mierzy to wprost na VOC07 (AP@50, budżet 200 obrazów): cechy obiektowe (RoI
pooling w ramkach GT, uśrednione per klasa w obrazie) dają **44,3%**, a metody na cechach
całego obrazu **38,6% / 32,3% / 26,0%**, przy losowaniu **37,9%**. Przy 500 obrazach
submodularna na całym obrazie ma **48,8%**, czyli **mniej niż losowanie (50,7%)**.
Cechy całego obrazu w detekcji potrafią być gorsze od losowania.
[Zha26e] robi to samo inaczej: **pooling mapy cech DINOv2 ważony maską obiektu**,
normalizacja L2, **bez PCA**.

Do tego 224 piksele: przy 512 już 56,9% ramek narzędzia jest pod progiem 32×32
(`13_STANDARD_DANYCH.md`), a przy 224 końcówka to kilka pikseli. CLS z takiego kadru nie
koduje narzędzia, tylko scenę.

Co zrobić: cechy z **patch tokenów uśrednionych w ramce GT** powiększonej 1,5x, wejście
518 albo 704, PCA do 32. Token CLS zostaje jako osobna grupa o małej wadze (kontekst scen).

### Dodatkowo: niespójność Fourier 9-D kontra 1-D

Artykuł opisuje wektor **9-wymiarowy** (`DETR_IEEE_FINAL.tex:163-199`), a kod liczy
**1 wymiar** (`protocol.yaml:196: fourier_dim: 1`). Do rozstrzygnięcia przed liczeniem cech,
bo wchodzi do tabeli wymiarów.

---

## 4. Ile realnie można wygrać (kalibracja oczekiwań)

Liczby przytoczone w przeglądzie dotyczą różnych zadań, skal AP i budżetów.
Nie wszystkie odpowiadają pozostawieniu 10–30% danych ani naszemu problemowi:

| Praca | Zadanie | Reżim | Losowanie | Metoda | Zysk |
|---|---|---|---|---|---|
| [Kan25] | detekcja 3D, nuScenes | zostaje 70% | 0,3663 mAP | 0,3728 | **+0,0065** |
| [Kan25] | detekcja, Waymo | odcięte 40% | — | — | degradacja mniejsza o 54% |
| [Yag25] | detekcja, VOC | zostaje 50% | 45,98 mAP | 47,08 | **+1,1 pp** |
| [Yag25] | detekcja, VOC | zostaje 30% | 40,81 | 42,75 | **+1,9 pp** |
| [Yag25] | detekcja, VOC | zostaje 10% | 28,09 | 30,05 | **+2,0 pp** |
| [Lee24b] | detekcja, VOC | 200 obrazów | 37,9 AP@50 | 44,3 | **+6,4 pp** |
| [Lee24b] | transfer na RetinaNet | 500 obrazów | 54,5 | 58,3 | **+3,8 pp** |
| [Lee24b] | transfer na FCOS | 500 obrazów | 47,9 | 53,1 | **+5,2 pp** |

Wnioski, które trzeba wpisać do artykułu zamiast dotychczasowej narracji:

1. **W tej tabeli zyski wynoszą kilka punktów AP lub mniej.** Nie wynika z niej
   uniwersalny limit przewagi w innej domenie i przy baseline bliskim zera.
2. **Sama trudność przy silnym przycinaniu jest niebezpieczna.** [Zhe22], CIFAR-10, 90%
   odcięte: losowanie **79,04%**, EL2N po samym wyniku **22,33%**, CCS na EL2N **86,68%**.
   Ta sama praca na CIFAR-100 i ImageNet: EL2N bez stratyfikacji spada do 8–13%. Czyli
   nasza grupa `E` wymaga kontroli pokrycia i porównania z random; jej wynik
   nie jest przesądzony przez doświadczenie klasyfikacyjne na innym zbiorze.
3. **Transfer sygnału trudności na inny detektor bywa ujemny.** [Yag25] tab. 3, subset
   wybrany przez Faster R-CNN, trenowany YOLOv5m, 90% odcięte: losowanie **22,87**,
   EL2N **18,44**, VPS_iou **23,08**. EL2N przeniósł się **gorzej niż losowanie**.
   [Kwo24] tłumaczy dlaczego: rankingi trudności zależą od inductive bias architektury.
4. **[Pau21] pokazuje transfer między architekturami tylko dla klasyfikacji** (ResNet18 →
   VGG13, ResNet50 → ResNet18, krzywe się pokrywają). Dla detekcji takiego dowodu nie ma,
   więc transfer rankingu trzeba zweryfikować. Scoring na architekturze docelowej
   jest jednym wariantem referencyjnym, nie warunkiem definicyjnym selekcji.

---

## 5. Protokół docelowy dla RF-DETR-S (i RT-DETRv2 w transferze)

### Krok 0 — pula
Kadry **źródłowe** z 35 filmów treningowych (~17,8 tys.). Rekord manifestu: `frame_id`,
`video_id`, `t` (czas znormalizowany 0–1), **`bboxes` jako lista** (Cataract-LMM ma średnio
1,4 narzędzia na klatkę; obecny kontrakt `bbox` gubi drugie — punkt 3 audytu),
`has_instrument`.

### Krok 1 — cechy
- **obiektowe** (główne): patch tokeny DINOv3 uśrednione w ramce GT powiększonej 1,5x,
  L2, PCA do 32 [Lee24b], [Zha26e];
- **kontekst**: CLS, PCA do 16, waga ~0,10;
- **Fourier / FastSAM**: tylko jako grupy ablacyjne o małej wadze — dla detekcji nie ma na
  nie dowodu, a [Cui25] (FAST) daje przesłankę tylko w klasyfikacji;
- **czas**: nie jako cecha, lecz ograniczenie w kroku 4.

### Krok 2 — trudność, licząc na architekturze docelowej
Proxy RF-DETR-S na **całej** puli treningowej, 10–17 epok, checkpoint co epokę
(kosztem jednego biegu na ziarno). Dla każdej ramki GT dopasowanie po IoU / Hungarian,
a potem:

- `VPS_iou(obiekt) = std_epok(IoU dopasowanej predykcji)` [Yag25], agregacja do obrazu
  przez **max**;
- wariant `DUAL = (1 − mean_okno p) * std_okno(p)`, okno J = 10 [Cho25];
- uśrednienie po ziarnach (namiastka 10 inicjalizacji z [Pau21]);
- **klatki tła nie wchodzą do rankingu** — osobna kwota (R2).

### Krok 3 — klastry
K-Means (FAISS) na `[obiektowe, kontekst]`, **k = B/3 … B/5**, ziarno = ziarno biegu.

### Krok 4 — budżet per klaster
`k_c = round(|c| * (1 − γ_c'))` z `Σ k_c = B` i `k_c ≤ |c|` [Kan25]. Korekta `γ_c'`:
wariant tani — z rozpiętości czasowej klastra (liczba filmów i zakres `t`) i średniej
trudności; wariant pełny — pętla AdaDeDup ze stratą proxy na zachowanych kontra odrzuconych.

### Krok 5 — wybór w klastrze
Stratyfikacja CCS po mierze trudności [Zhe22]: odcięcie najtrudniejszych `β`
(tabela: `β = 10%` przy 70–80% przycięcia, `20–30%` przy 90% — u nas przy ~25% pozostawienia
startowo `β = 0,1`), potem warstwy o równej szerokości i równy budżet na warstwę.
Dodatkowo **minimalny odstęp czasowy `dt`** między wybranymi z tego samego filmu.

### Krok 6 — augmentacja i trening
Augmentacja **po** wyborze, identyczna dla wszystkich wariantów. Trening bez zmian wobec
P5: ten sam checkpoint startowy, LR, 60 epok, bez early stopping, odczyt top-k.

### Krok 7 — baseline'y i transfer
Obowiązkowo: `B-RANDSRC` (**to jest główny baseline**, nie `B-RANDAUG`), `B-UNIFORM`,
`B-RANDAUG`, SemDeDup-lite, CCS na samej trudności, wariant v2 (obecny) jako punkt
odniesienia dla artykułu. Transfer: podzbiór wybrany sygnałami RF-DETR-S trenuje
**D-FINE-S, YOLO26 i RT-DETRv2**, tabela w układzie [Lee24b] tab. 7 i [Yag25] tab. 3.

---

## 6. Czego literatura nie wspiera

1. **Zlepek czterech grup cech z ręcznie dobranymi wagami** — nie znalazłem dla detekcji
   pracy walidującej dokładnie naszą konfigurację. Może być hipotezą i potencjalnym
   wkładem metodologicznym po kontrolowanym porównaniu; sama kombinacja nie dowodzi jakości.
2. **Uniwersalna przewaga jednej reguły wyboru** — centroid, pośrednia trudność
   i pokrycie odpowiadają różnym celom. Ich ranking dla naszego detektora trzeba zmierzyć.
3. **Odczyt z jednego zapytania** — zostaje wyłącznie jako obserwacja.
4. **Nazwa „EL2N" dla dowolnego score confidence** — trzeba podać wzór,
   output activation, matching i sposób estymacji, w tym liczbę inicjalizacji [Pau21].
5. **Gwarancja generalizacji z samej różnorodności** — ani metryka embeddingów,
   ani niekontrolowana historyczna ablacja jej nie potwierdza lub ogólnie nie obala.
6. **Dynamiczny wybór per epoka** ([Xie26], AFSS: min(precision, recall) na próg 0,85/0,55,
   1,4–1,7x szybszy trening bez straty AP) — ciekawe, ale to **inny problem** (przyspieszenie
   treningu na pełnym zbiorze), testowane tylko na YOLO, bez żadnego detektora z rodziny
   DETR i bez baseline'u losowego tej samej wielkości. Nie mieszać z naszą narracją.

---

## 7. Kolejność zmian w kodzie

Przed policzeniem cech na prawdziwej puli (rozszerza listę z audytu `18_...md`):

1. `configs/protocol.yaml` — sześć brakujących kluczy `features.*` (decyzja o modelach);
   `el2n.proxy_subset_size` → pełna pula; dodać `el2n.n_inits`; `dino_image_size` → 518/704;
   `features.object_features: gt_box_pool` + `box_expand: 1.5`; `features.cluster_k_divisor`;
   `selection.per_cluster_budget`, `selection.ccs_beta`, `selection.min_dt`; rozstrzygnąć
   `fourier_dim` (9 czy 1); obniżyć `budgets.tier_b` albo przedefiniować (R5).
2. `pool_manifest` + `materialize.py` + `train_el2n_proxy.py` — `bbox` → `bboxes`.
3. `pipeline/features/el2n_from_proxy.py` — klatki bez GT poza rankingiem; zmiana nazwy miary.
4. **nowy** `pipeline/features/difficulty_from_checkpoints.py` — VPS/DUAL z checkpointów proxy.
5. `pipeline/features/build_cache.py` — ekstraktor obiektowy (patch tokeny w ramce GT).
6. `pipeline/selection/select_subset.py` — `k = ceil(B / divisor)`, budżet per klaster,
   stratyfikacja CCS, odstęp `dt`.
7. `pipeline/selection/baselines.py` — `B-RANDSRC` awansuje na baseline główny.

---

## 8. Odnośniki

- [Zha26e] Zhang i in., *Foundation-Assisted Active Learning for Object Detection
  Annotation*, 2026 — RT-DETRv2, cechy obiektowe DINOv2 z maski, niepewność z dopasowania
  Hungarian: https://arxiv.org/abs/2607.16671
- [Lee24b] Lee i in., *Coreset Selection for Object Detection*, CVPRW 2024 —
  https://doi.org/10.1109/CVPRW63382.2024.00764
- [Kan25] Kang i in., *AdaDeDup: Adaptive Hybrid Data Pruning for Efficient Large-Scale
  Object Detection Training*, 2025 — https://doi.org/10.48550/arXiv.2507.00049
- [Yag25] Yagi i in., *Extending Dataset Pruning to Object Detection: A Variance-based
  Approach*, 2025 — https://doi.org/10.48550/arXiv.2505.17245
- [Cho25] Cho i in., *Lightweight Dataset Pruning without Full Training via Example
  Difficulty and Prediction Uncertainty*, ICML 2025 —
  https://doi.org/10.48550/arXiv.2502.06905
- [Pau21] Paul i in., *Deep Learning on a Data Diet*, NeurIPS 2021 —
  https://arxiv.org/abs/2107.07075
- [Zhe24b] Zheng i in., *ELFS: Label-Free Coreset Selection with Proxy Training Dynamics*,
  2024 — https://arxiv.org/abs/2406.04273
- [Zhe22] Zheng i in., *Coverage-centric Coreset Selection for High Pruning Rates*,
  ICLR 2023 — https://doi.org/10.48550/arXiv.2210.15809
- [Xie26] Xie i in., *Does YOLO Really Need to See Every Training Image in Every Epoch?*,
  2026 — https://doi.org/10.48550/arXiv.2603.17684
- [Col19] Coleman i in., *Selection via Proxy*, ICLR 2020 — https://arxiv.org/abs/1906.11829
- [Aye23] Ayed i Hayou, *Data pruning and neural scaling laws*, 2023 —
  https://doi.org/10.48550/arXiv.2302.06960
- [Kwo24] Kwok i in., *Dataset Difficulty and the Role of Inductive Bias*, 2024 —
  https://doi.org/10.48550/arXiv.2401.01867
- [Sor22] Sorscher i in., *Beyond neural scaling laws*, NeurIPS 2022 —
  https://doi.org/10.48550/arXiv.2206.14486
- [Gom25] Gomez-Zamanillo i in., *Reducing annotation effort in agricultural data*,
  Front. Plant Sci. 2025 — https://doi.org/10.3389/fpls.2025.1546756
- [Lv24] Lv i in., *RT-DETRv2*, 2024 — https://doi.org/10.48550/arXiv.2407.17140

Workspace Undermind (113 prac, deep search z 2026-09-10):
https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893

---

## Historia zmian

- **2026-09-11, refaktor** — skorygowano wymagania recenzenta, status RT-DETR,
  definicję EL2N i zbyt ogólne wnioski o literaturze. Parametry kroków 1–7 są
  propozycjami do sprawdzenia, nie wykonanym ani zwalidowanym pipeline'em.
- **2026-09-11** — pierwsza wersja. Weryfikacja kodu `pipeline/` wobec dziewięciu prac
  przeczytanych w całości; sześć rozbieżności, protokół docelowy, kalibracja oczekiwań.
