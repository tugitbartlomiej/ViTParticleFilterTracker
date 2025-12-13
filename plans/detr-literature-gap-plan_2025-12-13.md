# DETR literature gap analysis → upgrade plan for `access.tex`

## Scope
Plan dotyczy uzupełnień merytorycznych i eksperymentalnych do:
- `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`

Na podstawie wybranych prac z:
- `F:\Studia\Articles\Zbiory\DETR\`

## Przeczytane / wykorzystane prace (wycinek)
Poniżej tylko te, które dały konkretne „braki do uzupełnienia” (nie analizuję całego folderu, bo to nie daje dodatkowej wartości):

### DETR i linia rozwojowa
- `F:\Studia\Articles\Zbiory\DETR\2005.12872v3.pdf` — **DETR (Carion et al.)**: set prediction + Hungarian matching, brak NMS, ale **wymaga długiego treningu** i ma problem z małymi obiektami.

### Real-time DETR (DETR vs YOLO framing)
- `F:\Studia\Articles\Zbiory\DETR\2304.08069v3.pdf` — **RT-DETR**: real-time end-to-end detektor, **hybrid encoder (multi-scale)** + **uncertainty-minimal query selection** + tuning liczby warstw dekodera.
- `F:\Studia\Articles\Zbiory\DETR\2407.17140v1.pdf` — **RT-DETRv2**: “bag-of-freebies” (m.in. **dynamic data augmentation**, **EMA**, poprawki deployability przez discrete sampling).
- `F:\Studia\Articles\Zbiory\DETR\2409.08475v3.pdf` — **RT-DETRv3**: teza „Hungarian matching = zbyt rzadka superwizja”; proponuje **hierarchical dense positive supervision** (training-only moduły).

### Szybsze/efektywniejsze trenowanie DETR
- `F:\Studia\Articles\Zbiory\DETR\MS-DETR_Efficient_DETR_Training_with_Mixed_Supervision.pdf` + `F:\Studia\Articles\Zbiory\DETR\2401.03989v1.pdf` — **MS-DETR**: miesza one-to-one i one-to-many supervision bez dokładania gałęzi inferencyjnych.

### DETR w medycynie (design choices)
- `F:\Studia\Articles\Zbiory\DETR\2405.17677v2.pdf` — **DETR w obrazowaniu medycznym**: wiele „standardowych ulepszeń” z natural images (multi-scale fusion, IBBR, query init) bywa neutralne lub szkodliwe; często **prostsza konfiguracja** działa równie dobrze/lepiej.

### Wideo / temporal context (dla przyszłej wersji pracy)
- `F:\Studia\Articles\Zbiory\DETR\2408.14051v1.pdf` — **V2I-DETR**: video-to-image knowledge distillation (teacher wieloklatkowy → student jednoklatkowy) dla real-time.

### Kontekst chirurgii oka + data-efficiency
- `F:\Studia\Articles\Zbiory\DETR\Leveraging active learning techniques for surgical instrument recognition and localization_Revised.pdf` — active learning/annotation efficiency dla narzędzi chirurgicznych (YOLOv5).
- `F:\Studia\Articles\Zbiory\DETR\OPTH-438127-artificial-intelligence-in-ophthalmic-surgery--current-appli.pdf` — przegląd AI w chirurgii okulistycznej (motywacja/clinical framing).

**Uwaga:** `Healthcare Tech Letters - 2023 - Loza - ...pdf` wygląda na skan (pdftotext nie zwraca tekstu), więc nie wykorzystałem jej merytorycznie.

---

## Co jest „brakiem” w Twoim `access.tex` w porównaniu do tej literatury

### A) Brakuje mocniejszego powiązania Twoich obserwacji z aktualnym nurtem DETR
1. **„Sparse supervision” przez Hungarian matching** jest teraz głośnym tematem (RT-DETRv3, MS-DETR). U Ciebie jest opis „winner-take-all / query specialization”, ale bez mostu do tej literatury.
2. W Related Work masz RT-DETR jako cytat, ale brakuje **RT-DETRv2/v3** oraz wątku „bag-of-freebies” i „dense positive supervision”.

### B) Brakuje eksperymentu/baseline’u z nowszej rodziny DETR
W 2024–2025 standardem porównania do YOLO nie jest już tylko „vanilla DETR”. Minimum to **RT-DETR** (albo Deformable/DINO-family), nawet jeśli tylko jako „sanity baseline” na Twoim zbiorze.

### C) Brakuje „design-choice ablation” specyficznego dla medical domain
`2405.17677` pokazuje, że w medycynie pewne ulepszenia są neutralne/szkodliwe. U Ciebie jest świetna data-centric część, ale warto dodać choć małą ablacją:
- liczba encoder layers,
- liczba queries,
- multi-scale vs single-scale,
- box refinement / loss.

### D) Temporal context jest tylko w future work (bez osadzenia w literaturze)
Masz ograniczenie „static images”, ale możesz to wzmocnić o konkretne rozwiązania z literatury wideo (V2I-DETR) — to uwiarygadnia future work.

### E) Data-efficiency: masz mocny pipeline, ale brak osadzenia w active learning dla chirurgii oka
Aktywny dobór danych (active learning) dla narzędzi w okulistyce pojawia się w folderze — warto to zestawić z Twoim pipeline (Fourier/DINO/K-center/EL2N) jako „label-efficient curation”.

---

## Plan uzupełnień do artykułu (wersja „lepsza”, bo literatura-driven)

### P0 — uzupełnienia, które podnoszą wiarygodność claimów DETR-vs-YOLO
1) **Dodać mini-subsection w Related Work (Transformer Detectors, 2023–2025)**
- Dodać i omówić: RT-DETR (już masz), RT-DETRv2, RT-DETRv3, MS-DETR.
- Jedno zdanie „dlaczego to ważne dla tej pracy”: (a) convergence, (b) dense supervision vs Hungarian matching, (c) real-time bez NMS.

2) **Zrobić 1 baseline na Twoim zbiorze: RT-DETR**
- Cel: pokazać czy „transformer detector” może dogonić YOLO na strict IoU i FPS w praktyce.
- Minimalny wariant: RT-DETR-R18/R34 (jeśli zasoby ograniczone) + to samo splitowanie.
- Raportować te same metryki co dla YOLO/DETR.

3) **Uzupełnić dyskusję o „sparse supervision” jako możliwe źródło query specialization**
- Połączyć Twoją obserwację (dominant query) z RT-DETRv3/MS-DETR: rzadkie pozytywy → trudniejsze trenowanie → potencjalna koncentracja odpowiedzialności.

### P1 — uzupełnienia, które realnie wzmacniają wkład naukowy (bez przepisywania paperu od zera)
4) **Dodać ablation „DETR design choices for medical” (inspirowane `2405.17677`)**
Minimalny zestaw (wystarczą 3–4 punkty):
- `#queries`: 1 / 10 / 100 (masz już częściowo, ale ubrać to jako formalną ablacją)
- `encoder layers`: np. 2 / 6 (jeśli implementacja pozwala)
- `multi-scale`: single-scale vs (prosty) FPN/deformable jeśli dostępne
- `box loss`: GIoU vs (np. DIoU/CIoU/PIoU) — tylko jeśli to łatwe do wdrożenia i nie rozjeżdża porównania

5) **Zrobić „training-efficiency” add-on (tylko jeśli masz czas/compute)**
- Krótki eksperyment: „czy da się zejść z 160 epok bez spadku mAP”, inspirowany MS-DETR/RT-DETRv3 (dense positives).
- Jeśli nie robisz eksperymentu: dodać jako future-work z konkretnymi cytowaniami.

6) **Wzmocnić future work o konkret: video-to-image distillation (V2I-DETR)**
- Dopisać 1 akapit w Limitations/Future Work: „jak wykorzystać temporal context bez utraty real-time”.

### P2 — kliniczne i data-centric wzmocnienie narracji
7) **Dodać 2–3 cytowania „AI in ophthalmic surgery” do wstępu/dyskusji**
- Cel: lepsze osadzenie w real-time guidance i wymaganiach klinicznych (FP cost).

8) **Dodać „active learning vs intelligent selection” w Related Work**
- Pozycjonowanie: Twój pipeline jako alternatywa/uzupełnienie active learning (mniej iteracyjny, bardziej „offline curation” + difficulty + diversity).

---

## Konkretne deliverables (co finalnie ma się pojawić w `access.tex`)
- [ ] Nowe cytowania i akapity: RT-DETRv2, RT-DETRv3, MS-DETR, V2I-DETR, active learning w okulistyce, review AI w chirurgii oka.
- [ ] Jedna nowa tabela/sekcja: „DETR-family baseline (RT-DETR) vs YOLOv8 vs DETR” na Twoim zbiorze.
- [ ] Jedna nowa ablation tabela: „DETR medical design choices” (minimum: #queries).
- [ ] Zaktualizowana dyskusja: „sparse supervision / Hungarian matching” jako kontekst dla query specialization.

---

## Ryzyka / uwagi wykonawcze
- Jeśli nie jesteś w stanie uruchomić RT-DETR w tym samym pipeline co YOLO/DETR, to i tak warto dodać go jako **future work z jasnym uzasadnieniem** (real-time + bez NMS).
- W medycynie część „ulepszeń” DETR może nie pomagać (`2405.17677`) — dlatego ablacją powinna być mała, ale dobrze opisana protokołem.
