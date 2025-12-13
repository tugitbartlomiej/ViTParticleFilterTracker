# Plan dalszych poprawek do artykułu IEEE Access (`access.tex`) — 2025-12-13

## Kontekst
- Manuskrypt: `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`
- Powiązane materiały w repo:
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Plans\ieee-access-writing-audit_access-tex_2025-12-13.md` (audyt pod IEEE/Access)
  - `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Plans\detr-literature-gap-plan_2025-12-13.md` (braki vs literatura DETR/RT-DETR)
- Ostatnia zmiana w tekście paperu: doprecyzowanie „query concentration / winner-take-all” + dodanie kontekstu literaturowego, bez overclaimów.

## Co jeszcze warto poprawić (największe ryzyka recenzenckie)
Poniżej są rzeczy, które recenzent najprawdopodobniej zakwestionuje jako pierwsze (wiarygodność, spójność, protokół), zanim w ogóle przejdzie do „czy to jest ciekawe”.

### P0 — krytyczne: spójność liczb + protokołów (must-fix przed submission)
- [ ] **Ujednolicić „który checkpoint jest porównywany” (YOLO vs DETR).**
  - Problem: w tekście pojawia się jednocześnie `YOLOv8 (e100)` i opis/figura treningu „over 200 epochs”, a dla DETR: „best at e160”, ale inspekcja obejmuje epoki 60–170.
  - Co poprawić: wprowadzić jednoznaczną zasadę: „best checkpoint wybierany po walidacji” + konsekwentne oznaczenia (np. *best* zamiast *e100*), a epoki traktować jako informację pomocniczą.
  - Akceptacja: każda tabela/figura mówi, na jakim splocie i jak wybrano checkpoint.

- [ ] **Naprawić niespójności w analizie query (tabela vs figura vs opis).**
  - Obserwacja: w `Results/Query Specialization` występują jednocześnie wartości typu „96.3%” oraz w podpisie figury zakres „42–48%”, a liczba „active queries” różni się między tabelami/podpisami.
  - Co poprawić: dopisać *dokładnie* na jakim zbiorze (train/val/test, ile obrazów), przy jakim progu i jak zliczasz „detections” i „hit rate”. Jeśli dane pochodzą z różnych splitów — nazwać to wprost.
  - Akceptacja: jedna definicja `HitRate` + jeden pipeline zliczania + brak sprzecznych liczb.

- [ ] **Cross-patient generalization: spójność metryk i „zdrowy rozsądek” liczb.**
  - Ryzyko: bardzo niskie mAP/precision w tabelach cross-patient może stać w sprzeczności z wcześniejszymi wynikami (80%+ mAP), jeśli to nie jest jasno opisane jako *inny protokół / inny zbiór / inna metryka*.
  - Co poprawić: ujednolicić protokół ewaluacji (COCO mAP vs TP@IoU) i jawnie powiedzieć, że to jest osobny eksperyment pilotażowy (albo go przenieść do Appendix/Limitations).
  - Akceptacja: czytelnik rozumie „dlaczego tu są inne liczby” bez domyślania się.

- [ ] **Poprawić terminologię: Hungarian matching nie jest „greedy”.**
  - Co poprawić: zamienić sformułowania typu „greedy assignment” na „optimal bipartite assignment / cost-minimizing assignment”.
  - Akceptacja: brak merytorycznych „red flags” w opisie podstawowego algorytmu.

- [ ] **Ujednolicić opis seedów.**
  - Problem: część tekstu sugeruje „seed=42 dla wszystkich eksperymentów”, ale masz też tabelę z wieloma seedami (dominant query index).
  - Co poprawić: rozdzielić „main run (seed=42)” vs „analysis runs (multiple seeds)”.
  - Akceptacja: 1 zdanie w `Implementation Details` + 1 zdanie w sekcji query analysis.

### P1 — reprodukowalność i klarowność metod (wzmacnia wiarygodność)
- [ ] **Dopisać brakujące parametry pipeline selekcji danych (tak, by dało się to odtworzyć).**
  - Przykłady: progi/percentyle dla Fourier filtering, szczegóły DINO (model/warstwa/normalizacja), definicja `k` w K-Center, konfiguracja EL2N (proxy-model, liczba epok, split, augmentacje).
  - Akceptacja: osoba z zewnątrz jest w stanie uruchomić „Stage 1–4” bez zgadywania.

- [ ] **Uporządkować ablation pipeline tak, by odpowiadał realnej kolejności etapów.**
  - Ryzyko: jeśli ablation pokazuje etapy w innej kolejności niż metodologia, recenzent uzna to za „wykresy pod tezę”.
  - Akceptacja: tabela ablation = ta sama kolejność co opis + jasne „co jest dodane/wyłączone”.

- [ ] **Oddzielić mAP (metryka rankingowa) od „operacyjnego progu” (deployment).**
  - W tej chwili masz „confidence threshold analysis”; warto doprecyzować, czy mAP liczysz po odfiltrowaniu predykcji progiem (to nie jest standard COCO).
  - Akceptacja: jasne PR-curve/operating point (np. fixed precision) + opis, do czego służy threshold.

- [ ] **Dodać niepewność/statystykę dla kluczowych wyników.**
  - Minimum: 3 seedy dla głównych tabel (mAP, ablation, query concentration) i raportować średnia±std albo CI.

### P1 — urealnić i wzmocnić cross-patient generalization (żeby claim był obroniony)
- [ ] **Więcej pacjentów / więcej splitów** (np. leave-one-patient-out, albo ≥5 pacjentów).
- [ ] **Te same metryki co w głównym benchmarku** (mAP/AR), plus klinicznie ważne: FP/min, FROC lub precision przy zadanym recall.
- [ ] **Język wniosków ostrożniejszy**: unikać zdań typu „CNN nie transferuje”, tylko „w tym ustawieniu YOLO silnie traci”.
- [ ] **Kalibracja i koszty FP**: w okulistyce false positive może być „droższy” niż FN; warto to wyeksponować liczbowo.

### P2 — dodatkowe wzmocnienia naukowe (jeśli masz czas/compute)
- [ ] **Baseline z rodziny nowszych DETR** (najlepiej RT-DETR; alternatywnie Deformable/DINO) na tym samym splocie.
- [ ] **Bridge do literatury o sparse supervision** (RT-DETRv3, MS-DETR itd.) w Related Work + Discussion: połączyć Twoją obserwację koncentracji query z nurtem „dense positive supervision”.
- [ ] **„Medical design-choice ablation”**: liczba query (1/10/100) już masz — ubrać formalnie; opcjonalnie encoder layers / multi-scale.

### P2 — porządki redakcyjne/IEEE Access
- [ ] **Keywords**: zejść do ~6–8, ułożyć alfabetycznie, możliwie blisko IEEE Taxonomy.
- [ ] **Data/Code availability + ethics/privacy**: nawet krótka notka często uspokaja recenzenta (zwłaszcza medycyna).
- [ ] **Checklist figur**: spójne jednostki, czytelność fontów, brak sprzecznych liczb w captionach.

## Sugerowana kolejność pracy (praktyczna)
1) P0: spójność checkpointów + spójność query + poprawki terminologii + seedy.
2) P1: dopisać parametry pipeline + uporządkować ablation + doprecyzować metryki/threshold.
3) P1: przebudować cross-patient (albo jasno oznaczyć jako pilotaż i osłabić claim w Abstract/Conclusion).
4) P2: dodać RT-DETR (lub inny DETR-family baseline) i zaktualizować Related Work/Dyskusję.

## „Gotowe deliverables” (co powinno wyjść na końcu)
- Jedna tabela „Training/eval protocol summary” (split, seed, selection of checkpoint, thresholding).
- Jedna tabela „Query concentration across seeds + across splits” (bez sprzecznych liczb).
- Uporządkowana ablation pipeline zgodna z metodologią.
- Cross-patient: protokół + metryki + wnioski bez overclaimów (albo przeniesione do Appendix/Limitations).

