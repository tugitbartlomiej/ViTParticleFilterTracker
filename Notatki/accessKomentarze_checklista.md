# Checklist z komentarzy w `accessKomentarze.pdf` (DETR_IEEE/access.tex)

## Liczby i spójność danych
- [ ] Ujednolicić rozmiar zbioru: komentarze wskazują, że powinno być **20 000, nie 20 900** (str. 1–2, 7). Sprawdzić, skąd 20 900 i zdecydować jedną liczbę w całym tekście (abstract, tabele, podpisy).
- [ ] Zweryfikować redukcję 4.3× i **+8.5pp mAP** (str. 2, turkus: „Upewnić się czy prawda?”); potwierdzić bazę porównawczą i liczbę obrazów.
- [ ] Dataset composition table (str. 7): komentarz „Skąd te liczby???” – podać źródło/objaśnienie 90 141→20 900/32 000.

## Wyniki / porównania
- [ ] Tabela YOLO vs DETR (str. 8–9): „Kategorycznie do poprawy!” – zweryfikować Model Size (100 vs 475 MB), FPS, VRAM oraz IoU metryki; dodać protokół ewaluacji (sprzęt, batch, FP16?).
- [ ] Cross-patient generalization (str. 11): komentarz „Nie wiem czy tak ma być!” – doprecyzować protokół (split, liczba pacjentów, metryka TP@0.5 vs mAP) lub przenieść do dodatku.

## Query specialization / statystyki
- [ ] Tabela dominujących query (str. 6): komentarz „NIE ROZUMIEM DO KOŃCA PO CO?” – jasno napisać, czemu pokazujemy seedy, co oznacza hit rate i jak to wspiera wniosek.
- [ ] Gini coefficient 0.42 vs 0.98 (str. 6): krótko objaśnić metodę liczenia i źródło danych (COCO vs surgical).

## Pipeline / trening / ablation
- [ ] EL2N adaptacja (str. 5): dodać ref („~Tutaj trzeba ref”) i krótkie wyjaśnienie formuły.
- [ ] Ablacja (+1.8pp, +3.8pp, +6.2pp, +8.5pp) (str. 10): komentarze „Skąd te dane?” – podać protokół (split, seed, liczba obrazów) i/lub tabele w dodatku.
- [ ] Epoki/augmentacja: komentarze o „5k?”, „10 epochs” (str. 5) – uściślić liczby i konfiguracje użyte w pipeline (proxy model EL2N, liczba epok).
- [ ] „previously reported” (str. 7): złagodzić/uzasadnić stwierdzenie o długim trenowaniu DETR w imaging.

## Terminologia i skrótowce
- [ ] Wyjaśnić skróty: **BSS**, DC (komentarz „Co to?”), **CLS**, oraz spójnie używać „4-stage” vs „four-stage”.
- [ ] Rozszerzyć „DC”/„cosine similarity” tam, gdzie padają w równaniach.
- [ ] Spójny styl sekcji/figur: uwagi „FIGURE” (styl) i „Styl” (str. 4) – dostosować do wymogów IEEE Access.

## Cytowania / benchmarki
- [ ] Rozważyć benchmark/omówienie: Go-ELAN YOLOV9, CATARACTS (str. 2): komentarze „Może warto benchmark?”.
- [ ] Sprawdzić, czy warto zostawiać wzmiankę o udostępnieniu kodu („LEPIEJ NIE!” na str. 7) – jeśli zostaje, dodać informację o warunkach/licencji; jeśli nie, usunąć.

## Sekcja metod (problem formulation)
- [ ] Komentarz „Czy to ma być tak matematycznie?” (str. 3): rozważyć skrócenie/opis słowny lub dodać intuicyjny akapit przed równaniami.
- [ ] Dodać referencję do wzoru z Hungarian matching (`σ* = arg min ...`) – wskazano brak ref (str. 6).

## Spójność narracji
- [ ] Ujednolicić nazwy („4-stage” / „four-stage”; „Stage 1/2/3/4” w całym tekście).
- [ ] Doprecyzować „All code and pretrained models will be made available…” (str. 7) – rozważyć usunięcie lub podanie konkretów.

## Kluczowe pytania do decyzji
- [ ] Czy utrzymujemy liczbę 20 900 (wtedy poprawić komentarze 20 000) czy przechodzimy na 20 000? – decyzja globalna.
- [ ] Czy dodajemy dodatkowe benchmarki (YOLOV9/CATARACTS) czy tylko omówienie? – zależnie od czasu/compute.
- [ ] Jak prezentować cross-patient: mAP vs TP@0.5, ilu pacjentów, czy dołączyć do dodatku? – ustalić finalny protokół.
