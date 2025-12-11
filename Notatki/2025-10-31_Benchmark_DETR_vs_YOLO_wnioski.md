# Benchmark YOLO vs DETR – wyniki, interpretacja i wnioski (2025-10-31)

## Szybkie podsumowanie
- Z raportu benchmarku wynika bardzo niska absolutna skuteczność (mAP na poziomie ~1%), co jest nienaturalne w porównaniu do wcześniejszych analiz (rzędu 60–80%).
- Taka rozbieżność niemal zawsze oznacza problem ewaluacyjny (niedopasowanie kategorii, format bbox, rozjazd ścieżek/zbioru), a nie tylko słabą jakość modelu.
- Mimo to, względne porównanie w tych warunkach pokazało: YOLO ~1.25% mAP@0.5, DETR(100) ~1.23%, DETR(160) ~0.21% – różnice minimalne (lub regresja dla ep. 160), ale przez ewidentny błąd oceny nie należy ich traktować jako miarodajnych.

## Surowe wyniki z logów (skrócone)
- DETR epoch 100: 8250 detekcji na 4670 obrazów; mAP@0.5 ~ 0.012; mAP@[.50:.95] ~ 0.003; AR@100 ~ 0.014
- DETR epoch 160: 8493 detekcji na 4670 obrazów; mAP@0.5 ~ 0.002; mAP@[.50:.95] ~ 0.000; AR@100 ~ 0.013
- Porównanie (zestawienie skryptu):
  - YOLO ep100: mAP@0.5 = 1.25%, mAP@[.50:.95] = 0.34%, AR@100 = 0.42%
  - DETR ep100: mAP@0.5 = 1.23%, mAP@[.50:.95] = 0.34%, AR@100 = 1.44%
  - DETR ep160: mAP@0.5 = 0.21%, mAP@[.50:.95] = 0.03%, AR@100 = 1.29%

Uwaga: absolutne wartości są ekstremalnie niskie. To silna wskazówka błędu konfiguracji ewaluacji.

## Co oznaczają metryki
- mAP@0.5 – średnia precyzja przy progu IoU = 0.5 (wybacza luźniejsze boksy). Im wyższa, tym lepiej.
- mAP@0.5:0.95 – średnia z mAP po progach IoU od 0.5 do 0.95 co 0.05; premiuje dokładne dopasowanie boksów; trudniejsza metryka.
- mAP@0.75 – mAP przy IoU = 0.75; pośrednio mówi o precyzji lokalizacji (dokładność boksów).
- AR@K (tu AR@100) – Average Recall przy maksymalnie K detekcjach na obraz; mierzy odsetek obiektów, które udało się w ogóle „zahaczyć” predykcją (mniej wrażliwy na kalibrację score niż mAP).
- „small/medium/large” – mAP/AR liczone w rozróżnieniu rozmiaru obiektu wg COCO.

## Interpretacja obecnych wyników
1) Absolutny poziom mAP rzędu ~1% dla YOLO i DETR jednocześnie sugeruje problem w ewaluacji, a nie faktyczną jakość obu modeli.
   - Najczęstsze przyczyny:
     - Mismatch `category_id` (np. GT ma id=1 „tooltip”, a predykcje używają 0 lub innej mapy).
     - Zły format bbox w predykcjach (COCO oczekuje [x,y,width,height], a zapisano [x1,y1,x2,y2]).
     - Rozjazd ścieżek/błędny zbiór GT (inne obrazy, inna domena, puste adnotacje).
     - Skala współrzędnych poza wymiarami obrazu (np. znormalizowane wartości zamiast pikseli).
2) DETR ep. 160 ma jeszcze niższy mAP niż ep. 100 – to spójne z wcześniejszą obserwacją regresji po dłuższym treningu (przeuczenie/niestabilność), ale tutaj różnica może być także wzmacniana błędem mapowania.
3) AR@100 ~1.3–1.4% przy ~1.8 detekcji/obraz jest nielogicznie niskie, jeśli detekcje były sensowne. To dodatkowo wskazuje, że większość predykcji nie matchuje GT z powodu błędu „technicznym” (format/mapping), a nie braku obiektów.

## Dlaczego DETR „nie jest lepszy od YOLO” w tych wynikach
- Przy tych absolutnie niskich wartościach mAP nie da się wyciągać wniosków o przewadze architektur. Najpierw trzeba naprawić ewaluację.
- Po naprawie (na rzetelnym, spójnym zbiorze i poprawnym mapowaniu) realny obraz zwykle jest taki:
  - YOLO: lepsze FPS, bardzo dobra skuteczność na średnich obiektach dzięki FPN, mała zależność od skali danych.
  - Vanilla DETR: bez multi‑scale i przy małych zbiorach bywa gorszy; ma tendencję do „query specialization”, bywa wrażliwy na kalibrację score i wymaga staranniejszego datasetu (negatywy, różnorodność, brak duplikatów).
  - W Twojej domenie (jedna klasa „tooltip”, relatywnie mały zbiór) YOLO zazwyczaj wypada lepiej bez modernizacji DETR (np. Deformable/DINO) i bez znaczącej rozbudowy danych.

## Co się najpewniej dzieje „tu i teraz”
Najbardziej prawdopodobne źródła obecnych ~1% mAP:
1) Niezgodność `category_id` między predykcjami i GT.
   - W logach pojawia się też zmiana liczby klas (np. 92→2), co sugeruje remap labeli. Jeśli predykcja niesie id nieistniejące w GT – mAP spadnie do ~0.
2) Format bbox – COCO wymaga `[x, y, width, height]` w pikselach. Wystarczy, że pipeline poda `[x1, y1, x2, y2]` i mAP→0 (IoU ~0).
3) Zbiór/ścieżki – ewaluujesz na innym zbiorze niż detekcje były robione (nawet kosmetyczny rozjazd nazw/ścieżek = brak matchy).

## Checklist naprawcza (priorytet)
1) Kategoria
   - W pliku predykcji COCO sprawdź, czy każdy wpis ma `"category_id": 1` (jeśli GT ma `tooltip`=1). Zero/float/niewłaściwy id od razu psuje ewaluację.
2) Bbox
   - Upewnij się, że predykcje są w formacie COCO: `[x, y, width, height]`, nie `[x1, y1, x2, y2]`.
   - Sprawdź, że `x,y >= 0` i `x+width <= image_width`, `y+height <= image_height`.
3) Zbiór GT i predykcji
   - Czy `file_name` w `images` (GT) odpowiada nazwom obrazów użytych do predykcji? Brak zgodności → mAP≈0.
4) Skala współrzędnych
   - Powinny być w pikselach (nie w [0,1]).
5) Ścieżki i eval crash
   - Błąd `Path.relative_to` w skrypcie (mieszanie ścieżek absolutnych/względnych) nie wpływa na mAP, ale warto go poprawić (użyj `Path.resolve()` lub trzymaj się jednego typu ścieżek).

## Po naprawie ewaluacji – czego oczekiwać
- Absolutne wartości mAP wrócą do sensownych zakresów (dziesiątki procent), co pozwoli rzetelnie porównać YOLO i DETR.
- Detale:
  - mAP@0.5: szybka ocena ogólnej skuteczności.
  - mAP@0.5:0.95 i mAP@0.75: pokażą precyzję lokalizacji (DETR bywa tu słabszy bez multi‑scale/box refinementu).
  - AR@100: pokaże realny recall (czy w ogóle model „zahacza” obiekty); DETR w poprzednich testach miewał recall ograniczony przez query specialization.

## Gdy ewaluacja będzie naprawiona
- Jeśli celem jest przerastanie YOLO:
  - Rozważ Deformable DETR/DINO (multi‑scale, denoising – lepsze na średnie/małe obiekty).
  - Zadbaj o większą różnorodność danych i negatywy; unikać duplikatów;
  - Ewentualnie dodać test‑time augmentations i/lub prosty NMS na wyjściu DETR (koszt FPS).
- Jeśli celem jest stabilna produkcja (RT, ograniczona pamięć): YOLO jest naturalnym wyborem.

## Rekomendacje „co dalej” (konkretne kroki)
- Zweryfikuj predykcje JSON (dla obu modeli):
  - `category_id` – spójność z GT.
  - `bbox` – format `[x,y,w,h]` i zakresy w pikselach.
  - Próbka 10 obrazów: narysuj GT vs pred; sprawdź wzrokowo IoU > 0.5.
- Uruchom ponownie ewaluację po poprawkach;
- Dopiero wtedy porównaj mAP@0.5, mAP@[.5:.95], AR@100 i wnioskuj o przewadze architektur.

---

Wnioski z obecnego benchmarku: wartości ~1% mAP są nienaturalne i wskazują na błąd przygotowania/ewaluacji. Najpierw należy naprawić mapping kategorii oraz format współrzędnych bbox i zgodność nazw plików, a dopiero potem porównywać YOLO vs DETR.
