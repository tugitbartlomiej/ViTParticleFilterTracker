# 21 — Refaktor ewaluacji i dokumentacji

Data: 2026-09-11. Zakres: korekty wspólnego evaluatora, metryk transferu i niespójnych zaleceń w dokumentach. Zmiany wykonano na istniejącej kopii roboczej, zachowując wcześniejsze niezacommitowane pliki i konfigurację użytkownika.

## Poprawki kodu

- COCOeval ocenia wszystkie obrazy wskazane przez GT, także bez predykcji. Obcy `image_id` kończy obliczenie błędem zamiast znikać z wyniku.
- Import predykcji bez jawnej listy ocenionych obrazów oznacza **cały przekazany zbiór anotacji**. Podzbiór należy przekazać jako `--evaluated-images` albo `--images-dir`; musi odpowiadać faktycznie ocenionym obrazom. Lista predykcji nie jest manifestem obrazów.
- Brak `operating_point.json` pozwala dopasować próg tylko na `val`. Test i zbiory zewnętrzne wymagają zapisanego progu walidacyjnego; jego pochodzenie jest sprawdzane. Jawny próg CLI jest nadal dostępny i oznaczany jako `cli`, więc nie jest przedstawiany jako wynik strojenia na val.
- Remap pojedynczej klasy dopuszcza indeks modelu 0 i właściwy identyfikator GT; nie zamienia dowolnej klasy 99 na prawidłową. Analiza krzywych respektuje wyłączenie remapu w konfiguracji.
- Ponowne składanie metryk pomija agregaty, także zapisane pod własną nazwą `--metrics-name`.
- Import sprawdza identyfikatory ocenianych obrazów przed zastąpieniem poprzedniego pliku predykcji.
- Transfer korzysta ze wspólnej implementacji COCO AP i zliczania TP/FP/FN. Jego pomocniczy matcher wybiera najlepszy **wolny** GT; parsery grupują po parze `(image_id, category_id)`. Tuning progu i NMS nie łączą różnych klas.
- Domyślne kandydaty progów w transferze obejmują score predykcji, również poniżej 0,01.

## Testy lokalne

Zapis kontroli: [REFAKTOR_WERYFIKACJA_2026-09-11.json](pochodne/REFAKTOR_WERYFIKACJA_2026-09-11.json). Dodatkowo uruchomiono syntetyczne demo NMS, sprawdzono składnię sześciu zmienianych plików Python i 110 lokalnych odsyłaczy w indeksie, raporcie oraz notatkach archiwalnych. Demo nie jest wynikiem eksperymentu na danych rzeczywistych.

Z korzenia repozytorium:

```powershell
py -3.11 -m pytest SmallViTAblation/pipeline/eval/tests/test_evaluation_contract.py -q
```

Testy używają małych anotacji COCO, rzeczywistego pycocotools i plików tymczasowych. Nie wymagają modelu, OCR ani GPU. Przed poprawkami odtworzono błędy; po poprawkach przechodzą 23 przypadki, w tym pełna ścieżka CLI: walidacja, zapis progu, ocena testu i agregacja. AP50 dla dwóch GT i jednej idealnej predykcji wynosi około 0,50495, AR100=0,5. To wynik testu evaluatora, nie detektora.

Przechodzą również dwa warianty dry-run i trzy scenariusze kolejki uruchamiane własnymi skryptami:

```powershell
py -3.11 SmallViTAblation/pipeline/train/tests/test_dry_run.py
py -3.11 SmallViTAblation/eden/tests/test_queue.py
```

W teście dry-run poprawiono sprawdzanie SHA commita: wcześniej wymagał 64 znaków od 40-znakowego SHA-1. Skrypt kolejki wymaga uruchomienia jak wyżej; bezpośrednia kolekcja przez pytest zgłasza trzy błędy brakującego fixture `tmp`, więc nie deklarujemy przejścia całego drzewa przez pytest.

Bezpieczna kolejność oceny plikowej: `evaluate.py --set val`, następnie `operating_point.py --val-set val`, następnie `evaluate.py --set test` z tym samym katalogiem wyników. Końcowe krzywe dla pozostałych zbiorów można przeliczyć po ich ocenie. Przy dwóch checkpointach używać **oddzielnych katalogów wyjściowych**, np. `results/<run_id>/last_epoch/` i `results/<run_id>/best_val/`.

## Granice tego refaktoru

To nie jest ukończony adapter RT-DETR ani walidacja treningu RF-DETR. Otwarte pozostają m.in. rzeczywiste API i wersje modeli, rozpoznawanie epok checkpointów, automatyczne powiązanie progu z hashem checkpointu, pełny kontrakt wielu bboxów, produkcyjne ekstraktory DINO/SAM, cache i definicja sygnału trudności. Nie uruchomiono nowych treningów ani nie przeliczono historycznych checkpointów.

Evaluator nadal zwraca 0 dla metryk AP niezdefiniowanych z powodu braku GT — wymaga to osobnej migracji kontraktu wyników i agregacji na `null`/NA. Operacyjne liczniki zakładają zwykłe, wyczerpujące anotacje bbox; pełna semantyka `iscrowd`/ignore/area-range jest obsługiwana przez COCO AP, ale nie przez pomocnicze P/R/F1. Nie używać tych liczników do crowd benchmarku bez rozszerzenia kontraktu.

Limit 200 punktów krzywej F1 w głównym evaluatorze daje przybliżony wybór z siatki, nie gwarantowane maksimum po wszystkich score. W transferze domyślna siatka jest dokładna po unikalnych score, co dla dużych plików może być kosztowne. Samodzielny skrypt NMS wciąż wymaga audytu pochodzenia predykcji sprzed NMS i wskazania walidacyjnej roli danych. Jego pole `significant` oznacza przekroczenie arbitralnej różnicy F1 = 0,02, a nie test istotności statystycznej; nie używać tego komunikatu jako wniosku statystycznego.

## Korekty metodologiczne dokumentacji

- Istniejący split 35/8/7 i potwierdzenie autora o unikalnych pacjentach są użyteczne dla nowych biegów; nie zmieniają podziału historycznego checkpointu.
- Budżet unikalnych klatek źródłowych jest oddzielony od liczby augmentowanych obrazów. 5000 źródeł razy cztery warianty nie realizuje losowania 20 000 źródeł.
- Train-on-train scoring nie jest automatycznie wyciekiem. EL2N wymaga właściwej definicji błędu i opisu estymacji; liczba seedów nie jest uniwersalnym standardem.
- RT-DETR odpowiada na prośbę o nowoczesny detektor DETR-style; recenzent nie nakazał backbone'u ViT ani konkretnego małego RF-DETR.
- Różnice rozdzielczości, liczba seedów i podobieństwo embeddingów nie dają same z siebie gwarancji istotności, generalizacji ani równego kosztu.
- Grupy rozpoznane w Cataract-LMM to grupy do konserwatywnego podziału, a nie potwierdzone tożsamości pacjentów. Brak mapowania tracking–segmentacja wyklucza deklarację ich niezależności.

Materiały `notatki/` i `pochodne/evidence/` zachowują historię diagnozy. Stare wyniki reprodukcji opisane jako „problem odtworzony” nie są bieżącym wynikiem testów poprawionej wersji.
