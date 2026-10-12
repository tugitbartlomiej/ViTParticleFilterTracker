# Adaptacyjny dobór klatek według brakujących cech

**Cel:** po każdej rundzie treningu jednego narzędzia diagnozować cechy trudnych widoków, wyszukiwać odpowiadające im obrazy i wybierać następny podzbiór według przewidywanej korzyści zadaniowej.

**Architektura:** jeden nieprzerwany trening RF-DETR Nano; stały budżet obrazów na epokę; diagnoza cech i Koopman–GLE na aktualnych wagach, następnie wyszukiwanie i przybliżenie GLISTER na gradientach głowic. Niezależny zbiór operacji D steruje wyborem, a val mierzy jakość. Cząstka GLE pozostaje pojedynczym patchem.

**Technologie:** Python 3.11, NumPy, PyTorch, RF-DETR 1.10.1, PyTorch Lightning, Slurm/Eden.

**Specyfikacja:** zaakceptowany projekt z rozmowy: diagnoza brakujących umiejętności względem cech → wyszukanie obrazów z tymi cechami → wybór według przewidywanej korzyści → trening.

## Ograniczenia i decyzje

- Nowy katalog eksperymentu, poprzednie wyniki i kod bez zmian.
- Użytkownik zlecił kod i przygotowanie testów na Eden; nie uruchamiać lokalnych smoke testów, treningów ani kampanii na Edenie w tej rundzie.
- Testy zapisujemy przed implementacją; wykonanie red/green odroczone do Edena zgodnie z zakresem zlecenia. Nie deklarujemy zaliczenia testów.
- Praca w istniejącym checkoutcie, izolowana katalogiem eksperymentu; zależności poprzedniego pilota są jeszcze niewersjonowane, więc nowy worktree nie zawierałby ich. Bez commitów i bez zmian cudzych plików.
- Świeża wspólna inicjalizacja COCO. Stary L0 nie jest używany, bo mógł widzieć operacje wydzielone teraz do D.
- Podział po group_id: FIT/pula treningowa, D/sterowanie, val/ocena. Żaden gradient treningowy ani fit GLE/sondy nie używa D. Etykiety D mogą sterować selekcją. Test nie jest odczytywany.
- Geometria bbox, jasność, ostrość i udział bardzo jasnych pikseli są mierzalnymi cechami/proxy. Kąt, zasłonięcie i widoczność końcówki są opcjonalnymi jawnymi adnotacjami; nie zgadujemy ich z aspect ratio.
- Sondy cech mają przestrzenny opis tokenów (średnia i momenty x/y); to osobny opis diagnostyczny, nie zmiana definicji cząstki GLE.
- GLE opisuje czas filmu przy aktualnych wagach. Różnice między epokami mierzymy w zadaniu/sondach, nie dopasowujemy GLE do kilku checkpointów.
- GLISTER jest adaptacją z ograniczeniem do gradientów wybranych końcowych warstw klasyfikacji i bbox, z ponowną oceną gradientu D po wirtualnych krokach. Nie dziedziczy automatycznie gwarancji oryginalnej metody.
- Stałe archiwa wejściowe i hashe kodu, niezmienne katalogi rund, pełne checkpointy, raporty potrzeb i obrazów z uzasadnieniem wyboru.

## Zadania implementacyjne

1. [x] `dane.py`, `przygotuj.py`: podział operacji, manifesty, źródła cech, materializacja COCO, walidacja rozłączności i budżetów.
2. [x] `cechy.py`, `diagnoza.py`, `selekcja.py`, `glister.py`: słownik cech, sondy wyłącznie FIT, grupy braków wyłącznie D, wyszukiwanie kandydatów, eksploracja i wybór gradientowy. Testy obalają używanie cech brakujących jako zer, wyciek grup i wybór poza opisanymi potrzebami.
3. [x] `dynamika.py`, `vendor/`: zamrożona adaptacja poprzedniego D+JP+W+Z+N, ekstrakcja aktualnych patchy, wyłączenie D z fitu. Testy obalają pooling patchy, dostęp do przyszłości i fit na D.
4. [x] `rfdetr_backend.py`, `trenuj.py`: rzeczywista strata detektora, gradienty kandydatów i D, zmienny sampler o stałej długości, callback po epoce, pełne checkpointy bez restartu optymalizatora.
5. [x] `run_tests.slurm`, `run_experiment.slurm`, `preflight.py`, `submit_eden.sh`, `README.md`, `WYNIKI.json`: przygotowanie uruchomienia; testy kontraktowe i integracyjne wykonywane wyłącznie na Edenie.
6. [x] Przegląd integracji i statyczna kontrola składni; opis stanu bez sugerowania wykonanych pomiarów.

## Ramiona i kryteria

`random`, `glister`, `features_glister`, `features_kg_glister`, `features_placebo_glister`. Wszystkie widzą te same obrazy, etykiety i D; ramiona różnią się użytym kryterium. Eksploracja i limit kandydatów wspólne dla trzech ramion cechowych. Placebo tasuje przyrost GLE wewnątrz grup i przedziałów błędu, wyłącznie tam, gdzie istnieje historia.

Główne porównanie: `features_kg_glister` względem `features_glister` i `glister`; osobno względem placebo. Negatywny wynik: brak poprawy AP/czasu do ustalonego AP pomimo lepszej predykcji latentu. Liczyć czas pełnej procedury, liczbę aktualizacji, ekspozycji i unikalnych obrazów. Liczba epok sama nie wyrównuje kosztu. Plan pilota: 10 epok, budżet 1500, seedy 42/1337/2024; uruchomienie nie należy do bieżącego zlecenia.

## Stan weryfikacji

Kod i opis przygotowane. Statyczny przegląd integracji i składni wykonany. Testów wykonawczych nie uruchomiono; testy jednostkowe oraz integracja dwóch epok czekają na Eden. Wyników naukowych brak.

## Decyzje z integracji

- Trening używa niezmiennego FIT dataset i zmiennego samplera; faktyczne ID z każdego batcha są sprawdzane względem planu. Stałe skalowanie do kwadratu zapewnia wspólne współrzędne patchy i bbox.
- RF-DETR 1.10.1 ma dodatkowy slot głowicy: jedna klasa oznacza dwa kanały logitów. Zweryfikowano w źródłach zainstalowanych na Edenie.
- FIT/D w pierwszym pilocie wymaga jednego bbox na klatkę; dodanie selekcji tła będzie osobnym rozszerzeniem.
- Hashe obejmują także COCO w podkatalogach i rzeczywiste archiwa wejściowe. Marker zaliczonych testów jest związany z kodem, konfiguracją i archiwami.
- Pełne checkpointy są archiwalne; automatyczne resume częściowego biegu nie jest zaimplementowane. Sam trening między epokami nie jest restartowany.
- GLE używa max(e_obs-e_dyn,0), czyli różnicy BCE słabej sondy patchy. Nie nazywamy tego zmierzoną korzyścią treningową.
