# Eden: pauza, środowiska i plan pełnych treningów

Stan sprawdzony 2026-09-14 przez SSH `eden-cluster`. Decyzja autora: zakończyć
kontrolę, posprzątać smoke i nie uruchamiać kolejnych treningów bez nowego polecenia.
W tej kontroli nie zgłoszono żadnego nowego zadania SLURM.

## 1. Co jest gotowe

RF-DETR Nano zakończył rzeczywisty smoke na jednym H100 PCIe: zadanie 1807587,
COMPLETED, exit 0, 2 min 30 s. Sprawdzono 5 epok, spadek loss, zapis ostatnich
zwykłych wag i najlepszych wag EMA oraz osobną inferencję obu checkpointów.
Były 64 obrazy train i 64 val, rozdzielone po filmach. Szczegóły i ograniczenia:
[dokument 31](31_EDEN_RFDETR_SMOKE_WDROZENIE.md).

**Gotowy jest tor treningu i ewaluacji Nano na małej próbce starych danych.
Nie jest jeszcze gotowy cały eksperyment naukowy ani sprawdzony trening CATARACT-LMM.**
Smoke nie ocenia skuteczności kuracji; niskie AP po pięciu epokach nie uzasadnia
wniosków o przewadze metody. Nie wykonano nowego smoke, bo ponowienie tej samej
kontroli nie rozwiązałoby braków w danych i macierzy eksperymentu.

## 2. Znalezione środowiska

Sprawdzono `conda env list`, rejestr prefixów, katalog domowy do głębokości 6
(bez cache, .git, node_modules i site-packages), typowe wspólne lokalizacje
oraz metadane pakietów w poniższych interpreterach. To inwentarz znalezionych
środowisk, nie dowód braku innych prefixów na wszystkich dyskach klastra.

| Środowisko | Python | PyTorch | RF-DETR | Ocena do tego projektu |
|---|---|---|---|---|
| systemowe `base` | 3.12.4 | brak | brak | nieprzygotowane do treningu |
| `tsjepa_py311` | — | — | — | usunięte na późniejsze polecenie autora z 2026-09-14 |
| `rfdetr_python311` | 3.11.15 | brak | brak | nowy, minimalny interpreter bazowy dla venv Nano |
| `rfdetr_nano_clean_1_10_1` | 3.11.15 | 2.9.1+cu128 | 1.10.1 | wybrane; historyczny smoke H100, po migracji kontrola importów i `pip check` |

Interpreter do przyszłego treningu:

```text
/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
```

W tym środowisku są torchvision 0.24.1, Lightning 2.6.6, transformers 5.17.0,
NumPy 2.3.5 i pycocotools 2.0.11. Usunięte `tsjepa_py311` miało Lightning 2.5.6,
transformers 4.57.6 i NumPy 2.0.2. W pierwotnie sprawdzonych trzech środowiskach
nie znaleziono pakietu ultralytics ani osobnego zarejestrowanego `yolo_py310`.

Środowisko Nano początkowo współdzieliło pakiety GPU i interpreter z
`tsjepa_py311`. Na późniejsze polecenie autora usunięto tę zależność:
24 wpisy pakietów GPU/metadanych przeniesiono fizycznie do venv Nano,
a identyczny Python 3.11.15 z 32 pakietami bazowymi został odtworzony pod
`~/venvs/rfdetr_python311`. Interpreter venv wskazuje teraz ten nowy prefix.
`GPU_LINKS.json` w działającym venv jest pusty; nie ma już dowiązań GPU do JEPA.
Nie usuwać nowego `rfdetr_python311`, bo dostarcza interpreter i bibliotekę
standardową dla Nano. Ścieżka uruchamiania Nano pozostaje taka sama.

Przed usunięciem JEPA ukryto jego katalog i sprawdzono import RF-DETR/trenera,
PyTorch, torchvision, pycocotools, SSL i SQLite, odczyt wersji cuDNN oraz NMS
na CPU. Nie było odwołań do starego prefixu w `sys.path`, `sys.base_prefix`
ani załadowanych bibliotekach procesu. `pip check` przeszedł, a `pip freeze`
przed i po migracji był identyczny. Nie wykonywano treningu ani ponownego
smoke GPU po migracji; kolejny dopuszczony trening powinien wykonać preflight
CUDA na przydzielonym węźle. Usunięto cały stary katalog wraz z wpisem rejestru
Conda; dane, checkpointy wcześniejszych projektów i kod poza nim pozostawiono.

Dowody migracji i usunięcia:
[stan końcowy](evidence_remove_tsjepa_2026-09-14/final_state.json),
[przeniesione zależności](evidence_remove_tsjepa_2026-09-14/migration.json),
[dokładne pakiety Pythona](evidence_remove_tsjepa_2026-09-14/python_runtime_explicit.txt),
[pakiety Nano](evidence_remove_tsjepa_2026-09-14/rfdetr_freeze.txt).
Pierwotny inwentarz i symlinki pozostają dowodem historycznym:
[environments_final.json](evidence_eden_smoke_2026-09-14/environments_final.json),
[gpu_links.json](evidence_eden_smoke_2026-09-14/gpu_links.json).

## 3. Pauza i sprzątanie

Końcowy `squeue -u bpiotrowski` był pusty. Nie znaleziono procesu naszego
treningu na login node, wpisu cron dla SmallViT/RF-DETR ani timera użytkownika
uruchamiającego ten projekt. Nie było zadania wymagającego anulowania.
Historyczne zadania 1807583–1807587 mają stany końcowe; poprawny wynik to smoke05.
Końcowy odczyt o 20:15 czasu polskiego:
[final_state.json](evidence_eden_smoke_2026-09-14/final_state.json).

Usunięto 245 plików o łącznym rozmiarze 1 409 458 216 bajtów: wyniki smoke01,
smoke02, smoke03 i smoke05, ich checkpointy, robocze dane old64 i ich TAR,
pięć kolejek wraz z wejściowymi JSONL, robocze logi SLURM oraz katalog
`~/rfdetr_smoke_setup`. Dotyczy to wyłącznie artefaktów tej sesji smoke.
Nie ma już starego wpisu pending smoke01, który mógłby zostać podjęty przez worker.

Przed usunięciem zachowano 99 plików raportów, konfiguracji i logów w pasywnym
archiwum o rozmiarze 749 254 bajtów:

```text
~/smallvit_runs/2026-09-14_nano_smoke/audit_pause_2026-09-14/smoke_reports_and_logs.tar.gz
```

Zachowano kod, środowiska, oryginalne zbiory, wcześniejsze niezwiązane eksperymenty
i lokalne dowody smoke. Wagi smoke zostały usunięte; nie można już wykonać ich
ponownej inferencji bez odtworzenia treningu. Archiwum dowodów nie zawiera wag.
Lista sprzątniętych plików: [cleanup_manifest.json](evidence_eden_smoke_2026-09-14/cleanup_manifest.json).
Zapis `TRAINING_PAUSED.md` w katalogu wdrożenia opisuje decyzję autora;
to informacja organizacyjna, a nie techniczna blokada ręcznego `sbatch`.

## 4. Co będziemy trenować po nowym poleceniu

Pierwszy model: **RF-DETR Nano do detekcji obiektów**, z tą samą inicjalizacją
pretrained we wszystkich porównywanych wariantach. Trenujemy detektor na różnych
podzbiorach klatek, aby sprawdzić, czy selektor daje lepszy model przy tej samej
liczbie unikalnych źródeł. Na tym etapie wybór danych jest statyczną kuracją;
nie jest jeszcze aktywną pętlą uczenia ani badaniem dynamiki latentu.

### Dane i warunki wejścia

1. Główne nowe badanie: CATARACT-LMM, zaczynając od danych śledzenia obiektów
   z adnotacjami ramek. Przygotować jednoznaczne klasy, jednostki bbox, pełną pulę
   dostępnych oznaczonych klatek i identyfikatory pacjent/operacja/film/czas.
   Rozdzielić operacje, a gdy identyfikator pacjenta jest dostępny — pacjentów.
   Jeśli identyfikatora nie ma, jawnie ograniczyć deklarację do operacji.
   Nie uznawać nieoznaczonej klatki automatycznie za negatywną.
2. Stare dane: osobna replikacja jako odniesienie do wcześniejszego artykułu.
   Odtworzyć pełną pulę oryginalnych obrazów. Sam `20kSelectedImages.tar` jest
   już wynikiem selekcji i nie wystarcza jako bezstronna pula wszystkich metod.
3. Nie łączyć bez ustalenia wspólnego zadania etykiet końcówki narzędzia,
   całego instrumentu i instancji segmentacji. Nie mieszać wyników z różnych
   ontologii w jednej tabeli jako identycznego zadania detekcji.
4. Zamrozić train/val/test przed selekcją; cechy, PCA, klastrowanie i modele
   pomocnicze dopasowywać wyłącznie na train. Val służy do ustalenia receptury;
   test pozostaje zamknięty do końcowego potwierdzenia.

Po przygotowaniu docelowych danych potrzebny będzie ich własny krótki smoke,
ale dopiero po nowym poleceniu autora. Dzisiejszy sukces old64 go nie zastępuje.

### Pierwsza seria: przewaga kuracji

Proponowana macierz na jednym zbiorze, po sprawdzeniu jego pojemności:

| Wariant wyboru klatek | Co sprawdza |
|---|---|
| Losowe unikalne źródła train | czy metoda przewyższa zwykły losowy wybór |
| Równomierne próbkowanie w czasie | czy wystarcza ograniczenie sąsiednich, podobnych klatek |
| Pokrycie przestrzeni cech (`coverage`) | czy wystarcza prosty selektor różnorodności |
| Pełny, zamrożony pipeline autora | czy dodatkowe składniki dają przewagę nad powyższymi kontrolami |

Budżety: 500, 1500 i 2500 unikalnych klatek. Trzy sparowane seedy: 42, 1337,
2024. To **36 treningów na jeden zbiór**, bez automatycznego mnożenia przez
wszystkie architektury i oba zbiory. Trzy seedy to start; liczebność końcowego
potwierdzenia ustalić z pilota i oczekiwanej wielkości efektu, przed testem.

Istniejący profil `nano_static_pilot` zawiera tylko pierwsze trzy metody:
27 treningów po 30 epok. **Nie zawiera pełnej metody autora, więc sam nie
potwierdzi jej przewagi.** Rozszerzenie do 36 jest planem, nie wdrożoną kolejką.
Przed jego wykonaniem trzeba podłączyć pełny selektor do tej samej puli i
kontraktu budżetu oraz przygotować cache cech dla `coverage`.

Punkt startowy receptury: rozdzielczość 384, BF16, batch 16, jedna GPU na
trening, 30 epok. Sprawdzony sprzęt to H100; DGX wymaga osobnej kontroli modelu
GPU i zgodności środowiska. Nie zakładamy, że dostępność z `sfree` z 11 września
jest nadal aktualna. Przed zamrożeniem receptury skalibrować czas, VRAM i
zbieżność na danych pilota; 30 epok nie jest potwierdzonym czasem zbieżności.
Augmentacje muszą być identyczne między strategiami i wykonywane po selekcji.
Smoke celowo ich nie używał; recepturę naukową należy ustalić osobno.

Dla ustalonego K wszystkie metody dostają identyczny budżet kroków, batch,
optymalizator i harmonogram. Przy różnych K ta sama liczba epok oznacza różny
koszt: obok krzywej jakość–liczba klatek należy przeprowadzić kontrolę przy
stałej liczbie kroków optymalizatora i zapisać liczbę ekspozycji obrazów.
Pomiar kosztu obejmuje również selekcję, ekstrakcję cech i modele pomocnicze.
Nie przeliczać czasu 64-obrazowego smoke liniowo na koszt całej kampanii.

### Druga seria: ablacje i odniesienie do pełnej puli

Najpierw pełny pipeline oraz cztery warianty z usunięciem pojedynczej grupy
D/E/S/F. Potem, zgodnie z zakresem odpowiedzi recenzentowi, uzupełnić wszystkie
15 niepustych kombinacji z [macierzy](../02_MATRIX.md), przy jednym ustalonym K
i trzech seedach. To 45 treningów; trzy pełnego pipeline'u można wykorzystać
z pierwszej serii tylko przy identycznych danych, konfiguracji i protokole,
co pozostawia 42 dodatkowe. Nie zmieniać pozostałych wag dowolnie przy ablacji.
Analizę czułości wag planować osobno na walidacji.

Przed tą serią trzeba rozstrzygnąć DINOv3 względem wersji artykułu, wymiary
cech S/F i definicję E. Obecny score confidence nie staje się EL2N przez samą
nazwę. Historyczne `variants.yaml` i generator wymagają migracji do budżetu
unikalnych źródeł; nie uruchamiać ich jako gotowej macierzy v2.

Dodać pełną pulę train jako odniesienie, początkowo trzy seedy, i kontrolę
przy porównywalnym koszcie. Większa pula nie gwarantuje ani poprawy, ani
pogorszenia; to wynik do zmierzenia. Do mocnej tezy o nowości potrzebny jest
także wdrożony odpowiednik literaturowy uwzględniający obiekty lub zależność
czasową, zgodny z dostępem do etykiet i budżetem. Sam `coverage` nie zastępuje
tego porównania. Lista kandydatów i ograniczenia są we wcześniejszych analizach.

### Co mierzymy i kiedy przechodzimy dalej

- Główny wynik: standardowe COCO AP50:95; uzupełniająco AP50, AP75, AR,
  wyniki klas i małych obiektów, gdy adnotacje na to pozwalają.
- Jakość doboru: udział duplikatów i sąsiednich klatek, pokrycie operacji,
  klas i warunków, wraz z kosztem przygotowania podzbioru.
- Koszt: GPU-godziny treningu i selekcji, czas epoki, pamięć, liczba kroków
  oraz unikalnych klatek. Zamrożone cache i ich koszt raportować jawnie.
- Średnia i odchylenie między seedami oraz sparowane różnice względem
  baseline'ów. Bootstrap po operacjach/pacjentach z ponownym wyliczaniem AP;
  klatki z jednego filmu nie są niezależnymi powtórzeniami.
- Model do głównego raportu wybieramy ustaloną wcześniej regułą na val;
  last i best zapisujemy odrębnie. Nie wybieramy po fakcie lepszego wyniku test.

Dopiero po zamrożeniu metody i protokołu wykonujemy końcową ocenę na test.
Następnie sprawdzamy przenoszenie efektu na większy detektor i kontrolne YOLO
przy tych samych wybranych podzbiorach. To odrębna seria, bez obietnicy wyniku.
Koopman, Langevin, Mori–Zwanzig i aktywne dobieranie kolejnych klatek są kolejną
warstwą badawczą. Nie należą do pierwszej kampanii poprawiającej artykuł.

## 5. Co pozostaje wstrzymane

Nie uruchamiać obecnie smoke, pilota, pełnej macierzy, treningów modeli
pomocniczych ani automatycznego wznowienia. Ten dokument zapisuje następne
kroki; ich wykonanie wymaga nowego polecenia autora.
