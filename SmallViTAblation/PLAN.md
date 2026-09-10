PLAN — uporządkowany według zarzutów recenzentów
================================================

Wersja: 2026-09-09. To jest **dokument wejściowy**. Reszta plików w `docs/`
to rozwinięcia; ten porządkuje je według jedynej rzeczy, która decyduje
o przyjęciu artykułu — listy zarzutów.

---

## 0. Punkt wyjścia, sprawdzony w źródle

Obie recenzje przeszukane słowo po słowie (`odpowiedziRecenzentów/`):

- **obaj recenzenci: Reject**; recenzent 2 dopisał „do not encourage resubmit
  in the current form",
- recenzent 1: naprawa wymaga „reconstruction of the experimental protocol and
  extensive additional experiments rather than a conventional revision",
- **nikt nie prosił o udostępnienie danych ani kodu** — w obu plikach nie ma
  słowa o repozytorium, GitHubie ani oświadczeniu o dostępności danych,
- **nikt nie zarzucił, że zbiór treningowy jest za mały** — bo tezą pracy jest,
  że mniej danych wystarczy.

Wniosek: **nowy artykuł, nowe miejsce, przebudowany protokół.** Nie poprawka.

## 1. Trzy modele, każdy do czego innego

Częsty błąd to szukanie „jednego dobrego modelu". Tu potrzebne są trzy, i każdy
ma inne zadanie.

| Rola | Model | Wejście | Gdzie | Po co |
|---|---|---|---|---|
| **test hydrauliki** | **RF-DETR-Nano** | 384 | lokalnie, RTX 3070 | sprawdzić, że potok działa od końca do końca; minuty na bieg, mieści się w 8 GB |
| **sonda macierzy** | **RF-DETR-Small** | 512 | Eden, 8× H100 | 144 biegi macierzy; ma **uszeregować** warianty, nie wygrać |
| **potwierdzenie** | **RF-DETR** przy rozdzielczości z walidacji sondy | 512 albo 704 | Eden | tabele końcowe; do tego D-FINE-S i bieżąca YOLO jako test przeniesienia |

**RF-DETR-Nano jest do testowania, nie do wyników.** Trzydzieści milionów
parametrów, wejście 384, mieści się w sześciu gigabajtach pamięci karty.
Puszczasz go lokalnie na dwóch i pół tysiąca obrazów przez pięć epok i w kilka
minut wiesz, czy selekcja, manifesty, trening i ewaluacja się spinają. Dopiero
potem cokolwiek jedzie na Eden.

Uzasadnienie sondy: **Selection via Proxy** (Coleman i in., ICLR 2020) — mały
model zastępczy do wyboru danych dla dużego modelu docelowego. Że to działa
w naszym zadaniu, sprawdza eksperyment `P-RES` (`docs/02_MATRIX.md`, sekcja 6b):
pięć konfiguracji przy 512 i przy 704, korelacja kolejności.

Dlaczego to nie jest wolny wybór: przy wejściu 512 **56,9% ramek narzędzia
spada poniżej progu małego obiektu**, przy 704 tylko 24,6%
(`docs/13_STANDARD_DANYCH.md`).

## 2. Zarzuty już zamknięte — bez ani jednej godziny GPU

Siedem pozycji, wszystkie sprawdzone w danych albo rozstrzygnięte decyzją.

| Zarzut | Co go zamknęło | Dowód |
|---|---|---|
| R1 #1 losowy podział klatek | podział po pacjencie **już istnieje**: 50 pacjentów, jedno nagranie na pacjenta, 35/8/7 | `docs/05_OPEN_QUESTIONS.md` O1 |
| R2 #2 test ma tylko 589 obrazów | pod podziałem po pacjencie test to **7 pacjentów, 9 240 obrazów, 9 333 instancje** | policzone z anotacji |
| R2 #8 skąd 91 336, skoro 22 837 × 4 = 91 348 | różnica to **dokładnie 12 klatek**, które mają trzy warianty zamiast czterech | `docs/05_OPEN_QUESTIONS.md` O3 |
| R2 #6 brak baseline'u z klatek źródłowych | wykonalny na obu budżetach — w treningu jest **17 832 klatek źródłowych** | `docs/05_OPEN_QUESTIONS.md` O2 |
| R1 #5 inferencja z jednego zapytania | znika razem z DETR; standardowe dekodowanie | `docs/01_PROTOCOL.md` P7 |
| R1 #7 brak nowoczesnego detektora | RF-DETR, D-FINE-S, bieżąca YOLO | `docs/00_RESEARCH.md` §2 |
| R1 #8 porównanie non-iso-resource | ta sama rozdzielczość, epoki i zbiór dla wszystkich architektur | `docs/02_MATRIX.md` §6 |

Do tego trzy rzeczy, których **recenzenci nie zauważyli**, a które i tak trzeba
naprawić — i warto o nich napisać wprost, bo autor, który znalazł u siebie
więcej niż recenzent, czyta się inaczej:

1. selekcja jechała po **całej puli**, łącznie z klatkami pacjentów testowych,
2. EL2N pochodził z modelu trenowanego **170 epok na całej puli** — to nie jest
   EL2N w sensie oryginalnej definicji,
3. ablacja startowała z **checkpointu ep170 z wyciekiem**, więc wyciek wszedł
   przez inicjalizację, mimo czystych danych (`docs/08_AUDYT_WYCIEKOW.md`).

Audyt wycieków: osiem zbiorów ablacji **czystych**, ale produkcyjny zbiór 20k
ma **21,5% klatek od pacjentów walidacyjnych i testowych** oraz 48 obrazów
z benchmarku zewnętrznego.

## 3. Zarzuty wymagające obliczeń

| Zarzut | Eksperyment | Biegów |
|---|---|---:|
| R1 #2 jeden seed | 3 ziarna na macierzy, 5 na tabeli głównej, 10 losowań baseline'u | wliczone |
| R1 #3 wagi bez analizy czułości | zestaw `S-*`: wagi ±0,10, wymiar PCA, wybór reprezentanta, algorytm pokrycia, współczynniki trudności | ~18 |
| R1 #4 ablacje innym protokołem niż pełny pipeline | pełny pipeline **jest wariantem w tej samej macierzy**; stała liczba epok, jedna reguła checkpointu | wliczone |
| R1 #6 jeden arbitralny próg ufności | mAP z pełnej krzywej, próg operacyjny z walidacji, krzywa F1 od progu, test przeniesienia progu | bez GPU |
| R1 #9 brak analizy NMS | ustawienia jako jawny parametr plus wrażliwość | kilka |
| R1 #10 benchmark oglądany wielokrotnie | dyscyplina „zero-look" plus `external_access_ledger.jsonl` | bez GPU |
| R2 #1 czemu te pięć składowych FastSAM | uzasadnić albo uprościć miarę i pokazać w `S-*`, że wynik się nie zmienia | wliczone |
| R2 #5 ablacja niepełna | **macierz 15 kombinacji**: pojedyncze, pary, trójki, komplet | 45 |
| R2 #7 i #10 ablacja tylko na DETR | poziom B na drugiej i trzeciej architekturze | 10 |
| R2 #9 jedna instytucja | **drugi ośrodek z publicznego zbioru** — jedyny zarzut, którego nie da się zamknąć własnymi danymi | patrz §5 |
| drobne: czemu 20 000 | sweep 2 500 / 5 000 / 10 000 / 20 000 | ~18 |
| drobne: 71,3% to kopie augmentowane | analiza składu; docelowo znika, gdy pula powstaje z wideo | bez GPU |
| — (nasze) walidacja sondy | `P-RES`: 512 kontra 704, korelacja kolejności | 30 |

Razem około **174 biegi**. Koszt: `tools/estimate_time.py`, przy 512 pikselach
250–1000 godzin GPU, przy 704 mniej więcej dwa razy tyle.

## 4. Zarzuty, których nie zamyka eksperyment

- **R2 #17, interpretacja kliniczna** — oddzielić „lepsza generalizacja
  względna" od „użyteczne klinicznie". To jest akapit, nie tabela.
- **jedna instytucja jako ograniczenie** — nawet po dołożeniu drugiego ośrodka
  próba pozostaje mała. Napisać uczciwie w ograniczeniach.
- **nowość to kombinacja, nie składniki** — powiedzieć wprost. Macierz
  piętnastu kombinacji jest właśnie dowodem na wartość kombinacji.

## 5. Trzy etapy, żeby to się skończyło

Największe ryzyko tego planu nie jest naukowe, tylko organizacyjne: rozbudowa
korpusu może wchłonąć cały czas i artykuł nie powstanie.

| Etap | Co | Zamyka | Kiedy gotowe |
|---|---|---|---|
| **1** | macierz ablacji na **własnej puli** | wszystko poza R2 #9 | to jest artykuł |
| **2** | obce zbiory jako **nowe zbiory oceny** | R2 #9, jedna instytucja | dokładka, nie przebudowa |
| **3** | obce dane także do **treningu** | wersja wieloośrodkowa | następny artykuł |

Etapy 1 i 2 wystarczą. Etap 3 zaczyna się dopiero po wysłaniu.

Do etapu 2 najtaniej: **SICS-105** (23 godziny, pełne HD, inna technika
operacyjna, 12,3 GB) i **MIGS** (jaskra, narzędzia i anatomia, 11 GB, licencja
bez ograniczeń). Razem 23 GB i dwa nowe ośrodki
(`docs/14_SUROWE_FILMY_PLAN_POBRANIA.md`).

## 6. Kolejność robót

| # | Krok | Gdzie | Blokuje |
|---|---|---|---|
| 1 | **test hydrauliki na RF-DETR-Nano**, 2 500 obrazów, 5 epok | lokalnie | wszystko |
| 2 | pula: wycięcie klatek, pole widzenia, normalizacja | lokalnie | 3 |
| 3 | cechy DINO, Fourier, SAM | lokalnie, RTX 3070 | 5 |
| 4 | model wstępny EL2N i ocena puli | lokalnie | 5 |
| 5 | wygenerowanie podzbiorów wariantów | lokalnie | 6 |
| 6 | paczka ~7 GB na Eden | → Eden | 7 |
| 7 | **bieg kalibracyjny** i walidacja sondy `P-RES` | Eden | 8 |
| 8 | macierz poziomu A, kolejka na jednym węźle | Eden, 8 GPU | 9 |
| 9 | poziom B i przeniesienie architektur | Eden | 10 |
| 10 | **jedno** przejście po zbiorach zewnętrznych, wpis do rejestru | Eden | tabele |

Krok 1 jest tani i wykrywa większość błędów. Nic nie jedzie na Eden, dopóki
nie przejdzie.

## 6b. Podział pracy na pięciu agentów

Instrukcje: `agents/`. Każdy plik jest samodzielny — agent nie zna historii
projektu i nie musi jej znać.

| Kto | Katalog | Buduje |
|---|---|---|
| **Master** | `pipeline/ingest/`, `pipeline/annotate/` | pobieranie nagrań, klatki, normalizacja, propagacja ramek przez TSFA |
| Agent 1 | `pipeline/features/`, `pipeline/selection/` | cechy obrazów, model wstępny EL2N, wybór podzbiorów, baseline'y |
| Agent 2 | `pipeline/train/`, `eden/` | trening jednego biegu, kolejka i pracownicy na klastrze |
| Agent 3 | `pipeline/eval/` | inferencja, metryki, próg operacyjny, tabele, rejestr dostępów |
| Agent 4 | `pipeline/transfer/` | D-FINE-S i bieżąca YOLO przy iso-resource, analiza NMS |
| Agent 5 | `analysis/` | pokrycie rozkładu zewnętrznego, skład augmentacji, audyt wycieków, lista kontrolna |

Podział idzie **po kontraktach, nie po plikach**. Agenci nie rozmawiają ze sobą,
tylko wymieniają się plikami o ustalonym kształcie: `pool_manifest.jsonl`,
`variants/<run_id>.json`, `queue.jsonl`, `run_manifest.json`, `metrics.json`.
Kształt każdego z nich jest wypisany w `agents/README.md`, sekcja 4.

Trzy zasady, które trzymają to razem: **`configs/protocol.yaml` jest jedynym
źródłem prawdy** i nikt nie zaszywa hiperparametrów w kodzie; **nikt nie edytuje
cudzego katalogu**; każdy kończy zadanie **testem od końca do końca na małym
modelu**, zanim cokolwiek pojedzie na klaster.

## 7. Trzy rzeczy, które muszą być od pierwszego dnia

Nie da się ich dorobić po fakcie:

1. **`external_access_ledger.jsonl`** — każdy dostęp do zbioru zewnętrznego
   z datą i powodem. Bez tego zdanie o „zero-look" jest nie do obronienia.
2. **`run_manifest.json` przy każdym biegu** — hash commita, hash
   `protocol.yaml`, wersje bibliotek, hash listy plików. Poprzednio nie dało
   się ustalić, czym był wariant V4, i sprawa wisiała miesiącami.
3. **surowe wartości per ziarno w załączniku** — nie tylko średnie.

Nikt o kod ani dane nie prosił, ale przy nowym zgłoszeniu warto je dołączyć.
Rejestr i manifesty i tak powstają, więc paczka do odtworzenia wyników kosztuje
prawie nic — a przy pracy o kuracji danych to tanie zwycięstwo.

---

## Mapa dokumentów

| Chcesz wiedzieć | Czytaj |
|---|---|
| jaki protokół i dlaczego taki | `docs/01_PROTOCOL.md` |
| co dokładnie liczymy | `docs/02_MATRIX.md` |
| ile to potrwa i jak puścić na Edenie | `docs/03_COMPUTE_AND_OPTIMIZATION.md`, `docs/06_TODO_KOD_I_EDEN.md` |
| zarzut po zarzucie, ze wskazaniem miejsca | `docs/04_REVIEWER_MAP.md` |
| co zostało do decyzji | `docs/05_OPEN_QUESTIONS.md` |
| co już policzone na Edenie | `docs/07_EDEN_INWENTARZ.md` |
| gdzie były wycieki | `docs/08_AUDYT_WYCIEKOW.md` |
| skąd wziąć więcej danych i za ile | `docs/11`, `docs/12`, `docs/14` |
| jak budować pulę z wideo | `docs/10_TSFA_NOWA_PULA.md` |
| jaki format i rozdzielczość | `docs/13_STANDARD_DANYCH.md` |
| jak to przetwarzać i gdzie co leży | `docs/15_PIPELINE_PRZETWARZANIA.md` |
