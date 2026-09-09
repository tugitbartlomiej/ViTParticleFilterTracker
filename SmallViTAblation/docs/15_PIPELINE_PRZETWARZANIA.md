Pipeline przetwarzania — od filmu do tabeli
==========================================

Data: 2026-09-09. Odpowiedź na trzy pytania: jak przetwarzać najszybciej,
gdzie to trzymać, i na jakim modelu liczyć ablacje.

---

## 1. Zmierzone, nie zgadnięte

| Wielkość | Wartość | Skąd |
|---|---|---|
| procesor | i7-11700KF, 8 rdzeni / 16 wątków | zmierzone |
| pamięć | 32 GB | zmierzone |
| karta lokalna | RTX 3070, 8 GB | zmierzone |
| **wycinanie klatek** | **1 740 klatek/s na jeden proces ffmpeg** | zmierzone na `test01.mp4` |
| klatka JPEG przy 704 px | 30,0 kB | zmierzone |
| **wolne miejsce w domu na Edenie** | **65 GB** | sesja 2026-07-23 |
| scratch `/raid` na węźle hopper | 14 TB | `EDEN_CONSTRAINTS.md` |

Pomiar przepustowości jest rozstrzygający. Jedno trzynastominutowe nagranie
w pełnym HD przerobiłem w **dwanaście sekund**.

## 2. Wąskim gardłem nie jest przetwarzanie ani dysk lokalny

Z tej jednej liczby wynika reszta:

```
twoje 49 nagran, 943 353 klatki      ->  9 min na jednym rdzeniu
                                     ->  ~2 min na osmiu rownolegle
caly korpus publiczny ~1500 h @30fps ->  26 h na jednym rdzeniu
   (okolo 162 mln klatek do dekodowania)  ->  ~4 h na osmiu rownolegle
```

Pobieranie około 900 GB zajmie więcej niż przerobienie tego materiału.
**Wąskim gardłem jest łącze**, więc pobieranie i przetwarzanie muszą chodzić
równolegle, a nie jedno po drugim.

Drugie wąskie gardło jest w innym miejscu, niż się wydaje: **dom na Edenie ma
65 GB wolnego**. Nowy dysk sześciotelabajtowy tego nie zmienia. Ale to nie jest
problem, bo na Eden nie musi jechać cała pula — patrz punkt 5.

## 3. Trzy poziomy przechowywania

| Gdzie | Ile | Co tam leży | Czas życia |
|---|---|---|---|
| nowy dysk 6 TB | 6 TB | surowe filmy **przejściowo**, pełna pula klatek, cache cech | pula na stałe, filmy do skasowania |
| dom na Edenie | 65 GB | wybrane podzbiory, walidacja, test, wagi startowe | na czas eksperymentu |
| `/raid` na węźle | 14 TB | rozpakowana pula na czas jednego zadania | kasowane przez `trap cleanup` |

## 4. Potok pobierania: producent i konsument

Jeden proces pobiera, drugi przerabia i kasuje. Nigdy nie czekają na siebie.

```
[pobieracz]  paczka N+1 sciaga sie
     |
     v
[kolejka na dysku]
     |
     v
[przetwornik]  paczka N: rozpakuj -> wykryj pole widzenia -> wytnij klatki
               -> znormalizuj -> zapisz JPEG -> SKASUJ paczke i film
```

Szczytowe zużycie dysku to rozmiar dwóch paczek plus rosnąca pula klatek.
Przy sześciu terabajtach nie ma o czym mówić, ale nawyk zostaje: **film ginie,
gdy tylko klatki są zapisane**.

Gęstość wycinania zależy od roli nagrania:

- **twoje nagrania — wszystkie klatki**, bo wokół ziaren potrzebna jest
  propagacja klatka po klatce. 943 tysiące klatek to 28 GB, nie ma powodu
  oszczędzać.
- **obce zbiory — jedna klatka na sekundę.** Sąsiednie klatki są prawie
  identyczne, a to jest właśnie nadmiarowość, którą pipeline kuracji ma usuwać.

## 5. Co naprawdę jedzie na Eden

Tu jest oszczędność, której łatwo nie zauważyć. **Pula nie musi być na Edenie.**
Na Eden jedzie tylko to, na czym się trenuje:

| Co | Ile |
|---|---|
| walidacja i test (20 020 obrazów) | 0,6 GB |
| suma wszystkich wybranych podzbiorów poziomu A | poniżej 3 GB |
| to samo dla poziomu B | poniżej 3 GB |
| wagi startowe COCO | poniżej 0,2 GB |
| **razem** | **około 7 GB z 65 dostępnych** |

Podzbiory wariantów mocno się pokrywają, więc pakujemy **sumę zbiorów**, nie
każdy wariant osobno. Każdy bieg dostaje listę nazw plików, a obrazy leżą raz.

## 6. Gdzie liczyć cechy

Cechy DINO, Fouriera i FastSAM liczy się **raz** na całej puli i nie zależą od
wariantu ani od ziarna. Pytanie brzmi: lokalnie czy na Edenie.

**Twoja pula: lokalnie, przez noc.** Na RTX 3070 rzędu: Fourier to procesor
i jedna–dwie godziny, DINOv2 kilka godzin, FastSAM najdłużej. Całość mieści się
w jednej nocy, a wynik to kilka plików `.pkl` po kilkaset megabajtów.

**Pula z obcych zbiorów: na Edenie.** Jeśli urośnie do milionów klatek, karta
lokalna przestaje wystarczać. Wtedy pula jedzie spakowana na `/raid`, cechy
liczą się na H100, a wraca sam cache.

EL2N jest wyjątkiem, bo wymaga modelu. Model wstępny trenujemy **na losowym
podzbiorze puli treningowej** przez dziesięć epok, a potem tym modelem
oceniamy całą pulę — ocenianie to sama inferencja, więc jest tanie. To jest
zgodne z oryginalną definicją EL2N, która mówi o sygnale **z wczesnego
treningu**, nie z modelu douczonego do końca.

## 7. Na jakim modelu liczyć ablacje

**Macierz na RF-DETR-Small przy 512 pikselach.** To jest sonda, nie model
docelowy — jej zadaniem nie jest osiągnąć najlepszy wynik, tylko **uszeregować
warianty tak samo, jak uszeregowałby je model docelowy**. Dokładnie tak działa
Selection via Proxy (Coleman i in., ICLR 2020) i to jest cytowanie, które ten
wybór uzasadnia.

Ale samego założenia nie wolno przyjąć na słowo, bo przy 512 pikselach
**56,9% ramek narzędzia spada poniżej progu małego obiektu**, a przy 704 tylko
24,6% (`13_STANDARD_DANYCH.md`, punkt 1).

Dlatego dokładamy jeden eksperyment, tani i sam w sobie wart opisania:

> **Walidacja sondy.** Pięć konfiguracji z macierzy — pełny pipeline, dwie
> okrojone, losowy, losowy z klatek źródłowych — trenujemy **i przy 512, i przy
> 704**, po trzy ziarna. Liczymy korelację rang między obiema rozdzielczościami.
> Jeśli kolejność wariantów się zgadza, tania sonda jest uzasadniona i można
> to napisać wprost. Jeśli się nie zgadza, cała macierz idzie na 704 i kosztuje
> około dwa razy tyle.

To jest trzydzieści dodatkowych biegów przy budżecie poziomu A — mniej niż
jedna czwarta macierzy, a zamyka pytanie, które recenzent i tak zada.

Model do potwierdzenia i do tabel końcowych: **RF-DETR przy tej rozdzielczości,
którą wskaże walidacja sondy**, plus D-FINE-S i bieżąca YOLO jako sprawdzenie,
czy efekt przenosi się na inne architektury.

## 8. Kolejność całości

| # | Krok | Gdzie | Ile trwa |
|---|---|---|---|
| 1 | sformatować dysk 6 TB, przenieść nagrania z tacy | lokalnie | godzina |
| 2 | wyciąć wszystkie klatki z twoich 49 nagrań, znormalizować | lokalnie, 8 procesów | **minuty** |
| 3 | cechy DINO, Fourier, SAM na twojej puli | lokalnie, RTX 3070 | jedna noc |
| 4 | model wstępny EL2N i ocena puli | Eden, 1 GPU | godziny |
| 5 | wygenerować podzbiory wariantów | lokalnie, bez GPU | minuty |
| 6 | spakować sumę podzbiorów plus walidację i test, wysłać | → Eden | 7 GB |
| 7 | **bieg kalibracyjny** 512 kontra 704 | Eden, 1 GPU | godziny |
| 8 | macierz poziomu A, kolejka na jednym węźle | Eden, 8 GPU | dni |
| 9 | walidacja sondy, pięć konfiguracji przy obu rozdzielczościach | Eden | dzień |
| 10 | poziom B i przeniesienie architektur | Eden | dni |
| 11 | jedno przejście po zbiorach zewnętrznych | Eden | godziny |

**Kroki od 1 do 3 idą równolegle z pobieraniem obcych zbiorów.** Nie ma powodu
czekać na Cataract-LMM, żeby zacząć liczyć cechy na własnych danych.

## 9. Rozdzielenie, bez którego to się rozjedzie

Rozbudowa korpusu i macierz ablacji to **dwa osobne przedsięwzięcia**. Kuszące
jest wrzucić obce zbiory do puli treningowej od razu, ale wtedy macierz liczy
się na ruchomym celu i nigdy się nie skończy.

- **Etap 1** — macierz na twojej puli, tak jak w `02_MATRIX.md`. To jest artykuł.
- **Etap 2** — obce zbiory jako **nowe zbiory oceny**. Domyka zarzut o jedną
  instytucję, nie ruszając macierzy.
- **Etap 3**, jeśli starczy czasu — obce dane także do treningu, wersja
  wieloośrodkowa.

Etapy 1 i 2 wystarczą na artykuł. Etap 3 jest następnym artykułem.
