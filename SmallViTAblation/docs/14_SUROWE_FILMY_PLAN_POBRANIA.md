Surowe filmy — co pobrać, w jakiej kolejności i czy się zmieści
==============================================================

Data: 2026-09-09. Dokument dla jednego priorytetu: **surowe nagrania**.
Adnotacje są tu dodatkiem, nie kryterium.

---

## 1. Ile masz miejsca

| Dysk | Wolne | Razem |
|---|---:|---:|
| F: | **267 GB** | 2 048 GB |
| E: | 35,5 GB | 330,6 GB |
| C: | 18,8 GB | 351,5 GB |

Realnie **około 300 GB**, praktycznie wszystko na F:.

Można odzyskać jeszcze **99 GB**: katalog `E:\Cataract\videos\tray` to
pięćdziesiąt nagrań kamery pokazującej tacę z narzędziami. To osobne zadanie
(`10_TSFA_NOWA_PULA.md`, sekcja 6) i do tego artykułu nie wchodzi. Przeniesienie
go na dysk zewnętrzny podwaja zapas.

## 2. Które zbiory naprawdę dają pełne surowe nagrania

Sprawdzone w samych repozytoriach, nie w streszczeniach.

| Zbiór | Rozmiar pobrania | Nagrań | Godzin | Rozdzielczość | Licencja |
|---|---:|---:|---:|---|---|
| **MIGS (jaskra)** | **11 GB** | 21 plików zbiorczych | — | do sprawdzenia | **CC BY 4.0** |
| **Cataract-101** | **8,3 GB** | 101 | ~14 | do sprawdzenia | CC BY-NC 4.0 |
| **SICS-105** | **12,3 GB** (sam `videos_anonym.zip`) | 105 | 22,6 | 1920×1080 | **CC BY 4.0** |
| **Cataract-21** | mały | 21 | — | — | jak Cataract-101 |
| **Cataract-1K** | **89,9 GB** | 1 000 | ~119 | 512×324 | **CC BY 4.0** |
| **Cataract-LMM** | **772 GB całość** | 3 000 | 1 134 | 720×480 i 1920×1080 | CC BY-NC-**ND** |

Cztery pierwsze plus Cataract-1K to razem **około 122 GB** — mieści się na F:
z zapasem. Cataract-LMM się nie mieści i wymaga innego podejścia (punkt 4).

SICS-105 ma osobno `features.zip` na 5,2 GB — to gotowe cechy z sieci I3D,
nam niepotrzebne. Pobieramy **tylko `videos_anonym.zip`**.

MIGS ma w sobie 21 plików „MIGS video dataset" plus `Task II Raw Data.zip`
na 1,8 GB. Rozdzielczości repozytorium nie podaje — do sprawdzenia po pobraniu.

## 3. Klatki są tanie, filmy są drogie

Zmierzone na twoim nagraniu `test01.mp4`, pięć klatek, JPEG jakość 2:

| Szerokość | Bajtów na klatkę |
|---:|---:|
| 512 | 19,9 kB |
| 704 | 30,0 kB |
| 960 | 43,5 kB |

Stąd konkretne rachunki dla puli klatek przy 704 pikselach:

```
twoje 49 nagran, wszystkie 943 353 klatki   ->  28 GB
caly korpus publiczny ~1500 h przy 1 kl/s   ->  162 GB
caly korpus publiczny ~1500 h przy 2 kl/s   ->  324 GB
```

**Wniosek, który zmienia plan:** nie chodzi o to, żeby zmieścić 772 GB wideo.
Chodzi o to, żeby z każdego nagrania wyciągnąć klatki i **skasować film**.
Przy jednej klatce na sekundę cały dostępny publicznie materiał mieści się
w twoich dwustu sześćdziesięciu siedmiu gigabajtach.

Jedna klatka na sekundę wystarcza, bo sąsiednie klatki i tak są prawie
identyczne — to jest dokładnie ta nadmiarowość, którą pipeline kuracji ma
usuwać. Gęste klatki potrzebne są **tylko wokół ziaren do propagacji**, a te są
w twoich nagraniach, nie w obcych.

## 4. Cataract-LMM: pobierać partiami, nie w całości

772 GB nie zmieści się nigdy. Ale repozytorium na Hugging Face jest podzielone
na pięć katalogów, a surowe nagrania siedzą w `5_Raw_Videos/`. Pozostałe cztery
katalogi (rozpoznawanie faz, segmentacja, śledzenie, ocena umiejętności) są nam
niepotrzebne.

Procedura, którą trzeba napisać jako skrypt:

1. pobrać listę plików z `5_Raw_Videos/`,
2. wziąć jedną paczkę,
3. wyciągnąć klatki przy jednej klatce na sekundę, znormalizować wg
   `13_STANDARD_DANYCH.md`, zapisać jako JPEG,
4. **skasować pobraną paczkę**,
5. wrócić do punktu 2.

Szczytowe zużycie dysku to rozmiar jednej paczki plus rosnąca pula klatek.

**Uwaga licencyjna:** Cataract-LMM ma człon ND, czyli zakaz udostępniania
przeróbek. Trening i publikowanie wyników są w porządku, ale wyciągniętych
klatek **nie wolno redystrybuować**. W materiale uzupełniającym publikujemy
listę identyfikatorów nagrań i kod, który klatki odtwarza.

## 5. Kolejność pobierania

| # | Zbiór | GB | Po co akurat to |
|---|---|---:|---|
| 1 | **SICS-105** | 12,3 | pełne HD jak twoje, **inna technika operacyjna** i inny kraj — najtańszy test, czy model działa poza Brestem |
| 2 | **MIGS** | 11 | **inna operacja** (jaskra), narzędzia i anatomia oznaczone, licencja bez ograniczeń niekomercyjnych |
| 3 | **Cataract-101** | 8,3 | sto jeden nowych pacjentów za osiem gigabajtów |
| 4 | **Cataract-21** | mały | dokładka z tego samego serwera, przy okazji |
| 5 | **Cataract-1K** | 89,9 | tysiąc nagrań, ale 512×324 — **tylko do treningu, nigdy do oceny** |
| 6 | **Cataract-LMM** | partiami | tysiąc sto godzin i drugi ośrodek; jedyna rzecz, która zamyka zarzut o jedną instytucję |

Pierwsze cztery to **32 GB** i dają trzy nowe ośrodki plus dwie nowe procedury.
To jest najlepszy stosunek zysku do gigabajta w całej tabeli.

## 6. Czego nie pobierać

- **OphNet** — dwa tysiące nagrań, ale rozproszonych po YouTube. Pobieranie to
  osobna robota, jakość nierówna, a status prawny wątpliwy. Zostawić na koniec.
- **FOVEA** — tylko krótkie klipy, nie pełne nagrania.
- **`features.zip` z SICS-105** — 5,2 GB gotowych cech z sieci I3D, do niczego
  nam niepotrzebne.
- **Podkatalogi Cataract-LMM inne niż `5_Raw_Videos/`** — to adnotacje, których
  i tak nie używamy.

## 7. Do sprawdzenia po pobraniu

Każdy zbiór, zanim wejdzie do puli:

1. **rozdzielczość natywna i liczba klatek na sekundę** — repozytoria podają to
   niespójnie, a u nas to decyduje, czy zbiór trafia do treningu czy także do
   oceny,
2. **czy jeden pacjent ma jedno nagranie** — jeśli któryś ma dwa, na przykład
   oba oczy, podział po nagraniu przestaje być podziałem po pacjencie,
3. **kadr pola widzenia mikroskopu** — wykryć koło i zapisać prostokąt
   przycięcia, raz na nagranie (`13_STANDARD_DANYCH.md`, punkt 4).

Punkt drugi wyszedł z wyszukiwania w Undermind jako rzecz, której ze streszczeń
ustalić się nie da. Trzeba go sprawdzić w danych, nie w publikacji.
