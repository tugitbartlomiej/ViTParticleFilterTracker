# 12 — Lista linków do pobrania: zbiory do własnego oznaczania

Zebrane 2026-09-09. Cel: **materiał, na którym sam oznaczysz narzędzia i oko**.
Adnotacje w tych zbiorach są dodatkiem, nie warunkiem.

Uwaga o źródłach: Undermind (MCP) nie był podłączony w tej sesji, więc szukałem
przez Scite i wyszukiwanie sieciowe. Każdy link niżej otworzyłem i sprawdziłem.

---

## A. Surowe nagrania — tu jest właściwa wartość

| # | Zbiór | Nagrań | Rozdz. | fps | Licencja | Link |
|---|---|---:|---|---:|---|---|
| A1 | **Cataract-LMM** | **3 000** | 720×480 (2 930)<br>1920×1080 (70) | 30 / 60 | CC BY-NC-**ND** 4.0 | https://huggingface.co/datasets/mjahmadi/Cataract-LMM |
| A2 | **Cataract-1K** | 1 000 | 512×324 | 25 | CC BY 4.0 | https://www.synapse.org/Synapse:syn53404507 |
| A3 | **SICS-105** | 105 | **1920×1080** | 30 | Zenodo — sprawdzić rekord | https://doi.org/10.5281/zenodo.13847781 |
| A4 | Cataract-101 | 101 | — | — | CC BY-NC 4.0 | http://ftp.itec.aau.at/datasets/ovid/cat-101/ |
| A5 | Cataract-21 | 21 | — | — | jak wyżej | http://ftp.itec.aau.at/datasets/ovid/cat-21/ |
| A6 | CATARACTS (masz) | 50 | 1920×1080 | ~29 | niekomercyjna | https://cataracts.grand-challenge.org/Download/ |
| A7 | OphNet | 2 078 | zmienna | zmienna | linki do YouTube | https://github.com/minghu0830/OphNet-benchmark |

Rozmiary, o których warto wiedzieć przed kliknięciem: Cataract-1K to **89,9 GB**,
Cataract-101 to 8,3 GB, Cataract-LMM przy 1 134 godzinach będzie znacznie więcej.

Publikacje do cytowania:

- Cataract-LMM — https://doi.org/10.1038/s41597-026-07464-0 · preprint https://arxiv.org/abs/2510.16371 · kod https://github.com/MJAHMADEE/Cataract-LMM
- Cataract-1K — https://doi.org/10.1038/s41597-024-03193-4 · kod https://github.com/Negin-Ghamsarian/Cataract-1K
- SICS-105 — https://www.nature.com/articles/s41598-025-00303-z
- Cataract-101 — https://doi.org/10.5281/zenodo.1220951
- OphNet — https://arxiv.org/abs/2406.07471

## B. Gotowe maski **oka** — wzorzec dla twojej klasy anatomicznej

Chcesz oznaczać oko, więc warto wziąć od kogoś **gotowy podział na klasy**,
zamiast wymyślać własny. Recenzent porówna twoje wyniki z tymi zbiorami tylko
wtedy, gdy nazwy klas się zgadzają.

| # | Zbiór | Co zawiera | Link |
|---|---|---|---|
| B1 | **CaDIS** | **4 670 klatek**, 36 klas pikselowych: narzędzia **oraz tęczówka, źrenica, rogówka**; zbudowany na 25 nagraniach CATARACTS, czyli **na tych samych filmach, które masz** | https://cataracts-semantic-segmentation2020.grand-challenge.org/ |
| B2 | Iris and Pupil Segmentation | 82 klatki z Cataract-101, maski tęczówki i źrenicy | http://ftp.itec.aau.at/datasets/ovid/iris_pupil_seg/ |
| B3 | Relevance-Based Compression | 262 klatki z maskami **rogówki** + 216 z maskami narzędzi | http://ftp.itec.aau.at/datasets/ovid/CatRelevanceCompression/ |
| B4 | LensID | 401 klatek maski **soczewki**, 189 maski źrenicy, 2 400 klipów wideo | http://ftp.itec.aau.at/datasets/ovid/LensID/ |

**CaDIS jest tu pozycją numer jeden i to z konkretnego powodu.** Powstał na
nagraniach treningowych CATARACTS — czyli na twoich pięćdziesięciu pacjentach.
Masz więc gotowe maski oka **dla materiału, który już leży na twoim dysku**,
i nie musisz pobierać ani gigabajta wideo, żeby zacząć.

Praca opisująca CaDIS: https://arxiv.org/abs/1906.11586

## C. Narzędzia — ramki i maski

| # | Zbiór | Co zawiera | Link |
|---|---|---|---|
| C1 | InSegCat | 393 klatki z Cataract-101 plus 4 738 z CaDIS, **ramki i maski narzędzi** | http://ftp.itec.aau.at/datasets/ovid/InSegCat/ |
| C2 | Relevant cataract actions | lokalizacje czasowo-przestrzenne czterech faz operacji | https://ftp.itec.aau.at/datasets/ovid/relevant-cat-actions/ |

Na dysku `E:\CrossDatasetsForModelsEval` masz już cztery zbiory w formacie COCO
z Roboflow, w tym `InSegCat dataset 2.v10i.coco`, `Cataract_1K.v3i.coco`
i `Cataract-Tool-Segmentation.v6i.coco`. Zanim cokolwiek pobierzesz, sprawdź,
czy nie duplikujesz tego, co już jest.

## D. Kolejność pobierania

1. **CaDIS** — zero nowego wideo, maski oka na twoich nagraniach, natychmiastowy
   wzorzec klas.
2. **SICS-105** — 23 godziny, pełne HD, inna technika operacyjna, jeden plik
   z Zenodo. Najtańszy test, czy model działa poza Brestem.
3. **Cataract-1K** — najczystsza licencja, tysiąc nagrań, ale 512×324; przy
   wykrywaniu końcówki narzędzia to około trzynaście razy mniej pikseli na
   obiekcie niż w twoim materiale.
4. **Cataract-LMM** — dopiero gdy pierwsze trzy zadziałają; za to tysiąc sto
   godzin i drugi ośrodek, czyli jedyna rzecz, która zamyka zarzut o jedną
   instytucję.

## E. Dwie pułapki

**Licencja ND w Cataract-LMM.** Człon NoDerivatives znaczy, że trening i wyniki
są w porządku, ale **redystrybucja przetworzonych klatek już nie**.
W materiale uzupełniającym publikuj listę identyfikatorów i kod, nie klatki.

**Klasy oka trzeba ustalić raz.** CaDIS ma trzydzieści sześć klas, twoje zadanie
ma jedną. Zdecyduj na początku, czy oznaczasz oko jako jedną rzecz, czy jako
źrenicę, tęczówkę i rogówkę osobno — i trzymaj się tego. Zmiana taksonomii
w połowie oznacza oznaczanie od nowa.


---

## F. Przygotowane wyszukiwanie w Undermind (do uruchomienia po restarcie)

Undermind został 2026-09-09 przeniesiony do konfiguracji globalnej
(`~/.claude.json`, sekcja `mcpServers`), ale serwery MCP łączą się przy starcie,
więc w tej sesji jeszcze go nie ma. Po restarcie Claude Code do zrobienia:

1. `get_orientation()` — wymagane raz na rozmowę.
2. `list_workspaces()` — sprawdzić, czy workspace dla ViT już nie istnieje.
3. Utworzyć workspace: **„ViT: detekcja końcówki narzędzia w chirurgii zaćmy"**.
4. `inspect_deep_searches()` — obowiązkowe przed deep search, żeby nie palić
   limitu na duplikat.
5. Deep search, pytanie domknięte:

> *Publicly downloadable ophthalmic (cataract) surgery video datasets where
> recordings are organised per patient or per procedure, enabling patient-level
> train/test splits. Report for each dataset: number of videos, number of
> distinct patients, video resolution and frame rate, whether raw unannotated
> videos are distributed, licence, and the exact download URL or DOI.*

Czego szukamy ponad to, co już mamy w sekcjach A–C: zbiorów, w których
**jednostką jest pacjent**, a nie klatka — bo tylko takie pozwalają zrobić
podział po pacjencie bez zgadywania. Warto też sprawdzić, czy istnieją zbiory
z **więcej niż jednym nagraniem na pacjenta** (np. oba oczy), bo to zmieniałoby
regułę podziału.
