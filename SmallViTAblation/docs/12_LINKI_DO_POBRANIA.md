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

## F. Wyszukiwanie w Undermind — uruchomione

Undermind przeniesiony 2026-09-09 do konfiguracji globalnej (`~/.claude.json`,
sekcja `mcpServers`). Po restarcie podłączył się bez ponownego logowania.

**Workspace:** „ViT: detekcja końcówki narzędzia w chirurgii zaćmy"
https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893

**Deep search** uruchomiony 2026-09-09 12:38. Pytanie obejmuje: publicznie
pobieralne zbiory nagrań chirurgii okulistycznej zorganizowane **per pacjent
albo per zabieg**, z liczbą nagrań i pacjentów, informacją czy jeden pacjent
bywa nagrany więcej niż raz (np. oba oczy — to złamałoby naiwny podział po
nagraniu), rozdzielczością, liczbą klatek na sekundę, tym czy udostępniane są
pełne surowe nagrania czy tylko fragmenty, licencją, wymogiem umowy o użyciu
danych i dokładnym adresem pobrania. W pytaniu zaznaczone, że **adnotacje są
mniej ważne niż surowe wideo**, bo etykiety robimy propagacją, oraz że
rozdzielczość waży tu wyjątkowo dużo, bo końcówka narzędzia to mały obiekt.
Wykluczone jako znane: CATARACTS, CaDIS, Cataract-101, Cataract-21,
Cataract-1K, Cataract-LMM, SICS-105, LensID, OphNet.

Wyszukiwanie zakończone po dwóch minutach, 27 trafnych prac.

### F1. Nowe zbiory, których wcześniej nie mieliśmy

| Zbiór | Co to jest | Link |
|---|---|---|
| **MIGS multicenter** | pierwszy **wieloośrodkowy** zbiór nagrań chirurgii jaskry, miliony klatek, adnotacje **narzędzi i struktur anatomicznych**, Scientific Data 2026 | dane: https://doi.org/10.5281/zenodo.18231909 · praca: https://doi.org/10.1038/s41597-026-07535-2 |
| **Sankara-MSICS** | 53 nagrania ręcznego małego cięcia, Sankara Eye Hospital, X 2023 – X 2024; 18 faz i 3 527 klatek z 13 narzędziami na poziomie pikseli; MICCAI 2025 | kod: https://github.com/Sri-Kanchi-Kamakoti-Medical-Trust/ToolSeg · praca: https://arxiv.org/abs/2411.16794 — **dostęp do samych danych przez autorów** |
| **WetCat** | nagrania **z wet-labu** na sztucznych oczach, wysoka rozdzielczość, adnotacje faz i segmentacja anatomii | https://www.synapse.org/Synapse:syn66401174/files/ · praca: https://doi.org/10.1145/3746027.3758274 |
| **FOVEA** | 40 pacjentów z Moorfields, chirurgia witreoretinalna, obrazy przed- i śródoperacyjne plus krótkie klipy | https://doi.org/10.1038/s41597-025-04965-2 |
| **OphNet-3D** | 41 sekwencji RGB-D od 40 chirurgów, 7,1 mln klatek, 10 kategorii narzędzi, pełne pozycje 6-DoF | https://arxiv.org/abs/2505.17677 |

**MIGS jest tu najciekawszy.** To inna operacja niż zaćma, ale ma oznaczone
narzędzia **i** anatomię oka, jest wieloośrodkowy i leży na Zenodo bez umowy
o użyciu danych. Jako drugi zbiór do sprawdzenia odporności modelu na zmianę
dziedziny trudno o lepszy.

**WetCat rozwiązuje problem, o którym nie pomyślałem.** Sztuczne oczy w wet-labie
nie są danymi pacjenta, więc nie ma ograniczeń prywatnościowych ani zgód —
a rozdzielczość jest wysoka. Do testowania samego pipeline'u oznaczania to
materiał bez żadnych formalności.

### F2. Znalezisko metodologiczne, ważniejsze niż zbiory

**[Fox22]** — *The Impact of Dataset Splits on Classification Performance in
Medical Videos*, Fox i Schoeffmann, ICMR 2022, https://doi.org/10.1145/3512527.3531424

Autorzy powtórzyli badanie na czterech zbiorach obrazów z laparoskopii,
wyciągniętych automatycznie z nagrań. Pokazali, że klatki są **do siebie tak
podobne, że wysokie wyniki biorą się z samej konstrukcji zbioru**. Dowód:
zwykły algorytm SIFT, czyli metoda porównywania obrazów sprzed epoki uczenia
głębokiego, dostaje na tych danych wysokie oceny. Potem zbudowali podział po
nagraniach — i **sieć neuronowa na tym podziale wypadła gorzej niż SIFT na
podziale oryginalnym**.

To jest dokładnie nasza sytuacja i gotowe cytowanie do sekcji o protokole.
Zarzut 1 recenzenta 1 przestaje być kwestią opinii, a staje się zjawiskiem
opisanym w literaturze i zmierzonym przez innych.

Drugi przydatny przegląd: **[Kha20]**, *A global review of publicly available
datasets for ophthalmological imaging*, Lancet Digital Health,
https://doi.org/10.1016/S2589-7500(20)30240-5 — 94 zbiory otwarte,
507 724 obrazy i **tylko 125 nagrań wideo** od 122 364 pacjentów. Nagrania
chirurgiczne są w okulistyce rzadkością, co warto powiedzieć w artykule
jednym zdaniem.

### F3. Ostrzeżenie z podsumowania wyszukiwania

Agent zaznaczył wprost, że w większości publikacji **nie da się ustalić** ze
streszczeń: czy jeden pacjent ma więcej niż jedno nagranie, jaka jest natywna
rozdzielczość i liczba klatek na sekundę, czy udostępniane są pełne nagrania,
jaki jest rozmiar pobrania i licencja. To trzeba sprawdzić w samych
repozytoriach, zbiór po zbiorze, przed pobraniem.

Każde wydanie oparte na **wyciągniętych klatkach** trzeba traktować jako
niebezpieczne do ewaluacji, dopóki nie zachowano przypisania klatki do
nagrania źródłowego.
