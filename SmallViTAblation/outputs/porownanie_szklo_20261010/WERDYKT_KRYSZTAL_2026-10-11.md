# Werdykt: ruch tokenów ViT to pełzający, sprężysty kryształ na siatce patchy (2026-10-11)

Dokument podsumowuje fale 1–5 badania „ruch tokenów jak cząsteczki w szkle” i zastępuje werdykt „szkło” z sekcji 6–6d `README.md`.
Ostrożniejszą, niezależną ocenę tych samych plików zawiera `audyt_krysztalu_20261011/OCENA_KRYSZTALU.md`. Kopie źródeł i SHA-256 są w `audyt_krysztalu_20261011/zrodla/` i `MANIFEST_ZRODEL.json`.

## Odpowiedź w jednym zdaniu

Układ jest **w przybliżeniu w dwóch trzecich krystaliczny**. To dwuwymiarowa, zbiorowo skorelowana siatka węzłów, których adresy wyznacza kodowanie pozycji. Ma zamrożony nieporządek podstawieniowy pochodzący z treści obrazu i powoli pełznie pod wpływem uczenia. Nie jest szkłem, cieczą ani zbiorem niezależnych pułapek. Nazwa robocza do artykułu: **„odkształcająca się siatka reprezentacji związana z pozycjami patchy”**. Słowa „kryształ” używamy tylko jako analogii i zawsze podajemy przy nim zmierzone wartości.

## Model i dane

- MałyViT trenowany od zera na 35 000 klatkach: ziarno 47, lr 5e-4, 806 checkpointów do 60. epoki (`assets/RuchCzasteczek_ViT_szklo_2026-10/`, Eden `~/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok`).
- Cząstka = token (600 na zdjęcie, siatka 20 × 30). Położenie = wyjście GELU po fc1 ostatniego bloku MLP (768D) w rzucie losowym RP128 z ziarnem 0. Rzut wiernie zachowuje odległości (lemat JL): stosunek MSD 768/128 wynosi 0,94–1,01. PCA32 widzi tylko 4–14% ruchu i nie nadaje się do tych miar.
- 100 zdjęć (50 TRAIN, 50 VALID) przez 60 epok oraz gęste okna kroków 300–2188 i 12034–13128 z biegów kontrolnych na Edenie.

## Tabela: w ilu procentach kryształ

| Cecha kryształu | Miara | ViT bazowy | Bez kodowania pozycji | CNN | ViT lr 2e-3 („gorąco”) | Tło losowe |
|---|---|---:|---:|---:|---:|---:|
| Siatka węzłów | 6 najbliższych sąsiadów w latencie to sąsiednie patche | 70–72% (ziarna 47, 48, 49: 70,7, 70,6, 71,0%) | 34% | 57% | 59% | ok. 1,3% |
| Adres węzła | R² szablonu numeru patcha (ocena na innych sondach lub VALID) | 0,46–0,56 przy kroku 3280; 0,44 w 60. epoce | −0,03 | 0,12 | 0,24 | 0 |
| Ruch zbiorowy (długie fale) | udział mocy przyrostów w dwóch najniższych pasmach | 60–73% (od 73% w epokach 1–5 do 60% w epokach 50–60) | brak dyspersji | – | – | 3,2–3,5% po tasowaniu |
| Widmo fal | wykładnik mocy przy małym k | od −2,5 do −2,9 | −1,9 | −1,5 | – | 0 |
| Kierunek wiązań | średni cos wektora wiązania po 800 krokach | 0,60 (wcześnie), 0,71 (późno) | – | – | ok. 0,33 | 0 |
| Brak dyfuzji względnej | wykładnik wzrostu γ_L (Zahn–Maret) | 0,27–0,28 | – | – | γ_L ×2,5 | ciecz: ok. 1 |

Na starcie (krok 0) R² wynosi 0,998, a 97% sąsiadów pochodzi z siatki: kodowanie pozycji tworzy idealny kryształ. W pierwszej epoce uczenie częściowo go „topi”, do około 70%. Ten poziom utrzymuje się do 60. epoki.

Podsumowanie liczbowe: **porządek sąsiedztwa ok. 70%, ruch zbiorowy ok. 60–70%, adres z pozycji ok. 45–55%.** Reszta to treść obrazu i jej oddziaływanie z pozycją. Nie jest to szum.

## Jaki to kryształ: cztery przymiotniki i dowody

1. **Dwuwymiarowy, na siatce patchy.** Kodowanie pozycji (sincos) narzuca adresy węzłów. W kontroli bez kodowania pozycji R² spada do zera, a sąsiedztwo do 34%. Ziarna 48 i 49 powtarzają wynik.
2. **Sprężysty, czyli zbiorowo skorelowany.** Przyrosty położeń zmieniają się po siatce gładko: 60–73% mocy przypada na najdłuższe fale, wobec 3% po tasowaniu tokenów. We wczesnym oknie długie fale relaksują wolniej niż krótkie (300 wobec 120 kroków). W późnym oknie wszystkie pasma relaksują w 80 krokach, więc ta zależność nie występuje we wszystkich oknach.
3. **Z nieporządkiem podstawieniowym.** Każdy węzeł ma inną treść i inną sztywność. Heterogeniczność ruchliwości jest zamrożona (korelacja 0,87–0,89) i pozostaje po odjęciu wspólnego napędu. Ruchliwość tokenów przewiduje IoU: Spearman od −0,43 do −0,68, na TRAIN i VALID, od 5. epoki.
4. **Pełzający.** γ_L rośnie wolniej niż liniowo (wykładnik ok. 0,27) i nie osiąga plateau. Od 20. epoki γ_L po 4000 krokach wynosi około 0,57–0,67. Węzły przesuwa uczenie, a nie uderzenia atomów. Korelacja naporu atomów na ściany komórek z późniejszym ruchem ścian wynosi −0,01 (n = 65 520). Od 1. epoki ruch węzła nie zależy od drgań. Podprzestrzeń adresu obraca się: średni cos kątów głównych wynosi 0,38 w 60. epoce.

Najbliższy układ fizyczny to **kryształ koloidalny w siatce optycznej, którą zewnętrzna ręka powoli przesuwa i odkształca.** Odpowiedniki:

- siatka światła: kodowanie pozycji;
- temperatura: długość kroku optymalizatora;
- ręka: gradient straty;
- „wiązki”: wspólne wagi W fc1 (768 × 192);
- „kulki”: wejścia u tokenów.

## Czym ten układ nie jest

- **Szkłem.** Ma siatkę odniesienia. Objawy plateau MSD i szklistej relaksacji daje też sama cząstka w dołku OU (kontrola z sekcji 3 README).
- **Cieczą.** γ_L rośnie z wykładnikiem 0,27, a nie około 1. Sąsiedztwo zgodne z siatką utrzymuje się przez 60 epok.
- **Zbiorem niezależnych pułapek.** Sąsiedzi pozostają skorelowani (0,69–0,83) po odjęciu wspólnego napędu.
- **Prostym układem Markowa.** Operator jednokrokowy to niemal sama trwałość (0,98–0,99). Jądro pamięci ma K(1)/K0 ≈ 0,67, co odpowiada pędowi AdamW. Napęd jest niestacjonarny.

## Granice wniosków i korekty

- **Korekta wcześniejszej odpowiedzi w rozmowie:** 0,60 to średni cos kierunku wiązania po 800 krokach, a nie „60% zachowanych wiązań”. To samo dotyczy wartości 0,33 przy lr 2e-3.
- Jaccard sąsiedztwa względem kroku 0 wynosi 0,475. Nie oznacza to, że zachowało się 47,5% sąsiadów.
- Niewyjaśnionych przez szablon 56% położenia nie można automatycznie uznać za „czystą treść”, bo szablon zawiera też typową treść sceny.
- Analiza Fouriera nie odróżnia sił sprężystych od gładkiego wspólnego napędu przez wagi, więc „fonony” pozostają hipotezą. Widmo dotyczy przyrostów, a nie wychyleń równowagowych. FFT na siatce 20 × 30 nie ma kontroli wpływu brzegów.
- γ_L usuwa wspólną translację, ale nie wspólny obrót ani odkształcenie. Próg 0,033 pochodzi z fizycznego układu 2D i nie wyznacza tu granicy fazy.
- Kontrolę bez kodowania pozycji wykonano dla jednego ziarna.
- **Błąd w kodzie:** `eden/wspolny_naped_wagi.py` (wiersze ok. 43 i 88) bierze przyrost wag jako `dTh[idx[:-1]]` z pełnej listy checkpointów. Dla części przedziałów (np. 2100→2188 zamiast 2100→2200) przyrosty wag i ruch tokenów dotyczą wtedy różnych odcinków. Poprawka: `Theta[idx[1:]] - Theta[idx[:-1]]`. Wynik NAPED_WAGI (36% wobec 23% przypadkowo) trzeba policzyć ponownie.

## Które parametry ustalają IoU, a które pewność (test podatności)

- IoU ustala głowica, która daje 68–93% gradientu straty ramki.
- Pewność ustala stem (51–65% gradientu mapy ciepła) oraz końcowa norma. Szum w końcowej normie zawsze obniża pewność.
- Bloki 0–5 służą obu zadaniom: cos(gradient mapy ciepła, gradient ramki) wynosi od +0,1 do +0,5.
- Spearman pewność–IoU: +0,38 na TRAIN, +0,20 na VALID. Ruchliwość–IoU: −0,44 i −0,46.

## Testy, które rozstrzygną „kryształ” czy „gładki napęd”

1. Poprawić wyrównanie czasowe w `wspolny_naped_wagi.py` i powtórzyć test z prognozą poza oknem dopasowania.
2. Dopasować ruch sztywny (obrót i skalę) oraz osobno odkształcenie, a potem policzyć γ_L po ich odjęciu.
3. Sprawdzić, czy widmo przyrostów lepiej przewiduje model lokalnej odpowiedzi sprężystej, czy model samego skorelowanego wymuszenia.
4. Wytrenować drugie i trzecie ziarno kontroli bez kodowania pozycji.
5. Porównać rozstaw centrów węzłów z amplitudą drgań, żeby sprawdzić, czy klatki się nakładają.

## Pliki źródłowe

| Wynik | Plik |
|---|---|
| 60 epok, ruch i mobilność–IoU | `eden/wyniki/ruch60/RUCH60.json` |
| Test Markowa, jądro pamięci, VAMP | `eden/wyniki/markow/MARKOW_*.json` |
| Wspólny napęd | `eden/wyniki/naped/NAPED_*.json` |
| Pozycja patchy, R² i sąsiedzi | `audyt_krysztalu_20261011/zrodla/pozycja_patchy/POZYCJA_PATCHY.json` |
| Kontrole (bez pozycji, CNN, lr, ziarna) | `audyt_krysztalu_20261011/zrodla/kontrole_siatki/SIATKA_*.json` |
| Fale (widmo, dyspersja) | `audyt_krysztalu_20261011/zrodla/fonony/FONONY_*.json` |
| Lindemann | `audyt_krysztalu_20261011/zrodla/lindemann/LINDEMANN_*.json` |
| Podprzestrzenie | `audyt_krysztalu_20261011/zrodla/podprzestrzenie/PODPRZESTRZENIE.json` |
| Podatność IoU i pewności | `eden/wyniki/podatnosc/PODATNOSC.json` |
| Siły, adres i treść, IoU i pewność | `film_krysztal/SILY.json`, `ADRES_TRESC.json`, `IOU_PEWNOSC.json` |
| Filmy | `film_krysztal/krysztal_tokenow.mp4`, `adres_tresc.mp4`, `krysztal_sily.mp4`, `szescian_0.mp4`, `komorki_0.mp4`, `iou_pewnosc.mp4` |
| Literatura (listy linków) | `literatura_szklo_tokeny_20261011/linki/*.txt` |
| Dane wejściowe (trajektorie RP128 i 768D) | `assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/` |

---

## AKTUALIZACJA 2026-10-11, 23:00: testy przyczynowe i ślepa klasyfikacja (zastępuje część „Jaki kryształ”)

Wykonano trzy testy, które nie zakładają odpowiedzi z góry (`eden/odpowiedz_tokenow.py`, `eden/slepa_klasyfikacja.py`, wyniki w `eden/wyniki/odpowiedz/` i `eden/wyniki/slepa*/`).

### 1. Pchnięcie jednego tokenu po lodydze (sprzężenie przez bloki transformera)

Odpowiedź pozostałych tokenów jako ułamek odpowiedzi pchniętego, liniowa (nieliniowość 0,5%):

| model | r = 1 | r = 4–6 | r > 14 | udział 8 sąsiadów (losowo 1,3%) |
|---|---:|---:|---:|---:|
| start | 0,003 | 0,003 | 0,003 | 1,2% |
| epoka 10 | 0,018 | 0,020 | 0,013 | 1,6% |
| epoka 60 | 0,022 | 0,023 | 0,015 | 1,7% |
| bez kodowania pozycji, ep. 3 | 0,012 | 0,011 | 0,009 | 1,6% |

**Bloki transformera sprzęgają tokeny globalnie i słabo, nie z sąsiadami.** Sprzężenie rośnie z uczeniem.

### 2. Pchnięcie pikseli jednego patcha (przez lodygę konwolucyjną)

| model | r = 1 | r = √2 | r ≥ 1,5 | udział 8 sąsiadów |
|---|---:|---:|---:|---:|
| start (losowe wagi) | 0,52 | 0,28 | 0,005 | 50% |
| epoka 60 | 0,50 | 0,27 | 0,03–0,05 | 31% |
| bez kodowania pozycji, ep. 3 | 0,55 | 0,29 | 0,03 | 35% |

**Sprzężenie z sąsiadami istnieje, ale pochodzi z nakładania pól widzenia lodygi (wspólne piksele), jest obecne już przy losowych wagach i nie zależy od kodowania pozycji.** Uczenie dokłada tylko tło globalne.

### 3. Ślepa klasyfikacja (192 wzorce Langevina 2D o znanej klasie, 7 miar)

Runda 1 (bez nieporządku): tokeny w równej odległości od kilku klas (4,7–5,2 σ), wszystkie „poza wzorcami” — nierozstrzygnięta.
Runda 2 (z nieporządkiem statycznym, siatką zewnętrzną bez sprężyn i sprzężeniem globalnym; trafność LOO 0,73):

| tokeny | najbliższa klasa | odległość | następna |
|---|---|---:|---:|
| bieg główny, epoki 1–60 | kryształ napędzany z nieporządkiem | 1,8–2,1 | ≥ 3,1 |
| ziarna 47, 48, 49 (gęste okno) | kryształ napędzany z nieporządkiem | 2,2–2,6 | ≥ 3,3 |
| CNN | kryształ napędzany z nieporządkiem | 2,6 | 3,5 |
| bez kodowania pozycji | szkło / kryształ / ciecz | 3,6 / 3,9 / 3,9 | — |
| lr 2e-3 | szkło | 6,5 | 7,0 |

Wzorzec „siatka zewnętrzna bez sprężyn” nie pasuje (lokalność sztywności 0,045 wobec 0,15 u tokenów) — lokalność fluktuacji tokenów wymaga sprzężenia sąsiadów, a test 2 pokazuje, skąd ono jest.

### Werdykt po testach przyczynowych

**Fenomenologicznie:** dwuwymiarowa, nieuporządkowana, napędzana i pełzająca sieć — najbliżej kryształu sprężystego z nieporządkiem statycznym spośród 8 klas wzorców.
**Mechanistycznie to nie jest kryształ z oddziaływań wyuczonych:**
- **porządek (adresy węzłów)** narzuca kodowanie pozycji — zewnętrzny potencjał siatkowy; bez niego porządek znika;
- **„sprężyny” między sąsiadami** to nakładające się pola widzenia lodygi konwolucyjnej — są wbudowane w architekturę od inicjalizacji, nie powstają w uczeniu;
- **uczenie** dokłada słabe, globalne sprzężenie przez uwagę (z 0,003 do 0,02–0,05) i napęd (pełzanie).

Najkrótsza nazwa: **sieć adresów — kryształ wymuszony z zewnątrz** (jak koloidy w siatce optycznej, „laser-induced freezing”), z wiązaniami z architektury i globalnym sprzężeniem z uwagi. Wcześniejsze zdania „sprężysty kryształ” i „fonony” należy czytać jako opis fenomenologiczny, nie mechanizm.

Pozostaje: wzorce w rundzie 2 nadal nie obejmują tokenów w pełni (większość okien „poza wzorcami”); wzorzec łączący siatkę zewnętrzną, sprzężenie sąsiadów i globalne nie był jeszcze symulowany.

## AKTUALIZACJA 2026-10-12: wspólny napęd przez wagi po poprawce wyrównania czasu (naped_v2)

Ponowne liczenie `eden/wspolny_naped_wagi.py` (wersja v2, zadanie Slurm 1826939, wyniki `eden/wyniki/naped_v2/NAPED_WAGI.json`). Przyrosty wag są teraz liczone między tymi samymi checkpointami co ruch tokenów. Przed poprawką rozjechane były 0–9 przedziałów na etap, a w etapie 10–60 łącznie 48 z 546.

| etap | wyjaśnione w oknie, K = 10 | przypadek | prognoza poza oknem (R²) | kontrola |
|---|---|---|---|---|
| epoki 1–5 | 0,36 | 0,23 | ≈ 0 | ≈ 0 |
| epoki 10–60, każdy z 5 etapów | 0,14–0,16 | 0,09 | ≈ 0 (≤ 0,01) | ≈ 0 |
| epoki 10–60 razem (n = 546) | 0,04 | 0,02 | ≈ 0 (≤ 0,01) | ≈ 0 |

Wniosek: przyrosty wag „wyjaśniają” ruch tokenów tylko w oknie, w którym je dopasowano, niewiele ponad przypadek. Nie przewidują ruchu w drugiej połowie okna. Nie ma więc mierzalnego, trwałego kanału „wspólny napęd przez wagi”, który zastępowałby sprzężenia między tokenami. Wynik nie zmienia werdyktu z 23:00: adresy narzuca kodowanie pozycji, lokalne sprzężenie daje nakładanie pól recepcyjnych lodygi, a słabe sprzężenie globalne powstaje przez uwagę.
