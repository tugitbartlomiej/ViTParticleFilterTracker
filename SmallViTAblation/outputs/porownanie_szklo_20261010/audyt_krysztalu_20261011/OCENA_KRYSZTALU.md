# Czy ruch tokenów jest bardziej kryształem niż szkłem?

Data przeglądu: 2026-10-11. Zakres: odczyt istniejących wyników lokalnych i z Edena, sprawdzenie definicji miar w kodzie i interpretacji. Nie uruchamiano treningów, eksportów latentów ani ponownego liczenia miar trajektorii. Kopie źródeł znajdują się w `zrodla/`; pochodzenie i SHA-256 w `MANIFEST_ZRODEL.json`.

## Ocena

**Tak, w odniesieniu do geometrii analogia do odkształcającej się siatki krystalicznej jest lepiej uzasadniona niż pierwotna analogia do bezpostaciowego szkła.** Najważniejszą nową przesłanką jest kontrola bez kodowania pozycji: znacznie słabnie lokalność sąsiedztwa, a wspólny dla zdjęć szablon indeksu patcha przestaje dobrze przewidywać położenia. Dochodzi do tego długofalowy, skorelowany ruch na siatce.

**Nie jest to jeszcze identyfikacja fizycznej fazy krystalicznej.** Najbezpieczniejszy opis roboczy to „odkształcająca się siatka reprezentacji związana z pozycjami patchy, ze skorelowanymi fluktuacjami podczas uczenia”. „Kryształ reprezentacji” może być skrótem analogii, jeżeli podamy mierzone własności i zakres tego porównania.

Stała topologia przestrzenna i pamięć dynamiki to różne własności. Wynik o siatce nie usuwa możliwości pamięci optymalizatora, niestacjonarnego napędu ani pozornej pamięci powstającej przez pominięcie stanu sieci. Sąsiedztwo w latencie samo nie dowodzi bezpośrednich sił między cząsteczkami.

## 1. Najmocniejszy wynik: kontrola kodowania pozycji

Źródła: `zrodla/kontrole_siatki/SIATKA_*.json`, `zrodla/metadane/`, `zrodla/kod_eden/siatka_gesty.py`. Sprawdzono też lokalny `experiments/maly_vit_szklo_pokretla_20261010/kod/trening_pokretla.py`: opcja `--bez_pozycji` zastępuje funkcję `sincos_2d` zerami przed utworzeniem ucznia.

Odczyt dla kroku 3280, pod koniec trzeciej epoki, w oddzielnych biegach kontrolnych:

| Wariant | Najbliżsi sąsiedzi w latencie będący sąsiednimi patchami | R² szablonu numeru patcha |
|---|---:|---:|
| ViT, lr 0,0005, ziarno 47 | 70,73% | 0,459 |
| ViT bez kodowania pozycji | 34,26% | −0,030 |
| ViT, ziarno 48 | 70,56% | 0,559 |
| ViT, ziarno 49 | 71,03% | 0,507 |
| ViT, lr 0,002 | 59,01% | 0,244 |
| CNN, lr 0,0005 | 56,62% | 0,115 |

Bazowy ViT i wariant bez pozycji mają ten sam SHA manifestu 35 000 klatek, ziarno 47, lr 0,0005, trzy epoki, te same 12 sond oraz rzut RP128 z ziarnem 0. Starsze metadane bazowego biegu zawierają mniej pól niż metadane nowszej kontroli; nie wykonano porównania bitowego całego historycznego środowiska.

W tym teście szablon jest wyznaczany na sześciu sondach, a sprawdzany na pozostałych sześciu. To podział sond, **nie** pełna walidacja na osobnym zbiorze filmów. Ujemne R² oznacza, że ten predyktor jest gorszy od użytego predyktora stałego; nie oznacza ujemnej ilości informacji o pozycji.

Wniosek: kodowanie pozycji ma silny wpływ na wspólną organizację tokenów. Pozostające bez niego 34% lokalnych sąsiadów pokazuje, że lokalność ma także inne możliwe źródła: treść obrazu, konwolucyjny stem i efekty brzegowe. Kontrola dotyczy uczenia od początku bez pozycji; nie jest testem natychmiastowego wyłączenia pozycji w tym samym wytrenowanym modelu. Trzy ziarna wspierają powtarzalność lokalności zwykłego ViT, ale wariant bez pozycji ma tu jedno ziarno.

## 2. Główny bieg do 60 epok: trwały porządek, zmieniająca się geometria

Źródło: `zrodla/pozycja_patchy/POZYCJA_PATCHY.json`. Sto zdjęć, 50 TRAIN i 50 VALID, 600 tokenów, pomiar w RP128 reprezentacji 768D. Szablon pozycji liczony na TRAIN i oceniany na VALID.

| Moment | R² szablonu pozycji na VALID | Lokalni sąsiedzi patchy | Jaccard sąsiadów z krokiem 0 |
|---|---:|---:|---:|
| Start | 0,9979 | 96,99% | 1,000 |
| 1 epoka | 0,6977 | 73,40% | 0,492 |
| 10 epok | 0,5711 | 71,08% | 0,469 |
| 60 epok | 0,4416 | 72,24% | 0,475 |

Sąsiedztwo jest silnie związane z siatką obrazu przez cały pomiar. W kodzie „lokalny patch” oznacza jednego z maksymalnie ośmiu sąsiadów na siatce 20 × 30, a sąsiedztwo latentne zawiera sześć najbliższych tokenów. Podane tło 8/599 ≈ 1,34% pomija mniejszą liczbę sąsiadów na brzegu, więc jest nieco zawyżonym przybliżeniem tła losowego. Nie zmienia to dużej przewagi obserwowanej lokalności.

R² centrów etapów z samego indeksu patcha wynosi 0,730 dla epok 1–5 i 0,501 dla epok 50–60. Przewidywanie centrum z poprzedniego etapu osiąga odpowiednio około 0,90–0,92 w kolejnych etapach, ale korzysta z dopasowanej na TRAIN **dowolnej mapy liniowej z wyrazem wolnym**, nie tylko obrotu i przesunięcia.

Nieuprawnione rozszerzenia tych wyników:

- Jaccard 0,475 nie oznacza, że zachowano dokładnie 47,5% sąsiadów, ani że nie zachodzi wymiana sąsiadów. Jaccard ma w mianowniku sumę zbiorów.
- Wysokie R² po dopasowaniu mapy nie dowodzi małego dryfu, sztywności ani czystego obrotu całej chmury.
- Niewyjaśnione przez szablon 56% nie jest automatycznie czystą treścią obrazu. Szablon może zawierać typową treść sceny, a reszta także interakcje treści i pozycji.
- Wysoka przewidywalność średnich centrów nie dowodzi, że klatki poszczególnych tokenów nie nakładają się.
- Wnioski liczbowe tego testu dotyczą RP128. Nie wolno przedstawiać ich jako ponownego pomiaru w pełnych 768 wymiarach.

## 3. Ruch zbiorowy: silna gładkość przestrzenna, częściowa przesłanka modów sprężystych

Źródła: `zrodla/fonony/FONONY_*.json`, `zrodla/kod_eden/fonony.py`.

W dwóch najniższych pasmach przestrzennych skryptu mieści się znaczna część sumy kwadratów przesunięć po odjęciu średniego ruchu zdjęcia:

| Etap głównego biegu | Udział niskich częstotliwości przestrzennych | Po przetasowaniu tokenów | Czas relaksacji najniższego / najwyższego pasma |
|---|---:|---:|---:|
| Epoki 1–5 | 72,66% | 3,39% | 500 / 200 kroków |
| Epoki 10–20 | 65,71% | 3,17% | 500 / 300 kroków |
| Epoki 30–40 | 61,18% | 3,41% | 800 / 300 kroków |
| Epoki 50–60 | 59,58% | 3,25% | 800 / 300 kroków |

To oznacza, że duże obszary siatki poruszają się w sposób skorelowany. „Moc” jest tu sumą kwadratów przesunięć reprezentacji, bez kalibracji jako fizyczna energia.

W oddzielnym gęstym oknie kroków 300–2188 czas relaksacji wynosi 300 kroków dla najniższego pasma i 120 dla wysokich. Jednak w późnym gęstym oknie 12034–13128 wynosi **80 kroków dla wszystkich siedmiu pasm** przy zastosowanej rozdzielczości opóźnień. Nie ma więc uniwersalnej, wykazanej we wszystkich oknach zależności czasu relaksacji od długości fali.

Ograniczenia interpretacji:

- Fourierowska analiza gładkiego pola pokaże silne niskie częstotliwości także wtedy, gdy pole jest wspólnie wymuszane zmianami wag. Kontrola z losową permutacją usuwa gładkość; nie odróżnia sprężystych sił od gładkiego wspólnego napędu.
- Kod bada widmo **przyrostów** położenia dla danego opóźnienia. Nie należy bezpośrednio przykładać do niego równowagowej zależności 1/k² dla statycznych wychyleń.
- Siatka 20 × 30 jest skończona; FFT traktuje ją okresowo i nie ma tu kontroli wpływu brzegów ani okienkowania.
- Pasma są zdefiniowane przez całkowite indeksy modów obu osi. Przy jednakowym rozstawie patchy fizyczne liczby falowe należy skalować długościami osi 20 i 30; obecny promień w przestrzeni indeksów nie jest izotropową odległością falową w jednostkach patcha.
- Czas relaksacji to pierwsze z badanych opóźnień poniżej 1/e, bez interpolacji i bez przedziałów ufności. Identyczny wynik 80 nie dowodzi identycznych pełnych krzywych relaksacji.

Wniosek: są mocne dowody gładkiego ruchu zbiorowego na siatce. Określenie „fonony” wymaga dalszego rozdzielenia odpowiedzi układu i przestrzennej struktury napędu.

## 4. Względny Lindemann: ruch ograniczony w czasie pomiaru, ale brak wyraźnie stałego plateau

Źródła: `zrodla/lindemann/`, `zrodla/kod_eden/lindemann_wzgledny.py`. Gęste okna RP128 oddzielnych biegów pokręteł.

| Okno | gamma_L przy 30 krokach | gamma_L przy 800 krokach | Dopasowany wykładnik w późnej części zakresu | Kosinus kierunku wiązania przy 800 krokach |
|---|---:|---:|---:|---:|
| 300–2188 | 0,240 | 0,638 | 0,284 | 0,603 |
| 12034–13128 | 0,179 | 0,438 | 0,268 | 0,711 |

Względny ruch sąsiadów narasta wolniej niż liniowo, ale w badanym zakresie nadal rośnie. Wynik jest zgodny z częściowym ograniczeniem ruchu i odkształcaniem układu; nie wykazuje idealnie stałych klatek.

Miara odejmuje przesunięcia pary, co usuwa wspólną translację. **Nie usuwa wspólnego obrotu, skali ani odkształcenia**: nawet sztywno obracająca się para daje zmianę wektora wiązania. Mianownik opiera się na odstępie na początku okna, a sąsiedzi są ustaleni na jego początku. Zwiększenie gamma_L nie jest więc samo w sobie topnieniem ani przeskokiem do innej klatki.

Próg 0,033 narysowany przez skrypt pochodzi z określonego fizycznego układu 2D. Nie stanowi automatycznie granicy fazy dla kwadratowej siatki patchy osadzonej w RP128. Obliczany kosinus wiązania nie jest standardowym sześciokrotnym parametrem porządku orientacyjnego g6.

## 5. Siatka nie zajmuje stałej, nieruchomej podprzestrzeni

Źródło: `zrodla/podprzestrzenie/PODPRZESTRZENIE.json`.

Podprzestrzeń obejmująca 90% energii szablonu ma sześć kierunków w pierwszej epoce i dziewięć w sześćdziesiątej. Średni kosinus kątów głównych względem podprzestrzeni pierwszej epoki spada do około 0,38 na końcu. Geometria pozycyjna może zachowywać organizację indeksów, mimo że zmienia orientację w przestrzeni reprezentacji.

W etapach 50–60 około 24,46% miary lokalnych fluktuacji leży w ośmiowymiarowej podprzestrzeni szablonu; jej wymiar stanowi 6,25% RP128. Ruch nie jest zatem całkowicie oddzielony od kierunków pozycyjnych. Użyty szablon pochodzi ze średnich po zdjęciach, więc nazwy „czysta pozycja” i „czysta treść” są mocniejsze niż operacyjna definicja.

## 6. Znaleziony problem w regresji wspólnego napędu

Źródło: `zrodla/kod_eden/wspolny_naped_wagi.py`, wiersze 43 i 88; wyniki `zrodla/naped/NAPED_WAGI.json`.

Kod najpierw liczy `dTh = np.diff(Theta, axis=0)` dla wszystkich checkpointów, a następnie wybiera `dT = dTh[idx[:-1]]`. Ruch tokenów jest natomiast liczony pomiędzy kolejnymi **wybranymi** checkpointami co 100 kroków. Pełna lista zawiera też końce epok, więc sąsiednie elementy `idx` nie zawsze są sąsiednie w pełnej liście.

Przykład: dla ruchu tokenów od 2100 do 2200 istnieje dodatkowy checkpoint 2188. Obecne indeksowanie bierze wtedy przyrost wag 2100→2188 zamiast 2100→2200. Poprawny przedział odpowiada różnicy `Theta[idx[1:]] - Theta[idx[:-1]]`. Błąd dotyczy części przedziałów; nie unieważnia testów lokalności patchy, które nie używają tej regresji.

Ponadto udziały wyjaśnione przez czynniki wag są dopasowaniem w tym samym oknie czasowym. Kontrola z permutacją jest użyteczna, ale nie zastępuje prognozy poza okresem dopasowania. Słaby wynik takiej regresji nie wyklucza napędu wagami, bo odwzorowanie wag na latent może zmieniać się w czasie i być nieliniowe. Bufory BatchNorm są celowo wyłączone z wektora wag.

Problem odnotowano; skryptów i wyników historycznych nie zmieniano w ramach tego przeglądu.

## 7. Co jest rozstrzygnięte i co pozostaje hipotezą

| Twierdzenie | Ocena na podstawie sprawdzonych plików |
|---|---|
| Tokeny zachowują silny związek z siatką patchy | Mocno wspierane do 60 epok w RP128 |
| Kodowanie pozycji odpowiada za dużą część tego związku | Mocno wspierane kontrolą bez pozycji, z ograniczeniami małej próby i jednego ziarna kontroli |
| Ruch na siatce ma dużą składową zbiorową | Mocno wspierane widmem przestrzennym |
| Cała siatka tylko przesuwa się i obraca jako bryła | Niewykazane; nie wykonano tu testu ograniczonego do ruchu sztywnego |
| Każdy patch ma rozdzieloną, nieprzekraczalną klatkę | Niewykazane |
| Wykryto sprężyste oddziaływania i fonony | Częściowe przesłanki, bez rozstrzygnięcia względem wspólnego napędu |
| Wykryto fizyczny kryształ lub jego topnienie | Niewykazane |
| Pierwotne plateau i relaksacja dowodzą szkła | Nie; istniejące kontrole OU pokazują nieswoistość tych objawów |

Najbardziej rozstrzygające dalsze analizy, jeżeli zostaną zlecone: dopasowanie ruchu sztywnego i osobno odkształcenia na danych spoza dopasowania; porównanie rozstawu centrów z amplitudami pozostałego ruchu; poprawnie wyrównany czasowo test wspólnego napędu; porównanie modeli lokalnej odpowiedzi z modelem samego skorelowanego wymuszenia. Kontrola bez pozycji już istnieje i nie trzeba jej proponować jako jeszcze niewykonanej.

## Kontekst literaturowy odczytany w poprzednim etapie rozmowy

- [Positional Encodings Anchor Spatial Structure in Vision Transformers](https://doi.org/10.48550/arXiv.2606.00124): eksperymenty wspierają organizację związaną z indeksem patcha. Nie badają trajektorii podczas treningu ani fizycznego potencjału klatki.
- [Lattice dynamics of two-dimensional colloidal crystals subject to external light potentials](https://doi.org/10.1103/PhysRevE.75.051406): analogia układu uporządkowanego na zewnętrznym podłożu; tutaj dostępny był opis bibliograficzny i abstrakt.
- [Excess vibrational modes of a crystal in an external non-affine field](https://doi.org/10.1007/s12039-017-1305-3): rozdział odkształcenia afinicznego i reszty; fizyczne wnioski autorów opierają się na określonym modelu oddziaływań.
- [Extracting vibrational modes from fluctuations: a pedagogical discussion](https://doi.org/10.1039/C2SM07445A): założenia potrzebne do interpretacji fluktuacji jako modów drgań; tutaj dostępny był abstrakt.

Ocena jest migawką ukończonych plików odczytanych 2026-10-11. Podczas sprawdzania na Edenie nadal trwały zadania nazwane `kontrola_siatki` i `lindemann`; ich przyszłe wyniki nie są częścią powyższych wniosków.
