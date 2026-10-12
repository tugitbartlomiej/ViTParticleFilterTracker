---
date: 2026-10-02
type: mixed
tags: [vit, czasteczki, tozsamosc, koopman, pykoopman, edmd, langevin, gle, kalman, lama, native-channels, forecast, no-pca]
related: [Session_2026-10-01_042605, Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek]
status: in-progress
importance: high
---
# Session Summary: Session_2026-10-02_120420

## TL;DR

Zidentyfikowano 128 oryginalnych kanałów pozwalających odczytać obecność narzędzia w znanym ROI z punktową trafnością 92,83 procent. Jest to inne zadanie niż prognoza ruchu. Benchmark 99 konfiguracji nie doprowadził do 85 procent znaku przyszłego przemieszczenia. Następnie użytkownik zlecił rzeczywisty eksperyment PyKoopman na 30 obrazach fit oraz tych samych 4 holdoutach. Dane 34 obrazów są już wyekstrahowane; na moment tego zapisu brak potwierdzonych nowych wyników PyKoopman.

## Metadata

- **Date / Time:** 2026-10-02, 12:04:20 +02:00, Europe/Warsaw. To czas utworzenia zapisu, nie deklaracja początku całej rozmowy.
- **Type:** Mixed: analiza, literatura, pobranie istniejących danych, obliczenia lokalne, protokół i audyt.
- **Status:** zapis podsumowania wykonany; nowy etap PyKoopman pozostaje w toku.
- **Duration:** nieustalona. Dostępna historia i artefakty obejmują co najmniej 1–2 października 2026.
- **Zakres źródeł:** dostępna historia rozmowy i koordynacji agentów oraz wskazane raporty i manifesty. Pełny chronologiczny transkrypt wszystkich wypowiedzi nie jest dostępny; dokument nie udaje zapisu dosłownego.
- **Zasada dowodowa:** „potwierdzone” oznacza odczyt istniejącego artefaktu lub wcześniejszą kontrolę obliczeń. „Zlecone” i „planowane” nie oznaczają ukończenia. Sam komunikat interfejsu, istnienie środowiska albo prealokowanego pliku nie potwierdza gotowego modelu.

## Objective i Context

Użytkownik chce zrozumieć, które składowe reprezentacji obrazu kodują obecność narzędzia, jak poruszają się cząsteczki tego samego obrazu w trakcie uczenia oraz czy da się przewidywać ten ruch. Rozmowa przeszła od analogii fizycznej i biologicznej do konkretnych testów kontrfaktycznych, modeli dynamiki oraz wymogu zachowania oryginalnych kanałów bez PCA. Ostatnie zlecenie rozszerza wspólną identyfikację operatora na 30 obrazów fit z osobnym transferem na 4 dotychczasowe holdouty.

## Ustalona terminologia i mechanizm ruchu

1. **Cząsteczka** to reprezentacja jednego przestrzennego tokenu obrazu. Dla tego SmallViT obraz 480×320 daje siatkę 20×30, czyli **600 cząsteczek**. To nie jest liczba minibatchy.
2. Pomiar pochodzi z ostatniej GELU `bloki[-1].mlp.act`. Pełny wektor ma **768 współrzędnych**. Są to liczby aktywacji modelu; nie przypisano im fizycznych jednostek ani materialnych sił.
3. **Makrowektory** to wspólne wektory bazowe `e_j`. **Mikrowektory** konkretnej cząsteczki to składowe `z_j e_j`; implementacja przechowuje współrzędne `z_j`, nie osobne wielowymiarowe obiekty w środku tokenu.
4. Globalny układ oznacza wspólną przestrzeń współrzędnych, a nie średnią wszystkich tokenów. Średnie FG lub tła są dodatkowymi obserwacjami modeli, nie definicją globalnego układu.
5. **J128** oznacza wybór 128 spośród 768 oryginalnych kanałów każdej cząsteczki. Nie oznacza 128 cząsteczek ani 128 obrazów. Liczba tokenów pozostaje 600 na obraz, a dla maski wybiera się odpowiadający jej podzbiór.
6. Ruch tego samego tokenu na stałym obrazie między aktualizacjami wynika ze zmian parametrów sieci oraz zapisanych statystyk BatchNorm. Odrębne porównanie kolejnych obrazów przy zamrożonej sieci bada zmianę wejścia. Stały indeks patcha w różnych obrazach nie gwarantuje śledzenia tej samej materialnej części narzędzia.

Łańcuch uczenia omawiano jako: wykonana paczka obrazów → gradient → aktualizacja AdamW i jego momentów → zmienione parametry oraz bufory BN → inne reprezentacje stałych sond. Batch wynosił 32; jedna aktualizacja następuje po paczce. Pierwszorzędowo można rozważać `Δz ≈ J_theta Δtheta` z dodatkowym wpływem BN, ale nie jest to równoważność jeden do jednego: obserwacja tokenów traci informację o pełnym stanie uczenia.

## Actions Taken — etapy i ich status

### 1. Research: Koopman na wagach versus bezpośredni ruch tokenów

**Potwierdzone:** wykonano przegląd źródeł pierwotnych, internet i Undermind. Najbliższy kandydat na pamiętany przez użytkownika artykuł to Dogra i Redman, *Optimizing Neural Networks via Koopman Operator Theory*, NeurIPS 2020. Rozpatrzono także DMD na wagach i learned lifting/Koopman Message Passing. Tytuł zapamiętanej pracy nie został jednoznacznie potwierdzony przez użytkownika.

Literatura wspiera sens badania dynamiki uczenia, lecz nie dowodzi równoważności prognozy wag i prognozy tokenów. Zaprojektowano porównanie rollout wag plus forward stałego obrazu versus bezpośredni rollout tokenów, z tym samym horyzontem i bez przyszłych gradientów. **Takie pełne porównanie nie zostało tutaj wykonane.**

Źródła: [raport](../../SmallViTAblation/plan_na_przyszle_eksperymenty/research_koopman_wagi_i_czasteczki/RAPORT_2026-10-01.md), [rejestr źródeł](../../SmallViTAblation/plan_na_przyszle_eksperymenty/research_koopman_wagi_i_czasteczki/ZRODLA_2026-10-01.md), [protokół porównania](../../SmallViTAblation/plan_na_przyszle_eksperymenty/research_koopman_wagi_i_czasteczki/PROTOKOL_POROWNANIA_2026-10-01.md).

### 2. Pobranie istniejącej historii i wag

**Potwierdzone:** pobrano 101 checkpointów `k0000.pt`–`k1000.pt`, co 10 aktualizacji. Łącznie 1 184 784 641 bajtów; 101/101 nazw, kroków, rozmiarów i SHA256 zweryfikowano. Checkpointy zawierają stan sieci i bufory BN; nie pobrano pełnego stanu AdamW.

Następnie, na osobne polecenie użytkownika, pobrano **22 140 518 528 bajtów** pełnej historii `LATENT_GESTY.npy`, shape `[1001,12,600,768]`, float32. To 1001 stanów i 1000 aktualizacji, a nie 1001 wag zapisanych na dysku. Zgodny hash lokalny i zdalny: `d148b9649154b1ac4957da2c7d39184921ad852b195fdefe0125e28b284c5576`. Odczyt danych odbywa się przez mmap. Samo pobranie nie trenowało sieci ani nie uruchamiało nowych zadań Edena.

Dowody: [pobranie wag](../../SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pobranie_wag_2026-10-01/POBRANIE_WAG_2026-10-01.md), [pobranie pełnego latentu](../../SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pobranie_latentu_2026-10-01/POBRANIE_LATENTU_2026-10-01.md), [opis źródłowego biegu](../../SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/WYNIKI.md).

Uruchomiono także osobny pilot pełnych cząsteczek i prognoz na checkpointach. Jego wyniki oraz ograniczenia pozostają w [WYNIKI_PILOT.md](../../SmallViTAblation/experiments/maly_vit_dynamika_pelnych_czasteczek_checkpointy/WYNIKI_PILOT.md) i [AUDYT_METODY.md](../../SmallViTAblation/experiments/maly_vit_dynamika_pelnych_czasteczek_checkpointy/AUDYT_METODY.md). Nie należy mylić tego pilota z późniejszym wyborem kanałów tożsamości.

### 3. Analogia biologiczna i 400 par kontrfaktycznych

**Potwierdzone:** research objął concept cells, kodowanie populacyjne, mixed selectivity oraz metody AI identyfikacji i usuwania konceptów. Token nie jest biologicznym neuronem; selektywność nie jest automatycznie przyczynową odpowiedzialnością. Pary z obiektem/bez obiektu, kierunki konceptów i activation patching mają prior art. Oryginalność konkretnej kombinacji pozostaje hipotezą wymagającą porównania metod i powtarzalnych wyników.

Przygotowano **400 par**: oryginalne obrazy train z prawdziwą maską narzędzia oraz wersje po usunięciu i inpaintingu LaMa. Kontrole wykazały brak zmian poza maską edycji; nie gwarantuje to braku artefaktów wewnątrz maski. Zachowano stałe indeksy tokenów i maski obu stron pary. Przeprowadzono osobne kontrole sham, w tym 24 obrazy z filmów testowych, bez łączenia powielonych source-ID z wcześniejszą diagnostyką mieszaną.

Na k1000 wykonano forward 800 obrazów, analizę pełnych wektorów 6729 tokenów maski i 3200 kontrolnych tokenów tła. Wczesny wariant SVD U8 miał testową BA 88,61 procent, a osobna LDA 95,28 procent. **U8 to osiem kombinacji osi, nie osiem kanałów natywnych; energia podprzestrzeni nie oznacza procentu tożsamości.** Interwencje na aktywacjach porównano z losowymi zmianami o dopasowanej normie, także po `fc2`. Wyniki funkcjonalne dotyczą badanego odczytu detektora, nie pełnej semantycznej tożsamości ani rozpoznawania różnych instrumentów.

Dowody: [research biologii i prior art](../../SmallViTAblation/plan_na_przyszle_eksperymenty/research_czasteczki_koncepty_biologiczne/RAPORT_2026-10-01.md), [protokół kontroli](../../SmallViTAblation/plan_na_przyszle_eksperymenty/research_czasteczki_koncepty_biologiczne/PROTOKOL_WALIDACJI_2026-10-01.md), [podsumowanie analizy 400 par](../../SmallViTAblation/experiments/maly_vit_tozsamosc_czasteczki_pary_400/PODSUMOWANIE.md), [audyt](../../SmallViTAblation/experiments/maly_vit_tozsamosc_czasteczki_pary_400/AUDYT_ASTRA.md).

### 4. Kierunki U500 i modele Langevina ze wspólnym ruchem

**Potwierdzone:** kierunki wyznaczono ponownie na k500, aby U1000 nie wprowadzało końca badanej trajektorii do prognozy wcześniejszego okresu. Stała sonda 2 ma źródłową rolę fit i należy do train; wcześniejsze określenie jej jako pierwszego holdoutu skorygowano. Badano 31 tokenów narzędzia oraz kontrolne tło. Podstawa U500 była zamrożona, więc ruch współrzędnych nie oznaczał dynamicznego obracania osi.

W pierwszym GLE U8 wynik H1 wyniósł 78,61 procent znaku wszystkich współrzędnych i 60,24 procent redukcji MSE. Wynik 81,59 procent dotyczył wyłącznie filtrowanej metryki zmian powyżej progu wyznaczonego z treningu; odsiew według rzeczywistego celu nie jest pewnością dostępną przy prognozowaniu.

Następny eksperyment dodał pomocnicze kierunki PCA/PLS oraz własną i globalną pamięć. Prognozował FG i tło synchronicznie, bez przyszłych rzeczywistych stanów. Uzyskano poprawy części dłuższych horyzontów, lecz nie ogólną przewagę ani skuteczny transfer. Był to etap historyczny **przed późniejszym zakazem PCA**. Nie wolno przedstawiać jego wyników jako obliczonych wyłącznie na natywnych J128.

Źródła: [U500 i pierwszy GLE](../../SmallViTAblation/experiments/maly_vit_langevin_kierunki_tozsamosci/PODSUMOWANIE.md), [audyt U500](../../SmallViTAblation/experiments/maly_vit_langevin_kierunki_tozsamosci/AUDYT_ASTRA.md), [wspólny ruch](../../SmallViTAblation/experiments/maly_vit_langevin_wspolny_ruch/PODSUMOWANIE.md), [audyt rozszerzenia](../../SmallViTAblation/experiments/maly_vit_langevin_wspolny_ruch/AUDYT_ASTRA.md).

### 5. Wymóg użytkownika: ponad 90 procent rozpoznania i brak PCA

Użytkownik doprecyzował, że 90 procent ma oznaczać wspólną trafność rozpoznania narzędzia, nie udział energii. Następnie wyraźnie zażądał zachowania oryginalnych kanałów bez PCA. Uporządkowano kanały według efektu sparowanego kontrastu wyłącznie na danych train; wspólny klasyfikator logistyczny czyta wybrane współrzędne.

**Potwierdzone:** najmniejsze K w sprawdzonej siatce przekraczające próg na walidacji to **128**, C=0,01, próg decyzji 0,5. Split po filmach: 240/75/85 par oraz 69/23/24 filmów, rozłącznych między fit/val/test.

| Metryka rozpoznania ROI na k500 | Wynik |
|---|---:|
| Filmowo ważona BA walidacyjna | 90,6703% |
| Filmowo ważona BA testowa | 92,8274% |
| 95% bootstrap po całych filmach | 88,1745–96,9940% |
| Zwykła trafność po obrazach | 91,76% |

To rozpoznanie oryginału versus LaMa przy znanym ROI i reprezentacji ważonej maską. **Nie jest to trafność każdego tokenu, prognozy dynamiki, mAP ani rozpoznawania typu narzędzia.** Dolna granica przedziału nie przekracza 90 procent. Nie wykazano utrzymania wyniku na każdym checkpointcie. Test był oglądany we wcześniejszych etapach, co ogranicza niezależność potwierdzenia.

Późniejsze GLE na tych samych 128 kanałach uzyskało na H1 71,68 procent znaku i 47,75 procent redukcji MSE; prosta ostatnia prędkość była lepsza, 78,67 procent i 55,92 procent. Dłuższe horyzonty i transfer były słabe. Te wyniki dotyczą wcześniejszych startów 800–950, a nie późniejszego benchmarku od origin 500. Nie porównujemy ich wprost z metrykami U8, bo zmienił się cel.

Dowody: [podsumowanie J128](../../SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/PODSUMOWANIE.md), [wynik testu obecności](../../SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/OCENA_OBECNOSCI_TEST.json), [zamrożone kanały](../../SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/KANALY_MODEL.npz), [audyt](../../SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/AUDYT_ASTRA.md), [wykres rozpoznania](../../SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/ROZPOZNANIE_VS_K.png).

### 6. Poszukiwanie prognozy do 500 aktualizacji: 99 konfiguracji

**Potwierdzone:** wykonano nowy research z jednym celowanym Undermind Deep Search i odczytem źródeł pierwotnych. Zbadano 99 konfiguracji: 32 liniowe dt1, 32 liniowe dt10 oraz 35 obserwatorów/kontroli. Rodziny obejmowały natywny Koopman, skończoną i wykładniczą pamięć GLE, przycięcie widma, trendy, powrót do średniej i Kalmana. Kalman faktycznie znalazł się w tym benchmarku; jego obecności nie należy wstecznie przypisywać każdemu wcześniejszemu GLE.

Fit wewnętrzny: obrazy 0–5; walidacja: 6–7. Wszystkie cele wyboru kończyły się do 500. J wybrano na k500, więc wcześniejsze foldy były retrospektywną diagnostyką w tych kanałach, a nie symulacją ich znajomości w kroku 300. Centrowanie `q−q(0)` nie ujawniało przyszłej kotwicy k500. Zmianę jednego zdania planu o jednostce rho dt10 zapisano jawnie wraz z oboma hashami; nie zmieniono siatki ani fitów.

Zamrożono 16 wariantów do końcowego testu od **jednego originu 500** aż do 1000. Główną miarą był znak **skumulowanego przemieszczenia** `z(500+H)−z(500)` na wszystkich 128 kanałach, bez odsiewu małych zmian. Oddzielnie liczono MSEskill, normę i cosinus. MSEskill to średnia redukcji błędu per obraz, nie iloraz MSE po połączeniu tokenów. Model co 10 interpolował krótsze odstępy wyłącznie między własnymi prognozami.

**Cel ponad 85 procent nie został osiągnięty przez żaden z 16 testowanych modeli w grupach fit, transfer ani all na żadnym H1–500.** To wynik dla tej siatki i tego biegu, nie dowód niemożliwości prognozowania.

| Zamrożony wariant, transfer 8–11 | H | Znak skumulowanej zmiany | MSEskill |
|---|---:|---:|---:|
| Główny wybór: powrót, okno 200/czas 200 | 500 | 56,5201% | −7,5170% |
| Osobno wybrany najlepszy MSE: okno 50/czas 50 | 500 | 62,1930% | +9,8235% |
| Ostatnia prędkość | 1 | 75,7328% | +45,7229% |

Wariant najlepszy według MSE na H500 prognozował około 0,464 rzeczywistej normy ruchu. Nie zastąpiono nim po teście głównego zwycięzcy. Niezależny audyt potwierdził 120000 agregacji oraz 288 porównań wzoru powrotu do średniej z maksymalną różnicą 4,44×10⁻¹⁶. Historia była wcześniej oglądana, a wszystkie tokeny/obrazy współdzielą trening; brak podstaw do niezależnego CI po tysiącach współrzędnych.

Dowody: [PLAN](../../SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/PLAN.md), [WYNIKI](../../SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/WYNIKI.md), [literatura i końcowy audyt](../../SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/LITERATURA_I_AUDYT.md), [znak w transferze](../../SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/ZNAK_TRANSFER.png), [błąd i amplituda](../../SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/BLAD_I_AMPLITUDA.png), [trajektorie](../../SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/TRAJEKTORIE.png).

## Nowy etap: rzeczywisty PyKoopman, 30 fit + 4 holdout = 34 obrazy

### Zlecenie i zakres modelu

Użytkownik zlecił rzeczywiste obliczenia z biblioteką PyKoopman. Delegowano nowego agenta `/root/pykoopman_obliczenia` z modelem GPT-6.1 i reasoning high do środowiska, skryptu oraz fitów. Agent `/root/plan_poprawek` odpowiada za dobór 34 obrazów i ekstrakcję. Ten zapis sesji nie modyfikuje ich plików ani wcześniejszych wyników.

To ma być **jeden wspólny operator** działający na 128 oryginalnych współrzędnych, np. `x_next=A x+b`, z `A` 128×128 i `b` długości 128; opcjonalnie równoważna augmentacja stałą do 129 wymiarów. Nie chodzi o osobny operator na każdy obraz ani średnią 30 macierzy. Pary uczące łączą tę samą cząsteczkę tego samego obrazu w kolejnych checkpointach; nie wolno tworzyć przejść między końcem jednego obrazu a początkiem następnego. Ostateczne wagi próbek i reguły walidacji muszą być zapisane przed oceną nowego testu.

**EDMD nie jest automatycznie autoenkoderem.** EDMD dopasowuje liniową ewolucję wybranych obserwabli; przy natywnych współrzędnych i stałej nie uczy kodera ani dekodera. Autoenkoder oznacza osobno uczone kodowanie/rekonstrukcję i byłby inną metodą. Sam import biblioteki nie potwierdza, że jej estimator faktycznie dopasował operator; potrzebne są wersja, wywołanie, parametry i kontrola kształtu.

### Potwierdzony dobór obrazów

- Zachowano dokładnie stare obrazy 0–11, z ich pikselami, maskami i kolejnością.
- Stare fit 0–7 plus **22 nowe obrazy** daje **30 fit**. Holdouty to dokładnie stare indeksy **8–11**, nie nowe losowe cztery obrazy. Łącznie **34**, nie 26+4.
- Nowe 22 pochodzą z 22 filmów i należą do dokładnej bazy 1500 użytej w źródłowym biegu. Wykluczono filmy wszystkich 12 dotychczasowych sond; dobór był deterministyczny, oparty na danych i geometrii masek, bez oglądania latentów.
- Spośród ośmiu dawnych fit tylko indeksy 1, 4 i 7 należą dokładnie do bazy 1500. Pozostałe są train detektora poza tą bazą. Spośród holdoutów do bazy należy indeks 8. Tego członkostwa nie należy dopowiadać pozostałym obrazom.
- Holdout dotyczy **modelu dynamiki**. Wszystkie obrazy są z train detektora; nie jest to nowy niezależny test ViT.

Dokładne holdouty: `TR_0134_S1_P03/00000341`, `TR_0132_S1_P03/00000916`, `TR_0080_S1_P03/00000686`, `TR_0043_S1_P03/00001826`.

### Potwierdzony stan ekstrakcji w chwili zapisu

Odczytano [DOBOR_AUDYT.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DOBOR_AUDYT.json), [MANIFEST.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/MANIFEST.json), [PILOT_ZGODNOSC.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/PILOT_ZGODNOSC.json) oraz [DANE.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE.json).

Pilot k0/k500/k1000 został ukończony: maksymalna różnica J128 względem dawnego zapisu wynosiła 2,265×10⁻⁶, pełnego 768D 2,384×10⁻⁶, poniżej deklarowanego progu 10⁻⁵. Bufory BN pozostały niezmienione podczas pomiaru.

Raport `DANE.json` ma `status=complete`, `all_finite=true`, shape **[101,34,600,128]**, float32, rozmiar **1 054 924 928 bajtów**, hash latentu `9ad48dba618373aa0d1d1078907c12dd6449a2fc920977c6dff39e07f6f1fa5e`. Zawiera wszystkie checkpointy 0–1000 co 10, zgodność zachowanych sond i ustawienia pomiaru: eval, FP32, bez TF32/autocast, ostatnia GELU. Jest to dowód raportowy zakończenia ekstrakcji, mocniejszy niż samo istnienie prealokowanego pliku. Autor zapisu nie powtarzał hashowania 1 GB ani forwardów. Odczytany pomocniczy `EKSTRAKCJA_POSTEP.json` miał jeszcze `running` przy `completed_steps=101`; status końcowy oparto na finalnym raporcie, a rozbieżność zgłoszono wykonawcy.

### Liczba stanów i przejść — bez mieszania przyszłości

| Zakres | Liczba na obraz | Dla 30 obrazów fit |
|---|---:|---:|
| Wszystkie zapisane checkpointy 0..1000 co 10 | 101 stanów | 3030 stanów obraz–czas |
| Wszystkie sąsiednie przejścia tego zapisu | 100 | 3000 par obraz–czas |
| Prefiks do origin 500 włącznie | 51 stanów | 1530 stanów obraz–czas |
| Przejścia zakończone do 500 | 50 | **1500 par obraz–czas do fitu** |

Każda para obraz–czas zawiera wiele odpowiadających sobie tokenów. Liczba wierszy regresji zależy od tego, czy finalny protokół użyje wszystkich 600 tokenów czy tokenów maski; nie jest równa automatycznie 1500. To nie są niezależne treningi sieci. Do prognozy od 500 nie wolno użyć przejść 500→510 i późniejszych w dopasowaniu. H500 to 50 autonomicznych kroków modelu co 10. Późniejszy zapis służy wyłącznie ocenie.

### Co jeszcze nie jest wynikiem

Na moment pierwszego zapisu **nie ma `WYNIKI.json` ani potwierdzonych prognoz nowego PyKoopman**. Obecność `.venv-pykoopman` sama potwierdza tylko utworzenie katalogu środowiska; późniejszy status importu opisano w aktualizacji poniżej. Zapowiedzi „PyKoopman powinien działać lepiej”, „30 obrazów wystarczy” lub „osiągniemy 85 procent” pozostają niezweryfikowanymi hipotezami, a nie rezultatami.

Aktualny [PLAN ekstrakcji](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/PLAN.md) odsyła do odrębnego protokołu identyfikacji i walidacji. Biblioteka, wejścia oraz wspólny operator muszą być wykazane rzeczywistymi artefaktami kolejnego etapu.

### Jednorazowa aktualizacja przed zamknięciem zapisu

Wykonawca ekstrakcji potwierdził zakończenie wszystkich 101 pomiarów. Odczytano nowy [DANE_WERYFIKACJA.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_WERYFIKACJA.json): `passed`, pełne dane skończone, 30 fit, 4 zachowane holdouty, 22 nowe obrazy, dokładna zgodność 12 dawnych identyfikatorów i masek. Pomocnicze statusy postępu zmieniono na `complete`; zachowano snapshot kodu, który rzeczywiście wykonał ekstrakcję. Była to korekta końcowego statusu, bez ponownego obliczania latentów.

Agent PyKoopman zgłosił instalację wersji 1.2.1 wraz ze zgodnym sprawdzeniem zależności. Pierwszy import wymagał korekty `setuptools` z powodu braku `pkg_resources`; zastosowano wersję 80.9. Następna aktualizacja koordynatora potwierdziła import i pięć smoke tests. Są to przekazane statusy techniczne, **nie wynik dopasowania dynamiki**. Autor tego zapisu nie instalował pakietu i nie powtarzał testów.

Odczytano [PROTOKOL_MODELU.md](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/PROTOKOL_MODELU.md): wewnętrzne 24 obrazy fit do kroku 300, 6 obrazów walidacyjnych od 300 do 500, następnie refit 30 obrazów do 500 i test 4 zachowanych holdoutów. Plan wykorzystuje tokeny narzędzia z dodatnią coverage, równe masy obrazów i pełny natywny operator. PyKoopman Identity/EDMD ma rozwiązać ważoną regresję przez skalowane pary; wewnętrzna pełna SVD solvera nie redukuje końcowych kanałów. Regularyzacja wokół identyczności jest osobno deklarowaną konstrukcją, nie nieudokumentowaną właściwością domyślnego API.

Zgłoszono koordynatorowi rozbieżność wymagającą rozstrzygnięcia w końcowym raporcie: odczytana wersja protokołu nowego modelu pomijała w głównej miarze zmiany `|Δtrue|<=1e−8`, podczas gdy wcześniejszy cel obejmował wszystkie współrzędne. Dokładny znak bez odsiewu oraz diagnostyki z progami muszą być jawnie oddzielone. Zapis sesji nie zmieniał protokołu ani kodu i nie przyznaje nowemu modelowi żadnego wyniku.

Użytkownik zlecił także dokument pierwszych dwóch kroków pipeline: `SmallViTAblation/PIPELINE_TOZSAMOSCI_NARZEDZIA.md`. Odpowiada za niego agent plan_poprawek. W chwili sprawdzenia ścieżka jeszcze nie istniała; odnotowano zlecenie, bez dodawania martwego linku i bez deklarowania ukończenia.

## Lessons Learned i granice interpretacji

- Koopman może opisywać chwilowy dryf, a GLE dodawać pamięć nieobserwowanych zmiennych. Dopasowanie takiej regresji nie dowodzi dokładnego prawa Mori–Zwanziga ani fizycznych sił między tokenami.
- Jawne historyczne wejścia oznaczałyby cechy faktycznie wykonanych paczek **tego samego biegu**. Nie odnaleziono kompletnej historii paczek/gradientów pozwalającej na taki model w gęstym biegu; nie wolno podłączać sterowań z innych seedów ani przyszłych obrazów.
- Obserwator/Kalman może oszacować stan na originie. Po odcięciu pomiarów wykonuje samą predykcję; nie dostaje przyszłych korekt. Prognoza open-loop i filtracja z obserwacjami to różne zadania.
- Normalność reszt i ich białość w czasie są oddzielnymi własnościami. Poprzednie GLE pozostawiały silną autokorelację; słowo „Langevin” nie dowodzi gaussowskiego białego szumu.
- Filmowe rozdzielenie danych identyfikacji, holdout obrazów dynamiki i odłożony czas to trzy różne podziały. Nie należy ich łączyć w jedną deklarację niezależnego testu.
- Pozytywny wynik rozpoznania ROI nie przenosi się automatycznie na prognozowanie ruchu, identyfikację typu instrumentu ani przyczynową semantykę pojedynczych osi.

## Next Steps — do wykonania i udokumentowania

- [x] Zapis 34 obrazów, masek, J128 oraz raportowanej ekstrakcji 101 checkpointów; odczyt pilota zgodności.
- [x] Otrzymano potwierdzenie instalacji/importu PyKoopman 1.2.1 oraz smoke tests; nadal trzeba zachować ich dowody w artefaktach wykonawcy.
- [ ] Zamrozić pełny protokół doboru parametrów i wagi próbek przed oceną wyników nowego testu; zachować dokładne stare holdouty.
- [ ] Dopasować jeden wspólny operator z par tej samej cząsteczki i obrazu, wyłącznie w prefiksie do 500 dla prognozy od 500.
- [ ] Zapisać macierz, intercept, kształty, hashe i test niezależności od przyszłego sufiksu; wykazać brak PCA/autoenkodera.
- [ ] Obliczyć autonomiczne prognozy i porównać ze stałym stanem, ostatnią prędkością oraz prostym powrotem do średniej na identycznych celach.
- [ ] Raportować osobno znak wszystkich kanałów, MSEskill, amplitudę i transfer; nie zamieniać najlepszej konfiguracji po zobaczeniu testu.
- [ ] Wykonać inkrementalne indeksowanie tego zapisu `.sessions`; rzeczywisty status tej operacji zawiera [INDEX_STATUS.json](INDEX_STATUS.json), niezależnie od biegu PyKoopman.

## Mechanizm zapisu i powiązane sesje

Zastosowano [SESSION_RULES.md](../SESSION_RULES.md): folder `Session_YYYY-MM-DD_HHMMSS`, lokalny czas i dwa wymagane pliki. Użyto opisanego ręcznego tworzenia, ponieważ stary `save_session.py` nadal zapisuje UTC i historyczny układ `analysis/sesja_...`, sprzeczny z regułami v2. Nie zmieniano narzędzi ani starych sesji, nie kopiowano dużych danych i nie wykonywano commitów. Indeksowanie ma być inkrementalne zgodnie z [tools/README.md](../tools/README.md), bez pełnej przebudowy i bez obciążania GPU aktywnej ekstrakcji.

Powiązania: [animacje pipeline z poprzedniego zapisu](../Session_2026-10-01_042605/SESSION_SUMMARY.md), [wcześniejsze siły i kierunki cząsteczek](../Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek/SESSION_SUMMARY.md). Animacje i etykiety interfejsu pełnią funkcję objaśniającą; za dowody zakończenia przyjęto raporty, manifesty oraz wyniki kontroli wymienione powyżej.

## Keywords

`600 tokenów` `768 współrzędnych` `128 natywnych kanałów` `J128` `makrowektory` `mikrowektory` `LaMa 400 par` `tożsamość narzędzia` `AdamW` `BatchNorm` `Koopman na wagach` `GLE` `pamięć` `Kalman open-loop` `EDMD bez autoenkodera` `PyKoopman 30 fit 4 holdout` `H500` `bez PCA` `wspólny operator`

**Created:** 2026-10-02 12:04:20 +02:00.  
**Updated:** jedna aktualizacja statusu wykonawców po pierwszym zapisie; nowy eksperyment pozostaje w toku. Stan indeksowania i kontrolę linków zapisuje osobny plik techniczny.
