---
title: "Idealny latent, Stable Diffusion i równanie Langevina: co ma uzasadnienie naukowe i jak to połączyć w dobór danych dla małego ViT"
tldr: "Szukanie latentu, przy którym głowa detektora trafia, ma precedensy (LabelEnc, LGD, target propagation) i jest poprawną diagnozą. W obecnej postaci typowy cel leży jednak około 29 razy dalej od środka chmury prawdziwych latentów niż typowy prawdziwy token i prawie 6 razy dalej niż najbardziej nietypowy z 240 tysięcy, więc obraz ze Stable Diffusion przebył 1,3 procent drogi. Obraz, który by ten cel realizował, byłby dla sieci przykładem już rozwiązanym; literatura o generowaniu danych pod kontrolą uczonego modelu pokazuje, że uczą przykłady poprawne i umiarkowanie trudne. Siła tożsamości liczona wzdłuż kierunku z końca rozgrzewki nie idzie w kampanii 4.3 w parze z mAP. Propozycja: wyrocznia z ograniczonym zasięgiem daje kierunek, którego potrzebuje głowa; równanie Langevina mówi, w którą stronę ruszy latent każda kandydująca paczka; wybieramy paczkę o najlepszej zgodności i sprawdzamy na stracie kontrolnej. Prognozujemy skutek wybranej paczki w horyzoncie 10 kroków, a nie cały ruch latentu."
date_created: 2026-10-05
date_modified: 2026-10-05
type: synthesis
tags: [synteza, vit, latent, wyrocznia, stable-diffusion, langevin, koopman, dobor-danych, sterowanie-uczeniem, literatura]
sources:
  - Research/Audyty/audyt-vit-przyspieszenie-pipeline-2026-10-04.md (AU59-AU63)
  - experiments/vit_latent128_maksymalizacja (WYNIKI.md), experiments/vit_latent128_obraz_sd (WYNIKI_SD.md), experiments/vit_optymalizacja_latentu128 (WYNIKI.md)
  - "ViT: SmallViTAblation/docs/23, 25, 26; experiments/maly_vit_petla_sd_rownanie (WYNIKI.md, wyniki_eden43); experiments/maly_vit_sufit_przewidywalnosci_100 (WYNIKI_ROWNANIE_Z_PEDEM.md); experiments/maly_vit_horyzont_rownania_krok (WYNIKI.md)"
  - Research/syntheses/materialy/vit-idealny-latent-sd-langevin-2026-10-05 (OSIAGALNOSC.json, TELEMETRIA43_TOZSAMOSC.json i skrypty)
  - "Undermind, workspace ViT: trzy nowe wyszukiwania pogłębione z 2026-10-05 i siedem wcześniejszych; lista w rozdziale 10"
explored: false
confidence: medium
---

# Idealny latent, Stable Diffusion i równanie Langevina

Dokument odpowiada na trzy pytania użytkownika z 2026-10-05:

1. Czy szukanie takich ustawień latentu, przy których detektor ma najwyższe mAP, i rysowanie z nich obrazu przez
   Stable Diffusion ma uzasadnienie naukowe, potrzebne do publikacji.
2. Jak wykorzystać to do prognozowania zachowania wektora tożsamości narzędzia i do doboru zdjęć, żeby przyspieszyć
   uczenie.
3. Jak podłączyć do tego równanie Langevina i Koopmana.

**Status.** Rozdziały 2-5 to przegląd i dwa obliczenia na już zapisanych plikach. Rozdział 6 to PROPOZYCJA pomiarów.
Nie uruchomiono żadnego uczenia, generowania obrazów ani zadania na Edenie. Każdy pomiar z rozdziału 6 wymaga osobnej
zgody.

## Słownik (zwykłymi słowami)

- **Latent**: liczby wewnątrz sieci opisujące jeden token obrazu. Tutaj 768 liczb po funkcji GELU w MLP ostatniego
  bloku małego ViT; obraz ma 600 tokenów.
- **Głowa**: ostatnia część sieci, która z latentu robi ramkę narzędzia i pewność wykrycia.
- **Wyrocznia**: procedura z eksperymentu `vit_latent128_maksymalizacja`. Wagi sieci są zamrożone, a zmieniane są same
  liczby latentu (128 wybranych kanałów w każdym z 600 tokenów), osobno dla każdego obrazu, tak żeby głowa trafiła w
  prawdziwą ramkę. Wyrocznia zna etykietę, więc nie jest detektorem.
- **Idealny latent**: wynik wyroczni, czyli „ustawienia latentu, przy których detektor jest najlepszy”.
- **Siła tożsamości**: rzut latentu narzędzia na jeden kierunek (LDA) oddzielający narzędzie od tła, wyznaczony na końcu
  rozgrzewki i stały do końca biegu (docs/26, krok 2).
- **Sondy**: stałe obrazki, których latent śledzimy w czasie uczenia. **Kotwica**: sonda użyta jako wzorzec dla obrazu SD.
- **Siła paczki** `S(B)`: o ile przesunie się latent sond, gdy sieć zrobi jeden krok uczenia na paczce `B`. Liczona
  przed krokiem, przez „wirtualny krok” (docs/23). **Pchnięcie**: to samo dla siły tożsamości.
- **Zbiór kontrolny**: oznaczone klatki, na których mierzymy błąd detektora, ale na których sieć się nie uczy.
- **Umiejętność prognozy**: jaka część błędu prognozy „nic się nie zmieni” została usunięta (1 = prognoza idealna,
  0 = tyle co brak zmiany).

## 0. Werdykt

1. **Pierwsza połowa pomysłu ma uzasadnienie.** Szukanie latentu, przy którym głowa trafia, to znana konstrukcja:
   cechy wyprowadzone z etykiet jako cel pośredni dla sieci (LabelEnc, LGD, LabelDistill) i cele warstw ukrytych
   liczone ze straty wyjścia (target propagation). Jako diagnoza „czego głowa potrzebuje od reszty sieci” jest poprawna.
2. **W obecnej postaci idealny latent leży poza zasięgiem obrazów.** Policzone z zapisanych plików: typowa zmiana
   współrzędnej to 22 naturalne odchylenia kanału, a 99 procent celów jest bardziej nietypowych niż najbardziej
   nietypowy z 240 tysięcy prawdziwych tokenów. Dlatego obraz SD przebył 1,3 procent drogi do celu, w kierunku
   praktycznie losowym (kosinus 0,03).
3. **Obraz, który by ten cel realizował, uczyłby mało.** Przy idealnym latencie strata detekcji jest 11 razy mniejsza,
   więc sygnał uczący jest bliski zera. Prace o generowaniu danych pod kontrolą uczonego modelu zgodnie pokazują, że
   pomagają obrazy poprawne i umiarkowanie trudne, a obrazy „najpewniejsze” dają najmniej. Wyjątek to wczesna faza i
   mało danych, gdzie przykłady łatwe i typowe pomagają.
4. **Siła tożsamości nie jest miarą postępu detektora.** W kampanii 4.3 ramię bez dokładania klatek ma najwyższą siłę
   tożsamości i najniższe mAP, a zmiana siły w obrocie nie mówi nic o zmianie mAP (korelacja 0,006). Prognozowanie tej
   wielkości nie pomoże w doborze.
5. **Jak to połączyć.** Wyrocznia z ograniczonym zasięgiem daje kierunek, w którym latent powinien się ruszyć, żeby
   głowa trafiała. Równanie Langevina mówi, w którą stronę ruszy latent każda kandydująca paczka. Wybieramy paczkę o
   najlepszej zgodności obu kierunków i sprawdzamy ją na stracie kontrolnej. To jest odpowiednik reguły
   „wszechwiedzącego nauczyciela” z teorii machine teaching.
6. **Co prognozować na 10 kroków.** Nie cały ruch latentu (sufit 0,22, bo rządzą nim nieznane przyszłe paczki), tylko
   skutek paczki, którą sami wybieramy. Ta część jest sterowalna i według dwóch prac z 2026 roku przewidywalna w
   horyzoncie 8-12 kroków. U nas jest to niezmierzone; rozstrzyga pomiar P1.
7. **Rola Stable Diffusion.** Uzasadniona tam, gdzie prawdziwa pula klatek nie potrafi pchnąć latentu w brakującym
   kierunku. Cel dla SD musi leżeć w zasięgu obrazów i mieć regulowaną trudność.

## 1. Skąd pochodzą odpowiedzi

- **Własne pomiary projektu**: audyt z 2026-10-04, trzy eksperymenty J128, dokumenty 23, 25 i 26 repozytorium ViT,
  kampania 4.3 (10 ziaren), pomiary sufitu przewidywalności i równania z pędem z 2-5 października.
- **Dwa nowe obliczenia na zapisanych plikach** (bez sieci, bez uczenia; skrypty i wyniki w
  `Research/syntheses/materialy/vit-idealny-latent-sd-langevin-2026-10-05/`): odległość celów wyroczni od prawdziwych
  latentów oraz związek siły tożsamości z mAP w 840 rekordach kampanii 4.3.
- **Literatura**: trzy nowe wyszukiwania pogłębione Undermind (99, 191 i 118 prac) oraz siedem wcześniejszych z tego
  samego workspace. Pełne teksty 32 prac przeczytane w tej sesji przez podagentów Undermind, do tego 8 odczytów z
  audytu; reszta na poziomie streszczeń. W rozdziale 9 każda praca ma oznaczenie P (pełny tekst) albo S (streszczenie).

## 2. Co mówią własne dane

### 2.1 Jak daleko idealny latent leży od latentów prawdziwych zdjęć

Źródło: `outputs/vit_latent128_maksymalizacja/TARGETS_FULL400.npz` (400 obrazów, 600 tokenów, 128 kanałów, razem
240 000 prawdziwych tokenów i tyle samo celów). Skrypt `osiagalnosc_celu.py`, wynik `OSIAGALNOSC.json`.

| wielkość | prawdziwe tokeny | cele wyroczni |
|---|---:|---:|
| naturalny rozrzut kanału (odchylenie standardowe), mediana po 128 kanałach | 0,17 | |
| typowa (średniokwadratowa) zmiana jednej współrzędnej | | 3,75, czyli 22 odchylenia |
| zmiana współrzędnej w odchyleniach kanału: mediana, 90. centyl, 99. centyl | | 1,4; 23; 91 |
| udział współrzędnych zmienionych o ponad 10 odchyleń | | 17 procent |
| udział celów powyżej największej wartości kanału widzianej w prawdziwych tokenach | | 20 procent |
| tokeny z choć jedną współrzędną poza zakresem prawdziwych | | 99,8 procent |
| nietypowość tokenu (odległość Mahalanobisa na wymiar): mediana, centyl 99,9, maksimum | 0,90; 2,9; 4,6 | 26; 222; 593 |
| cele bardziej nietypowe niż najbardziej nietypowy prawdziwy token | | 99,1 procent |
| długość zmiany tokenu wobec typowej odległości tokenu od średniej (mediany) | 2,4 | 28, czyli 12 razy więcej |
| energia zmiany skupiona w 10 procentach współrzędnych | | 94 procent |

Rozrzut kanału liczony jest po wszystkich prawdziwych tokenach, narzędzia i tła razem. Nietypowość mierzy, jak daleko
token leży od środka chmury prawdziwych tokenów, w jednostkach jej własnego rozrzutu; liczba 1 oznacza „typowy”.
Typowy cel leży więc około 29 razy dalej niż typowy prawdziwy token i prawie 6 razy dalej niż najbardziej nietypowy. Przebieg optymalizacji: średnia strata spada z 2,38 do 0,32 po 100 krokach (największa zmiana
współrzędnej 38) i do 0,21 po 800 krokach (największa zmiana 201). Po optymalizacji mediana straty obrazu to 0,054, a
48,5 procent obrazów ma stratę poniżej 0,05.

Wcześniejszy eksperyment `vit_optymalizacja_latentu128` z ograniczeniem zmiany do 1 umownego odchylenia kanału dał na
85 obrazach części test AP50 0,226 → 0,277 (punkt kontrolny 500, inny niż wyżej). To jest rząd wielkości zysku
dostępnego blisko naturalnego zakresu: około +0,05, a nie +0,45.

### 2.2 Co zrobił obraz ze Stable Diffusion

Źródło: `outputs/vit_latent128_obraz_sd/LATENT_cel_0000.npz`, `LATENT_obrazu_sd.npz`, `RESULT.json`.

| wielkość | wartość |
|---|---:|
| długość ruchu latentu (128 kanałów) wobec długości drogi do celu | 1,35 procent |
| kosinus ruchu z kierunkiem celu | 0,029 |
| część drogi pokonana w kierunku celu | 0,04 procent |
| ruch 128 kanałów wobec ich normy w obrazie źródłowym | 17 procent |
| ruch pozostałych 640 kanałów wobec ich normy | 17 procent |
| gradient prowadzenia (wartość średniokwadratowa) | od 2e-6 do 1e-4 |

Prowadzenie było tego samego rodzaju co w pętli 4.3: bez uczenia generatora, gradientem energii liczonej na
przewidywanym czystym obrazie, w duchu FreeDoM [Yu23]. Obraz zmienił latent o wielkość naturalną (17 procent normy),
ale cel leży 74 razy dalej i w kierunku, z którym ruch obrazu nie ma związku. To nie jest kwestia zbyt małej liczby kroków prowadzenia: realistyczny obraz zmienia latent o
ułamek rozrzutu, a cel wymaga dziesiątek rozrzutów.

Dla porównania, w pętli 4.3 cel dla SD leżał blisko: latent kotwicy przesunięty wzdłuż kierunku tożsamości o
przewidzianą zmianę. Tam SD pokrywało cel w medianie w 88,5 procent (docs/26). Różnica między 88 procent a 1,3 procent
to różnica między celem w zasięgu i celem poza zasięgiem.

### 2.3 Siła tożsamości a mAP w kampanii 4.3

Źródło: `OBROT.json` i `OCENY.json` z `wyniki_eden43/petla43b_s47..s56` (10 ziaren, 6 ramion, 14 obrotów po 500 kroków).
Skrypt `telemetria43_tozsamosc_a_map.py`, wynik `TELEMETRIA43_TOZSAMOSC.json`.

| ramię | średnia siła tożsamości | średnie przewidziane pchnięcie | pole pod krzywą mAP@0,5 | mAP@0,5 na końcu |
|---|---:|---:|---:|---:|
| bez dokładania | **4,88** | 0 | **0,422** | 0,451 |
| kolejne | 4,52 | +0,029 | 0,465 | 0,519 |
| losowe | 4,51 | +0,022 | 0,472 | 0,544 |
| metoda | 4,41 | +0,032 | 0,488 | 0,566 |
| metoda_kl3 | 4,30 | +0,022 | 0,487 | 0,557 |
| sd_bez_prognozy | **4,19** | +0,025 | **0,487** | 0,554 |

Kolejność jest odwrócona: ramię, które uczy się najgorzej, ma najwyższą siłę tożsamości, a ramiona uczące się
najlepiej mają najniższą (korelacja rang na 6 ramionach −0,77). W czasie dzieje się to samo: średnia siła spada z 5,26
w obrocie 3 do 3,95 w obrocie 16, a mAP rośnie z 0,36 do 0,53.

Korelacje liczone po odjęciu średniej z ramion w każdej parze (ziarno, obrót), z przedziałem z losowania ziaren:

| pytanie | korelacja | przedział 95 procent | n |
|---|---:|---|---:|
| czy przewidziane pchnięcie zapowiada zmianę siły tożsamości w obrocie | +0,21 | od 0,12 do 0,32 | 700 |
| czy zmiana siły tożsamości idzie w parze ze zmianą mAP | +0,006 | od −0,07 do 0,08 | 840 |
| czy wyższa siła tożsamości to wyższe mAP (to samo ziarno i obrót) | −0,19 | od −0,37 do 0,01 | 840 |
| czy przewidziane pchnięcie zapowiada zmianę mAP | −0,06 | od −0,15 do 0,03 | 700 |
| czy siła teraz zapowiada przyrost mAP w następnym obrocie | −0,03 | od −0,10 do 0,05 | 780 |

Pierwszy obrót, w którym wszystkie ramiona ruszają z tego samego stanu (50 par): pchnięcie wobec siły 0,03, pchnięcie
wobec mAP −0,25.

Zastrzeżenia: kierunek tożsamości jest stały od końca rozgrzewki, więc spadek rzutu może oznaczać, że reprezentacja
odeszła od starego kierunku, a nie że narzędzie jest gorzej zakodowane. Siła liczona jest na 180 sondach (12 wycinków
w 16 wersjach), a mAP na 3000 klatkach walidacji. To korelacje, nie dowód przyczyny.

### 2.4 Co z tego wynika

- Cel wyroczni bez ograniczenia zasięgu nie jest opisem „jak wygląda narzędzie, gdy detektor widzi je najlepiej”. Jest
  opisem tego, co głowa chciałaby odczytać, w obszarze, którego żaden obraz nie wytwarza. To odpowiednik zaburzenia
  przeciwnego (adversarial) w przestrzeni cech, tylko skierowanego na korzyść.
- Siła tożsamości wzdłuż stałego kierunku LDA jest raczej znacznikiem tego, jak mało reprezentacja odeszła od stanu po
  rozgrzewce, niż miarą jakości detekcji. To liczbowe potwierdzenie ustalenia AU62 audytu dla małego ViT.
- Jednokrokowe pchnięcie niesie słabą, ale istotną informację o zmianie w skali obrotu (około 4 procent zmienności).
  Równanie jednego kroku jest trafne (96 procent w każdym ramieniu), lecz pojedyncze pchnięcia w dużej części się
  znoszą w ciągu kilkunastu kroków.

## 3. Czy szukanie latentu o najwyższym mAP ma uzasadnienie w literaturze

### 3.1 Precedensy

| praca | co robi | wynik | warunek powodzenia |
|---|---|---|---|
| LabelEnc [Hao20] | koder zamienia etykiety (ramki) na „idealne” cechy, z których głowa detektora odtwarza etykiety; cechy te są dodatkowym celem dla sieci podczas uczenia | około +2 AP na COCO dla RetinaNet, FCOS i FPN, zero kosztu w inferencji | cechy z etykiet muszą być uczone razem z prawdziwą siecią; bez tego zysk spada o około 1 AP; bez rozgrzewki strata pomocnicza „oszukuje” i wynik jest zły; liczba iteracji się podwaja |
| LGD [Zha21p] | nauczyciel budowany z etykiet i z bieżących cech ucznia, bez osobnej dużej sieci | od +1,8 do +2,8 AP; koszt części nauczycielskiej o 51 procent niższy niż w klasycznej destylacji | zysk LabelEnc znika na mocnych detektorach (44,0 wobec 44,1), LGD go utrzymuje (44,9) |
| LabelDistill [Kim24k] | koder etykiet jako przybliżona odwrotność zamrożonej głowy | +5,1 mAP na nuScenes (test) | zależność od jakości etykiet |
| Target propagation [Lee14, Meu20] | cel dla warstwy ukrytej liczony ze straty wyjścia; niższe warstwy uczą się go osiągać | na małych sieciach wynik bliski zwykłemu uczeniu (CIFAR-10, mała sieć splotowa: 24,0 wobec 24,4 procent błędu), niższa strata ucząca na MNIST; liczby z [Meu20] | teoria (związek z metodą Gaussa-Newtona) działa tylko dla **małych** przesunięć celu względem obecnej aktywacji |
| Informacja uprzywilejowana [Lop15] | nauczyciel widzi więcej niż uczeń (tu: etykietę) | ramy teoretyczne | uczeń musi umieć odtworzyć cel bez tej informacji |
| Unadversarial examples [Sal20e] | wzór naklejany na obiekt, optymalizowany tak, by zamrożony model mylił się jak najmniej | wyraźny wzrost trafności na ImageNet (autorzy podają 26,3 procent przy wzorze zajmującym 5 procent obrazu) i większa odporność na zakłócenia | wzór musi być wykonalny (optymalizacja po realnych przekształceniach); narzędzie do projektowania obiektów, nie dane uczące |
| Odwracanie detektorów [Cao21b] | obraz optymalizowany tak, by detektor dał zadany układ ramek | wgląd: głowa klasyfikacji patrzy na teksturę, głowa ramek na kontur | tylko interpretacja, bez użycia w uczeniu |

### 3.2 Wniosek dla wyroczni J128

- **Tak jako diagnoza.** Wynik 0,44 → 0,88 AP50 pokazuje, że głowa potrafi trafić, jeśli dostanie właściwe liczby.
  Poprawa dotyczy głównie położenia ramki: AP75 rośnie z 0,02 do 0,77, średnie pokrycie ramki z 0,47 do 0,87, a
  średnia pewność prawie się nie zmienia (0,69 → 0,73). Interpretacja: wyrocznia wpisuje w latent dokładne wymiary
  ramki, w języku, który czyta głowa. To odpowiada temu, co LabelEnc i LabelDistill nazywają kodowaniem etykiety w
  przestrzeni cech.
- **Tak jako cel pośredni dla zdjęć uczących.** Użycie etykiety do wyznaczenia celu dla klatki uczącej jest uczciwe,
  bo uczenie i tak z etykiety korzysta. Warunki z literatury: cel blisko tego, co sieć umie wytworzyć (LabelEnc), małe
  przesunięcia (target propagation), dobre etykiety (LabelDistill).
- **Nie jako wynik detektora.** 0,88 to wynik interwencji z etykietą, nie detekcji. Audyt (AU61) już to zaznaczył.
- **Obecny cel łamie warunek małych przesunięć** o dwa rzędy wielkości (rozdział 2.1). Żeby był użyteczny, trzeba go
  liczyć z ograniczeniem zasięgu.

## 4. Czy rysowanie obrazu z idealnego latentu ma uzasadnienie

### 4.1 Jako wizualizacja tego, co widzi detektor

Prowadzenie dyfuzji tak, by cechy obrazu w zamrożonej sieci trafiły w zadany wektor, jest opisane i działa, **gdy cel
pochodzi z naturalnego obrazu**. W pracy [Shi25f] odległość cech spada do pojedynczych jednostek przy 100 procentach
udanych prób, kosztem 2,5-6 godzin na obraz. Ta sama praca pokazuje granicę: dla celów dalekich od cech naturalnych
obrazów (osadzenia tekstu w CLIP) realistyczny obraz nie powstaje, bo „prawdziwe obrazy nie zamieszkują tych
obszarów”. Dekodery uczone (RCDM [Bor21], FeatInv [Neu25]) odtwarzają obraz z cech naturalnych obrazów, więc dotyczy
ich ta sama granica. Z kolei [Sab15] pokazuje, że niewidocznym dla oka zaburzeniem pikseli można nadać obrazowi
reprezentację innego obrazu. Takie zaburzenia leżą poza zbiorem naturalnych obrazów. Stable Diffusion ich nie rysuje,
i dobrze, bo sieć uczona na takich wzorach reagowałaby na szum.

Wniosek: żeby pokazać „co detektor chce zobaczyć”, cel musi leżeć w naturalnym zakresie, albo optymalizować trzeba
wejście generatora (szum, warunek), tak by każdy pośredni wynik był obrazem [Wal23b, Aug23]. Nasze liczby z 2.1 i 2.2
zgadzają się z tą granicą.

### 4.2 Jako dane uczące

**Argument z minimum.** Wyrocznia szuka minimum straty głowy względem latentu. Blisko minimum nachylenie jest bliskie
zera, więc przez te kanały do reszty sieci płynie znikomy sygnał uczący. Liczbowo: strata przy idealnym latencie jest
11 razy mniejsza niż wyjściowa (0,21 wobec 2,38). Obraz realizujący ten latent byłby więc przykładem, który sieć już
„umie”.

**Literatura o generowaniu danych pod kontrolą uczonego modelu.**

| praca | sygnał prowadzący generator | wynik (wobec generowania bez prowadzenia, o ile nie zaznaczono inaczej) |
|---|---|---|
| [Hem23] | niepewność (entropia) klasyfikatora | ImageNet-LT: 64,7 wobec 58,9; rzadkie klasy 59,1 wobec 51,1. Zwykłe prowadzenie klasyfikatorem ciągnie do środka klasy, czyli do przykładów już pewnych |
| [Hem25] | entropia bieżącego ucznia, generowanie w trakcie uczenia | ta sama trafność przy 7,5 raza mniejszej liczbie próbek; bez prowadzenia 61,6 wobec 68,0; zbyt mocne prowadzenie szkodzi (68,0 → 66,9) |
| [Wan24m] | zadany poziom trudności | łatwe +0,2 punktu, średnie +0,8, bardzo trudne −0,4 |
| [Zha22i] GIF | latent obrazu wyjściowego optymalizowany w małej kuli tak, by klasa została, a niepewność i różnorodność rosły; potem dekodowanie przez SD | obrazy naturalne 67,8 wobec 54,3; medyczne 81,7 wobec 79,3 |
| [Hay25] | sygnał rzadkości z uczonego modelu | 82,4 wobec 59,9; siła prowadzenia dobierana tak, by pewność klasy spadła, ale nie do zera |
| [Fri25] GAUDA | niepewność modelu w trakcie uczenia; dane chirurgiczne, w tym zaćma (CaDISv2) | +1,6 i +1,5 punktu IoU wobec porównywanych metod; czas uczenia 15,3 → 22,6 godziny |
| [Ven25e] | wybór próbek ocenianych wysoko przez model zadaniowy, dostrojenie generatora do nich | dane syntetyczne bez dopasowania **obniżały** wynik segmentacji o 1-2 punkty (Endoscapes) i 4-14 punktów (AutoLaparo); po dopasowaniu zysk 2-10 procent |
| [Kim26i] | zgodność gradientu próbki z gradientem walidacji | AUC 0,894 wobec 0,866 dla generowania bez tego sygnału i 0,873 dla samych danych prawdziwych (mało danych medycznych) |
| [Guo25c] | użyteczność = spadek straty walidacyjnej | 62,0 wobec 58,7 |
| [Kar26] | generowanie z reprezentacji; filtr odrzuca 25 procent próbek najbliższych środka klasy i 25 procent najdalszych | cztery razy większy zbiór syntetyczny: 79,3 wobec 77,3 dla danych prawdziwych |

Dwie rzeczy powtarzają się we wszystkich tych pracach. Po pierwsze, obraz musi być **poprawny** (klasa i położenie
zgodne z etykietą); temu służy kotwica klasy w GIF, dolna granica pewności w [Hay25] i dopasowanie w [Ven25e]. Po
drugie, obraz musi być **niebanalny** dla uczonej sieci; najlepiej działa trudność umiarkowana, a przesada szkodzi.
Obraz z idealnego latentu spełnia pierwszy warunek w skrajnej postaci i zupełnie nie spełnia drugiego.

Dane odwrócone z modelu (DeepInversion [Yin19], wersja dla detekcji [Cha21d]) służą w literaturze do uczenia
**innego** modelu bez dostępu do danych. Wyszukiwanie nie znalazło dowodu, że obrazy odwrócone z modelu przyspieszają
uczenie tego samego modelu.

### 4.3 Kiedy przykłady łatwe pomagają

- Gdy danych jest mało, lepiej zostawić przykłady łatwe, a gdy dużo, trudne; teoria i potwierdzenie na CIFAR-10 i
  ImageNet [Sor22]. To samo w destylacji zbiorów: przy małym budżecie wzorce łatwe dają 60 procent, trudne 10 procent,
  a przy dużym jest odwrotnie [Guo23f].
- Łatwiejsze, bardziej typowe wersje trudnych przykładów działają jako rozgrzewka, po której przechodzi się do
  prawdziwych obrazów; poziom dobierany jest według zmierzonego postępu ucznia (DisCL [Lia24e]: rzadkie klasy
  ImageNet-LT z 4,4 do 23,6 procent).
- W detekcji próbki „pierwszorzędne” (dobrze pokrywające obiekt) ważą dla mAP więcej niż najtrudniejsze: +1,7 AP wobec
  kopania trudnych przykładów [Cao19].
- Nasz własny wynik: klatki podobne do wycinków narzędzia przyspieszają mały ViT wobec losowych (4.3: +0,015 pola pod
  krzywą, 9 z 10 ziaren), a mały ViT po rozgrzewce jest właśnie we wczesnej fazie i przy małym zbiorze.

### 4.4 Wniosek i co trzeba zmienić, żeby to obronić w publikacji

Do obrony jest zdanie: „optymalizacja latentu z etykietą wskazuje, czego głowa potrzebuje; obraz realizujący cel w
zasięgu jest łatwym, poprawnym wzorcem”. Nie do obrony jest zdanie: „obraz, przy którym detektor jest najlepszy, jest
najlepszym obrazem uczącym”.

Metoda użytkownika (docs/26: tożsamość → prognoza → SD → wyszukanie → uczenie) zostaje linią główną. Poniższe różnice
są propozycjami; żadnej nie wprowadzono.

- **Z1. Cel w zasięgu.** Wyrocznia z ograniczeniem zmiany każdej współrzędnej do `r` naturalnych odchyleń kanału
  (`r` = 0,5; 1; 2) i z karą za nietypowość tokenu. Powód: tylko taki cel obraz może osiągnąć (2.1, 2.2, [Shi25f]).
- **Z2. Cel po dodaniu gałęzi resztkowej albo optymalizacja wejścia SD.** Powód: nowy obraz zmienia też gałąź, która
  omija MLP (AU61), więc trafienie w sam latent nie gwarantuje detekcji.
- **Z3. Pokrętło trudności.** Cel `z_λ = z + λ·d`, gdzie `d` to przesunięcie z Z1, a `λ` od 0 do 1. `λ = 1` to wersja
  najłatwiejsza. Poziom dobierany według zmierzonego postępu, nie z góry. Powód: [Wan24m], [Hem25], [Lia24e].
- **Z4. Kontrole.** SD bez prowadzenia; wyszukanie z prawdziwej kotwicy bez SD (brak wskazany w AU60); obrazy SD jako
  dane uczące i osobno jako zapytania; losowe klatki. Powód: bez nich nie wiadomo, co daje zysk, a w porównaniach z
  klasyfikacji wyszukane prawdziwe obrazy uczyły lepiej niż syntetyczne [Gen24].

## 5. Jak podłączyć równanie Langevina i Koopmana

### 5.1 Trzy elementy

1. **Mapa braków** (z wyroczni): dla każdej klatki kontrolnej kierunek, w którym jej latent powinien się ruszyć, żeby
   głowa trafiła.
2. **Model reakcji** (równanie Langevina): dla każdej kandydującej paczki kierunek, w którym ruszy ona latent klatek
   kontrolnych, policzony przed uczeniem.
3. **Wybór**: paczka, której przewidziany ruch najlepiej pokrywa mapę braków, z zachowaniem różnorodności i ze
   sprawdzeniem na stracie kontrolnej.

W języku sterowania: stan to latent i pęd optymalizatora, wejście to skład paczki, cel to mapa braków. Koopman z
wejściem jest liniową postacią tego samego modelu.

### 5.2 Wzory

Oznaczenia: `θ` wagi, `x_p` klatka kontrolna, `z_p` jej latent, `ℓ_p` jej strata detekcji, `L_C` średnia strata
zbioru kontrolnego, `B` kandydująca paczka, `Δθ_B` część kroku AdamW pochodząca z gradientu paczki `B` przy obecnym
stanie optymalizatora (funkcja `czesci_kroku` w `maly_vit_petla_sd_rownanie/kod/rownanie.py`).

    siła paczki:                 S_p(B) = z(θ + Δθ_B; x_p) − z(θ; x_p)
    kierunek potrzebny głowie:   g_p = ∂ℓ_p / ∂z_p            (jedno przejście wstecz przez samą głowę)
    brak w zasięgu:              d_p = argmin_δ ℓ_p(z_p + δ),  |δ_c| ≤ r · sd_c

Trzy oceny paczki, od najtańszej do wzorcowej:

    ocena przez latent (pierwszy rząd):   A1(B) = −(1/P) Σ_p  g_p · S_p(B)
    ocena „nauczyciela”:                  A2(B) =  (1/P) Σ_p [ 2 d_p · S_p(B) − |S_p(B)|² ]
    użyteczność zmierzona wprost:         u1(B) = L_C(θ) − L_C(θ + Δθ_B)

- `A1` mówi, o ile spadnie strata kontrolna dzięki samemu ruchowi latentu. Dla małego `r` brak `d_p` jest
  proporcjonalny do `−g_p`, więc `A1` i `A2` się pokrywają.
- `A2` ma postać reguły wszechwiedzącego nauczyciela z [Liu17]: nauczyciel wybiera przykład, który najbardziej
  zmniejsza odległość ucznia od celu, czyli waży „użyteczność” (zgodność kroku z kierunkiem do celu) przeciw
  „trudności” (wielkości kroku). U nas odległość liczona jest w latencie, a celem jest latent w zasięgu. W wersji, w
  której nauczyciel sam tworzy przykłady [Qiu22b], przykłady tworzone swobodnie wychodzą poza rozkład danych, a
  tworzone blisko danych uczą słabiej; to ta sama granica co w rozdziale 2.
- `u1` obejmuje wszystkie drogi wpływu, także te, które omijają latent (wagi samej głowy, gałąź resztkowa). Kosztuje
  tyle samo co `S`, bo to ten sam wirtualny krok. To znana konstrukcja: krok na próbę oceniany na zbiorze walidacyjnym
  [Ren18] i wpływ przykładu śledzony po krokach uczenia [Pru20]. Jest wzorcem; `A1` i `A2` mają sens tylko, jeśli
  przewidują `u1` i skutek po wielu krokach nie gorzej, a dają coś więcej: rozkład na kanały tożsamości i resztę oraz
  celowanie w konkretne braki.
- Tańsze przybliżenie skutku dołożenia przykładu przez linearyzację sieci [Moh22b] jest ponad 100 razy szybsze niż
  uczenie na próbę, ale zakłada szerokie sieci i wymaga okresowego odświeżania. U nas tanie przybliżenia bez
  przeliczania całej sieci już zawiodły (docs/23: ostatni blok niesie 7,5 procent ruchu latentu).
- Do sprawdzenia w tym samym pomiarze: pchnięcie siły tożsamości wzdłuż kierunku LDA (obecny cel metody), sama strata
  paczki, RHO-loss [Min22] i losowanie.

Kod dla `u1` i pierwszego rzędu już istnieje (`Kontrola`, `ocen_grupe`, `zysk_bezposredni` w `rownanie.py`; ramiona
`sd_strata` i `kotwice_strata` w `petla4.py`). Kampanii z tymi ramionami nie było.

**Nowa definicja kierunku tożsamości do sprawdzenia.** Zamiast stałego kierunku LDA: kierunek, wzdłuż którego strata
głowy spada najszybciej, ograniczony do tokenów narzędzia i 128 kanałów, przeliczany w każdym obrocie. Zachowuje
pojęcie „wektora tożsamości”, ale wiąże je z bieżącą głową. Powód: rozdział 2.3.

### 5.3 Co prognozować na 10 kroków: skutek wybranej paczki

Stan faktyczny z pomiarów projektu:

| co przewidujemy | co znamy w chwili prognozy | umiejętność | źródło |
|---|---|---:|---|
| cały ruch latentu, 1 krok | historię i stan AdamW | 0,68 | równanie z pędem, 3 października |
| cały ruch latentu, 10 kroków | historię i stan AdamW | 0,22 | jak wyżej |
| cały ruch latentu, 10 kroków | prawdziwą pełną siłę każdego kroku (wagi z każdego kroku) | 0,985 | jak wyżej |
| siłę tożsamości, 1-2 kroki | dodatkowo paczki tych kroków | 0,98 | horyzont równania kroku, 30 września |
| siłę tożsamości, 5 kroków | paczki tych kroków, siły liczone przy wagach z chwili startu | 0,06 | jak wyżej |

Cały ruch na 10 kroków jest zdominowany przez paczki, których jeszcze nie znamy, a ich siły zależą od wag, które też
dopiero powstaną. Tej części nie da się przewidzieć bez wykonania uczenia.

Do doboru potrzebna jest inna wielkość: **różnica**, jaką robi nasza decyzja. Dwa przebiegi z tego samego stanu, jeden
z paczką `A`, drugi z paczką odniesienia `B0`, dalej te same paczki. Przyszłe paczki działają na oba przebiegi prawie
tak samo, więc w różnicy się znoszą:

    D_p(h)  = z_p po h krokach z paczką A − z_p po h krokach z paczką B0
    prognoza: D̂_p(h) = R(h) · [ S_p(A) − S_p(B0) ]

`R(h)` to funkcja odpowiedzi: ile z jednorazowego pchnięcia zostaje po `h` krokach. Z równania z docs/23 (bezwładność
0,83, siła 0,98) wynika `R(h) = 0,98 · (1 − 0,83^h) / (1 − 0,83)`, czyli około 4,9 dla 10 kroków: paczka działa dalej
przez pęd optymalizatora i jej łączny skutek narasta do około pięciokrotności pierwszego kroku. To wniosek z
dopasowania do zwykłego uczenia, nie z interwencji. Dokładniejsze dopasowanie z 3 października pokazuje odbicie po 3-5
krokach, więc prawdziwą `R(h)` trzeba zmierzyć (P1).

Co mówi literatura o tej wielkości:

- [Liu26p] (preprint, sierpień 2026): zaburzenie jednej paczki w AdamW wraca przez pęd ze szczytem po 7,5-9 krokach.
  Model liniowy zachowuje kolejność kandydatów nawet przy pełnej skali zaburzenia (korelacja rang 0,76-0,89 w
  horyzoncie 8-12 kroków), choć wielkości przewiduje słabo. Wskaźniki jednego kroku nie przewidują wielkości
  opóźnionego skutku. Od 70 do 92 procent zmienności skutku zależy od samego kandydata, a nie od wylosowanej
  przyszłości.
- [Guo26] (preprint, sierpień 2026): uwzględnienie 8 kroków transportu przez stan AdamW zmienia wybór w 60 z 96 okien
  i daje niższą stratę kontrolną w 10 z 12 historii. Wymaga jednak znajomości przyszłych paczek i kosztuje około
  10,6 raza więcej niż zwykłe uczenie przy pełnym przeglądzie kandydatów.
- [Den26b]: prognozy pierwszego rzędu psują się z długością horyzontu i ze współczynnikiem uczenia. Najlepszy horyzont
  to 5 kroków przy współczynniku 1e-2, 10 przy 1e-3, 25 przy 1e-4. Traktowanie AdamW jak zwykłego SGD daje korelację
  z prawdą 0,02-0,12, więc krok trzeba liczyć dokładnie tak jak optymalizator. Nasz współczynnik 5e-4 leży między
  drugim i trzecim przypadkiem (to moje przełożenie, nie wynik tej pracy).
- [Guu23] Simfluence: strata przykładu testowego jako prosty układ liniowy napędzany kolejnymi przykładami uczącymi;
  korelacja rang z prawdą około 0,89, dwa razy lepiej niż metody czysto addytywne. To najbliższy opublikowany
  odpowiednik „równania z danymi jako wejściem”, ale dla straty, nie dla latentu.

**Hipoteza do pomiaru P1:** umiejętność prognozy `D̂_p(10)` wobec prognozy „paczka nic nie zmienia” przekracza 0,85 dla
rzutu na kierunek potrzebny głowie. To jest ten sam próg 85 procent, którego szuka użytkownik, ale postawiony na
części sterowalnej. Na jednym kroku ta wielkość powinna wynosić około 0,98, bo część pędu jest w obu przebiegach ta
sama.

**Reguła o przyszłych klatkach.** Prognoza `D̂` używa tylko stanu z chwili `t` i kandydatów ocenianych w chwili `t`,
tak jak w docs/25. Wspólne przyszłe paczki służą wyłącznie do pomiaru prawdy. Gdyby do prognozy miał wejść plan
przyszłych paczek (tak robi [Guo26]), byłby to trzeci tryb obok „wyjaśnienia” i „prognozy”, wymagający osobnej decyzji
użytkownika i osobnej nazwy.

### 5.4 Dwie skale czasu

- **Skala kroków (1-20).** Odpowiedź na jedno pchnięcie. Służy do rankingu kandydujących paczek.
- **Skala obrotu (500 kroków).** Dołożenie 134 klatek zmienia skład zbioru, czyli średnią siłę wszystkich następnych
  paczek. Latent krąży wokół wolno płynącego poziomu i wraca do niego z połową odchylenia w 13-23 kroki (pomiar z
  2 października). Trwała zmiana średniej siły przesuwa ten poziom. W telemetrii 4.3 jednokrokowe pchnięcie zapowiada
  zmianę w obrocie słabo (0,21), więc w tej skali potrzebny jest zmierzony współczynnik przełożenia (pomiar P2), a nie
  samo pchnięcie.
- Uczenie AdamW nie jest układem w równowadze, więc związku między wahaniami a odpowiedzią nie wolno zakładać; obie
  odpowiedzi trzeba zmierzyć interwencją.

### 5.5 Rola Koopmana

- **Model liniowy z wejściem** [Pro14, Kor16]: `s(t+1) = A·s(t) + B·u(t)`, gdzie `s` to współrzędne tożsamości
  (dotąd siła LDA, w propozycji rzut na kierunek potrzebny głowie), a `u` to siła paczki w tych współrzędnych. Równanie
  Langevina z jądrem pamięci jest tym samym modelem zapisanym na opóźnieniach. Ten model pozwala planować kilka
  dołożeń naprzód i poprawiać plan po każdym pomiarze (sterowanie predykcyjne). Plan ułożony z góry, bez poprawiania
  po pomiarach, kumuluje błędy; to ograniczenie podejścia z [Les18b] przy głębokich sieciach.
- **Postać biliniowa** [Bru20, Iac22b]: wzmocnienie wejścia zależy od stanu. To formalny zapis naszej obserwacji, że
  siła tej samej paczki zależy od stanu sieci i dlatego składanie sił przy zamrożonych wagach psuje się po kilku
  krokach.
- **Osiągalność** [Yeu17c, Liu17]: które kierunki latentu dostępne dane w ogóle potrafią pchnąć. Praktycznie: siły
  `S(B)` dla kilkuset kandydujących grup tworzą macierz; brak `d` rozkładamy na część leżącą w jej zasięgu i resztę.
  Teoria machine teaching mówi to samo: nauczyciel wybierający z puli jest tak szybki jak nauczyciel tworzący
  przykłady tylko wtedy, gdy pula pokrywa potrzebny kierunek.
- **Prognoza siły w skali obrotu** zostaje tym, czym jest: Langevin z jądrem przewiduje około 29 procent zmiany na 500
  kroków. Nowością byłby człon sterowany: przewidziana zmiana = dryf własny + przełożenie razy średnia siła dołożonego
  zestawu.

### 5.6 Gdzie wchodzi Stable Diffusion

- **Kandydaci dla braków poza zasięgiem puli.** Jeśli znaczna część braku leży poza zasięgiem sił prawdziwych klatek,
  generator jest uzasadniony właśnie tam. Etykieta pochodzi z maski. Precedensy w tej samej dziedzinie: rzadkie układy
  w operacjach zaćmy [Fri23b], generowanie tego, czego uczony model jest niepewny [Fri25].
- **Ta sama ocena dla klatek prawdziwych i narysowanych.** Kandydat z SD wchodzi do zestawu tylko, jeśli jego
  przewidziana użyteczność jest wyższa niż klatek prawdziwych.
- **Prowadzenie latentem w zasięgu jako pokrętło trudności i jako filtr poprawności** (Z1, Z3): narysowany obraz musi
  dać ramkę zgodną z maską, a jego trudność ustawia `λ`.
- **Później, opcjonalnie: prowadzenie użytecznością.** Energia prowadzenia równa zgodności gradientu obrazu z
  gradientem kontrolnym [Kim26i], w duchu dopasowania gradientów w przestrzeni generatora [Zha20, Caz23]. Drogie, bo
  wymaga różniczkowania przez wirtualny krok.

### 5.7 Bezpieczniki

- **Różnorodność i część losowa** (25-50 procent zestawu). Losowanie z przetasowaniem i dopasowanym harmonogramem to
  mocny punkt odniesienia dla czasu do jakości [Oka23]; lepsze dopasowanie gradientu nie dało lepszego uczenia [Bal23b].
- **Krótki horyzont bywa krótkowzroczny.** Wybory oceniane kilka kroków naprzód systematycznie faworyzują szybki, płytki
  zysk; pokazano to dla harmonogramu współczynnika uczenia [Wu18c]. Przeniesienie na dobór paczek to moje wnioskowanie.
  Dlatego ranking trzeba sprawdzać także po 50 i 500 krokach.
- **Kalibracja i powrót do losowania**, gdy przewidziany zysk nie przekracza błędu prognozy (T6 z docs/25).
- **Szum etykiet.** Wysoka strata to nie zawsze przykład do nauczenia; model odniesienia jak w RHO-loss [Min22, Hu21].
- **Koszt.** Ocena kandydatów potrafi potroić czas iteracji [Sun25]. Krótka lista z taniego wyszukiwania, drogi test
  tylko dla finalistów, zegar rzeczywisty dla każdego etapu (AU59).

## 6. Plan pomiarów (PROPOZYCJA, nic nie uruchomiono poza P0)

| etap | co mierzy | koszt | kryterium ustalone przed wynikiem |
|---|---|---|---|
| **P0** (zrobione) | odległość celów od prawdziwych latentów; ruch obrazu SD; siła tożsamości wobec mAP | sekundy, same zapisane pliki | brak; wyniki w rozdziale 2 |
| **P1** | odpowiedź latentu i straty kontrolnej na jedną paczkę: pary przebiegów z zapisanego pełnego stanu (wagi i AdamW) w krokach 1000 i 2000, 32 kandydujące grupy, horyzonty 1, 2, 5, 10, 20, te same dalsze paczki | lokalnie, szacunek 1-2 godziny na RTX 3070 | umiejętność prognozy różnicy na 10 kroków ≥ 0,85 dla rzutu na kierunek potrzebny głowie; korelacja rang przewidzianego i zmierzonego spadku straty kontrolnej ≥ 0,5; porównanie ocen A1, A2, u1, pchnięcia LDA, straty paczki i losowania |
| **P2** | odpowiedź w skali obrotu: z jednego stanu 8 różnych zestawów po 134 klatki (losowe, wyszukane, o najwyższej i o najniższej przewidzianej użyteczności), 500 kroków, 2-3 ziarna | lokalnie albo mały przydział Edenu | zestaw o najniższej przewidzianej użyteczności wypada gorzej od zestawu o najwyższej w co najmniej 80 procentach par; znak przewidzianej zmiany straty zgodny ze zmierzonym |
| **P3** | zasięg puli: siły 256 kandydujących grup wobec mapy braków | jak P1 | raport: jaka część braku leży w zasięgu puli; decyzja, czy generator jest potrzebny |
| **P4** | wyrocznia z ograniczonym zasięgiem (`r` = 0,5; 1; 2; wariant po gałęzi resztkowej) na punkcie kontrolnym 1000; potem ramię z celem pośrednim jak w LabelEnc | lokalnie; ramię uczące wymaga planu | ile AP daje cel w zasięgu; czy ramię z celem pośrednim skraca czas do progu mAP |
| **P5** | SD z celem w zasięgu dla `λ` = 0,25; 0,5; 1: jaka część drogi jest pokonana; obrazy jako dane i jako zapytania; kontrole z Z4 | lokalnie, potem Eden | pokrycie celu ≥ 80 procent; przewaga nad SD bez prowadzenia i nad wyszukaniem z prawdziwej kotwicy |
| **P6** | pętla zamknięta: czas do progu mAP z poprawnymi zegarami (E0, E2, E3 audytu) | Eden, 5 potem 10 ziaren | progi z audytu: skrócenie mediany czasu o co najmniej 15 procent przy utracie AP50:95 nie większej niż 0,005 |

P1 i P3 korzystają z istniejącego kodu (`rownanie.py`). Jeśli P1 nie przejdzie, równanie nie nadaje się do rankingu na
10 kroków i zostaje ranking jednokrokowy z testem bezpośrednim. Jeśli P2 nie przejdzie, cały dobór według
przewidzianej użyteczności nie ma podstaw i właściwym wyborem jest różnorodne losowanie.

## 7. Co można napisać w publikacji

| twierdzenie | stan |
|---|---|
| Jeden krok ruchu latentu w AdamW jest przewidywalny w 95-98 procentach przy znanej paczce; współczynniki przenoszą się między ziarnami | zmierzone (docs/23, kampania 4.3) |
| Cały ruch na 10 kroków ma sufit około 0,22 bez wiedzy o paczkach; z pełną siłą każdego kroku 0,985 | zmierzone (2-3 października) |
| Latent optymalny dla głowy, liczony bez ograniczeń, leży o rząd wielkości poza naturalnym zakresem, więc wynik wyroczni jest miarą możliwości głowy, a nie celem dla obrazu | zmierzone w P0 na jednym punkcie kontrolnym |
| Rzut na stały kierunek tożsamości nie śledzi jakości detekcji w czasie uczenia | zmierzone w P0 (korelacyjnie, 10 ziaren) |
| Sterowalna część ruchu latentu jest przewidywalna na 10 kroków | hipoteza, rozstrzyga P1 |
| Dobór według przewidzianej użyteczności skraca czas do zadanej jakości wraz z kosztem doboru | hipoteza, rozstrzygają P2 i P6 |
| Generator pomaga tam, gdzie siły puli nie pokrywają braku | hipoteza, rozstrzygają P3 i P5 |

Nowość. Wyszukiwania nie znalazły pracy, która łączy cele latentu wyprowadzone z głowy detektora, zmierzoną odpowiedź
reprezentacji sond na paczkę w AdamW, dobór paczek i generator dla braków poza zasięgiem puli. Najbliżej są [Guo26] i
[Liu26p] (małe modele, strata zamiast reprezentacji, bez generatora), [Kim26i] (dyfuzja sterowana wpływem, klasyfikacja
medyczna), [Fri25] (niepewność, segmentacja chirurgiczna) i [Hao20] (cechy z etykiet). Ocena opiera się na
streszczeniach, nie na pełnym przeglądzie systematycznym.

Czego nie pisać: że obraz, przy którym detektor jest najlepszy, jest najlepszym obrazem uczącym; że siła tożsamości
napędza uczenie; że latent przewidujemy na 10 kroków w 85 procentach bez dopowiedzenia, że chodzi o skutek wybranej
paczki.

## 8. Czego ten przegląd nie dowodzi

- Obliczenia z 2.1 i 2.2 dotyczą jednego punktu kontrolnego, 400 obrazów, 128 kanałów i jednego obrazu SD.
- Związki z 2.3 to korelacje w jednej kampanii; kierunek tożsamości był stały, a nie odświeżany.
- Prace z 2026 roku ([Liu26p], [Guo26], [Den26b], [Kim26i], [Kar26], [Xia26c]) to preprinty na małych modelach; żadna
  nie dotyczy detektora ani wideo chirurgicznego.
- Wnioski o tym, czego w literaturze brakuje, wynikają z rankingów i streszczeń Undermind.
- Funkcja odpowiedzi `R(h)` i przełożenie w skali obrotu są hipotezami do zmierzenia.
- Nie sprawdzono, czy cel w zasięgu (Z1) daje jeszcze istotny zysk AP na punkcie kontrolnym 1000.

## 9. Literatura

Odczyt: P = pełny tekst przeczytany przez podagenta Undermind, S = streszczenie i metadane.

### Cele latentu z etykiet i cele warstw ukrytych

| klucz | praca | odczyt | odnośnik |
|---|---|---|---|
| Hao20 | Hao i in., LabelEnc: A New Intermediate Supervision Method for Object Detection, ECCV 2020 | P | https://doi.org/10.1007/978-3-030-58595-2_32 |
| Zha21p | Zhang i in., LGD: Label-guided Self-distillation for Object Detection, AAAI 2022 | P | https://doi.org/10.1609/aaai.v36i3.20240 |
| Kim24k | Kim i in., LabelDistill: Label-guided Cross-modal Knowledge Distillation for Camera-based 3D Object Detection, ECCV 2024 | P | https://doi.org/10.48550/arXiv.2407.10164 |
| Lee14 | Lee i in., Difference Target Propagation, ECML PKDD 2015 | S | https://doi.org/10.1007/978-3-319-23528-8_31 |
| Meu20 | Meulemans i in., A Theoretical Framework for Target Propagation, NeurIPS 2020 | P | https://doi.org/10.5167/UZH-198834 |
| Lop15 | Lopez-Paz i in., Unifying distillation and privileged information, 2015 | S | https://arxiv.org/abs/1511.03643 |
| Sal20e | Salman i in., Unadversarial Examples: Designing Objects for Robust Vision, NeurIPS 2021 | P | https://arxiv.org/abs/2012.12235 |
| Cao21b | Cao, Johnson, Inverting and Understanding Object Detectors, 2021 | P | https://arxiv.org/abs/2106.13933 |
| Sab15 | Sabour i in., Adversarial Manipulation of Deep Representations, 2015 | S | https://arxiv.org/abs/1511.05122 |

### Obraz z zadanych cech

| klucz | praca | odczyt | odnośnik |
|---|---|---|---|
| Shi25f | Shirahama i in., Feature Space Analysis by Guided Diffusion Model, 2025 | P | https://doi.org/10.48550/arXiv.2509.07936 |
| Neu25 | Neukirch i in., FeatInv: Spatially resolved mapping from feature space to input space using conditional diffusion models, 2025 | S | https://doi.org/10.48550/arXiv.2505.21032 |
| Bor21 | Bordes i in., High Fidelity Visualization of What Your Self-Supervised Representation Knows About, 2021 | S | https://arxiv.org/abs/2112.09164 |
| Wal23b | Wallace i in., End-to-End Diffusion Latent Optimization Improves Classifier Guidance, ICCV 2023 | S | https://doi.org/10.1109/ICCV51070.2023.00669 |
| Aug23 | Augustin i in., DiG-IN: Diffusion Guidance for Investigating Networks, CVPR 2024 | S | https://doi.org/10.1109/CVPR52733.2024.01055 |
| Yu23 | Yu i in., FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model, ICCV 2023 | S | https://doi.org/10.1109/ICCV51070.2023.02118 |

### Generowanie danych pod kontrolą uczonego modelu

| klucz | praca | odczyt | odnośnik |
|---|---|---|---|
| Hem23 | Askari Hemmat i in., Feedback-guided Data Synthesis for Imbalanced Classification, 2023 | P | https://doi.org/10.48550/arXiv.2310.00158 |
| Hem25 | Askari Hemmat i in., Improving the Scaling Laws of Synthetic Data with Deliberate Practice, 2025 | P | https://doi.org/10.48550/arXiv.2502.15588 |
| Zha22i | Zhang i in., Expanding Small-Scale Datasets with Guided Imagination, NeurIPS 2023 | P | https://doi.org/10.48550/arXiv.2211.13976 |
| Wan24m | Wang i in., Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data, AAAI | P | https://doi.org/10.1609/aaai.v40i12.38007 |
| Hay25 | Hayden i in., Generative Data Mining with Longtail-Guided Diffusion, 2025 | P | https://doi.org/10.48550/arXiv.2502.01980 |
| Fri25 | Frisch i in., GAUDA: Generative Adaptive Uncertainty-Guided Diffusion-Based Augmentation for Surgical Segmentation, WACV 2025 | P | https://doi.org/10.1109/WACV61041.2025.00370 |
| Ven25e | Venkatesh, Speidel, Towards Application Aligned Synthetic Surgical Image Synthesis, 2025 | P | https://doi.org/10.48550/arXiv.2509.18796 |
| Fri23b | Frisch i in., Synthesising Rare Cataract Surgery Samples with Guided Diffusion Models, MICCAI 2023 | S | https://doi.org/10.48550/arXiv.2308.02587 |
| Kim26i | Kim i in., Steering Diffusion Models via Class-Contrastive Influence for Few-Shot Medical Classification, 2026 | P | https://doi.org/10.48550/arXiv.2607.12464 |
| Guo25c | Guo i in., UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation, 2025 | P | https://doi.org/10.48550/arXiv.2510.24262 |
| Kar26 | Karthikeyan i in., Representation-Conditioned Diffusion Models for Guided Training Data Generation, 2026 | P | https://doi.org/10.48550/arXiv.2605.27495 |
| Xia26c | Xiang i in., Learning-State-Aware Dynamic Generative Data Augmentation on Small-Scale Datasets, 2026 | P | https://arxiv.org/abs/2608.18907 |
| Lia24e | Liang i in., Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion, ICCV 2025 | P | https://doi.org/10.1109/ICCV51701.2025.00166 |
| Yin19 | Yin i in., Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion, CVPR 2020 | S | https://doi.org/10.1109/cvpr42600.2020.00874 |
| Cha21d | Chawla i in., Data-free Knowledge Distillation for Object Detection, WACV 2021 | S | https://doi.org/10.1109/WACV48630.2021.00333 |
| Zha20 | Zhao i in., Dataset Condensation with Gradient Matching, 2020 | S | https://arxiv.org/abs/2006.05929 |
| Caz23 | Cazenavette i in., Generalizing Dataset Distillation via Deep Generative Prior, CVPR 2023 | S | https://doi.org/10.1109/CVPR52729.2023.00364 |
| Gen24 | Geng i in., The Unmet Promise of Synthetic Training Images: Using Retrieved Real Images Performs Better, 2024 | P (audyt) | https://doi.org/10.48550/arXiv.2406.05184 |

### Przykłady łatwe i trudne

| klucz | praca | odczyt | odnośnik |
|---|---|---|---|
| Sor22 | Sorscher i in., Beyond neural scaling laws: beating power law scaling via data pruning, NeurIPS 2022 | P | https://doi.org/10.48550/arXiv.2206.14486 |
| Guo23f | Guo i in., Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching, 2023 | P | https://doi.org/10.48550/arXiv.2310.05773 |
| Cao19 | Cao i in., Prime Sample Attention in Object Detection, CVPR 2020 | P | https://doi.org/10.1109/cvpr42600.2020.01160 |
| Min22 | Mindermann i in., Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt, ICML 2022 | P (audyt) | https://doi.org/10.48550/arXiv.2206.07137 |
| Hu21 | Hu i in., When does loss-based prioritization fail?, 2021 | S | https://arxiv.org/abs/2107.07741 |
| Oka23 | Okanovic i in., Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning, 2023 | P (audyt) | https://doi.org/10.48550/arXiv.2305.18424 |
| Bal23b | Balles i in., A Negative Result on Gradient Matching for Selective Backprop, 2023 | P (audyt) | https://doi.org/10.48550/arXiv.2312.05021 |
| Sun25 | Sun i in., Online Data Curation for Object Detection via Marginal Contributions to Dataset-level Average Precision, 2025 | P | https://doi.org/10.48550/arXiv.2511.14197 |

### Nauczanie jako sterowanie i skutek paczki w czasie

| klucz | praca | odczyt | odnośnik |
|---|---|---|---|
| Liu17 | Liu i in., Iterative Machine Teaching, ICML 2017 | P | https://arxiv.org/abs/1705.10470 |
| Qiu22b | Qiu i in., Iterative Teaching by Data Hallucination, 2022 | P | https://doi.org/10.48550/arXiv.2210.17467 |
| Les18b | Lessard i in., An Optimal Control Approach to Sequential Machine Teaching, AISTATS 2019 | P | https://arxiv.org/abs/1810.06175 |
| Moh22b | Mohamadi i in., Making Look-Ahead Active Learning Strategies Feasible with Neural Tangent Kernels, 2022 | P | https://doi.org/10.48550/arXiv.2206.12569 |
| Liu26p | Liu, Li, Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW, 2026 | P | https://arxiv.org/abs/2608.19762 |
| Guo26 | Guo, Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions, 2026 | P | https://arxiv.org/abs/2608.24593 |
| Den26b | Deng i in., How Faithful Is Trajectory-Based Data Attribution? Error Sources, Remedies, and Practical Guidelines, 2026 | P | https://doi.org/10.48550/arXiv.2605.18814 |
| Guu23 | Guu i in., Simfluence: Modeling the Influence of Individual Training Examples by Simulating Training Runs, 2023 | P | https://doi.org/10.48550/arXiv.2303.08114 |
| Wu18c | Wu i in., Understanding Short-Horizon Bias in Stochastic Meta-Optimization, ICLR 2018 | P | https://arxiv.org/abs/1803.02021 |
| Ren18 | Ren i in., Learning to Reweight Examples for Robust Deep Learning, ICML 2018 | S | https://arxiv.org/abs/1803.09050 |
| Pru20 | Pruthi i in., Estimating Training Data Influence by Tracking Gradient Descent, 2020 | S | https://arxiv.org/abs/2002.08484 |

### Koopman i układy z wejściem

| klucz | praca | odczyt | odnośnik |
|---|---|---|---|
| Pro14 | Proctor i in., Dynamic Mode Decomposition with Control, SIAM J. Appl. Dyn. Syst. 2016 | S | https://doi.org/10.1137/15M1013857 |
| Kor16 | Korda, Mezić, Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control, Automatica 2018 | S | https://doi.org/10.1016/j.automatica.2018.03.046 |
| Bru20 | Bruder i in., Advantages of Bilinear Koopman Realizations for the Modeling and Control of Systems With Unknown Dynamics, IEEE RA-L 2021 | S | https://doi.org/10.1109/LRA.2021.3068117 |
| Iac22b | Iacob i in., Koopman Form of Nonlinear Systems with Inputs, 2022 | S | https://doi.org/10.48550/arXiv.2207.12132 |
| Yeu17c | Yeung i in., A Koopman Operator Approach for Computing and Balancing Gramians for Discrete Time Nonlinear Systems, ACC 2018 | S | https://doi.org/10.23919/ACC.2018.8431738 |

## 10. Pochodzenie

- **Nowe wyszukiwania pogłębione Undermind** (workspace „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”, 2026-10-05):
  - [Label-informed optimal internal activations as training targets for a detector](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Label-informed%20optimal%20internal%20activations%20as%20training%20targets%20for%20a%20detector) (99 prac)
  - [Learner-guided synthetic training images easy versus hard and effect on training speed](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Learner-guided%20synthetic%20training%20images%20easy%20versus%20hard%20and%20effect%20on%20training%20speed) (191 prac)
  - [Counterfactual effect of one training batch over the next optimizer steps and its use for online batch selection](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Counterfactual%20effect%20of%20one%20training%20batch%20over%20the%20next%20optimizer%20steps%20and%20its%20use%20for%20online%20batch%20selection) (118 prac)
- **Wcześniejsze wyszukiwania użyte bez powtarzania**: „Accelerating detector training through useful data selection and
  reachable latent targets” (audyt), „Stable Diffusion generating images whose representation in an external frozen
  network matches a target latent”, „latent particle of a network in training as a controlled state-space system”,
  „linear descriptions of a nonlinear system with inputs and linearized network training dynamics”, „Joint
  identification of controlled Koopman dynamics with state and input memory”, „Koopman prediction of neural network
  weights during training and fixed probe representations”, „when does data selection accelerate training early phase
  and model size”.
- **Scite**: 20 prac z numerem DOI sprawdzonych pod kątem not redakcyjnych (wycofania, korekty); brak not. Z wyniku
  odczytano tylko tytuły, lata i pole not.
- **Obliczenia**: `Research/syntheses/materialy/vit-idealny-latent-sd-langevin-2026-10-05/osiagalnosc_celu.py` →
  `OSIAGALNOSC.json`; `telemetria43_tozsamosc_a_map.py` → `TELEMETRIA43_TOZSAMOSC.json`. Ziarno losowania przedziałów
  20261005, 2000 powtórzeń po ziarnach.
- Linków nie dopisano do kolejki PdfDownloadera.
