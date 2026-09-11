# Recenzje a audyt i plan eksperymentów RT-DETR

> Archiwum audytu sprzed refaktoru kodu. Aktualne poprawki i testy: [21_STAN_REFAKTORU.md](../21_STAN_REFAKTORU.md). Numery linii i wyniki reprodukcji odnoszą się do zapisanej wersji źródeł.

Data: 2026-09-11. Dokument uzupełnia [RAPORT.md](RAPORT.md) i [PROTOKOL_RT_DETR.md](PROTOKOL_RT_DETR.md). Jest analizą uwag i specyfikacją pracy, **nie odpowiedzią deklarującą wykonanie nowych treningów**.

## 1. Źródła i wniosek

Przeczytano pełny wyodrębniony tekst dwóch PDF-ów oraz cały `RESPONSE_TO_REVIEWERS_DETAILED.md` z folderu `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\odpowiedziRecenzentów`.

- **R1** oznacza tutaj `Comments_to_the_Author.pdf`: 6 stron, 10 uwag głównych i 7 drobnych. Rekomendacja w dokumencie: Reject (do not encourage resubmit).
- **R2** oznacza `Reviewer_Comments_to_Authors.pdf`: 5 stron, 10 uwag głównych i 3 nienumerowane uwagi końcowe. Rekomendacja: Reject.
- R1/R2 to lokalne etykiety tej mapy; pliki nie potwierdzają oficjalnych numerów recenzentów w systemie redakcji. R2-m1–m3 poniżej to nadana tutaj numeracja trzech akapitów końcowych.
- Teksty i hashe źródeł: [evidence/reviewers](../pochodne/evidence/reviewers/source_hashes.json). Wizualnie sprawdzono stronę 3 R2: brak liczby w pierwszym zdaniu punktu 6 występuje również w PDF; dalszy tekst podaje próg 0,3.

**Obie recenzje podważają identyfikację przyczyny poprawy, a nie sam sens selekcji danych.** Potrzeba podziału po operacjach/pacjentach, porównywalnych treningów, niezależnych powtórzeń, standardowej ewaluacji i jawnej historii użycia testu. Audyt znalazł dodatkowo błędy nowego pipeline'u, które trzeba usunąć przed powtórzeniem eksperymentów. Nie ustalono, czy historyczny evaluator artykułu zawierał te same błędy.

## 2. Projekt odpowiedzi nie jest potwierdzeniem wykonania badań

Plik odpowiedzi ma 10 połączonych punktów głównych i 7 drobnych. Nie zachowuje osobnych list obu recenzentów: jego punkt 10 dotyczy Query 81, podczas gdy R1-10 dotyczy izolacji efektu selekcji w obu architekturach, a R2-10 historii dostępu do zewnętrznego testu. W przyszłej odpowiedzi należy zachować źródłowe identyfikatory i odsyłać między powtarzającymi się punktami.

Poniższe deklaracje **nie mają potwierdzenia w materiale zweryfikowanym w tym audycie**. Nie jest to dowód, że nigdzie nie istnieją inne logi; nie przeprowadzono przeszukania wszystkich dysków ani zdalnego serwera treningowego. Przeszukano przekazany folder wersji Final i folder odpowiedzi oraz zestawiono treść z wcześniejszym audytem lokalnego kodu i manifestów.

| Deklaracja w projekcie odpowiedzi | Co trzeba sprawdzić lub poprawić |
|---|---|
| Punkt 1: wykonane perturbacje Dirichleta ±0,05 i korelacja r>0,94 | Brak wyników i definicji eksperymentu. Dirichlet sam z siebie nie ogranicza zmian do ±0,05. Trzeba podać rozkład/odrzucanie prób, seed, rodzaj korelacji. Stabilność rankingu nie dowodzi stabilności wybranego zbioru ani AP; R1 pyta przede wszystkim o wybór pięciu składowych. |
| Punkt 2: już stosowano video/surgery/patient split i brak próbek pacjentów testowych w treningu | Przekazany TEX opisuje random frame-level split. Bieżący manifest 35/8/7 nie dowodzi historii starych checkpointów. Jeden film na operację nie dowodzi sam w sobie unikalnego pacjenta; potrzebna udokumentowana mapa pacjent–operacja–film. |
| Punkty 3 i 7: wykonano nowoczesne benchmarki i pełną ablację YOLO | Potrzebne manifesty biegów, checkpointy, predykcje, wersje modeli i identyczny protokół porównywanych wierszy. Nie potwierdzono nowych badań. |
| Punkty 5–7: nowe średnie, SD i wyniki, m.in. 26,04±0,42; 4,12±0,38; 8,45±0,51; 1,15; 2,40; 18,30 | Nie kopiować do artykułu bez wyników jednostkowych i definicji seedów. Historyczne 26,04 lub 34,17 nie stają się średnią z powtórzeń przez dopisanie SD. |
| Punkt 5: rozmyte klatki dominują po usunięciu Fouriera; brak trudności powoduje pomijanie rzadkich kątów | To hipotezy mechanistyczne. Potrzebne pomiary rozmycia, składu wybranych danych, podgrup błędów i powtórzenia. Sama różnica AP ich nie potwierdza. |
| Punkt 8: odrzucono dokładnie 12 uszkodzonych/czarnych klatek na końcach trzech filmów | Różnica arytmetyczna wynosi 12, ale jej przyczyna nie została ustalona. Wymagany log odrzuceń i ścieżki źródeł; nie można dopisywać przyczyny z samej różnicy. Także opisy oświetlenia i materiałów narzędzi wymagają danych źródłowych. |
| Punkt 9: seedy 42,123,999 i paired t-test p<0,001 | Brak surowych par wyników, statystyki, liczby jednostek i założeń. Nie traktować skorelowanych klatek jako niezależnych obserwacji. Wynik p nie może być założeniem planu. |
| Punkt 10: wszystkie tabele już przeliczone na wszystkie 100 queries z bipartite matching | Nie potwierdzono przeliczenia. Hungarian służy do przypisania targetów w treningu; COCO AP wykorzystuje dopasowanie predykcji posortowanych po score dla kolejnych progów IoU. Liczba queries RT-DETR zależy od konfiguracji; nie narzucać 100 ze starego DETR. |
| Drobne 13: przeprowadzono PCA 16/32/64/128 i wykazano optimum 32 | Brak krzywej i wyników; R1 prosi także o mniejsze wymiary około 10–20. Wariancja PCA nie jest miarą jakości detektora. |

Sformułowania „irrefutable statistical proof”, „proving” i „benefit identically” są zbyt mocne także po wykonaniu badań. Warto raportować wielkość efektu w punktach procentowych, niepewność oraz zakres architektur, danych i budżetów, w którym został zmierzony.

## 3. Pełna mapa uwag R1

Status „eksperyment” oznacza wymagane nowe dowody, nie uruchomiony bieg. „Dokumentacja” może wymagać odtworzenia historii, a nie tylko redakcji zdania.

| ID | Przedmiot uwagi | Działanie i dowód zamknięcia |
|---|---|---|
| R1-1 | Sens pięciu składowych FastSAM | Eksperyment: każda składowa, prosty segment-count baseline i pełny score; uzasadnienie czego miernik nie mierzy. Sprawdzić union masek, skalę i normalizację. Korelacja rankingu plus overlap zbiorów i downstream AP na walidacji; osobno wpływ wag. |
| R1-2 | Mały test 589 klatek, nieznana walidacja, split po klatce | Nowy split przed selekcją i augmentacją po niezależnych operacjach/pacjentach. Tabela liczby grup i oryginalnych klatek, negatywów, adnotacji. Liczba klatek nie zastępuje liczby operacji. Powtórzyć główne treningi. |
| R1-3 | Aktualność YOLOv8L | Zachować go jako historyczny punkt odniesienia i uzasadnić wariant L; dodać współczesny YOLO ze wskazaną wersją i kosztem. W połączeniu z R2-7 samo uzasadnienie starego modelu nie zamyka całości problemu. |
| R1-4 | Niejednakowe metryki w tabelach | Wspólny COCO evaluator: AP50 i AP@[.50:.95] we wszystkich tabelach, zalecane AP75 i AR100; P/R/F1 z progiem wybranym na walidacji oraz TP/FP/FN pomocniczo. Zachować wszystkie ocenione obrazy, także bez detekcji. |
| R1-5 | Pełna analiza czterech grup cech | Wszystkie 15 niepustych kombinacji czterech bloków, plus random i pełna pula. Te same dane źródłowe, budżet, trening i evaluator. Leave-one-out nie identyfikuje wszystkich interakcji; reprezentację i skalowanie opisać jednoznacznie. |
| R1-6 | Jeden random i brak random z oryginałów | Losować unikalne oryginalne klatki wyłącznie z train, potem jednakowo augmentować. Powtórzyć losowanie i trening; rozdzielić ich seedy. Historyczny random z augmentowanej puli może być analizą pomocniczą. Budżet 20k musi być wykonalny po nowym podziale. |
| R1-7 | Brak ablacji YOLOv8 | Powtórzyć macierz doboru danych także dla YOLOv8, jeżeli utrzymujemy pierwotny zakres twierdzeń; użyć tych samych manifestów podzbiorów. Ablacja tylko dla nowego YOLO nie jest dosłowną odpowiedzią na ten punkt — wymaga jawnej zmiany zakresu pracy. |
| R1-8 | 91 336 vs 22 837×4; pochodzenie 2333 zewnętrznych obrazów | Odtworzyć tabelę source→augmentation→exclusion z manifestów/logów. Opisać zewnętrzne źródło, filmy/operacje, adnotacje i warunki pozyskania; dodać obrazy porównawcze. Wyjaśnić 48 nazw zgodnych z lokalną listą przecieków. |
| R1-9 | Jeden ośrodek, seed 42, niepewność | Powtórzenia treningów, SD między biegami, CI po niezależnych operacjach; ostrożne wnioski z jednego ośrodka. Dodatkowy niezależny zbiór wzmacnia dowody, ale jest osobnym przedsięwzięciem, nie istniejącym wynikiem. |
| R1-10 | Izolacja curation głównie w DETR | Dla każdej badanej architektury osobno porównać full-pool/random/curated z równym protokołem wewnątrz architektury. Efekt selekcji i porównanie recept architektur raportować osobno. |
| R1-11 | Ilustracja trudności chirurgicznych | Wybrać reprezentatywne przykłady glare, blur, occlusion, zmiennego światła; jawne kryteria doboru i split. Nie wybierać obrazów tylko według korzystnego wyniku własnego modelu. |
| R1-12 | Trzy fazy tekstu vs cztery bloki rysunku | Jednoznaczne przyporządkowanie faz do bloków (a)–(d), wejścia i wyjścia każdego etapu; zgodność z faktycznym kodem. |
| R1-13 | Mniejszy wymiar PCA | Walidacyjna krzywa np. 10/16/20/32/64, fit PCA tylko na train; retained variance, czas/pamięć, overlap selekcji i downstream AP. To propozycja siatki, nie pomiar. |
| R1-14 | Alternatywy reprezentatywnej selekcji | Omówić k-center, facility location, CSOD/OFDS; porównać co najmniej sensowny baseline obiektowy. K-center musi zachować wybrane greedy centra — obecny nowy kod zmienia je po fakcie. |
| R1-15 | Pełne liczebności splitów | Tabela train-source/train-selected/val/internal-test/external: grupy, źródła, obrazy po augmentacji, bboxy i negatywy. Nie mieszać liczebności historycznych z nowym podziałem. |
| R1-16 | Niejasne znaczenie wierszy ablacji | Każdy wiersz wskazuje manifest, architekturę/wagi, seed, split oceny, checkpoint/hash/epokę/EMA, budżet i protokół. Oznaczyć historyczne wyniki nieporównywalne z nową macierzą. |
| R1-17 | Zbyt silne wnioski kliniczne | Oddzielić względną poprawę generalizacji od użytecznej dokładności lokalizacji. AP@[.50:.95], AP75 i analiza błędów małych obiektów; błąd w mm wymaga kalibracji, a środek bbox nie jest automatycznie anotacją rzeczywistej końcówki. |

## 4. Pełna mapa uwag R2

| ID | Przedmiot uwagi | Działanie i dowód zamknięcia |
|---|---|---|
| R2-1 | Fundamentalny problem frame-level split | Jak R1-2: odtworzyć zbiory i powtórzyć główne biegi. Samo dopisanie ograniczenia lub zmiana nazwy splitu nie wystarcza. |
| R2-2 | Brak stabilności ogromnych różnic | Seedy selekcji, klastrowania i treningu, wyniki jednostkowe i mean±SD; wyjaśnić awarie treningu przed interpretacją synergii. Kryteria uznania biegu za awarię ustalić bez wyników testu. |
| R2-3 | Ręczne wagi, PCA, 20k, próg, centroid | Walidacyjne analizy wrażliwości osobno dla wag bloków, składowych FastSAM, trudności, wymiaru PCA i budżetu. Raportować faktyczny udział bloków w odległości. Próg AP i próg użytkowy rozdzielić. Selekcja centroidowa wymaga porównania, nie deklaracji optymalności. |
| R2-4 | Inny trening full-pipeline niż ablacji | Nowy full-pipeline trenować dokładnie w protokole macierzy. Nie używać starego 26,04 jako kontroli przy innym podziale/stopping. Hipotezy blur/easy-samples testować, nie przyjmować za wynik. |
| R2-5 | Query 81 i audyt 100 slotów na teście | Główny wynik: standardowy postprocessor modelu i COCOeval. Historyczny Q81 porównać na tym samym checkpointcie, oznaczyć wybór i chronologię. Nie przenosić tożsamości slotu między seedami/modelami. |
| R2-6 | Confidence, krzywe PR/F1, kalibracja i transfer progu | Zapisać predykcje z odpowiednio niskim cutoff. AP z rankingu; próg P/R/F1 wybrać na val, zamrozić na obu testach. Krzywe testowe tylko jako końcowa analiza opisowa. Reliability diagram i metryka kalibracji detekcji z jawnie ustalonym matchingiem/IoU/binami; kalibrator fit tylko na danych do tego przeznaczonych. |
| R2-7 | Stare modele i brak uzasadnienia skali L | Co najmniej współczesny YOLO i nowoczesny DETR-style; **RT-DETR/RT-DETRv2 bezpośrednio odpowiada na tę uwagę**. D-FINE sugerowany, nie nakazany. Recenzent nie wymaga testowania każdego modelu. Podać parametry, pretraining, rozdzielczość i koszt. |
| R2-8 | Różna rozdzielczość i zasoby | Ujednolicić protokół selekcji wewnątrz architektury. Między architekturami raportować komplet recept i koszt; dodać kontrolę rozdzielczości/budżetu dla mocnych twierdzeń porównawczych. Nie narzucać jednego LR wszystkim modelom; jednakowy budżet strojenia jest uczciwszy niż przypadkowo jednakowy parametr. |
| R2-9 | Wpływ NMS w YOLO | Podać confidence cutoff, NMS IoU, class-aware/agnostic, max_det, wersję kodu. Walidacyjna krzywa dla NMS IoU, opcjonalnie Soft-NMS z surowych kandydatów sprzed hard-NMS. Standardowego RT-DETR nie uzupełniać automatycznie NMS. |
| R2-10 | Wielokrotne oglądanie zewnętrznego testu | Chronologia decyzji o checkpointach, Q81, selektorze i progach. Benchmark wielokrotnie analizowany opisać uczciwie jako taki; nowy ledger nie przywraca niezależności historycznej. Do potwierdzenia nowej metody preferować odłożone niezależne operacje. |
| R2-m1 | Nowość połączenia znanych metod | Rozdzielić nową hipotezę/konfigurację od znanych Fourier, DINO, SAM, PCA, k-means i trudności. Porównać z literaturą selekcji obiektowej. Nie nazywać wyboru rzeczywistych klatek syntetyczną dataset distillation. |
| R2-m2 | Zależność od budżetu 20k | Krzywe jakości i całkowitego kosztu vs udział unikalnych źródeł, np. 10/25/50/75/100% train. Metody porównywać przy jednakowym K; jasno określić stałe epoki czy stałe kroki. |
| R2-m3 | 71,3% wybranych obrazów to augmentacje | Historycznie: udział typu augmentacji w puli i wyborze, enrichment i liczbę źródeł powtarzanych. Kontrola z takim samym rozkładem augmentacji. W nowym eksperymencie podstawowym wybierać źródła, augmentować identycznie po wyborze. |

## 5. Macierz pracy i kolejność

**Brama 0 — poprawność pomiaru.** Naprawić potwierdzone w audycie błędy COCOeval, threshold-on-test, class mapping i checkpointów. Smoke test rzeczywistego adaptera RT-DETR oraz predykcji pustych, wieloobiektowych i małych bboxów. Celem nie jest uzyskanie wysokiego AP, lecz zgodność z referencją i pełna identyfikowalność wyniku.

**Brama 1 — dane.** Zamrozić split grupowy oraz manifest źródeł. W aktualnej lokalnej anotacji train znaleziono 17 832 unikalne nazwy źródłowe przed augmentacją. Jeśli ten kontrakt pozostaje, losowanie 20 000 różnych źródeł z train jest niemożliwe. Ustalić wspólny K≤N_train, np. udziały puli; nie dołączać val/test, aby osiągnąć arbitralne 20k. To zmiana względem propozycji recenzenta opartej na starej liczebności 22 837, wymagająca jawnego wyjaśnienia.

**Brama 2 — dwa różne pytania badawcze.** Wierna kontrola starego DESF zachowuje bloki Fourier9/DINO-PCA32/SAM1/difficulty1, kolejność PCA i normalizacji oraz definicję trudności. Adaptacja RT-DETR z ROI i natywnym loss jest nową metodą do osobnego testu. Nie wolno w tej samej ablacji zmienić selektora, modelu, augmentacji i splitu, a wyniku przypisać usunięciu jednego bloku. Nowy poprawny split jest wspólny wszystkim wierszom nowego eksperymentu.

**Brama 3 — pilotaż.** Random-source, pełna pula i proponowana selekcja na RT-DETR, ze stałym protokołem. Weryfikacja małego overfit testu i uczenia baseline'u poprzedza interpretację załamania AP. Trzy biegi mogą ujawnić duży problem, ale nie są uniwersalną gwarancją mocy statystycznej.

**Brama 4 — zasadnicza ablacja.** Dla czterech bloków D/F/S/E pełna macierz ma 15 niepustych kombinacji; z random-source i full-pool daje 17 warunków. Dwa detektory i trzy powtórzenia oznaczają **102 treningi**, zanim doliczymy sensitivity i alternatywne selektory. To przykład kosztu, nie uruchomione zadanie ani obowiązkowa liczba seedów. Pełna macierz RT-DETR+nowy YOLO potwierdza nowy zakres architektur; jeśli bronimy dokładnie starej tezy DETR+YOLOv8, trzeba także odpowiedzieć na R1-7 i jasno uzasadnić zakres dodatkowych kontroli.

Nie traktować trzech treningów jednego random-subset jako trzech niezależnych losowań zbioru. Minimum dokumentacyjne to osobne `selection_seed`, `clustering_seed`, `train_seed`. Praktyczny projekt może parować wiersze po seedzie i wykonywać kilka niezależnych selekcji, ale przy splątaniu źródeł losowości raportuje się całkowitą zmienność eksperymentu, nie osobną estymację każdej składowej. Główną różnicę AP między curated a random zdefiniować przed testem; dodatkowe interakcje oznaczyć jako eksploracyjne lub uwzględnić wielokrotność porównań.

**Brama 5 — walidacja i ocena końcowa.** Wagi, PCA, checkpoint, próg użytkowy, kalibrator i NMS wybierać na walidacji przy jawnym budżecie strojenia. Gdy wiele decyzji korzysta z tego samego val, preferować wewnętrzne podziały grupowe/cross-validation; nie stroić na zewnętrznym benchmarku. Na końcu wykonać zarejestrowaną ocenę, z CI po operacjach i SD po treningach raportowanymi osobno. Przy zaledwie siedmiu operacjach testowych pokazać wyniki każdej i ograniczenia niepewności.

## 6. Uzupełnienie o kalibrację i postprocessing

AP, F1 i kalibracja odpowiadają na inne pytania. Dla niezmiennego zbioru predykcji ściśle monotoniczna transformacja score zachowuje ranking AP; może natomiast zmienić działanie stałego progu. Kalibracja zależna od położenia/rozmiaru może zmienić ranking. Dlatego nie należy wnioskować, że poprawa kalibracji automatycznie poprawi AP, ani że oracle-localization dowodzi dobrej kalibracji.

W raporcie kalibracji określić zdarzenie poprawnej detekcji: zgodna klasa i dopasowanie przy ustalonym IoU, z zasadą obsługi duplikatów. Reliability diagram oraz D-ECE mogą ujawnić zależność od położenia i skali bboxów. Podawać liczność binów, sposób binningu, rozdzielczość i cutoff. Kalibracja predykcji nie zastępuje recall: niewykryte GT wymagają nadal osobnych metryk. Schemat i ograniczenia powinny odpowiadać definicji wybranej implementacji. [Küppers et al., CVPR Workshops 2020](https://arxiv.org/abs/2004.13546).

Soft-NMS osłabia score nakładających się kandydatów zamiast od razu odrzucać je przez hard-NMS. Można go zbadać jako kontrolę postprocessingu YOLO, lecz poprawy z oryginalnej publikacji nie są obietnicą poprawy w tooltip detection. Nie da się wiarygodnie ocenić tej zamiany, mając jedynie końcowe bboxy po hard-NMS. [Bodla et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html).

Literatura podstawowa i bliskie selektory pozostają w [ZRODLA.md](ZRODLA.md). Nowe zapytania wykonano przez paper-search (arXiv/Crossref), a trafne prace zweryfikowano na oficjalnych stronach CVF i arXiv. Nie wszystkie surowe trafienia były trafne. Zakres nowej lektury: abstrakty i fragmenty definicji/metody udostępnione przez strony źródłowe, nie pełna reprodukcja implementacji. Dodatkowe linki: [linki_recenzje_uzupelnienie.txt](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/kalibracja-detekcji-i-soft-nms-2026-09-11.txt); logi: `search_logs/reviewers_calibration.txt`, `search_logs/reviewers_soft_nms.txt`.

## 7. Warunek rzetelnej przyszłej odpowiedzi

Każdy punkt oznaczony jako wykonany powinien wskazywać: zmieniony fragment artykułu, rzeczywisty artefakt wynikowy oraz ograniczenie wniosku. Dla treningu potrzebne są manifest/hash danych, wersja kodu i konfiguracji, log, seed, checkpoint oraz predykcje i wynik referencyjnego evaluatora. Dla wyjaśnienia liczebności potrzebny jest rejestr danych, a nie domyślna historia 12 klatek. Do czasu zebrania tych dowodów używać opisu planowanej korekty, bez konkretnych nowych wyników i wartości p.

Oryginalnych recenzji, projektu odpowiedzi, artykułu ani kodu produkcyjnego nie zmieniono. W tej części pracy nie uruchomiono nowych treningów i nie potwierdzono deklarowanych w projekcie odpowiedzi eksperymentów.
