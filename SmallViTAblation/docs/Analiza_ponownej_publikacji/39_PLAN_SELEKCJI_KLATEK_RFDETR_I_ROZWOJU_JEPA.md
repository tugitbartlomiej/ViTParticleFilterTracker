# 39. Dobór klatek do zwykłego RF-DETR: plan eksperymentu i dalszych badań

**Data:** 2026-09-21. **Status:** poprawiony projekt badania; bez nowych treningów i bez wdrożenia opisanych poprawek kodu.

**Decyzja autora:** wykorzystać Koopmana i model z pamięcią inspirowany uogólnionym równaniem Langevina do wyboru już oznaczonych klatek. Docelowy RF-DETR nadal analizuje pojedynczy obraz. Selekcja i diagnostyka działają podczas przygotowania danych lub uczenia, nie podczas inferencji.

Ten dokument zastępuje wcześniejszy wariant selekcji z sekcji 9 [planu 37](37_PLAN_EKSPERYMENTU_DIAGNOSTYKA_LATENTU_W_WIDZENIU.md). Plan 37 zachowuje protokół diagnostyki i testy poprawności. Dokumenty [36](36_KOOPMAN_LANGEVIN_TOZSAMOSC_NARZEDZIA_I_DOBOR_KLATEK.md) i [38](38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md) są źródłami wyników eksploracyjnych; nie zastępują porównania selektorów na nowych operacjach. W kwestii selekcji i kryteriów jej oceny obowiązuje niniejszy dokument.

## 1. Co badamy teraz, a co później

Pierwsze pytanie brzmi: **czy informacja o dynamice reprezentacji narzędzia pozwala wybrać zbiór, na którym standardowy RF-DETR uczy się lepiej niż na klatkach wybranych według błędu detektora i różnorodności?**

To nadzorowany dobór danych treningowych z oznaczonej puli, nie eksperyment oszczędzania nowych anotacji. Etykiety całego train są dostępne wszystkim selektorom. Budżet `B` oznacza liczbę unikalnych obrazów użytych w końcowym treningu. Oddzielnie liczymy koszt proxy, ekstrakcji cech, dopasowania dynamiki i treningu końcowego.

Nie dodajemy do detektora pamięci, filtra ani rekonstrukcji 3D. Zachowanie architektury, rozdzielczości i ustawień inferencji usuwa dodatkowy koszt selektora podczas detekcji; identycznego czasu ściennego nie zakładamy bez pomiaru. Dobór danych może poprawić detekcję częściowo widocznego narzędzia, ale nie daje jednoklatkowemu modelowi dostępu do przeszłości.

| Hipoteza | Kontrast rozstrzygający | Granica wniosku |
|---|---|---|
| H-S: wartość selekcji | Loss + różnorodność + dynamika z pamięcią kontra loss + różnorodność | Wartość dla detekcji, a nie tylko dla przewidywania latentu. |
| H-M: wartość historii | Model z pamięcią kontra ten sam model bez pamięci | Historia wnosi sygnał przy tej reprezentacji i skali czasu. |
| H-K: wartość konkretnego modelu | Model Koopmana z pamięcią kontra zwykły model autoregresyjny o kontrolowanej pojemności | Sama przewaga nad modelem bez historii nie dowodzi szczególnej roli formalizmu GLE. |
| H-T: transfer | Te same listy klatek na drugim małym detektorze | Selekcja nie jest korzystna wyłącznie dla jednego proxy lub architektury. |

Nie zapisujemy z góry, że wybrane klatki są „najbardziej informacjonośne”. To skrót do hipotezy, której miarą jest jakość modelu po treningu oraz koszt jej osiągnięcia.

## 2. Co już wiemy z lokalnych wyników

Odczytano 36 istniejących raportów `validation_report.json` z kampanii Static36; wszystkie mają `evaluation_set=val`, `test_evaluated=false`, `status=passed`. Ponowne zestawienie liczb i hashe plików zapisano w [rejestrze lokalnym](evidence_latent_cv_2026-09-21/selection_update_local_validation_summary.json). To agregacja istniejących raportów, nie nowe treningi, ponowna ewaluacja predykcji ani test istotności.

Wyniki `best_val` wybrano na tym samym zbiorze walidacyjnym, na którym je raportowano. Nie traktować ich jako niezależnego oszacowania generalizacji. Wynik `last_epoch` i `best_val` musi mieć osobne kolumny; nie porównywać jednego z drugim.

Przy 1500 klatkach, średnie AP z trzech seedów:

| Selekcja Static36 | Ostatnia epoka | Najlepszy checkpoint na val |
|---|---|---|
| Random | 0,6472 | 0,6818 |
| Uniform time | 0,6476 | 0,6832 |
| DINO coverage | 0,6508 | 0,6838 |
| DINO temporal hybrid | 0,6524 | 0,6821 |

Rozstęp średnich wynosi około 0,52 punktu AP dla ostatniej epoki i 0,20 punktu dla `best_val`. To małe różnice opisowe, bez przedziałów dla sparowanych kontrastów. Wyniki uzasadniają mocne proste kontrole; nie wykazują jeszcze statystycznej przewagi któregoś selektora.

Dokument 38 i lokalny `REPORT_kalman_ut_b2500_best.json` wskazują ważne ograniczenia pilota:

- AUC sygnału innowacji wynosi około 0,603 dla ramek i 0,678 dla odczytu latentu, podczas gdy confidence daje około 0,827. To argument za badaniem **przyrostu ponad prosty sygnał**, nie za jego zastąpieniem resztą.
- Odczyt prędkości z ramek detektora jest lepszy niż z liniowej sondy latentu. Geometria ramek musi być tanim baseline; latent trzeba uzasadnić dodatkowym wynikiem.
- Dokument 38 raportuje około 13–15 procent chybień w najszybszych 5 procentach klatek wobec około 3–5 procent w wolniejszych. Część skoków jest związana z adnotacjami lub powrotem obiektu. To eksploracyjna przesłanka dla analizy trudnych zdarzeń, nie czysty dowód błędu spowodowanego ruchem.
- Historia 3 kontra 8 kroków nie jest porównaniem „bez pamięci” kontra „z pamięcią”. Globalne PCA i standaryzacja wymagają powtórzenia w poprawnym podziale.

**Konsekwencja:** pierwszy selektor powinien łączyć informację zadaniową z dynamiką obiektu. Maksymalizacja samej nieprzewidywalności całej sceny mogłaby wybierać odblaski, ruch tła i błędy przypisania.

## 3. Dane i rozdzielenie dostępu

1. Zachować podział Static36: **117/25/25 grup operacji**. Sprawdzić mapowanie pacjentów; do tego czasu nie deklarować rozłączności pacjentów.
2. Selekcja, uczenie proxy i modeli dynamiki używają wyłącznie train. Walidacja służy zamrożeniu receptury. Końcowy test nie służy do ustawiania wag, progów ani wyboru reprezentacji.
3. Historyczne analizy torów obejmowały 167 grup, w tym grupy testowe; dokument 38 potwierdza to w sekcji 4e. Raportować zakres wcześniejszego dostępu. Bez nowego, rzeczywiście odłożonego materiału końcowe wyniki mają to ograniczenie; nie opisywać testu jako nigdy nieoglądanego.
4. Obecna pula Static36 zawiera klatki z ramką cystotomu. Brak ramki nie jest automatycznie etykietą „brak narzędzia”. Negatywy, pełne zasłonięcia i wejścia/wyjścia wymagają osobnego audytu oraz rozszerzenia wspólnej puli wszystkich metod.
5. Jednostką wyboru jest oryginalna klatka przed augmentacją. Zduplikowane eksporty i warianty augmentacji należą do tego samego źródła. Augmentacje generować dopiero podczas treningu.
6. Do wyliczenia dynamiki potrzebne są ciągłe, oznaczone czasem odcinki w train. Pula kontekstu może być gęstsza niż pula klatek dopuszczonych do końcowego treningu. Zapisać obie listy i ich koszt. Te same odcinki są dostępne wszystkim wariantom temporalnym.

Minimalny manifest obejmuje `frame_id`, `source_id`, `video_id`, `group_id`, `split`, PTS, indeks klatki, źródłowy FPS, identyfikator obiektu/odcinka, klasę, ramkę GT i status widoczności, jeżeli jest wiarygodnie znany. Osobno zapisać checkpoint, warstwę cech, preprocessing i listę grup dopasowania. Brak mapowania grupy kończy walidację błędem.

**Odczyt bieżących danych:** pierwszy rekord lokalnego `feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl` ma `time_unit=frames`, `timestamp=frame_index`, `native_fps=null` i `patient_id=null`. Zawiera też `tip_point`, którego pochodzenie i jakość trzeba sprawdzić przed użyciem do oceny końcówki. Nie stosować poniższych progów w sekundach przed ustaleniem czasu źródłowego. Paczka diagnostyczna dokumentu 38 obejmuje 12 klipów val i 4 train; nie jest właściwą pulą do dopasowania selektora.

Budżety startowe: **500, 1500 i 2500** źródeł, zgodnie z istniejącą kampanią. Główny pilot: **1500**. Wielkości są decyzjami projektowymi, nie optimum wyprowadzonym z teorii.

## 4. Wspólne proxy i reprezentacja narzędzia

### 4.1. Co zamrażamy

Domyślny pilot: jeden RF-DETR Nano trenowany z COCO na wspólnych 500 klatkach train wybranych równomiernie w czasie, seed 42, przez ustalone 10 epok. Checkpoint po 10. epoce jest wspólny dla wszystkich selektorów; 10 epok to punkt startowy eksperymentu, nie gwarancja dobrego proxy. Przed zamrożeniem protokołu sprawdzić, czy loss i pokrycie detekcji nie są zdegenerowane. Ewentualną zmianę checkpointu ustalić na development i stosować jednakowo.

Końcowe detektory zawsze zaczynają od tych samych wag COCO. Klatki użyte do proxy **nie muszą** wejść do końcowego podzbioru. Jest to poprawne dla już oznaczonej puli; należy ujawnić 500 dodatkowych źródeł użytych do uczenia proxy i ich unię z końcowym podzbiorem. Nie nazywać takiego eksperymentu uczeniem przy całkowitym dostępie tylko do `B` etykiet.

Istniejący checkpoint można wykorzystać zamiast nowego wyłącznie po audycie grup, receptury, epoki i dostępu do walidacji. Historyczny `best.pth` po 30 epokach nie jest checkpointem z 10. epoki. Jedno proxy wystarcza do pilota; potwierdzenie może powtórzyć jego inicjalizację osobno od seedu końcowego treningu.

### 4.2. Cechy

- **Wariant główny:** cechy backbone/encodera RF-DETR z obszaru ramki GT oraz geometria tej ramki. Adnotacje są legalnym wejściem selektora treningowego. Pozwala to analizować narzędzie również wtedy, gdy proxy go nie wykrywa.
- **Kontrola tania:** sama geometria ramek GT i przewidywanych oraz miary confidence/ruchu.
- **Kontrola reprezentacji:** cechy DINOv2 ROI na tej samej puli; globalne cechy sceny tylko jako kontrola tła.
- **Wariant bez GT lokalizacji:** przewidziane ROI z jawnym przypisaniem obiektu. To dodatkowa ablacja, nie warunek pierwszego eksperymentu.

GT-bbox może opisywać końcówkę lub fragment narzędzia, a nie całe narzędzie. Ustalić semantykę anotacji przed nazwaniem cech „reprezentacją całej bryły”. Środek ramki nie jest automatycznie końcówką, a kierunek dłuższego boku ramki nie jest prawdziwą orientacją 3D.

Ekstrakcja w `eval()`, bez losowej augmentacji. Pooling tokenów w ROI i osobne kodowanie cropu to różne eksperymenty; nie zmieniać sposobu między ramionami. Zmniejszenie cropu usuwa położenie globalne, więc geometria pozostaje oddzielnym kanałem.

## 5. Dwa różne błędy i kandydat na selektor

### 5.1. Trudność detekcji

`L_det(t)` jest stratą pojedynczego obrazu policzoną ze standardowym przypisaniem zbioru predykcji do GT: składnik klasyfikacji, L1 i GIoU zgodnie z tą samą recepturą we wszystkich ramionach. Zapisać składniki oddzielnie. Nie odejmować nieuporządkowanych zbiorów ramek i nie wybierać stale jednego numeru query.

Pomiar loss wymaga adaptera zwracającego wartości per obraz, przy deterministycznym preprocessingu. Główne rozstrzygnięcie wykorzystuje pełną stratę detekcyjną, obejmującą również błędy klasyfikacji tła w dostępnej puli. Raport FN/FP/IoU jest warstwą interpretacji, a nie dowolnym zamiennikiem funkcji straty.

Proxy widziało część train, więc jego błędy na tych klatkach mogą być zaniżone. To nie przeciek testowy, ale możliwa stronniczość rankingu. Zapisać `proxy_seen`; sprawdzić wynik z wyłączeniem tych 500 źródeł z puli wyboru we wszystkich metodach albo z proxy OOF w potwierdzeniu. Jeżeli uczymy dodatkowy kalibrator błędu, potrzebne są zagnieżdżone predykcje OOF; nie kalibrować go na własnych błędach treningowych.

### 5.2. Dynamika

Nie porządkować losowych zdjęć i nie nazywać takiej kolejności czasem. Używać PTS oraz tożsamości tego samego narzędzia. Domyślny krok wynosi 0,20 s z tolerancją 10 procent, jak w planie 37. Dla nieregularnych odcinków jawnie zerwać historię; nie wypełniać ukrytego obiektu zerową ramką. Dodatkowe kroki czasu oceniać dopiero w analizie wrażliwości.

Porównujemy na tych samych oknach. N/K/KM/AR mają wspólny oceniany target stanu; V jest oddzielną kontrolą samej geometrii:

| Model | Konstrukcja | Co sprawdza |
|---|---|---|
| V | Stała prędkość / odporna innowacja geometryczna | Czy wystarcza prosty ruch, bez cech głębokich? |
| N | Nieliniowa regresja następnego stanu z bieżącego stanu | Czy liniowy model nie jest po prostu zbyt słaby? |
| K | Ustalony lifting `u=g(z)` i regresja `u[t+1] ≈ A u[t]` | Skończone przybliżenie Koopmana/EDMD. |
| KM | Ten sam lifting i bieżący człon plus `sum_j B[j] u[t-j]` | Dodatkowa informacja z historii. |
| AR | Zwykła regularyzowana autoregresja nieliniowa na tych samych opóźnieniach | Czy nazwa/formalizm wnosi coś ponad ogólne modelowanie temporalne? |

K i KM mają identyczny bieżący słownik, target, normalizację i wspólne okna. Kontrola N/AR ma jawnie porównany budżet parametrów oraz strojenia. Jeżeli KM po zapisaniu równań okazuje się dokładnie tą samą klasą co wybrane AR, są jednym baseline, nie dwiema niezależnymi metodami. Nowością może być wtedy empiryczny selektor, nie nowa klasa modeli dynamiki.

Do porównania N/K/KM/AR użyć tego samego stanu `z`: cechy ROI po redukcji dopasowanej na fold-train oraz geometria ramki. Lifting ma zawierać także stan, np. `g(z)=[z, phi(z), 1]`; błąd rankingu liczymy w bloku `z`, a nie dla K w większej przestrzeni obserwabli i dla AR w mniejszej. Proponowana agregacja to połowa średniego błędu standaryzowanych cech i połowa średniego błędu standaryzowanej geometrii, niezależnie od liczby ich wymiarów. Te same wagi i ta sama odporna/kwadratowa strata obowiązują w każdym ramieniu. V jest osobną kontrolą geometryczną, nie izomorficznym wariantem targetu latentu.

Obecne `RFF + ridge` przewidujące przyrost nie jest samo z siebie dowodem istnienia niezmienniczej podprzestrzeni Koopmana. Najpierw nazwać je nieliniowym predyktorem. Użycie nazwy EDMD wymaga wskazania liftingu, liniowej ewolucji obserwabli i jej sprawdzenia poza próbką. KM nazywamy modelem inspirowanym GLE; nie odzyskujemy fizycznego tarcia ani termicznego szumu.

Modele i transformacje dopasowywać z walidacją po grupach **wewnątrz train**. Reszty do rankingu głównego wariantu pochodzą z przewidywań OOF: model nie widział danej operacji przy dopasowaniu. Wykorzystywanie obrazu bieżącego targetu do obliczenia jego reszty jest legalne w selekcji offline i musi być tak opisane.

OOF dotyczy tu regresji dynamiki i jej transformacji przy wspólnym zamrożonym encoderze. Sam encoder mógł widzieć etykiety tej operacji w 500 obrazach proxy; nie jest to niezależna walidacja całej reprezentacji. Dla takich wniosków służy rozłączny podział B/C planu 37 albo cross-fitting również proxy. W selekcji train wspólny encoder jest legalny i jednakowy dla porównywanych metod.

### 5.3. Prosta reguła do pierwszego testu

Najpierw ustalić wspólne kwoty po operacjach, proporcjonalne do pierwiastka liczby kandydatów, z zaokrągleniem metodą największych reszt. We wszystkich selektorach poza jawną kontrolą czysto losową stosować te same reguły odstępu czasowego i uzupełniania niedoboru. Dla głównych kontrastów reguły są identyczne.

W ramionach H i H+M każda kwota ma **25 procent kotwic równomiernych w czasie, 25 procent pokrycia cech ROI, 50 procent rankingu zadaniowego.** Kontrola confidence + pokrycie stosuje ten sam podział, zastępując jedynie ranking loss przez `1 - max_confidence` dla klasy narzędzia; brak detekcji dostaje 1. U wybiera 100 procent kwoty równomiernie w czasie, random losowo, a samo pokrycie przez 100 procent wyboru k-center. Wspólne pozostają pula, kwoty, ograniczenia czasu i zasady uzupełniania.

Liczby są prostym, zamrażanym punktem startu. Nie są wnioskiem z literatury. Wypełnianie bez powtórzeń, deterministyczne zaokrąglenie, jawne reguły dla małych grup. Najpierw odrzucić identyczne źródła; minimalny odstęp 0,20 s można rozluźnić tylko według wspólnej, logowanej reguły, gdy nie da się osiągnąć `B`.

Ranking zadaniowy:

```text
H:        rank(L_det)
H + M:    0.5 * rank(L_det) + 0.5 * rank(r_M)
r_M:      błąd OOF przewidywania wspólnego stanu, po skalowaniu wyłącznie na fold-train
```

`M` oznacza V, N, K, KM albo AR. Użyć średnich rang dla remisów; stały score nie może otrzymać sztucznego trendu przez kolejność plików. Rangi liczyć w obrębie grupy; w grupach bez zmienności dany składnik jest neutralny. Kolejność przy ostatecznym remisie rozstrzyga wspólny seed.

Sygnał ma być reprezentowany identycznie we wszystkich ramionach temporalnych. Wybrać przed treningiem jedną miarę błędu stanu; kwadrat błędu i odporną stratę Hubera porównać na development ze względu na grube ogony. Nie dodawać wyłącznie do KM dodatkowego score poprawy względem K i nadal interpretować kontrastu jako samego efektu historii.

Przy braku wiarygodnego okna wszystkie ramiona temporalne wracają do `rank(L_det)` i zapisują powód. Odsetek fallback, wejścia narzędzia i wyjścia raportować. Nie usuwać trudnych klatek tylko dlatego, że nie mają pełnej historii. Nie odrzucać automatycznie najwyższych 5 procent reszt/loss jako „szumu”.

Diagnostyka rankingu przed treningiem: zestawić score z prostą zmianą `||z[t]-z[t-1]||`, ruchem ramek, wielkością obiektu, confidence, jasnością i ostrością. Sprawdzić, czy KM nie jest tylko droższym rankingiem prędkości lub zakłóceń. Permutowana historia jest dodatkową kontrolą zależności od kolejności czasu, a nie dowodem przyczynowości. Nie wymaga to kolejnego detektora; jeśli prosta zmiana w pełni wyjaśnia ranking, dodać ją jako ramie selekcji przed mocnym twierdzeniem o metodzie i jawnie zwiększyć koszt macierzy.

Jest to **hipoteza użyteczności reszty**, nie jej gwarancja. Jeżeli sygnał reaguje głównie na odblaski lub błędy GT, najpierw raportować tę porażkę. Bardziej złożony kalibrowany selektor pozostaje późniejszą ablacją, z własnym kosztem i OOF.

## 6. Minimalny pilot i rozszerzenia do artykułu

### 6.1. Najpierw mały eksperyment

Główne porównanie przy `B=1500`, jednym seedzie selekcji i trzech seedach treningu `42, 1337, 2024`:

| ID | Selektor | Znaczenie |
|---|---|---|
| U | Równomiernie w czasie | Istniejący silny baseline wideo. |
| H | Loss + różnorodność i kotwice | Najważniejszy baseline dla już oznaczonego zbioru. |
| V | H + prosta dynamika geometrii | Czy wystarcza ruch/innowacja ramek? |
| K | H + reszta modelu K | Kandydat bez pamięci. |
| KM | H + reszta modelu KM | Główny kandydat z pamięcią. |
| AR | H + reszta zwykłego modelu AR | Kontrola szczególnego wkładu Koopmana/GLE. |

To **18 końcowych treningów + jedno wspólne proxy**, jeśli nie ma checkpointu spełniającego ustaloną recepturę, przy sześciu rzeczywiście różnych ramionach. Liczby 18 i maksymalne 87 poniżej zakładają odrębną konstrukcję AR oraz KM; jeśli zostaną scalone jako ta sama metoda, manifest przelicza mniejszą macierz. Najpierw można wykonać sześć treningów z seedem 42 jako test techniczny; nie ogłaszać z nich istotności ani wybierać zwycięskiego seedu. Jeżeli protokół pozostaje bez zmian, te sześć należy do osiemnastu, a nie dodaje się do nich.

Wszystkie końcowe modele: RF-DETR Nano, ta sama inicjalizacja COCO, rozdzielczość 384, efektywny batch 16, BF16 i identyczne augmentacje. Punktem odniesienia jest odczytana receptura Static36: AdamW, LR 0,0001, LR encodera 0,00001, 30 epok przy `B=1500`. Przed uruchomieniem przypiąć również EMA, harmonogram LR, wersje, matching i regułę checkpointu z pełnego `effective_train_config.json`.

Pierwotna miara pilota używa `last_epoch`; `best_val` jest osobną analizą. Żadnego wyboru reguły po wynikach. Przy stałym `B` wyrównać liczbę aktualizacji i walidacji oraz `drop_last`. Dla różnych `B` główny eksperyment wyrównuje **liczbę aktualizacji optymalizatora**, nie samą liczbę epok; warm-up i scheduler również liczymy w krokach. Dotychczasowe 30 epok dla każdego `B` to inny, oddzielnie opisany eksperyment kosztu.

### 6.2. Rozszerzenia uruchamiane po pilocie

| Rozszerzenie | Dodatkowy zakres | Liczba treningów końcowych |
|---|---|---|
| Pełne kontrole przy 1500 | Random, samo pokrycie ROI, confidence + pokrycie, H + N; po 3 seedy | 12 dodatkowych; razem z pilotem 30. |
| Krzywa budżetu | H, K, KM, AR przy 500 i 2500, po 3 seedy | 24 dodatkowe. |
| Stabilność wyboru | H, K, KM, AR przy 1500; 2 dodatkowe seedy selekcji x 3 seedy treningu | 24 dodatkowe; łącznie po 3 seedy selekcji. |
| Transfer | U, H, KM na jednym drugim małym detektorze, 1500, po 3 seedy | 9 dodatkowych; te same listy klatek. |

Całość tabeli z pilotem daje maksymalnie **87 treningów końcowych**, przed dodatkowymi proxy i referencją pełnej puli. Nie jest to zlecenie uruchomienia całej macierzy. Pierwsza decyzja badawcza dotyczy 18 biegów. Kolejne pakiety mają osobne kryterium sensowności i koszt.

W miarę zasobów dodać referencję treningu na pełnej puli przy tym samym budżecie kroków. Bez niej nie twierdzić, że selekcja zastępuje trening na wszystkich danych. RHO-loss/GLISTER można dodać jako osobne adaptacje, z jawnym kosztem i dostępem do pomocniczych danych.

Nie wstawiać bezpośrednio historycznych wyników Static36 w miejsce nowego sparowanego ramienia. Najpierw sprawdzić identyczność puli, kwot, listy źródeł, receptury, inicjalizacji, checkpointu i ewaluacji. Wynik U w nowej tabeli musi odpowiadać nowemu eksperymentowi, a nie starej stałej 0,6832.

## 7. Co mierzymy i jak analizujemy

**Główna miara:** AP@[0.50:0.95] standardowego ewaluatora, na odłożonych operacjach. Dodatkowe: AP50, AP75, recall, FN/FP, liczba błędów per operacja i zależność od wielkości obiektu. Próg operacyjny zamrozić na validation; AP liczyć ze wspólnym niskim progiem eksportu, nie z wysokim progiem alarmu.

**Trudne sytuacje:** osobno szybki ruch, odblaski, mały obiekt, częściowe zasłonięcie i wejście/wyjście, o ile istnieją niezależne oznaczenia. Prędkość z GT liczyć w jednostkach obrazu na sekundę przy prawidłowym czasie i ciągłości obiektu. Progi „szybko” ustalić na development i przenieść na test; testowy górny kwantyl podać ewentualnie jako analizę opisową. Prędkość to nie etykieta zasłonięcia.

Jeśli nie ma oznaczeń zasłonięć, przygotować jeden wspólny, ślepo anotowany podzbiór oceny z zamrożonymi zasadami widoczności. Nie dobierać go na podstawie sukcesów KM. Wyniki syntetycznych zasłonięć są odrębnym testem i nie zastępują rzeczywistych zdarzeń. Błędy skokowych anotacji sprawdzić na próbce przed ustaleniem progów; analiza bez odrzucania skoków pozostaje jawna.

**Selekcja:** liczyć udział powtarzalnych sąsiadów, pokrycie operacji, histogramy loss i ruchu, udział ramek niewykrytych przez proxy, odsetek fallback i pokrycie rzadkich stanów. Pokazać przykłady dobrane według zapisanej reguły, również niekorzystne. Duża różnorodność sama w sobie nie dowodzi wzrostu AP.

**Koszt:** osobno czas proxy, ekstrakcji, CV dynamiki, wyboru i końcowego treningu, pamięć oraz rozmiar cache. Nie pomijać kosztu przeczytania całej puli w twierdzeniu o oszczędności. Raportować krzywą jakość–liczba klatek oraz jakość–całkowite GPU/CPU-godziny. Zsumowanie czasu 36 istniejących biegów nie jest prognozą czasu nowej metody.

**Inferencja:** ta sama rozdzielczość, batch, precision, urządzenie, eksport i postprocessing; rozgrzewka, synchronizacja GPU i mediana/p95 na tej samej puli. Osobno czas samej sieci i całej detekcji. Selektor nie jest wywoływany na teście. Nie zakładać dokładnie identycznego FPS, gdy liczba detekcji zmienia koszt postprocessingu.

### Kryteria wniosku

- Główny kontrast: **KM–H** przy 1500. Proponowany minimalny efekt praktyczny: **0,004 AP**, czyli 0,4 punktu AP. To decyzja do zamrożenia przed wynikami, nie oczekiwany rezultat.
- Przy dodatnim głównym wyniku sprawdzić **KM–K** i **KM–AR**; dla tych dwóch kontrastów atrybucji kontrolować wielokrotne testowanie, np. procedurą Holma. Pełne twierdzenie o szczególnej korzyści modelu wymaga również porównania z V i N; nie ukrywać mocniejszego prostego baseline.
- Raportować sparowane różnice dla każdego seedu. Bootstrap po całych operacjach, 2000 replik, ponowne liczenie AP z predykcji, a nie średnia AP klatek. W potwierdzeniu resampling hierarchiczny po seedach wyboru/treningu i operacjach; trzy seedy dają ograniczoną precyzję składowej treningowej.
- Pilot na validation służy rozwojowi, nie potwierdzeniu hipotezy na tej samej wielokrotnie oglądanej walidacji. Dla wniosku potwierdzającego potrzebny jest zamrożony protokół i odpowiednio odłożone dane, z ujawnieniem wcześniejszych analiz testowych.
- Poprawa praktyczna: oszacowanie powyżej 0,004 i przedział różnicy powyżej zera. Jeśli dolna granica przekracza 0,004, można dodatkowo mówić o potwierdzeniu co najmniej minimalnego efektu. Przedział obejmujący zero i wartości istotne praktycznie oznacza wynik nierozstrzygający, nie równoważność.

| Wynik | Wniosek do artykułu |
|---|---|
| KM wygrywa tylko z U/random | Lepszy dobór od prostego próbkowania; brak dowodu przewagi ponad nadzorowany baseline. |
| H dogania KM | Informacja z etykiet i różnorodność wystarczyły w badanej puli; koszt dynamiki nie jest uzasadniony wynikiem. |
| V lub AR dogania KM | Temporalny sygnał może pomagać, lecz nie wykazano przewagi konkretnej konstrukcji Koopmana/GLE. |
| KM wygrywa z H i K, ale nie AR | Historia pomaga; nie wykazano wyjątkowej roli proponowanego modelu pamięci. |
| KM wygrywa z mocnymi kontrolami przy podobnym koszcie | Wsparcie dla dodatkowej wartości selektora w określonej domenie, budżecie i architekturze. |
| Poprawa tylko szybkiego ruchu / zasłonięć | Wąski efekt na oznaczonej podgrupie; główny AP i koszty pozostają jawne. |
| Niższy błąd latentu, bez wzrostu AP | Lepszy model dynamiki nie przełożył się na lepszy zbiór treningowy. |

## 8. Co trzeba poprawić w kodzie przed takim badaniem

Audyt dotyczy wersji odczytanej 2026-09-21. Wskazane funkcje istnieją; lista nie oznacza, że zostały już poprawione.

Przy końcowym sprawdzeniu workspace pojawiły się równoległe zmiany kodu: `history=False` w `latent_langevin.py` oraz `selection_matrix.py`, `pool_confidence.py` i skrypty kolejki stage1. Nie zmieniano ich w ramach tej aktualizacji dokumentacji. Odczytana macierz stage1 nadal opisuje wariant bez GT w rankingu, z wymuszonym wspólnym L0 i DINO; nie jest implementacją nadzorowanego H/K/KM/AR z tego planu. Nowy przełącznik bez historii jest przydatny, ale sam nie rozwiązuje izolacji foldów ani nie zmienia RFF w EDMD. Przed implementacją ponowić audyt aktualnej wersji, zamiast polegać wyłącznie na numerach linii z wcześniejszych dokumentów.

| Miejsce względem SmallViTAblation | Problem / brak | Warunek odbioru |
|---|---|---|
| `pipeline/koopman_langevin/latent_langevin.py`, `track_dynamics.py` | Skalowanie przed CV, zagnieżdżone korektory reszt, użycie bieżącego targetu w `reducible_gain` | Kontrakty A1–A8 planu 37; jawny score offline, bez utożsamiania z błędem nieusuwalnym. |
| `pipeline/koopman_langevin/detector_latent.py` | Wybór maksymalnego logitu po klasach i query; puste ROI zastępowane całym obrazem | Jawna klasa narzędzia, GT-ROI, flaga fallback, zachowanie wszystkich obiektów; COCO nie zna automatycznie cystotomu. |
| `pipeline/koopman_langevin/epoch_dynamics.py` | Globalne PCA i dopasowanie operatora przed sondą CV | Outer-test nie wpływa na transformację, mody ani dobór wymiaru; identyczne okna modeli. |
| `pipeline/koopman_langevin/kalman_state.py` | Składanie OOF sondy pozycji i filtra może ujawnić outer-test; remisy w AUC | Zagnieżdżone dopasowanie; porównanie AUC z implementacją uwzględniającą remisy. |
| `pipeline/koopman_langevin/select.py`, `pipeline/active_learning/acquisition.py` | Manifest starego selektora nie opisuje użycia GT i feedbacku modelu; rangi i niejawne grupy | Prawdziwe `labels_used_for_ranking`, jawny checkpoint, średnie rangi remisów, walidacja grup, dokładnie B źródeł. |
| Nowy adapter loss per obraz | Brak wspólnego score nadzorowanego dla H i rozszerzeń | Zgodność agregacji per-image z referencyjną stratą, deterministyczne przypisanie i jawne składniki. |
| `pipeline/train/train_rfdetr.py`, `run_with_validation.py` | Potrzebna blokada wspólnej liczby kroków i receptury | Te same aktualizacje/scheduler; zapis rzeczywistych kroków i wszystkich odchyleń. |
| `pipeline/eval/group_bootstrap.py` | Bootstrap grup nie pokrywa całej zmienności treningu i wyboru | Sparowanie metod i jawne rozdzielenie źródeł niepewności. |

Testy konieczne: perturbacja outer-test nie zmienia fit; stały score ma same remisy; syntetyczny proces Markowa nie udaje pamięci z powodu zbyt słabego baseline; AR(2) wykazuje znaną korzyść historii; zmiana query nie staje się ruchem obiektu; brak ROI nie przechodzi cicho w cechy tła; selektor nie używa val/test i zwraca dokładnie B unikalnych źródeł. Test forecast niezależnego od przyszłej klatki dotyczy tylko wariantu deklarowanego jako forecast; score offline ma dostęp do bieżącej obserwacji.

Proponowane nowe artefakty: `protocol.lock.json`, audyt puli, manifest proxy i cech, predykcje OOF, tabela score z powodami fallback, listy wybranych klatek, konfiguracje treningu, surowe predykcje oceny i raport sparowanych różnic. To lista do implementacji, nie nazwy gotowych runnerów.

## 9. Co dołożyć do artykułu

1. Diagram: oznaczona pula train → proxy i cechy → selektor → zwykły RF-DETR; osobna strzałka test → RF-DETR. Selektor nie występuje w inferencji.
2. Tabela porównań selektorów z AP, niepewnością, liczbą źródeł oraz pełnym kosztem przygotowania i treningu.
3. Krzywe jakości względem budżetu oraz tabela trudnych zdarzeń z licznościami operacji i klatek.
4. Ablacja obszaru narzędzia kontra globalna scena, geometrii kontra latentu, modelu bieżącego stanu kontra pamięci i zwykłego AR.
5. Analiza, jakie przykłady trafiają do podzbiorów i czy efekt przenosi się na drugi detektor.

Proponowane twierdzenie, wyłącznie po dodatnim wyniku: „Informacja o dynamice cech obiektu uzupełnia nadzorowaną trudność i pokrycie danych przy selekcji klatek chirurgicznych, poprawiając jakość standardowego detektora bez dodawania modułu temporalnego do inferencji”. Nie wpisywać poprawy, nowości ani znaczenia statystycznego do streszczenia przed wynikiem.

## 10. Dalszy rozwój: adaptacja, JEPA i sztywna bryła

**Etap drugi — wybór podczas uczenia.** Po zamrożonym selektorze porównać aktualizację rankingu po równych liczbach kroków treningu z tym samym podzbiorem statycznym i adaptacją według samej straty. Każda metoda ma taki sam budżet odświeżeń, unikalnych źródeł, ekspozycji i obliczeń. Zapisywać całą historię użycia danych, aby „mały końcowy podzbiór” nie ukrywał wcześniejszego treningu na całej puli.

**Etap trzeci — reprezentacja typu JEPA.** JEPA jest sposobem uczenia przez przewidywanie reprezentacji, a ViT jest architekturą; nie są synonimami. Uczyć na krótkich spójnych klipach, zachowując znaczniki czasu. Wybór wyłącznie odległych klatek może usunąć przejścia konieczne do poznania ruchu. Jeśli używamy GT do ROI lub doboru klipów, ujawnić to: sam cel JEPA może być samonadzorowany, lecz cała procedura nie jest wtedy pozbawiona nadzoru.

**Etap czwarty — bryła 3D i zasłonięcia.** Model ma rozdzielać wspólny, sztywny kształt narzędzia, zmienną pozę, projekcję kamery i widoczność. Koopman/pamięć może regularizować lub przewidywać stan; sama mała strata przewidywania JEPA nie identyfikuje jednoznacznie poprawnej bryły. Potrzebne są geometryczne ograniczenia, różne ujęcia i niezależne cele oceny. Palce, uchwyt i anatomia mogą być kontekstem, ale należy sprawdzić przenoszenie na innych operatorów i ujęcia. Standardowe ramki fragmentu narzędzia nie są pełną anotacją amodalną ani 3D.

**Etap piąty — stereo i jednostki fizyczne.** Skalibrowane, zsynchronizowane stereo i znane wymiary umożliwiają pomiar metryczny przy spełnieniu założeń obserwacji. Jedna długość z katalogu ustala skalę dopiero dla prawidłowej geometrii odtworzonej do podobieństwa; nie naprawia ogólnej niejednoznaczności projekcyjnej. Raportować błąd względem niezależnego wzorca; wartości w milimetrach nie oznaczają automatycznie milimetrowej dokładności. Przewidywana część pod tkanką pozostaje estymacją z niepewnością.

Każdy etap ma osobne porównanie kontrolne. Dodatni wynik selekcji nie jest dowodem uczenia bryły, a dobra rekonstrukcja 3D nie dowodzi użyteczności selektora.

## 11. Literatura i granice nowości

Przegląd jest celowany. Nie znalezienie identycznego pipeline w kilku zapytaniach nie dowodzi pierwszeństwa. Rejestr zakresu odczytu i surowe wyniki: [evidence](evidence_latent_cv_2026-09-21/README.md).

| Praca | Co uzasadnia | Czego nie dowodzi |
|---|---|---|
| Shrivastava, Gupta, Girshick, 2016, [Training Region-Based Object Detectors with Online Hard Example Mining](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html) | Silny baseline wyboru trudnych przykładów w detekcji. | Nasz offline loss + różnorodność nie jest ścisłą reprodukcją OHEM. |
| Mindermann i in., 2022, [Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt](https://arxiv.org/abs/2206.07137) | RHO-loss odróżnia duży błąd od użyteczności dla uczenia. | Reszta dynamiki nie jest RHO-loss ani oszacowaniem błędu Bayesowskiego. |
| Killamsetty, Sivasubramanian, Ramakrishnan, Iyer, 2021, [GLISTER](https://ojs.aaai.org/index.php/AAAI/article/view/16988) | Dobór oznaczonych danych, również aktualizowany podczas treningu. | Zbiór referencyjny używany w selekcji nie jest niezależnym testem; adaptacja do DETR ma dodatkowy koszt. |
| Wu i in., 2024, [Correlation-Aware Active Learning for Surgery Video Segmentation](https://arxiv.org/abs/2311.08811) | Redundancja wideo, różnorodność i niepewność jako kontrola. | Dotyczy aktywnego anotowania i segmentacji; tutaj etykiety już istnieją. |
| Lin, Tian, Anghel, Livescu, 2021, [Data-driven learning for the Mori–Zwanzig formalism](https://arxiv.org/abs/2101.05873) | Formalny związek projekcji Koopmana i pamięci. | Nie potwierdza zysku selekcji chirurgicznych obrazów. |
| Comas i in., 2023, [Learning Object-Centric Dynamic Modes from Video and Emerging Properties](https://proceedings.mlr.press/v211/comas23a.html) | Dynamika obiektów z wideo i obserwable Koopmana mają precedens. | Nie wykazuje poprawy RF-DETR przez dobór klatek. |
| Assran i in., 2025, [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Kierunek uczenia reprezentacji i predykcji z filmów. | Nie gwarantuje identyfikacji sztywnej bryły narzędzia. |
| Guo i in., 2024, [SurgTrack](https://arxiv.org/abs/2409.02598) | Wyuczona reprezentacja 3D narzędzi i śledzenie z historią. | Sam cel rekonstrukcji i odporności na zasłonięcia nie jest nowy. |
| Tian i in., 2026, [Panning for Gold: Finding Representative Water Quality Dynamic Data for Data-Efficient Machine Learning through Koopman Analysis](https://www.sciencedirect.com/science/article/pii/S3050472426000171), DOI 10.1016/j.wateco.2026.100049 | Bezpośredni precedens doboru danych z użyciem Koopmana i klasteryzacji do uczenia innych modeli. | Inna dziedzina: dynamika jakości wody, bez chirurgicznego wideo i RF-DETR. Wyklucza szerokie twierdzenie o pierwszej selekcji danych Koopmanem. |
| Li, Silva, Hsieh, 2024, [EnKode: Active Learning of Unknown Flows with Koopman Operators](https://arxiv.org/abs/2410.16605) | Operator Koopmana i ensemble przy aktywnym wyborze pomiarów. | Próbkowanie pól przepływu to inne zadanie niż dobór już oznaczonych obrazów. |

**Kandydat na wkład:** kontrolowany test dodatkowej wartości dynamiki i pamięci w nadzorowanej selekcji klatek do szybkiego, jednoklatkowego detektora, z kontrolą redundancji, geometrii, trudności, kosztu i transferu. Nowość konkretnej konstrukcji trzeba jeszcze zestawić z najbliższymi metodami, a jej wartość wykazać pomiarem.

## 12. Pierwsza kolejność prac

1. Zamrozić pulę, grupy, stan wcześniejszego dostępu do testu i definicję ramki narzędzia.
2. Naprawić krytyczne kontrakty CV, ROI, czasu i rang; powtórzyć tani test mechanizmu na train.
3. Przygotować wspólne proxy, loss per obraz i cache ROI; sprawdzić zakres sygnałów oraz fallback.
4. Wygenerować sześć list po 1500 źródeł, bez uruchamiania treningów przy błędnych manifestach.
5. Wykonać pilot 18 treningów według jednej blokady receptury i sporządzić pełną tabelę, także przy wyniku zerowym.
6. Zdecydować na podstawie pilota, które pakiety potwierdzające są uzasadnione; zamrozić je przed odczytem wyników końcowych.

Najbliższy kamień milowy to **poprawny, porównywalny selektor i pilot jego wpływu na RF-DETR**. Rozbudowana analiza widma, oś epok, JEPA i model bryły nie są warunkiem wykonania tego pierwszego badania.
