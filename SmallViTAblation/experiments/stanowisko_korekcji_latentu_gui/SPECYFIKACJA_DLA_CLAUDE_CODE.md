# Stanowisko korekcji latentu i identyfikacji dynamiki — opis dla Claude Code

Data: 2026-10-07. Repozytorium: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`.

Dokument przekazuje wymagania do przyszłej implementacji. W ramach przygotowania tego opisu nie zmieniono kodu GUI i nie uruchamiano eksperymentów. Użytkownik chce kolejną zakładkę istniejącej aplikacji, która pozwoli badać ruch tokenów potrzebny do poprawienia detekcji oraz porównywać go z rzeczywistym uczeniem ViT.

## 1. Cel badawczy

Pytanie główne: jaka mała zmiana reprezentacji tokenów wystarcza do poprawy lokalizacji narzędzia, które składowe tej zmiany są widoczne dla detektora i czy podobną zmianę da się uzyskać przez rzeczywiste uczenie?

Etykiety pozwalają optymalizować aktywacje osobno dla danego obrazu. Taki pomiar oznaczać jako „korekta z etykietą — oracle”. Nie przedstawiać wyniku jako jakości detektora działającego bez etykiet. Wspólna korekta lub regulator oceniany na niewidzianych filmach jest osobnym eksperymentem.

Nie obiecywać globalnego maksimum mAP, unikalnego kierunku ani najmniejszej możliwej korekty. Wynik optymalizacji to najlepsza znaleziona korekta przy podanym celu, ograniczeniach i budżecie. Porównywać również różne początki i ustawienia optymalizatora.

## 2. Istniejąca aplikacja i miejsca integracji

Aplikacja korzysta z PyQt6, PyTorch, NumPy i Matplotlib. Nie zastępować istniejącego GUI innym frameworkiem.

Przeczytać przed implementacją:

- `app/app.py`: tworzenie głównych zakładek, klasa `Okno`, metoda `_buduj_ui`.
- `app/zakladka_uczenia.py`: wybór biegu, checkpointu, sondy i tokenów, istniejące wykresy trajektorii.
- `app/zakladka_pchniecia.py` oraz `app/pchniecie.py`: pomiar wpływu obrazów i gradient straty względem mierzonego latentu.
- `app/panel_rownania.py`, `app/rownanie_czastki.py`, `app/rownanie_krokowe.py`: istniejące modele Langevina, GLE i Hankel-DMD oraz ocena prognoz. Nie tworzyć drugiej niezależnej implementacji tych samych podstawowych modeli bez uzasadnienia.
- `app/stan_uczenia.py`, klasa `Odtworzenie`, oraz `app/paczki_uczenia.py`: odtwarzanie rzeczywistego stanu i paczek treningu.
- `app/zakladka_filmu.py`, klasa `WatekFilmu`: wzorzec liczenia w tle i komunikacji z GUI.
- `app/silnik.py`, funkcja `granice_z_wag`: ograniczenia aktywacji wynikające z konkretnego checkpointu.
- `experiments/vit_optymalizacja_latentu128/core.py`: istniejąca interwencja w aktywacje i odtworzenie końcówki sieci z zachowaniem residualu.
- `experiments/vit_latent128_maksymalizacja/maximize.py`: historyczna optymalizacja oracle i sposób zapisu wyników. Nie uruchamiać historycznego skryptu automatycznie; jego cel, ograniczenia i evaluator wymagają jawnego opisu.

Proponowana nazwa zakładki: **„Korekta latentu: cel i dynamika”**. Nowe elementy GUI umieścić w `app/`. Pomocnicze eksperymenty i plany w `experiments/stanowisko_korekcji_latentu_gui/`. Wszystkie nowe pomiary, modele, obrazy i logi zapisywać w unikalnych podkatalogach `outputs/stanowisko_korekcji_latentu_gui/`. Nie nadpisywać historycznych wyników ani nie uruchamiać ich ponownie dla zmiany ścieżek.

## 3. Terminologia i punkt interwencji

Cząsteczka oznacza reprezentację jednego przestrzennego tokenu obrazu. W badanym modelu jest 600 tokenów, po 768 współrzędnych. Globalny układ to wspólna przestrzeń współrzędnych. Makrowektory to jej osie, a mikrowektory cząsteczki to składowe wzdłuż tych osi. Liczby są właściwościami tego modelu i punktu pomiaru.

Mierzony latent znajduje się po GELU, przed `fc2` w MLP ostatniego bloku. Nie jest pełnym stanem sieci. Gałąź residual musi być zachowana i identyfikowana osobno. W trybie oracle zamrozić wszystkie wagi i bufory, ustawić ewaluację i zmieniać wyłącznie zadeklarowane aktywacje. Zerowa interwencja ma odtwarzać pełny forward.

Kontrola osiągalności ma kilka poziomów:

1. Swobodne aktywacje: interwencja diagnostyczna, bez deklaracji osiągalności.
2. Aktywacje w zakresach wyliczonych z checkpointu: ograniczenia każdej współrzędnej są konieczne, lecz nie wystarczają do wspólnej osiągalności całego wektora.
3. Korekta przez wejście badanego MLP: używać rzeczywistej LayerNorm, `fc1` i GELU. Jawnie określić, czy interwencja dotyczy tylko gałęzi MLP, czy również wspólnego wejścia residualu. To różne eksperymenty. Żaden nie gwarantuje istnienia rzeczywistego zdjęcia realizującego stan.
4. Korekta uzyskana przez rzeczywistą aktualizację wag: mierzyć skutki również na innych sondach.

Nie traktować samego dolnego ograniczenia GELU jako gwarancji osiągalności. Historyczna maksymalizacja stosowała przede wszystkim to ograniczenie.

## 4. Dane, które można od razu obejrzeć

Historyczny eksperyment `experiments/vit_latent128_maksymalizacja/WYNIKI.md` opisuje 400 obrazów, checkpoint 1000 i osobną korektę 600 × 128 aktywacji każdego obrazu z użyciem jego etykiety. W jego protokole AP50 wzrosło z 0,437555 do 0,884748, AP50:95 z 0,120881 do 0,770525, a średnia IoU najpewniejszej ramki z 0,471680 do 0,872713. To wynik oracle na optymalizowanej puli; nie dowód generalizacji ani globalnego optimum.

`outputs/vit_latent128_maksymalizacja/TARGETS_FULL400.npz` zawiera klucze:

- `source_indices`: identyfikatory 400 obrazów w źródłowym manifeście.
- `channels0`: 128 indeksów kanałów.
- `base`, `delta_raw`, `target`: tablice 400 × 600 × 128.

Manifest obrazów i etykiet jest w `outputs/vit_optymalizacja_latentu128/input/manifest.json`. Powiązanie z checkpointem weryfikować przez metadane i hash, a nie tylko numer kroku. Pełne 768 współrzędnych odtwarzać z bazowego forwardu i podmieniać wskazane kanały.

`outputs/vit_latent128_maksymalizacja/DETECTIONS_FULL400.npz` zawiera wyniki głowy `baseline`, `optimized`, `random`. Historyczna kontrola `random` wykorzystuje moduły korekt oracle, więc nie jest kontrolą niezależną od etykiet.

Te pliki zawierają końce korekt. Nie przedstawiać interpolacji między końcami jako zapisanej trajektorii optymalizacji. W historycznym logu są wybrane miary postępu, a nie pełna sekwencja aktywacji wszystkich kroków. Import powinien pokazać „przed/po”; odtwarzanie dynamiki wymaga rzeczywiście zapisanych stanów nowego pomiaru.

Przeczytać również `outputs/vit_kierunek_nadzorowany_detekcja128/REPORT.md`: wspólny kierunek nie potwierdził poprawy AP na teście. Nie przenosić sukcesu indywidualnego oracle na hipotezę uniwersalnej korekty.

## 5. Zakres zakładki

### Obraz, cel i sterowanie

- Wybór checkpointu, obrazu z manifestu i jego etykiety; opcjonalne przejęcie sondy i zaznaczenia z zakładki uczenia.
- Podgląd obrazu z GT oraz detekcjami przed i po korekcie. Zachować mapowanie tokenów do patchy.
- Kanały: historyczne J128, wszystkie 768 lub jawnie wybrany zbiór. Tokeny: wszystkie, zaznaczone, obiekt, granica, tło. Jeżeli maskę wyboru wyznacza GT, zapisać tę informację jako część interwencji oracle.
- Cele: zmniejszenie pełnej straty detekcji; poprawa lokalizacji przy kontroli fałszywych detekcji; najmniejsza znaleziona korekta osiągająca zadany próg jakości.
- Parametry: ograniczenie normy, skala regularizacji, liczba kroków, krok optymalizacji, seed i sposób ograniczania aktywacji.
- Przyciski: „Wczytaj historyczny wynik”, „Oblicz korektę”, „Zatrzymaj”, „Porównaj z ruchem uczenia”, „Zapisz pomiar”. Nowe obliczenia uruchamiać po świadomym działaniu w GUI, nie przy otwarciu zakładki.
- Obliczenia w tle, bez blokowania interfejsu. Zatrzymanie zachowuje częściowe wyniki i ich status. Model zadania nie może zmieniać modelu używanego w innych zakładkach.

### Metryki i wizualizacja

- Dla obrazu: strata i jej składniki, IoU przewidzianej ramki, confidence, liczba pominięć i fałszywych detekcji według jawnego protokołu.
- Dla zestawu: AP50 i AP50:95 według jawnie zapisanego evaluatora. Nie nazywać pojedynczego IoU mAP. Historyczny evaluator nie jest automatycznie pełnym COCO evaluatorem; obsługę wielu obiektów i obrazów bez obiektu trzeba zweryfikować.
- Wybór ramki do zwykłego podglądu musi wynikać z predykcji. Ramka wybrana przez najlepsze dopasowanie do GT wymaga osobnego oznaczenia diagnostycznego.
- Mapa normy korekty tokenów, strzałki ruchu, odległość od stanu początkowego, wielkość naruszenia ograniczeń, krzywa jakość–norma korekty.
- Widok 3D w stałej bazie projekcji dla porównywanych trajektorii. Miary liczyć w pełnym latencie lub jawnie nazwanej podprzestrzeni. Projekcja dopasowana z przyszłych stanów jest retrospektywna.
- Dla rzeczywiście nagranej trajektorii: suwak, odtwarzanie, prędkość i przyspieszenie względem właściwej osi kroków. Nie wyciągać tych miar z dwóch punktów przez udawaną interpolację.
- Losowe kontrole o porównywalnym budżecie korekty; po projekcji na ograniczenia mierzyć faktyczną normę, bo może się zmienić.

### Ruch niewidoczny dla końcówki

Dodać diagnostykę wykorzystującą macierz `fc2`: W2 ma w tym modelu rozmiar 192 × 768. Przy zamrożonej końcówce, stałym residualu i wyłączonym dropout zmiana w jądrze W2 nie zmienia jej wyniku. Wymiar jądra wynosi co najmniej 576; rzeczywisty rząd policzyć numerycznie z jawną tolerancją.

Za pomocą SVD rozłożyć korektę na część w przestrzeni wierszowej W2 i część w jego jądrze. Pokazywać normy i udział sumy kwadratów norm. Zero całkowitej korekty daje brak określonego udziału, a nie dzielenie przez zero.

Nazwy: „część przechodząca przez fc2” i „część niewidoczna dla fc2”. Pierwsza nie musi poprawiać zadania. Jest to własność badanego punktu i zamrożonej sieci; w treningu W2 także się zmienia. Dla J128 można dodatkowo badać macierz W2 ograniczoną do wybranych kolumn; jej jądro nie musi być duże. Nie narzucać wielowymiarowego jądra podproblemowi 128 kanałów.

## 6. Dwie osie czasu — obowiązkowe rozróżnienie

**Krok korekty latentu:** zmieniamy aktywacje, wagi i residual są zamrożone. Tor zależy od optymalizatora aktywacji, etykiety, celu i projekcji na ograniczenia. Jeżeli używany jest Adam, jego momenty również należą do stanu tej optymalizacji.

**Krok uczenia ViT:** zmieniają się wagi, momenty AdamW, bufory i odpowiedzi na stałe sondy. Stan latentu sam w sobie na ogół nie domyka dynamiki. Paczka, augmentacje i harmonogram są częścią warunków ewolucji.

Pole `time_axis` zapisywać jako `latent_optimization` albo `network_training`. Klatek filmu nie traktować jako trzeciej próbki tej samej osi czasu. Każda prognoza, pochodna, jądro pamięci i porównanie musi wskazywać swoją oś.

Nie interpretować ruchu sztucznej optymalizacji jako odkrytego prawa uczenia wag. Porównanie obu torów jest pytaniem badawczym.

## 7. Modele dynamiki: rola i warunki użycia

### Langevin i GLE — pierwszy poziom

Modelować krótką prognozę przyrostów tokenu lub wybranych modów zbiorowych. Porównać model bez pamięci z modelem z opóźnieniami oraz z persystencją i stałą prędkością. Ponownie wykorzystać istniejące funkcje po sprawdzeniu zgodności osi, wymiarów i długości próbek.

Jeżeli istnieje rzeczywiście znane wejście, porównać model autonomiczny z modelem uwzględniającym wejście. W modelu uczącym się z samych trajektorii resztę nazywać błędem modelu. Bez sprawdzenia rozkładu i autokorelacji nie nazywać jej białym szumem lub temperaturą. Dopasowane współczynniki opóźnień nie stanowią automatycznie wyprowadzonego jądra Mori–Zwanziga.

Wartość badawcza: określenie, kiedy pamięć poprawia prognozę oraz czy dodanie informacji o optymalizatorze zmniejsza potrzebę długiej historii latentu.

### Koopman / DMD — drugi poziom

Modelować wspólną ewolucję niewielkiego zestawu obserwabli. Zacząć od Hankel-DMD dostępnego w aplikacji. DMDc stosować dopiero z dobrze określonym i dostatecznie zróżnicowanym wejściem; pojedyncza naturalna trajektoria nie gwarantuje identyfikowalności wpływu sterowania.

Dopasowanie redukcji wymiaru, skalowania i operatora wyłącznie na części treningowej. Test na późniejszych odcinkach oraz innych obrazach/przebiegach. Pokazywać błąd przyrostu i błąd wyniku detektora, a nie tylko zgodność dużej, niemal stałej reprezentacji. Wartości własne skończonego modelu nie są dowodem globalnego widma Koopmana ani praw zachowania sieci.

Wartość badawcza: czy mało modów przewiduje zbiorową korektę i jej znaczenie dla detekcji; gdzie taki opis traci ważność.

### Kalman / obserwator stanu — po zbudowaniu modelu

Obserwator służy estymacji stanu modelu z niepełnych lub rzadkich pomiarów. Nie potrzebujemy go wyłącznie do odczytania latentu, który jest dokładnie dostępny po każdym forwardzie. Ma sens, gdy chcemy mierzyć mniej tokenów, rzadziej uruchamiać sondy lub odtwarzać ukryte współczynniki modów.

Pierwszy sprawdzian: zachować pełny zapis jako prawdę odniesienia, ukryć część pomiarów i porównać samą prognozę z prognozą korygowaną dostępnymi obserwacjami. Nie używać ukrytych przyszłych stanów do strojenia obserwatora. Oddzielić szum obserwacji od błędu modelu; dokładnym aktywacjom nie przypisywać dowolnego dużego szumu dla lepszego wykresu.

Zacząć od filtru liniowego w małym modelu stanu. EnKF/EKF rozważyć po wykazaniu ograniczenia tego wariantu. Członek zespołu EnKF to realizacja stanu/prognozy, a nie jeden przestrzenny token. Sprawdzić obserwowalność przy wybranym zestawie pomiarów i kalibrację przedziałów.

Wartość badawcza: podobna jakość estymacji lub prognozy przy mniejszym koszcie sondowania. Wynik filtrowania po nowej obserwacji pokazywać osobno od prognozy wykonanej przed nią.

### Lapunow: funkcja stabilności i wzrost zaburzeń to dwa różne pomiary

Funkcja Lapunowa może służyć analizie zbieżności wybranego modelu korekty do celu. Sama malejąca strata lub odległość do oracle nie daje dowodu stabilności oryginalnego treningu. Dla lokalnego liniowego modelu można badać dodatnio określoną funkcję kwadratową, ale wnioski dotyczą modelu, jego otoczenia i założeń o wejściu.

Wzrost małych zaburzeń badać na dwóch kopiach stanu poddanych tym samym dalszym wejściom i losowości. W treningu odtwarzać również momenty optymalizatora i bufory. Badać kilka skal zaburzenia oraz horyzontów. Po różnej paczce początkowy dystans określić po kroku wprowadzającym zaburzenie.

Różnica obserwowana tylko w tokenach to miara wzrostu odchylenia reprezentacji. Nie nazywać jej automatycznie największym wykładnikiem Lapunowa pełnego układu. Dodatni wzrost w krótkim oknie nie wystarcza do stwierdzenia chaosu.

Wartość badawcza: horyzont przewidywalności, wrażliwe kierunki oraz rozbieżność między wrażliwością latentu i stabilnością detekcji.

### Odpowiedź impulsowa i transmitancja — po kontrolowanych interwencjach

Wejście musi mieć definicję: np. mała zmiana wagi wybranego przykładu w paczce albo podmiana jednego obrazu przy stałej liczebności paczki. Wyjście: określony mod reprezentacji, strata sondy, współrzędne ramki lub IoU według ustalonego dopasowania. Obraz nie jest sam z siebie skalarnym sygnałem sterującym; zapisać jego parametrację.

Dwie gałęzie startują z identycznego pełnego stanu. Jedna otrzymuje interwencję, obie potem te same paczki i augmentacje. Rejestrować różnicę wyjść po 1, 5, 20 krokach i ewentualnie dalej. Powtórzyć dla kilku przyszłych sekwencji. Odpowiedź podmiany o skończonej amplitudzie nie jest automatycznie liniową odpowiedzią impulsową; sprawdzić skalowanie i superpozycję, jeśli ma powstać model liniowy.

Najpierw prezentować zmierzoną odpowiedź i jej niepewność. Klasyczna transmitancja ma sens jako lokalny model w dostatecznie stałym reżimie. Przy zmianie LR, checkpointu lub parametrów dynamiki używać zależnego od kroku modelu odpowiedzi zamiast jednej globalnej transmitancji. mAP z progami i rankingiem nie jest dobrym pierwszym odczytem do liniowej identyfikacji; mierzyć też gładkie wyjścia/stratę, a AP oceniać jako rezultat zadania.

Wartość badawcza: znak, opóźnienie, czas maksimum i trwałość wpływu obrazu. Zgodność prognozy musi być sprawdzona na nowych interwencjach.

## 8. Kolejność budowy

1. Import istniejących par przed/po, identyfikacja modelu i etykiet, mapy korekt, pomiar składowej niewidocznej dla fc2 i naruszeń ograniczeń.
2. Nowa optymalizacja oracle z regularizacją, rejestracją rzeczywistej trajektorii, losowymi kontrolami i eksportem. Start od pojedynczego obrazu; większe zbiory są osobnym uruchomieniem.
3. Porównanie rzeczywistego ruchu uczenia z korektą oracle oraz podłączenie Langevina/GLE i Koopmana do prawdziwych sekwencji. Horyzonty 1, 5, 20; za mało próbek oznacza „brak danych do oceny”.
4. Kontrolowane odpowiedzi gałęzi treningu, wzrost zaburzeń i lokalna identyfikacja wejście–wyjście.
5. Obserwator Kalmana jako sprawdzian redukcji kosztu pomiarów, po zweryfikowaniu modelu i protokołu brakujących obserwacji.

Każdy etap powinien dawać działający pomiar. Nie dodawać pozornie działających paneli z przypadkowymi parametrami lub wynikami zastępczymi. Nie przedstawiać etapu jako wykonanego, jeżeli działa tylko jego interfejs.

## 9. Zapis pomiaru i odtwarzalność

Każdy nowy run otrzymuje własny identyfikator i folder. Zapisywać:

- checkpoint, hash, źródłowy kod modelu, tryb interwencji, punkt wpięcia i stan residualu;
- identyfikatory obrazów/filmów, pochodzenie etykiet, skalowanie obrazu i ramek, wybór tokenów i kanałów;
- cel, ograniczenia, normę i skalowanie, optimizer oraz harmonogram, seed, oś czasu i rzeczywiste indeksy zapisanych kroków;
- bazowe i końcowe aktywacje, rzeczywiście zarejestrowane trajektorie i predykcje;
- surowe wyjścia do oceny AP, wersję evaluatora, maski obserwacji i podział fit/validation/test;
- zmierzone wyniki i prognozy osobno, ograniczenia osiągalności, błędy, status complete/cancelled/failed, czas i zużycie pamięci;
- dla gałęzi treningu pełny stan wznowienia, wspólne przyszłe paczki i losowość, definicję interwencji.

Pełna klatka latentu 600 × 768 float32 zajmuje około 1,76 MiB. Przed nagrywaniem wielu obrazów oszacować koszt. Przechowywać dane porcjami na dysku; nie trzymać wszystkich trajektorii na GPU. Przy rzadszym zapisie podać rzeczywisty interwał i nie udawać obserwacji co krok. Błędy i przerwanie nie mogą niszczyć wcześniejszych pomiarów.

## 10. Kryteria akceptacji i rzetelności

- Zerowa interwencja daje wynik zgodny z pełnym forwardem w jawnej tolerancji numerycznej; w oracle wagi i bufory nie zmieniają hashy.
- Kanały i tokeny poza zakresem interwencji pozostają niezmienione. Stan residualu zgadza się z zadeklarowanym trybem.
- Mała perturbacja z jądra fc2 nie zmienia wyjścia zamrożonej końcówki poza tolerancją; test nie wymaga, żeby perturbacja była naturalnie osiągalna.
- Na przypadkach referencyjnych: identyczne ramki mają IoU 1, rozłączne 0; evaluator obsługuje duplikaty, brak detekcji, wiele GT i obraz bez obiektu albo jawnie odrzuca nieobsługiwany przypadek. Nie pomijać niewygodnych obrazów.
- Historyczny import pokazuje tylko dostępne dane i ostrzega o niezgodnym checkpointcie lub indeksach. Interpolacja nie jest traktowana jako pomiar.
- Wszystkie redukcje, hiperparametry i obserwatory są dopasowane bez dostępu do testowej przyszłości. AP zbioru nie jest średnią „AP tokenów”.
- Model dynamiki musi mieć porównanie z persystencją i stałą prędkością. Osobno ocenić prognozę bez dalszych obserwacji i estymację po korekcie obserwatorem.
- Rozbieżność dwóch identycznych gałęzi bez interwencji wyznacza numeryczny poziom odniesienia. Wyniku poniżej tego poziomu nie interpretować jako efektu obrazu.
- Miary niepewności i podziały uwzględniają filmy oraz niezależne treningi. Tokeny jednego obrazu nie są niezależnymi powtórzeniami eksperymentu.
- GUI pozostaje responsywne, obsługuje brak etykiety, brak checkpointu, brak GPU, błędną adnotację, anulowanie oraz ponowne wczytanie wyniku. Brak zasobu ma dać jasny status; nie podstawiać innego modelu lub losowych danych.

## 11. Literatura do interpretacji metod

- GIoU jako strata lokalizacji: https://arxiv.org/abs/1902.09630
- AP-loss i problem optymalizacji AP: https://arxiv.org/abs/1904.06373
- DMD z wejściem: https://arxiv.org/abs/1409.6358
- Redukcja z pamięcią, Koopman–Mori–Zwanzig: https://arxiv.org/abs/2101.05873
- Obserwacja stanu za pomocą ensemble Kalman filter: https://arxiv.org/abs/physics/0511236
- Chaos podczas uczenia SGD: https://papers.neurips.cc/paper_files/paper/2022/hash/222a2a46018a1e7b55ba48ba11932d04-Abstract-Conference.html
- Odpowiedź AdamW na lokalną interwencję treningową: https://arxiv.org/abs/2608.19762

Samo zastosowanie tych nazw/metod nie jest deklaracją nowości artykułu. Potencjalny wkład to zmierzona struktura korekt tokenów, ich osiągalność, wiarygodna prognoza wpływu na detekcję lub oszczędność kosztu pomiaru.
