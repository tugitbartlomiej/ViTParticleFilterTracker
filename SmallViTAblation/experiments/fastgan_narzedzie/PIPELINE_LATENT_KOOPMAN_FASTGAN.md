# Od cząstek latentu do obrazu wskazującego brak i do prawdziwych klatek

Data: 2026-09-24. Status: projekt metody, nie raport z wykonania całego pipeline’u.

Ten dokument zapisuje ustalenia rozmowy. Jego utworzenie nie uruchamia obliczeń, uczenia adaptera, generatora ani detektora. Zakres obecnej pracy to dokumentacja.

## 1. Cel

Chcemy ustalić, jak sieć opisuje oznaczone narzędzie w różnych pozycjach, pod różnymi kątami, na różnych tłach i przy różnym oświetleniu. Następnie chcemy śledzić rozwój tego opisu podczas uczenia. Jeżeli znajdziemy konkretną słabość, spróbujemy wygenerować obraz, który ją ujawnia. Obraz będzie zapytaniem do zbioru prawdziwych, oznaczonych klatek. Do ewentualnego dalszego treningu trafią prawdziwe klatki, nie obrazy generatora.

Łańcuch zależności:

**Stałe obrazy i maski → cząstki w kolejnych checkpointach → kierunki związane z narzędziem → sprawdzona prognoza zmian → mierzalny brak → sterowanie generatorem → weryfikacja obrazu → wyszukanie prawdziwych klatek.**

To nie jest założenie, że każdy brak można odwrócić do obrazu. Generator ma ograniczony zakres obrazów, a ten sam opis latentu może odpowiadać wielu obrazom. Sukces techniczny generowania nie dowodzi jeszcze przydatności znalezionej klatki do uczenia.

### Pierwsza wersja wykonawcza: wyłącznie Koopman

**Warunek początkowy: model we wczesnej fazie uczenia.** Zaczynamy od RF-DETR Nano wstępnie dostrojonego do narzędzia oraz zapisów kolejnych etapów tego samego treningu. Nie wymagamy modelu doprowadzonego do końca uczenia. „Wstępnie nauczony” oznacza tutaj, że model już wykrywa narzędzie na części przykładów, a jego reprezentacja nadal może się rozwijać. Mała architektura ani mała liczba epok nie gwarantują braku przeuczenia.

Aktualne ustalenie autora: wstępne dostrojenie trwa cztery epoki, z kilkoma zapisami w każdej epoce (proponowane pięć, w równych odstępach aktualizacji optymalizatora), wraz ze stanem początkowym. To etap startowy pętli, a nie wcześniejsza propozycja kalibracji z sufitem 20 epok. Mniejsze tempo uczenia pozostaje kandydatem wymagającym sprawdzenia konfiguracji. Przed wykonaniem sprawdzamy dostępne zasoby i manifest; sam zapis ustaleń nie jest raportem z uruchomienia.

Przed wyborem początkowego okna checkpointów sprawdzamy przebieg jakości na danych uczących i oddzielnych filmach walidacyjnych. Poprawa na treningu przy pogarszaniu walidacji jest sygnałem możliwego przeuczenia, a nie oczekiwanego domykania reprezentacji. W takim przypadku nie traktujemy dalszej trajektorii jako wzorca poprawy. Dobór okna i kryteria oceny ustalamy przed końcowym testem prognozy; późniejsze checkpointy pozostają do jej sprawdzenia. Nie zakładamy z góry, że konkretna liczba epok oznacza właściwy stan modelu. Sam pojedynczy checkpoint wystarcza do analizy odpowiedzi, ale nie do wyznaczenia historii uczenia.

Doprecyzowanie z rozmowy: pierwsza próba dotyczy prognozy w czasie uczenia i odtworzenia jej w obrazie. Bez członu pamięci Langevina, modelowania reszt, filtra Kalmana i uczenia adaptera. Korzystamy z istniejących checkpointów oraz zamrożonych generatora i detektora.

1. Na identycznych obrazach w kolejnych checkpointach mierzymy porównywalne odpowiedzi w wybranych kierunkach związanych z narzędziem.
2. Na wcześniejszej historii dopasowujemy wspólny model afiniczny Koopmana. Prognozę sprawdzamy na późniejszych checkpointach wobec braku zmiany i prostego trendu.
3. Dla obrazu kotwiczącego wyznaczamy prognozowany opis oraz przyrost: `delta_K = prognoza_K(c_k) - c_k`. Cel generatora to `c_cel = c_k + delta_K`, a nie sam przyrost. Wszystkie wielkości muszą być w tej samej ustalonej bazie.
4. Zmieniamy tylko wejście FastGAN, aby zamrożony detektor odczytał z wygenerowanego obrazu opis możliwie bliski celowi. Zachowujemy ograniczenia obecności narzędzia i realizmu z sekcji 9A. Prognoza musi dostarczać cel osiągalny przez generator; nie zakładamy tego z góry.
5. Porównujemy z generowaniem pod obecny opis, bez przyrostu, oraz pod losowy przyrost o tej samej długości. Osobno mierzymy osiągnięcie celu, poprawność obrazu i błędy detekcji. To sprawdza, co wnosi kierunek prognozy.
6. Udane obrazy mogą następnie służyć jako zapytania do prawdziwej puli treningowej według sekcji 11. Nie rozpoczynamy tym samym dalszego treningu detektora.

Pierwszy wynik ma odpowiedzieć, czy potrafimy przełożyć sprawdzony przyrost Koopmana na zmianę obrazu. Nie zakładamy, że taki obraz będzie trudny lub najlepszy do uczenia. To oddzielny wynik do zmierzenia. Jeżeli historia checkpointów nie wystarcza do wiarygodnej prognozy, raportujemy ograniczenie i nie zastępujemy jej operatorem zmian tła z poprzedniego pilota.

### Dłuższa historia uczenia przed przeuczeniem: propozycja do sprawdzenia

Przegląd `experiments/krystalizacja_latentu/README.md` potwierdza zawracanie straty walidacyjnej w okolicach epok 6–7 w trzech biegach z pełną telemetrią. Nie jest to uniwersalna granica: szczyt jakości w szerszym zestawie biegów występował w różnych epokach. Celem jest dłuższy odcinek użytecznych zmian, nie sztuczne zwiększenie numeru epoki.

Proponowany pierwszy wariant kalibracyjny pozostawia ten sam zbiór 1500 klatek, inicjalizację, batch, augmentacje i regularyzację, lecz zmniejsza oba tempa uczenia trzykrotnie względem rzeczywistego manifestu biegu odniesienia. W lokalnym protokole widnieją `lr=1e-4` i `lr_backbone=1e-5`; odpowiadałyby im około `3.33e-5` i `3.33e-6`. Przed uruchomieniem należy sprawdzić nadpisania profilu i manifest, a nie zakładać, że sam YAML opisuje wykonany bieg. To kandydat do próby, nie ustawienie o potwierdzonej przewadze. Nie zamrażamy backbone, bo chcemy obserwować zmiany jego latentu.

Niezależnie zagęszczamy zapis i pomiar stałej sondy: orientacyjnie pięć razy na epokę, w ustalonych odstępach aktualizacji optymalizatora. Zapis obejmuje krok, rzeczywisty learning rate i informację, czy analizowane są wagi zwykłe, czy EMA; nie mieszamy ich trajektorii. Krótszy odstęp nie zastępuje dostatecznie długiej i zróżnicowanej historii. W pierwszej próbie sufit 20 epok jest limitem kosztu, nie obietnicą poprawy przez cały ten czas. Pełną walidację wykonujemy co epokę, zapisując zarówno stratę, jak i jakość wykrywania.

Drugi wariant, jeśli potrzebny, rozszerza zbiór do około 3000 unikalnych, poprawnie oznaczonych klatek wyłącznie z operacji treningowych, zachowując dotychczasowe klatki. Nowe mają wnosić inne położenia, oświetlenie, zasłonięcia i tła; ograniczamy prawie identyczne sąsiednie klatki oraz dominację jednego filmu. Dostępność i jakość takiego podzbioru trzeba potwierdzić manifestem przed jego utworzeniem. Większy zbiór daje więcej aktualizacji na epokę, więc porównujemy również kroki optymalizatora, liczbę prezentacji obrazów i czas GPU. Nie zakładamy, że samo zwiększenie liczby klatek przesunie optimum do późniejszej epoki.

Najpierw zmieniamy tempo na stałym zbiorze, dopiero potem badamy rozszerzenie danych. To pozwoli rozdzielić ich wpływ. Kryterium sukcesu: późniejsze stabilne lub poprawiające się wyniki na osobnych operacjach oraz mierzalna, przewidywalna zmiana latentu. Płaski przebieg spowodowany zbyt małymi aktualizacjami nie jest sukcesem. Nie zwiększamy liczby epok przez usunięcie dobrych wag wstępnych ani przez celowe pogorszenie modelu.

Status: propozycja kalibracji, bez uruchomienia treningu. Zmniejszenie tempa nie gwarantuje usunięcia przeuczenia. Parametry obu części modelu opisuje [dokumentacja RF-DETR](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/); zastosowanie w lokalnej wersji należy potwierdzić w kodzie i manifeście.

## 2. Co już istnieje, a co dopiero proponujemy

Stan pilota sprawdzono w lokalnych dokumentach:

- [Plan próby](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/fastgan_narzedzie/PLAN_PROBY_TRUDNYCH.md).
- [Wyniki próby](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/fastgan_narzedzie/WYNIKI_PROBY_TRUDNYCH.md).

| Element | Stan |
|---|---|
| Wytrenowany FastGAN z wyjściem RGB i maską | Istnieje model EMA po 50 000 kroków. Maska generatora nie jest prawdą odniesienia. |
| Gradient przez zamrożony generator i detektor do wejścia generatora | Zrealizowany w pilocie zmiany otoczenia prawdziwego narzędzia. |
| Zachowanie prawdziwych pikseli narzędzia | Sprawdzone w pilocie, także po zapisaniu obrazów. |
| Odczyt tokenów z uwzględnieniem rzeczywistej maski | Zastosowany w pilocie. |
| Koopman opisujący rozjaśnienie otoczenia | Zastosowany, ale nie pokonał prognozy braku zmiany. |
| Koopman przewidujący rozwój reprezentacji przez checkpointy | Etap projektowany w tym dokumencie. To inny operator. |
| Pełny rozdział tożsamości, kąta, skali, światła i tła | Hipoteza do sprawdzenia. Nie jest wynikiem samego wyboru wolnych modów. |
| Sterowanie pod konkretny, zweryfikowany brak reprezentacji | Projektowane rozszerzenie. |
| Mały adapter z opisu latentu na wejście FastGAN | Projektowany, nie wytrenowany. |
| Automatyczny dobór prawdziwych klatek na podstawie wygenerowanych zapytań | Projektowany. |
| Pamięć Langevina i niepewność Kalmana dla historii uczenia | Opcjonalne hipotezy, nie wynik pilota. |

Pilot obejmował wycinki z 12 operacji. Oryginały dawały 8 trafień, najlepsze z 20 losowań dawało 3, sterowanie samą detekcją dawało 4, a sterowanie detekcją i Koopmanem dawało 2. Mniej trafień oznaczało trudniejsze obrazy, nie lepszą sieć. Ciągła miara trudności nie dawała przewagi wariantowi z Koopmanem. Jego błąd przewidywania wynosił 0,2555 wobec 0,2519 dla braku zmiany. Odczyt kierunku RDZEN z wybranych modów dawał R²=0,029.

Obrazy pilota zawierały rozmycia, przenikanie struktur i obwódki. Generator widział wcześniej operacje użyte w próbie. Wynik potwierdza możliwość sterowania wejściem generatora, ale nie potwierdza przewagi Koopmana, jakości samodzielnie wygenerowanych narzędzi ani poprawy uczenia.

## 3. Co nazywamy cząstką i kierunkiem

W analizowanej warstwie obraz daje siatkę 24 × 24 tokenów. Każdy z 576 tokenów jest wektorem 384 liczb. Cząstką nazywamy taki token związany z komórką siatki. Jego cecha zależy również od otoczenia poprzez działanie sieci. Nie jest to izolowany opis pikseli małego kwadratu.

W dotychczasowym zapisie profil tokena wyznaczamy z końcowej normalizacji backbone:

\[
r=(z-\beta)/|\gamma|,\qquad h=P^\top(r-\mu).
\]

P oznacza zamrożoną bazę PCA, obecnie redukującą opis do 32 współrzędnych. To wybór analizy, nie wymiar natywnego latentu. Trzeba sprawdzać, czy redukcja nie usuwa informacji potrzebnej do badanego zadania. Małe lub zerowe wartości gamma wymagają jawnego zabezpieczenia i raportowania odrzuconych współrzędnych.

Kierunek w jest wspólnym wektorem odczytu. Liczba wᵀh mówi, jak dany token odpowiada w tym kierunku. Kierunek nie jest jednym z tokenów siatki. Informacja o narzędziu może zajmować kilka kierunków, czyli podprzestrzeń, a nie jedną szczególną współrzędną.

Na nieruchomym obrazie obserwujemy zmianę opisu tej samej komórki przez checkpointy. Przy obrocie narzędzia komórki trzeba dopasować znaną transformacją i maską. Stały numer tokena nie oznacza śledzenia tego samego fragmentu metalu.

## 4. Stały zestaw kontrolowanych obrazów

Wybieramy obrazy z ręcznymi maskami z wielu operacji. Dla każdego przygotowujemy oryginał i kontrolowane warianty. Najpierw zmieniamy osobno położenie, kąt, skalę, pozorną długość, tło oraz oświetlenie. Później badamy kombinacje czynników. Zmiana pozornej długości nie musi odpowiadać rzeczywistej zmianie geometrii narzędzia; deformacje muszą być osobno oznaczone i ocenione.

Każdy wariant, jego maska i parametry przekształcenia pozostają identyczne we wszystkich checkpointach. Kontrola wklejenia na własne tło musi odtwarzać oryginał. Unikamy prostokątnego wklejania z obcym otoczeniem jako rzekomej zmiany samego narzędzia.

### Kontrolowany zestaw syntetyczny: narzędzie jak wskazówka zegara

Ustalenie z rozmowy: wycinamy prawdziwe narzędzie według ręcznej maski, bez prostokątnego fragmentu obcego tła. Ustalamy punkt zakotwiczenia wycinka w środku obrazu i obracamy narzędzie wokół tego środka jak wskazówkę zegara. Zapisujemy położenie kotwicy względem wycinka, kąty oraz macierze transformacji. Narzędzie i maska przechodzą tę samą transformację; kontrolujemy obcięcie narzędzia przez granice obrazu.

Osobno zmieniamy skalę całego narzędzia, jego pozorną długość wzdłuż osi oraz rzutowanie płaskiego wycinka. Te operacje mają oddzielne etykiety: rozciągnięcie nie jest tym samym co zmiana odległości od kamery, a transformacja perspektywiczna płaskiego obrazu nie odtwarza nowych widoków trójwymiarowego narzędzia. Nie tworzy też niewidocznych powierzchni ani właściwych im refleksów.

Stosujemy trzy rodzaje tła: jednolite kolory, w tym czarny i żółty; rzeczywiste tła z danych treningowych; oraz tła proponowane przez FastGAN i sprawdzone pod kątem dodatkowych narzędzi lub artefaktów. Jednolite tła tworzymy bezpośrednio, bez uczenia ani wymuszania ich na FastGAN. Zachowujemy prawdziwe piksele narzędzia z wyjątkiem jawnie określonej transformacji; zmiany oświetlenia narzędzia są osobnym czynnikiem. Nie zakładamy, że obecny generator potrafi niezawodnie tworzyć dowolnie wskazane tło.

Najpierw zmieniamy jeden czynnik naraz, przy pozostałych stałych, a następnie przygotowujemy wybrane kombinacje kąta, długości i tła. Gotowy zestaw, wraz z maskami i parametrami, zamrażamy i pokazujemy identycznie każdemu checkpointowi. Cel: znaleźć kierunki reagujące na poszczególne zmiany oraz kierunki zachowujące odczyt narzędzia mimo tych zmian. To zestaw diagnostyczny, nie syntetyczny zbiór do treningu detektora.

Kontrole obejmują samo tło bez narzędzia, wklejenie na własne tło oraz kontrolę brzegów maski. Czarne i żółte tła są celowym testem skrajnym; reakcja na nie nie dowodzi zachowania na filmach chirurgicznych. Znalezione zależności sprawdzamy na rzeczywistych, odłożonych operacjach oraz na wielu rzeczywistych wyglądach narzędzia, nie tylko na jednym wycinku.

Podział wykonujemy po operacjach:

1. Zbiór dopasowania kierunków, skal i operatora.
2. Zbiór kontroli ich jakości oraz ustalenia progów.
3. Odłożone operacje do sprawdzenia gotowej procedury.

Końcowy test detektora nie służy do wybierania klatek ani ustawiania metody. Jeżeli istniejące dane testowe były już używane do podejmowania decyzji, nie przedstawiamy ich ponownie jako nietkniętego testu.

Maska pozwala wskazać tokeny narzędzia i ich udział powierzchniowy. Nie gwarantuje, że ich cechy nie zawierają informacji o tle. Dlatego potrzebne są również kontrole: narzędzie na zmienionym tle, tło bez narzędzia oraz dostępne podobne obiekty. Przy jednej klasie nie deklarujemy rozróżniania wszystkich rodzajów narzędzi.

## 5. Historia reprezentacji podczas uczenia

Niech k oznacza checkpoint, i obraz źródłowy, a wariant, a p token. Rejestrujemy:

\[
h_{k,i,a,p}=H_{\theta_k}(x_{i,a})_p,
\]

oraz maskę, pozycję tokena, wynik detekcji i błąd względem adnotacji. H obejmuje określony odczyt i preprocessing. Wagi theta są parametrami sieci. Wektory h są wynikiem jej działania na obraz; nie istnieje przypisanie jednej wagi do jednego wektora.

W każdym checkpointcie stosujemy tryb ewaluacji, identyczne obrazy i deterministyczny preprocessing. Powtórzenie tego samego wejścia musi dawać różnice dużo mniejsze od badanych zmian. Zapisujemy dokładną warstwę, wagi normalizacji, bazę PCA i skróty checkpointów.

Porównywalność współrzędnych wymaga kontroli: beta i gamma zmieniają się z treningiem. Oddzielamy wpływ zmian normalizacji od zmian reprezentacji. Porównujemy ustaloną konwencję odniesienia z normalizacją właściwą checkpointowi. Nie dopasowujemy nowej PCA niezależnie w każdej epoce i nie traktujemy jej osi jako tych samych.

Kierunki można zamrozić na wcześniejszym checkpointcie albo dopasowywać ponownie i dopasować ich podprzestrzenie na obrazach treningowych. W drugim przypadku raportujemy także niedopasowane wyniki: wyrównanie nie może ukrywać rzeczywistego rozwoju. Podobieństwo CKA nie jest samo w sobie macierzą wyrównania.

## 6. Identyfikacja informacji związanej z narzędziem

### Nazwa własna: Pełny zbiór wektorów tożsamości narzędzia

Na życzenie autora tak nazywamy poszukiwany zestaw kierunków w przestrzeni latentu, które wspólnie opisują rozpoznawanie narzędzia oraz jego postać w badanych pozycjach, kątach, skalach, pozornych długościach, tłach i oświetleniu. Obejmuje on zarówno odpowiedzi zachowujące rozpoznanie przy zmianie warunków, jak i odpowiedzi opisujące te zmiany. Nie ograniczamy go do jednego kierunku ani do niezmienników Koopmana.

„Wektory” w tej nazwie oznaczają kierunki odczytu wspólne dla tokenów, nie wybrany podzbiór komórek obrazu. Kierunki mogą wspólnie kodować kilka właściwości i nie muszą mieć jednoznacznej bazy. Sam kierunek opisujący tło nie staje się przez to wektorem tożsamości; sprawdzamy jego związek z rozpoznawaniem narzędzia przy kontrolowanych zmianach tła.

„Pełny” jest nazwą celu badania. Otrzymany zestaw uznajemy za wystarczający dopiero względem jawnie określonych warunków, miar i tolerancji na odłożonych danych, z kontrolą wpływu na odpowiedź detektora. Dopóki tego nie sprawdzimy, raportujemy go jako kandydata na Pełny zbiór wektorów tożsamości narzędzia, bez deklaracji kompletności dla wszystkich możliwych obrazów.

Szukamy kierunków lub podprzestrzeni, z których można odczytać obecność narzędzia i jego właściwości na innych operacjach. Do obecności używamy prostego klasyfikatora; do geometrii regresji. Kąt reprezentujemy przez sinus i cosinus, z właściwą okresowością: oś bez wskazanej końcówki jest czym innym niż kierunek końcówki.

Oprócz średniej odpowiedzi wewnątrz maski zachowujemy rozkład przestrzenny. Sama średnia może zgubić zwrot narzędzia. Przykładowe odczyty dla kierunku w_j to:

\[
c_j^{(0)}=\frac{\sum_p m_p w_j^\top h_p}{\sum_p m_p},\qquad
c_j^{(1)}=\frac{\sum_p m_p u_p w_j^\top h_p}{\sum_p m_p}.
\]

m_p jest udziałem maski, a u_p położeniem względem ustalonej osi narzędzia. Przy braku wiarygodnego zwrotu osi nie interpretujemy znaku momentu jako kierunku końcówki.

Odczyty sprawdzamy pod kątem odporności na tło, reakcji na obrót i związku z poprawną detekcją. Dobra sonda pokazuje dostępność informacji, nie dowodzi jej używania przez detektor. Dlatego osobno usuwamy wybraną składową lub zachowujemy tylko wybraną podprzestrzeń i przepuszczamy zmieniony latent przez dalszą część detektora. Porównujemy z losowymi podprzestrzeniami o tej samej liczbie wymiarów i porównywalnej energii. Interwencje mogą tworzyć nienaturalny latent; sprawdzamy także łagodne tłumienie i podmianę na aktywacje z dopasowanego prawdziwego obrazu.

Redundancja reprezentacji oznacza, że brak efektu wyłączenia jednego kierunku nie dowodzi jego nieistotności. Zachowanie dotychczasowych odpowiedzi detektora nie oznacza poprawności tych odpowiedzi względem etykiet. Mierzymy obie rzeczy.

## 7. Koopman w czasie uczenia

Łączymy wybrane odczyty w wektor c_k dla tego samego obrazu i wariantu. Pierwszy model jest wspólny dla kierunków:

\[
c_{k+1}=A c_k+b+e_k.
\]

Nie dopasowujemy osobnego prawa do każdej liczby bez sprawdzenia sprzężeń. Model afiniczny można zapisać liniowo po dodaniu stałej współrzędnej. Bardziej złożone funkcje obserwacji lub deep Koopman wymagają osobnego uzasadnienia przewagi nad tym punktem odniesienia.

Formalnie pełny stan uczenia s_k obejmuje wagi, stan optymalizatora i potrzebne informacje o harmonogramie oraz danych. Latent ustalonego obrazu jest obserwacją g(s_k). Koopman opisuje ewolucję takich obserwacji. Dla losowego treningu prognoza może dotyczyć warunkowej średniej. Nie wynika z tego, że mały wektor c_k sam tworzy zamknięty, autonomiczny stan.

Dziesięć checkpointów daje tylko dziewięć przejść czasowych na obraz. Wiele obrazów pomaga dopasować wspólną regułę przy odpowiednich założeniach, lecz nie zastępuje długiej historii uczenia. Zaczynamy od małego wymiaru, regularyzacji i ograniczonego rzędu. Zachowujemy przyszłe checkpointy do kontroli prognozy. Nie dopasowujemy kierunków ani operatora z użyciem checkpointów, które udajemy, że przewidujemy.

Porównania obowiązkowe: brak zmiany, prosty trend, operator z potasowanymi przejściami oraz prognoza bez wyrównania i po uzasadnionym wyrównaniu. Sprawdzamy prognozę wielokrokową oraz jakość rozpoznawania, nie tylko błąd wektorów. Dla wartości własnej lambda bliskiej +1 odpowiedź zmienia się powoli. Nie jest to certyfikat tożsamości ani pełnego wytrenowania. Rosnąca odpowiedź nie musi oznaczać poprawy.

### Opcjonalna pamięć i niepewność

Jeżeli obecna odpowiedź nie wystarcza, sprawdzamy opóźnienia oraz dyskretny model z pamięcią inspirowany rozszerzonym Langevinem:

\[
c_{k+1}=A c_k+b+\sum_{\ell=1}^{L}B_\ell c_{k-\ell}+Uu_k+\eta_k.
\]

u_k może opisywać znany harmonogram uczenia. W tym doświadczeniu na stałych obrazach nie jest ruchem kamery. Ruch kamery dotyczy osobnego modelu czasu filmu. Pozostałe człony autorskiego równania wymagają jawnego przypisania mierzalnych wielkości przed dopasowaniem.

Interpretacja Mori–Zwanziga uzasadnia rozważenie pamięci po pominięciu części stanu. Nie dowodzi, że konkretne dopasowane B są dokładnym jądrem fizycznym. Reszta może zawierać pominiętą informację, błąd modelu i zmienność treningu; nie nazywamy jej od razu szumem nieusuwalnym.

Filtr Kalmana lub jego rozszerzenie może łączyć prognozę z kolejnym odczytem i estymować niepewność. Kowariancje wymagają kalibracji na rzeczywistych błędach prognoz. Mała deklarowana niepewność filtra nie jest dowodem dobrej kalibracji. Dodatki przyjmujemy tylko po przewadze na odłożonej historii.

## 8. Co dokładnie oznacza brak

Nie istnieje ogólna skala, według której aktywacja 0,6 oznacza wytrenowanie w 60 procentach. Skalę kierunku można zmienić bez zmiany funkcji całej sieci. Definiujemy brak operacyjnie: w określonych warunkach występuje błąd rozpoznawania, a zmiana określonego opisu jest powiązana z jego usunięciem w kontrolowanym porównaniu.

Źródłem docelowego opisu c* może być poprawnie rozpoznany dopasowany przypadek lub zweryfikowana reprezentacja lepszego checkpointu. Prognoza Koopmana jest kandydatem na przyszły opis, nie automatycznie poprawnym celem. Sieć może nadal się zmieniać i jednocześnie pogarszać wyniki.

Pomocnicza wielkość różnicy:

\[
\delta=c^*-c_k,\qquad D=\sum_j \frac{a_j\delta_j^2}{s_j^2+v_j+\varepsilon}.
\]

s_j to skala odpowiedzi wyznaczona na danych dopasowania, v_j niepewność różnicy w tych samych jednostkach, a a_j waga kierunku uzasadniona testem związku z błędem. D jest ważoną różnicą, nie liczbą brakujących epok ani procentem niewiedzy. Duża niepewność osłabia interpretację prognozy. Cel i skale muszą być ustalone przed oceną odłożonych przykładów.

Checkpoint referencyjny, jeśli użyty, musi rzeczywiście poprawiać rozpoznawanie w badanych warunkach. Późniejszy checkpoint użyty retrospektywnie jest kontrolą z dostępem do przyszłości; nie wolno przedstawiać go jako dostępnego podczas wcześniejszej selekcji online.

## 9. Sterowanie FastGAN: dwa różne zadania

Niech G(xi) oznacza zamrożony generator, a F_theta(x) wybrany odczyt latentu zamrożonego detektora. Optymalizujemy wejście xi, nie wagi modeli. Gradient przepływa przez oba modele. Zamrożenie parametrów nie oznacza wyłączenia zapisu gradientu całego przejścia.

### A. Odtworzenie zadanego opisu

\[
\xi^*=\arg\min_\xi\;\|S^{-1}(F_\theta(G(\xi))-c^*)\|^2
+\lambda_p R_{prior}(\xi)+\lambda_a R_{anchor}(G(\xi)).
\]

S skaluje współrzędne. R_prior ogranicza odejście od typowych wejść generatora. R_anchor utrzymuje wymagane podobieństwo do obrazu kotwiczącego, geometrię lub obecność narzędzia. Dobieramy kilka startów, stały szum warstw i ograniczony budżet. Raportujemy także nieudane próby oraz minimalną osiągalną różnicę.

To zadanie pokazuje obraz o zadanym opisie. Nie oznacza znalezienia obrazu trudnego. Dopasowanie obecnego detektora do kodu dobrze rozpoznanego narzędzia może wręcz dać łatwy obraz. Nie podajemy samej delty jako pełnego kodu obrazu: potrzebna jest kotwica, cel i ograniczenia.

### B. Wygenerowanie obrazu ujawniającego słabość

Tu wymagamy zachowania poprawnego narzędzia i szukamy błędu obecnego modelu związanego z wybranym opisem. Przykładowy cel z wiarygodną referencją:

\[
\max_\xi\; L_{det}(\theta,x_\xi,y_\xi)
+\alpha\|S^{-1}(F_{ref}(x_\xi)-F_\theta(x_\xi))\|_W^2
-\lambda R_{valid}(x_\xi,\xi).
\]

W wybiera wcześniej zbadane kierunki. Referencja musi poprawnie rozpoznawać ten sam przykład, a jej współrzędne muszą być porównywalne. Sama rozbieżność dwóch sieci nie mówi, która ma rację. Bez referencji używamy prawdziwych etykiet i kontrolowanych przekształceń oraz celu wymuszającego konkretny, uprzednio zweryfikowany wzorzec błędu. Nie maksymalizujemy dowolnej reszty Koopmana jako rzekomej wartości próbki.

y_xi musi pochodzić z zachowanej prawdziwej maski lub jej znanej transformacji. Własna maska generatora nie wystarcza do uznania obrazu za poprawnie oznaczony. Zamiast nieciągłego progu trafienia do gradientu stosujemy różniczkowalną stratę z jawnym dopasowaniem ramek. Twarde trafienia mierzymy dopiero w ocenie.

W istniejącym wariancie tła:

\[
x_\xi=M\odot x_0+(1-M)\odot[(1-\rho)x_0+\rho B(G(\xi))].
\]

M chroni narzędzie i ustalony margines, B wygładza tło. Ten wariant nie zmienia kąta, długości ani oświetlenia zachowanych pikseli metalu. Takie czynniki wymagają osobnej kontrolowanej transformacji obrazu i maski albo zweryfikowanego generowania geometrii. Nie obiecujemy ich uzyskania poprzez istniejący mechanizm podmiany tła.

Koopman może wskazać, którą odpowiedź warto badać i z jaką niepewnością. Gradient steruje obrazem. Prognoza dla jednej kotwicy nie jest automatycznie prognozą rozwoju sieci dla każdego nowo wygenerowanego obrazu.

## 10. Opcjonalny adapter: bezpośrednie przejście z opisu do wejścia generatora

Po sprawdzeniu pętli optymalizacji można wytrenować małą sieć:

\[
\xi=A_\psi(c^*,q,\nu),
\]

gdzie q opisuje kotwicę i wymagane warunki, a nu daje różnorodność. Generator i detektor pozostają zamrożone; uczymy tylko psi. Dane do adaptera mogą pochodzić z par wejście generatora–odczyt detektora oraz z udanych wyników optymalizacji.

Podstawowa strata sprawdza cykl: zadany opis → adapter → generator → detektor → odzyskany opis. Dodajemy ograniczenia realizmu i zachowania warunków. Nie polegamy wyłącznie na błędzie między przewidzianym xi a jednym zapisanym xi: wiele wejść może spełniać ten sam cel, a ich średnia może być zła. Adapter może inicjalizować krótką dalszą optymalizację.

To bezpośrednie przełożenie w sensie wyuczonej funkcji warunkowej, nie jednoznaczny odwrotny enkoder. Adapter nie stworzy obrazów poza możliwościami generatora. Po zmianie checkpointu detektora ponownie sprawdzamy jego trafność albo jawnie uwzględniamy checkpoint w modelu. Uczenie adaptera jest osobnym, jeszcze nieuruchomionym etapem.

## 11. Wyszukiwanie prawdziwych klatek

Aktualne ustalenie autora: wyszukujemy przypadki podobne wizualnie do obrazu FastGAN. Nie wymagamy podobnej odpowiedzi w kierunkach badanego detektora. Opisy do wyszukiwarki liczy osobny, zamrożony DINOv2, niezależny od dostrajanych checkpointów RF-DETR. To nadal wektorowy opis obrazu, ale nie analizowany latent detektora ani jego mody Koopmana.

### Przygotowanie na początku: indeks FAISS

Jeszcze przed generowaniem zapytań tworzymy indeks wyłącznie z dozwolonej, oznaczonej puli treningowej. FAISS jest indeksem opisów zdjęć, a nie nowym zbiorem zdjęć ani modelem uczonym na etykietach. Walidacja i końcowy test nie trafiają do indeksu.

1. Zapisujemy manifest klatek: identyfikator, operacja, czas w filmie, ścieżka, adnotacja i współrzędne wycinka. Dla obecnego generatora wycinków indeksujemy porównywalne wycinki narzędzia z otoczeniem według stałej reguły marginesu; nie porównujemy małego wycinka z opisem całej klatki.
2. Jednorazowo liczymy opisy zamrożonym DINOv2 przy stałym preprocessingu. Zachowujemy orientację i proporcje obrazu. Nie obracamy narzędzi do wspólnej osi, ponieważ kąt ma pozostać kryterium podobieństwa. Zapisujemy także metadane pierwotnej skali i pozycji, które samo kadrowanie może usunąć.
3. Normalizujemy opisy do długości jednostkowej i zapisujemy je wraz z mapowaniem na oryginalne zdjęcia. W pierwszej wersji używamy dokładnego indeksu FAISS `IndexFlatIP`: iloczyn skalarny znormalizowanych opisów odpowiada podobieństwu kosinusowemu. Przy tej skali puli nie zakładamy potrzeby przybliżonego indeksu ani osobnego GPU do wyszukiwania.
4. Zapisujemy wersję modelu, preprocessing, manifest i skróty danych. Zmiana opisującego modelu lub kadrowania wymaga przeliczenia odpowiednich opisów. Samo dostrajanie RF-DETR nie zmienia tego indeksu.

### Zapytanie wygenerowanym obrazem

Dla obrazu FastGAN liczymy jeden opis tym samym niezależnym DINOv2, a FAISS zwraca najbliższe kandydatury. Początkowy ranking opiera się na:

\[
s(x,x_q)=\frac{E(x)^\top E(x_q)}{\|E(x)\|\,\|E(x_q)\|}.
\]

E jest niezależnym opisem obrazu, nie funkcją F_theta używaną do sterowania generatorem. DINOv2 może traktować obrazy jako podobne mimo różnic kąta czy światła, dlatego jego ranking jest listą kandydatów, nie gwarancją szczegółowej zgodności wyglądu.

Tylko na krótkiej liście kandydatów można wykonać dokładniejsze porównanie LPIPS oraz geometrii, koloru i oświetlenia. LPIPS porównujemy przy zgodnym zakresie wycinków i bez usuwania obrotu, który chcemy zachować. Sprawdzamy oddzielnie narzędzie i otoczenie, aby podobne tło nie zdominowało innego kształtu narzędzia. Maska generatora pozostaje przybliżeniem, więc nie traktujemy opartych na niej pomiarów geometrii jako prawdy odniesienia. Sposób łączenia miar kalibrujemy na ocenach podobieństwa, nie na błędach RF-DETR.

Wynik to obraz zapytania obok grupy najbardziej podobnych prawdziwych klatek, z punktacją i odnośnikami do pełnych obrazów. Ograniczamy prawie identyczne sąsiednie klatki oraz dominację jednego filmu. Jeśli żadna kandydatura nie jest dostatecznie podobna według skalibrowanego kryterium, raportujemy brak trafienia; zawsze istniejący najbliższy sąsiad nie musi być dobrym odpowiednikiem.

Ocena detektora na znalezionych klatkach jest późniejszym, osobnym pomiarem przydatności. Nie zmienia definicji podobieństwa wizualnego i nie jest obowiązkowym filtrem wyszukiwania. Do ewentualnego treningu trafiają oryginalne, oznaczone klatki. Obrazy FastGAN pozostają zapytaniami.

Źródła: [DINOv2 — karta modelu, wyszukiwanie obrazów](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [FAISS](https://github.com/facebookresearch/faiss), [LPIPS](https://richzhang.github.io/PerceptualSimilarity/). To projekt etapu wyszukiwania; indeks nie został jeszcze utworzony.

## 12. Schemat algorytmu i warunki przejścia

```text
Ustal operacje do dopasowania, kalibracji i niezależnej oceny.
Przygotuj niezależne opisy DINOv2 i indeks FAISS puli treningowej.
Zapisz stałe obrazy, warianty, prawdziwe maski i transformacje.
Dla każdego dostępnego checkpointu:
    zarejestruj tokeny, odczyty kierunków i błędy detekcji;
    sprawdź powtarzalność i porównywalność współrzędnych.
Wyznacz i sprawdź podprzestrzenie narzędzia oraz warunków obrazu.
Dopasuj prognozę po wcześniejszych checkpointach.
Oceń ją na późniejszych checkpointach i wobec prostych kontroli.
Jeżeli nie przewiduje lepiej, nie używaj jej jako źródła przyszłego celu.
Wybierz brak potwierdzony błędem na oznaczonych obrazach.
Zdecyduj: wizualizacja kodu czy obraz ujawniający ten brak.
Optymalizuj wyłącznie wejście zamrożonego generatora.
Odrzuć obrazy z utratą narzędzia, błędną maską lub artefaktami.
Wyszukaj wizualnie podobne prawdziwe klatki przez DINOv2 i FAISS.
Oceń podobieństwo; osobno zmierz zachowanie detektora na tych klatkach.
Zapisz kandydatury oraz porównania bez generatora i bez Koopmana.
Przed ewentualnym treningiem detektora przygotuj osobny plan.
```

Warunki oceny ustalamy na zbiorze kalibracji przed końcowym porównaniem. Raport musi oddzielać:

- trafność identyfikacji informacji w latencie;
- wpływ tej informacji na decyzję detektora;
- błąd i niepewność prognozy przez checkpointy;
- osiągalność docelowego kodu przez generator;
- realizm i zgodność obrazu z maską;
- występowanie tego samego problemu w prawdziwych klatkach;
- ewentualną późniejszą poprawę uczenia.

W porównaniach sterowania zachowujemy te same kotwice, starty i zakres zmian. Raportujemy zarówno liczbę prób, jak i czas GPU, ponieważ krok gradientowy kosztuje więcej od pojedynczego losowania. Kontrole obejmują losowanie, samą stratę detekcji, losowe kierunki, wariant bez prognozy oraz bez generatora. Wyniki grupujemy po operacjach, a nie traktujemy sąsiednich klatek jako niezależnych powtórzeń.

### Walidacja „checkpoint — checkpoint prim”: kierunek i dodatkowy wpływ klatki

Ustalenie autora: porównujemy zwykły checkpoint końcowy z checkpointem prim, uzyskanym po dodaniu znalezionej prawdziwej klatki. Obie gałęzie wychodzą z tego samego wcześniejszego checkpointu. Oznaczenia dotyczą zapisów modelu, niekoniecznie całych epok.

- `C_k`: wspólny checkpoint początkowy, np. trzeci zapis.
- `C_{k+1}`: kontynuacja bez nowej klatki, np. czwarty zapis.
- `C'_{k+1}` (checkpoint prim): kontynuacja z tego samego `C_k`, z dodaną klatką, do porównywalnego punktu końcowego.

Jeśli zwykły przebieg został już wykonany, można wrócić do wcześniejszego zapisu i wykonać tylko gałąź prim, o ile zapis i metadane pozwalają odtworzyć warunki. Potrzebne są wagi, stan optymalizatora, harmonogram tempa uczenia, licznik aktualizacji, stan generatorów losowych, sposób próbkowania danych oraz — jeżeli używane — stan skalera precyzji i EMA. Same wagi `best.pth` nie gwarantują odtworzenia kontynuacji. Gdy brak pełnego stanu, obie gałęzie trzeba uruchomić według wspólnej, jawnej reguły restartu; stary wynik pozostaje wtedy tylko pomocniczym odniesieniem.

Prognozę Koopmana, bazę kierunków, normalizację i wybór klatki zamrażamy z użyciem wyłącznie informacji dostępnej do `C_k`. Nie używamy rzeczywistego `C_{k+1}` do wybrania klatki, która ma wywołać przewidzianą zmianę. Dopuszczalność metody można wcześniej kalibrować na odrębnych przebiegach lub wcześniejszych oknach czasowych.

Klatkę pobieramy z dozwolonej puli treningowej spoza dotychczas użytego podzbioru. Indeks FAISS może obejmować całą pulę, ale zapytanie filtruje już użyte identyfikatory. Ponowne częstsze pokazywanie znanej klatki jest osobnym wariantem. Obie gałęzie mają ten sam budżet aktualizacji i rozmiar paczek. Ustalamy i zapisujemy liczbę prezentacji nowej klatki oraz zmienione miejsca w strumieniu próbek; pozostałe losowania i augmentacje parujemy, gdzie to możliwe. W takim porównaniu mierzymy efekt dodania klatki przy stałym budżecie, łącznie z zastąpieniem części wcześniejszych prezentacji. Pełna epoka po powiększeniu zbioru może mieć inną liczbę kroków — tej różnicy nie ukrywamy pod wspólną nazwą epoki.

Po obu kontynuacjach ponownie pokazujemy dokładnie te same obrazy diagnostyczne. Dla każdego obrazu, we wspólnej zamrożonej bazie Pełnego zbioru wektorów tożsamości narzędzia, zapisujemy:

```text
c_start     = odpowiedź w C_k
c_koniec    = odpowiedź w C_{k+1}
c_prim      = odpowiedź w C'_{k+1}
delta_K     = prognoza Koopmana dla ustalonego horyzontu - c_start
delta_bez   = c_koniec - c_start
delta_z     = c_prim - c_start
efekt_klatki = c_prim - c_koniec
```

`delta_bez` porównana z `delta_K` ocenia trafność prognozy kontynuacji bez interwencji. `delta_z` pokazuje całkowitą zmianę po uczeniu z nową klatką. Dopiero `efekt_klatki` oddziela ją od kontynuacji bez dodania. Model Koopmana dopasowany do starego doboru danych nie jest automatycznie prognozą nowego doboru.

Raportujemy błąd prognozy, długość obu rzeczywistych przyrostów, ich podobieństwo kierunkowe do `delta_K` oraz rzut `efekt_klatki` na kierunek `delta_K`. Dodatni rzut oznacza dodatkową zmianę w przewidywaną stronę; ujemny — przeciwną. Osobno mierzymy, czy checkpoint prim zbliżył się do ustalonego celu bardziej niż zwykły checkpoint, bo zgodny kierunek może również przestrzelić cel. Dla przyrostów bliskich zeru nie interpretujemy niestabilnego kąta; podajemy normy i błąd bezwzględny. Skale współrzędnych ustalamy przed rozgałęzieniem.

Wyniki pokazujemy per obraz, warunek (kąt, długość, tło, światło) i operacja, a obok mierzymy jakość wykrywania na osobnych filmach. Nie uznajemy zgodnego ruchu latentu za automatyczną poprawę rozpoznawania. Jedna para gałęzi jest wynikiem pilota; mocniejszy wniosek wymaga sparowanych powtórzeń. Dla oceny jakości samego wyboru potrzebna jest także kontrola z losowo dodaną klatką przy tej samej liczbie jej prezentacji.

Artefakty walidacji: oba końcowe checkpointy i wspólny początek, pełne stany wznawiania, manifest zmiany danych, zamrożona prognoza, zapis odpowiedzi na stałych obrazach, tabela przyrostów oraz wykres strzałek `delta_K`, `delta_bez` i `delta_z` w tej samej bazie. Pełne miary liczymy w całej badanej podprzestrzeni, nie tylko na wizualizacji dwuwymiarowej. Status: zaplanowana walidacja, bez uruchomienia treningu.

### Końcowe porównanie checkpointu i checkpointu prim na zbiorze testowym

Na życzenie autora porównanie latentu uzupełniamy oceną rozpoznawania narzędzia: oba końcowe checkpointy oceniamy na dokładnie tych samych rzeczywistych filmach testowych, z tym samym preprocessingiem, ewaluatorem, etykietami i ustawieniami inferencji. Test nie uczestniczy w uczeniu, indeksie FAISS, ustalaniu kierunków, prognozie Koopmana, generowaniu celów ani wyborze dodatkowej klatki.

Przed otwarciem wyników testowych ustalamy główną miarę jakości ramek `mAP@0.5:0.95`. Dodatkowo raportujemy `AP@0.5`, odsetek pominiętych narzędzi, fałszywe alarmy na klatkę oraz IoU dopasowanych ramek razem z pokryciem detekcjami, aby poprawa IoU tylko na łatwych trafieniach nie ukrywała pominięć. Do miar wymagających progu pewności używamy wspólnego progu ustalonego na walidacji; próg dopasowania IoU również ustalamy wcześniej. Nie stroimy progów na teście.

Dla każdej miary zapisujemy wynik zwykłego checkpointu, wynik checkpointu prim i różnicę. Niepewność różnicy szacujemy przez sparowane losowanie całych filmów/operacji, zachowując te same wylosowane grupy dla obu modeli. To nie zastępuje powtórzeń treningu z różnymi ziarnami; pojedyncza para pozwala ocenić tę konkretną kontynuację, a nie gwarantuje przewagi metody ogólnie. Raport musi dopuszczać poprawę, pogorszenie i wynik nierozstrzygający.

Zmiana latentu w przewidywanym kierunku i poprawa detekcji na teście są dwoma osobnymi wynikami. Zgodność z Koopmanem nie rozstrzyga z góry, który checkpoint będzie lepszy.

Jeżeli procedurę wielokrotnie poprawiamy i ponawiamy dobór klatek, bieżące decyzje opieramy na walidacji, a końcowy test otwieramy dla ustalonego porównania. Filmy, których wyniki wcześniej wykorzystano do zmiany metody, nie są ponownie nazywane nietkniętym testem; ich historię ujawniamy i w miarę dostępności odkładamy nowe operacje. Zapis tej sekcji nie uruchamia oceny ani treningu.

### Powtarzana pętla doboru klatek i dalszego uczenia

Po pierwszym porównaniu powtarzamy procedurę na kolejnych etapach uczenia. Wersja pierwsza korzysta wyłącznie z prognozy Koopmana; Langevin, Kalman i uczony adapter pozostają poza nią.

0. Jednorazowo przygotowujemy stały zestaw diagnostyczny, podziały operacji, niezależny indeks DINOv2–FAISS całej dozwolonej puli treningowej i historię czterech epok wstępnego dostrojenia z gęstymi checkpointami.
1. Na początku rundy zapisujemy przyjęty checkpoint, pełny stan wznawiania, aktualny podzbiór treningowy i listę jeszcze niewykorzystanych klatek.
2. Na stałych obrazach mierzymy historię odpowiedzi w Pełnym zbiorze wektorów tożsamości narzędzia. Prognozę Koopmana dopasowujemy tylko do dostępnej historii. Ustalamy cel jako obecny opis plus prognozowany przyrost dla horyzontu następnej kontynuacji.
3. Optymalizujemy wejście zamrożonego FastGAN przez zamrożony bieżący detektor, aby zbliżyć odpowiedź na wygenerowany obraz do celu. W pierwszej wersji zaczynamy od losowego wejścia; inwersja prawdziwego obrazu jest opcjonalnym, osobno ocenianym wariantem. Ustalamy limit prób i regułę zakończenia. Osiągnięcie tego limitu nie oznacza znalezienia globalnie najlepszego obrazu.
4. Sprawdzamy jakość obrazu i wyszukujemy wizualnie podobne prawdziwe przypadki w indeksie FAISS. Filtrujemy już wykorzystane klatki. Gdy brak dostatecznie podobnego poprawnego przykładu, zapisujemy brak trafienia zamiast wymuszać dodanie dowolnego sąsiada.
5. Wybraną prawdziwą, oznaczoną klatkę dodajemy do dotychczasowego zestawu. Z jednego stanu początkowego wykonujemy kontynuację bez dodania i kontynuację z dodaniem, uzyskując checkpoint oraz checkpoint prim. Reguła liczby prezentacji i budżet aktualizacji są jawne, zgodnie z sekcją walidacji. Domyślny horyzont to budżet aktualizacji jednej epoki dotychczasowego zbioru, identyczny dla obu gałęzi.
6. Ponownie mierzymy stały zestaw diagnostyczny i porównujemy przyrost przewidziany, przyrost bez dodania, przyrost z dodaniem oraz ich różnicę. Równolegle porównujemy detekcję na walidacji, nie na końcowym teście używanym do podejmowania decyzji.
7. Przed rozpoczęciem pętli ustalamy regułę przyjęcia: kandydatem do kontynuacji jest checkpoint prim, jeżeli poprawia główną miarę walidacji ponad ustaloną tolerancję i nie przekracza dopuszczalnego pogorszenia miar pomocniczych. W przeciwnym razie kontynuujemy gałąź bez dodania, a próbę zachowujemy w raporcie. Progi ustalamy przed rundami; nie dopasowujemy ich po obejrzeniu wyniku. Nieudane próby nie znikają z oceny skuteczności metody.
8. Dla przyjętej gałęzi ponawiamy cały cykl. Aktualizujemy historię Koopmana, ponieważ zmiana zbioru może zmienić dynamikę uczenia. Nie sklejamy dwóch alternatywnych gałęzi w jeden szereg czasowy. Kierunki i bazę zamrażamy w obrębie każdego porównania; ich ewentualną aktualizację między rundami wersjonujemy i kontrolujemy porównywalność. Niezależny indeks FAISS pozostaje ten sam, zmienia się filtr wykorzystanych klatek.

Przed uruchomieniem określamy maksymalną liczbę rund, budżet GPU, tolerancje walidacji i liczbę kolejnych nieudanych prób, po której kończymy. Przerywamy też, gdy prognoza przestaje pokonywać proste kontrole, nie udaje się wygenerować poprawnego obrazu lub brak odpowiednich prawdziwych kandydatur. Nie zakładamy nieograniczonej poprawy ani możliwości całkowitego wyczerpania braków modelu.

Końcowy test służy ocenie ustalonej procedury i zapisanych modeli po zakończeniu decyzji. Jeśli porównujemy na nim checkpointy z wielu rund, listę porównań ustalamy wcześniej i nie wykorzystujemy wyników do kolejnej selekcji. Bieżąca walidacja, używana wielokrotnie do podejmowania decyzji, nie jest bezstronnym końcowym oszacowaniem jakości. Dla oceny całej pętli utrzymujemy również referencyjny przebieg bez selekcji oraz dobór losowy przy porównywalnym budżecie; pojedyncze porównania checkpoint–prim nie zastępują tej oceny.

W każdej rundzie zapisujemy identyfikatory gałęzi i checkpointu rodzica, prognozę sprzed interwencji, wejście generatora i obraz, ranking FAISS, wybraną klatkę, manifest danych i prezentacji, oba przyrosty latentu, wyniki walidacji oraz decyzję o kontynuacji. Ten dokument opisuje pętlę; żaden z tych treningów nie został uruchomiony w ramach jego redakcji.

## 13. Artefakty przyszłego wykonania

Przyszła implementacja powinna zapisywać manifest obrazów i masek, parametry przekształceń, identyfikatory checkpointów, bazę współrzędnych, definicje kierunków, podziały operacji, prognozy z niepewnością i wyniki prostych kontroli. Dla każdej generacji: kotwicę, definicję braku, funkcję celu, początkowe i końcowe wejście generatora, przebieg strat, obrazy, kontrolę maski oraz listę znalezionych prawdziwych klatek.

Ten dokument nie tworzy tych artefaktów ani nie potwierdza istnienia całej implementacji. Najmniejszy sensowny następny etap to sprawdzenie porównywalnej historii odpowiedzi na stałych obrazach. Dopiero dobra prognoza uzasadnia sterowanie generacją według przewidywanego rozwoju, zamiast według bieżącego błędu.

## 14. Podstawy literaturowe i granice analogii

1. [Dogra i Redman, Optimizing Neural Networks via Koopman Operator Theory, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/169806bb68ccbf5e6f96ddc60c40a044-Paper.pdf). Modelują historię wag i zastępują część aktualizacji prognozą. U nas obserwacją byłyby odpowiedzi latentu na stałe obrazy. Praca nie identyfikuje semantycznych kierunków narzędzia.
2. [Manojlović i in., Applications of Koopman Mode Analysis to Neural Networks](https://arxiv.org/abs/2006.11765). Analiza dynamiki i zbieżności sieci. Widmo nie nadaje samo znaczenia semantycznego modom.
3. [Raghu i in., SVCCA](https://arxiv.org/abs/1706.05806) oraz [Kornblith i in., Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). Podstawa porównywania reprezentacji; podobieństwo nie zastępuje pomiaru poprawności rozpoznawania.
4. [Kim i in., TCAV](https://proceedings.mlr.press/v80/kim18d/kim18d.pdf). Kierunki pojęć i wrażliwość odpowiedzi modelu. Uzasadnia badanie kierunków odczytu, ale nie gwarantuje wyłącznego ani przyczynowego kodowania tożsamości.
5. [Yeh i in., On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://arxiv.org/abs/1910.07969). Kompletność opisu w odniesieniu do zachowania modelu. Nie oznacza, że model jest bezbłędny lub w pełni wytrenowany.
6. [SEIS, preprint 2026](https://arxiv.org/abs/2602.04054). Badanie niezmienności i ekwiwariancji reprezentacji, także w trakcie uczenia. Wspiera pomiary na obrazach i przekształceniach, nie gwarantuje prognozy ani jednoznacznej identyfikacji wektorów.
7. [When Do Concepts Become Functionally Sufficient During Language-Model Training?, preprint 2026](https://arxiv.org/abs/2608.15323). Wystarczalność funkcjonalna reprezentacji przez etapy uczenia. Analogia dotyczy sposobu pomiaru; wyników językowych nie przenosimy bez testu na detektor.
8. [Probing the Probes: Methods and Metrics for Concept Alignment](https://arxiv.org/abs/2511.04312). Kontrole zgodności kierunków pojęć z obiektem, w tym wpływu tła. Dobra trafność sondy nie wyklucza skrótów kontekstowych.
9. [Rousselot, Wendebourg i Köthe, Show Me What You Don’t Know: Efficient Sampling from Invariant Sets for Model Validation](https://arxiv.org/abs/2603.21782). Generowanie obrazów ograniczonych reprezentacją z użyciem uprzednio wyuczonego modelu generatywnego. Nasza pętla FastGAN jest inspirowaną propozycją, nie implementacją ich metody dyfuzyjnej ani dowodem skutecznej kuracji danych.

## Wersja do przeczytania na głos

Najpierw przygotujemy stały zestaw zdjęć oznaczonego narzędzia. Zmienimy osobno jego położenie, kąt, wielkość, tło i oświetlenie. Te same obrazy pokażemy sieci zapisanej na kolejnych etapach uczenia.

Dla każdego obrazu zapiszemy opis małych fragmentów, czyli naszych cząstek. Sprawdzimy, które kierunki tego opisu pozwalają rozpoznawać narzędzie i odróżniać jego położenia. Sprawdzimy też, czy detektor rzeczywiście korzysta z tych informacji.

Następnie spróbujemy Koopmanem przewidywać, jak te odpowiedzi zmienią się podczas dalszego uczenia. Prognozę porównamy z późniejszymi zapisami sieci. Sam wzrost odpowiedzi nie oznacza poprawy. Brak będziemy rozpoznawać po konkretnym błędzie, na przykład gubieniu narzędzia pod określonym kątem.

Taki brak wykorzystamy do sterowania FastGANem. Generator stworzy obraz, detektor go oceni, a gradient podpowie, jak zmienić wejście generatora. Wagi obu sieci pozostaną zamrożone. Osobno potraktujemy odtwarzanie wskazanego opisu i tworzenie obrazu ujawniającego słabość. To nie jest to samo zadanie.

Później można nauczyć małą sieć szybszego przekładania zadanego opisu na wejście generatora. Najpierw jednak sprawdzimy prostszą pętlę.

Na początku przygotujemy wyszukiwarkę prawdziwych zdjęć treningowych, korzystając z osobnego modelu opisującego wygląd obrazu. Wygenerowany obraz posłuży potem do znalezienia wizualnie podobnych klatek. Pokażemy je obok siebie i osobno sprawdzimy działanie detektora. Prawdziwe, oznaczone klatki będą kandydatami do dalszego uczenia. Obrazy sztuczne pozostaną wskazówkami. Obecny pilot potwierdza możliwość sterowania generatorem, ale nie potwierdza jeszcze działania całego tego łańcucha.

Po dodaniu prawdziwej klatki porównamy dalsze uczenie z nią i bez niej, wychodząc z tego samego checkpointu. Zmierzymy kierunek zmiany latentu oraz jakość wykrywania. Następnie powtórzymy cały cykl według wcześniej ustalonej reguły. Decyzje w kolejnych rundach podejmiemy na walidacji, a końcowy test zachowamy do oceny gotowej procedury.
