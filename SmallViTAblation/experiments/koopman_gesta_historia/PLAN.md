# Gęstsza historia uczenia RF-DETR Nano

## Wykonane najnowsze zlecenie: osobna macierz per obraz i wektor

Na zlecenie z 2026-09-25 wykonano [koopman_osobno_obraz_token](../koopman_osobno_obraz_token/DIAGNOZA.md): 41472 niezależne operatory, po jednym dla każdej pary obraz/token, na wszystkich 384 współrzędnych bez PCA. Każda końcowa macierz używa 72 par własnej historii z 73 dostępnych checkpointów. Nie łączono obrazów ani tokenów w regresji. Błędy prognozy dotyczą osobnych modeli z wcześniejszego fragmentu, ocenionych na 24 późniejszych przejściach; końcowe operatory z całej historii nie mają jeszcze oceny przyszłości. Wynik negatywny względem kontroli pozostaje zapisany. To zastępuje założenie wspólnej macierzy w bieżącej analizie, ale nie usuwa wcześniejszych wyników ani ograniczeń całego eksperymentu.

## Wykonany audyt sześciu prac i test pamięci pełnego stanu — 2026-09-24

Na zlecenie użytkownika zakończono pełne odczyty sześciu prac przez Undermind, weryfikację kluczowych twierdzeń oraz lokalny test uzasadnionej literaturą poprawki. Źródła, diagnoza i wyniki: [AUDYT_LITERATURY_6_KOOPMAN.md](AUDYT_LITERATURY_6_KOOPMAN.md). Protokół zapisany przed oceną: LITERATURA6_PROTOCOL.json. Kod: koopman_pamiec_full.py, audyt_literatura6.py, test_literatura6.py; dodatkowa kontrola odtwarzalności: weryfikacja_literatura6.py.

Zakończono też analizę 73 checkpointów i 72 obrazów, czyli 5184 par tego samego obrazu między kolejnymi checkpointami. Raporty: wyniki/macierze_wiele_obrazow oraz wyniki/audyt_literatura6. Nowy model zachowuje wszystkie 576×384 współrzędne i dołącza pełny poprzedni przyrost, bez PCA i obcinania rzędu. Korzysta ze 192 wcześniejszych par na dopasowanie; regularyzację wybrano tylko na wewnętrznych grupach i wcześniejszych momentach. To model liniowy z pamięcią, nie deep Koopman.

Wynik diagnostyczny: dla jednego checkpointu RMSE holdout 0,0045273 wobec 0,0089230 poprzedniej macierzy, ale kontrola ostatniego przyrostu jest lepsza (0,0042506). Dla 100 aktualizacji nowy model jest gorszy również od braku zmian. Nie zmieniono bramki ani parametrów po ocenie. Nie przypisujemy poprawy samej pamięci: zmieniono też liczbę par, regularyzację i dynamikę bazową. Osobno zapisano porównanie horyzontów na tych samych początkach. Sześć testów unittest i odtworzenie błędu z zapisanego operatora przeszły poprawnie.

Ten audyt nie kończy całego wcześniejszego eksperymentu: pełne pobranie Edena, identyfikacja na całej palecie RGB oraz pokaz FastGAN pozostają osobnymi etapami. Nie uruchomiono nowych treningów RF-DETR ani generatora. Dalsze warianty wymagają własnej oceny na nowych danych; nie wolno przedstawiać wielokrotnie oglądanego holdout jako niezależnego potwierdzenia.

## Wykonane lokalnie: operator przyrostowy bez zerowania niewidzianych składowych

Na zlecenie użytkownika wykonano diagnostyczny wariant K=I+B, w którym regresja uczy przyrost zamiast całego następnego stanu. Wyprowadzenie i granice gwarancji: [MATEMATYKA_OPERATORA_PRZYROSTOWEGO.md](MATEMATYKA_OPERATORA_PRZYROSTOWEGO.md). Wyniki: [wyniki/koopman_przyrostowy_full/RAPORT.md](wyniki/koopman_przyrostowy_full/RAPORT.md).

Użyto 161 zweryfikowanych przez SHA256 pełnych zrzutów jednego obrazu, bez PCA. Dopasowania chronologiczne wykorzystywały 100/120/140 par; osobny końcowy operator ma 160 par, a jego przyszłe prognozy są niezweryfikowane. Testy potwierdziły zachowanie niewidzianych perturbacji wejścia i poprawność obliczeń dualnych. Krótkoterminowe RMSE spadło z 0,025764 do 0,007261 po zmianie modelu i aktualizowaniu historii, ale nadal jest wyższe od kontroli braku zmian 0,005133. Dla horyzontu 100 aktualizacji nowy wariant ma RMSE 0,066176, wobec 0,060871 starego operatora i 0,047449 braku zmian. Nie ogłaszamy sukcesu prognozowania; to nie jest pełny test na osobnych obrazach ani test tożsamości. Pozostałe etapy planu pozostają niewykonane przez ten skrypt.

## Zaplanowany test wpływu kierunków na decyzję detektora

Protokół i warunkowy dowód zapisano w [TEST_WPLYWU_KIERUNKOW_LATENTU.md](TEST_WPLYWU_KIERUNKOW_LATENTU.md). Bada osobno odczytywalność informacji, wpływ interwencji i korzystny wpływ na błąd detekcji. Zachowuje pełny tensor bez PCA. Wymaga kontroli losowych, zamrożonego odczytu i rozdzielenia zmian latentu od zmian dalszej części sieci. Status: zaplanowany, niewykonany; niniejszy wpis nie uruchamia dodatkowej kampanii w monitorze. Podstawą jest dziesięć udanych odczytów pełnych PDF przez Undermind, z ograniczeniami opisanymi w protokole.

## Obowiązująca zmiana: bez kompresji także w prognozie Koopmana

Najnowsze zlecenie użytkownika zastępuje wcześniejsze plany redukcji. Główna nowa analiza zachowuje pełny tensor 576 x 384, bez PCA, uśredniania tokenów jako stanu, wyboru tylko ośmiu kierunków, obcięcia SVD ani kompresującego autoenkodera. Kierunki tożsamości służą do interpretacji i dodatkowych miar, nie zastępują pełnego wejścia i wyjścia prognozy. Maska waży ocenę obszaru narzędzia, ale nie usuwa kontekstu ze stanu. Nie utożsamiać braku kompresji z gwarancją trafności.

Pełny stan obrazu ma 221184 współrzędne. Nie mylić wspólnego operatora 384 x 384 stosowanego osobno do tokenów z dowolnym operatorem na całym tensorze: pierwszy pomija ogólne zależności między miejscami. Przed implementacją oszacować pamięć, liczbę par i identyfikowalność. Preferować matematycznie równoważne obliczenie regularizowanej regresji pełnego stanu w postaci dualnej lub blokowej, bez materializacji ogromnej macierzy i bez obcięcia rzędu. Przy niedostatecznych zasobach zgłosić konkretny koszt i ograniczenie, nie wprowadzać po cichu redukcji. Skończona próba sama ogranicza identyfikowalność, nawet bez PCA. Zapisać dokładną definicję operatora i jego reprezentację obliczeniową; nie wymagać pełnego widma ogromnej macierzy kosztem wykonalności.

Potrzebne są ponowne lokalne odczyty pełnych tokenów z checkpointów dla stałego zestawu sond: zapisane agregaty 1536 nie odtwarzają pełnego tensora. Przetwarzać porcjami, kontrolować miejsce i zachować grupy fit/holdout oraz chronologiczne granice. Nie udawać, że pełna historia wszystkich sond już istnieje.

Wcześniejsze sekcje o ośmiu kierunkach, PCA kontekstu 16/32 oraz EDMD i autoenkoderze na ośmiu współrzędnych są zachowanym zapisem poprzedniego planu i NIE są teraz zleceniem wykonania. Zachować istniejące wyniki historycznego pilota, nie nadpisywać ich nowymi miarami. Progi ustalić przed oceną, a zmianę przestrzeni błędu raportować jawnie. Kontrole braku zmian i trendu muszą używać tego samego pełnego stanu, podziału i horyzontu 100 aktualizacji.

Pokaz FastGAN pozostaje zlecony niezależnie od wyniku prognozy. Dostosować cel do pełnej reprezentacji przy zachowaniu dotychczasowego limitu 40 kroków i dwóch restartów; zapisać także osobny błąd w obszarze narzędzia. Nie zastępować celu ośmioma współrzędnymi. Nie zmieniać bieżącego treningu detektora na Edenie ani nie uruchamiać nowego treningu jego wag.

## Najnowsze ustalenie: identyfikacja narzędzia bez wstępnego PCA

Identyfikację „Pełnego zbioru wektorów tożsamości narzędzia” wykonywać na pełnych wektorach tokenów analizowanej warstwy (576 miejsc, po 384 współrzędne), z maskami segmentacyjnymi. Nie wykonywać redukcji PCA przed tą identyfikacją. Maska określa udział narzędzia w obszarze tokenu; nie dowodzi, że jego cechy nie zawierają kontekstu spoza maski. Osobno traktować wnętrze narzędzia, brzeg i tło. Zachować oryginalne współrzędne oraz ewentualne parametry skalowania, aby kierunki można było odnieść do pełnego latentu.

Porównywać to samo narzędzie na zmienionych tłach oraz kontrole bez narzędzia, używając rozłącznych źródeł obrazów fit/holdout. Odróżnić informację odczytywalną z kierunków od przyczynowego wpływu na detekcję. Nie zakładać jednego tokenu ani pojedynczego kierunku wyłącznie odpowiedzialnego za tożsamość. Ograniczony zbiór i rodzina odczytów nie pozwalają zagwarantować kompletności.

Audyt kodu: obecny fit_directions w koopman_fastgan_petla/dynamika.py wykonuje nadzorowaną regresję wszystkich 1536 współrzędnych deskryptora na etykiety, a potem SVD jej współczynników z ograniczeniem do ośmiu kierunków. To nie jest PCA latentu przed identyfikacją. Deskryptor jednak agreguje tokeny (średnie i momenty), więc nie zastępuje identyfikacji na pełnych tokenach. Nie wykazano, że PCA spowodowało wcześniejszy nieudany wynik prognozy.

Zapisane deskryptory zachować dla porównania bazowego bez zmiany jego bramki. Nową analizę identyfikacji oznaczyć osobno i wykonać potrzebne odczyty pełnych tokenów z pobranych checkpointów: istniejący zapis opcjonalnych tokens dotyczy tylko wybranej sondy, nie wszystkich obrazów. Przetwarzać partiami zamiast zakładać, że cała historia tokenów mieści się w pamięci.

PCA kontekstu w późniejszych ramionach diagnostycznych pozostaje dozwolone dopiero po identyfikacji kierunków narzędzia; nie może zastąpić tego etapu. Nie zmieniać trwającego treningu na Edenie. Powyższe ustalenie ma pierwszeństwo w sposobie wykonania identyfikacji po pobraniu.

Zlecenie użytkownika: restart na Edenie, dłuższy i wolniejszy trening oraz gęstsze checkpointy.

- Start od tych samych wag COCO, nie od końca poprzedniego treningu.
- Osiem epok; 100 aktualizacji na epokę; 1500 oznaczonych klatek, ten sam sampler i seed.
- Oba learning rate zmniejszone o połowę względem poprzedniego pilota.
- Zapis wag i latentu co pięć aktualizacji: 160 checkpointów wag oraz 161 stanów latentu wraz ze stanem początkowym.
- Pełny stan optymalizatora na granicy każdej epoki i na końcu.
- Ten sam zamrożony zestaw sond skopiowany z poprzedniego biegu.
- Walidacja każdej epoki. Osiem epok nie gwarantuje braku przeuczenia.
- Eden wykonuje tylko pretrening; Koopman i ewentualny FastGAN lokalnie po pobraniu.

Pytanie: czy dłuższa, gęstsza historia wolniejszego dostrajania pozwoli przewidywać latent?
Kontrole późniejszej oceny: brak zmiany i trend liniowy przy tym samym horyzoncie w aktualizacjach.
Ostatni fragment historii i osobne grupy obrazów pozostają poza dopasowaniem.
Podstawowe ramię zachowuje poprzedni wymiar opisu. Zlecone niżej porównanie rozszerza kontekst wejściowy w osobnych ramionach. Zmiana LR i długości równocześnie nie pozwala przypisać poprawy wyłącznie gęstości zapisu.
Można dodatkowo porównać pełną historię z jej przerzedzoną wersją w ramach tego samego treningu.
Nie zmieniamy progów bramki prognozy po obejrzeniu wyników.
Brak osobnego testu w tej puli pozostaje ograniczeniem.

Stare pliki wag usuwamy wyłącznie z pretrain biegu 1810448 po pobraniu i potwierdzeniu SHA256 każdego pliku. Pozostałe wyniki oraz model FastGAN pozostają dostępne.

## Dodatkowe tła RGB — zlecenie użytkownika
Osobny zestaw wyniki/probe_rgb256 powstaje przez paleta_tla.py. Paleta zawiera 256 kolorów na regularnej siatce RGB (8 poziomów R, 8 G, 4 B), z czernią, bielą i żółcią. To próbka przestrzeni RGB, nie wszystkie możliwe kolory. Każdy oryginalny wycinek otrzymuje każde tło; zachowujemy maskę i podział grup fit/holdout. Piksele wnętrza narzędzia są identyczne, półprzezroczysty brzeg jest mieszany zgodnie z maską. Nie zmieniamy sond trwającego treningu. Nowe obrazy wymagają osobnego lokalnego odczytu latentu z pobranych checkpointów przed wykorzystaniem w identyfikacji kierunków. Dotychczasowe zrzuty nie zawierają tych wariantów. Jednolite tła badają odporność na kolor; nie zastępują sprawdzenia prawdziwych teł operacyjnych.

## Po pobraniu: kierunki narzędzia czy szerszy kontekst latentu

Zlecenie użytkownika: wykonać lokalnie po pobraniu i sprawdzeniu SHA256 wszystkich checkpointów. Nie zmieniać trwającego treningu.

1. Najpierw identyfikować i ocenić kandydatów na „Pełny zbiór wektorów tożsamości narzędzia”, używając etykiet, masek i kontrolowanych wariantów obrazu. Zachować rozróżnienie pomiędzy odczytywalną informacją a udowodnioną przyczynową rolą i kompletnością. Nową paletę RGB ocenić osobno przez lokalne odczyty detektora; wcześniejsze LATENT nie zawierają tych obrazów. Sama obecność narzędzia na każdym kolorowym tle nie wystarcza do wyuczenia rozróżnienia obecność/nieobecność; wykorzystać również istniejące kontrole bez narzędzia. Nie mieszać duplikatów tych samych źródeł między fit a holdout.
2. Podstawowe porównanie korzysta z istniejącej historii deskryptorów i tej samej zamrożonej bazy ośmiu kierunków narzędzia we wszystkich ramionach. Bazę wyznaczać wyłącznie na wczesnych danych fit. Szerszy opis nie oznacza udowodnionej identyfikacji wszystkich surowych tokenów: zapisany deskryptor agreguje narzędzie, tło i momenty przestrzenne.
3. Ramię A: operator na samych ośmiu współrzędnych narzędzia. Ramiona B: te same współrzędne plus dodatkowo 8 lub 24 współrzędne kontekstu, czyli stany o łącznym wymiarze 16 lub 32. Kontekst uzyskać przez PCA reszty deskryptora po usunięciu składowej w bazie narzędzia. Skalowanie, PCA i regresję dopasowywać tylko na danych fit sprzed początku prognozy, po czym zamrozić. Odrzucać zerowe i liniowo zależne kierunki; raportować rzeczywisty rząd. Nie próbować zwiększać wymiaru przez sam parametr istniejącego fit_directions: ta funkcja ogranicza rząd do ośmiu.
4. Każdy operator przewiduje własny pełny stan, ale wynik porównujemy WYŁĄCZNIE na tych samych ośmiu współrzędnych narzędzia. Wspólne obrazy, początki prognoz, horyzont 100 aktualizacji, skala błędu z danych fit oraz kontrole braku zmian i trendu liniowego. Zachować poprzednią regularyzację i progi podstawowego testu; podać wzrost liczby parametrów większych operatorów. Kierunek i długość przewidywanego przyrostu mierzyć obok błędu prognozy.
5. Podstawową bramkę oryginalnego wariantu raportować bez zmiany. Porównanie dodatkowych wymiarów jest diagnostyczne: najlepszy wynik na odłożonym fragmencie nie staje się automatycznie niezależnym potwierdzeniem wybranego wariantu. Wybór konfiguracji wymaga wcześniejszego osobnego fragmentu do doboru i późniejszego nietkniętego fragmentu do końcowej oceny; jeśli historia nie wystarczy, raportować brak niezależnego potwierdzenia. Nie wybierać największej macierzy z założenia.

Wyniki: macierze, widma, konfiguracja podziału, jakość identyfikacji, błędy prognoz i niepewność liczona grupami filmów, opis ograniczeń. Ewentualna prognoza poza ostatni checkpoint pozostaje niezweryfikowana. Po analizie wykonujemy oddzielny pokaz FastGAN opisany poniżej, bez kolejnego treningu detektora.

## Dodatkowe przybliżenia nieliniowe — zlecone po pobraniu

Użytkownik zlecił także Extended DMD i wariant autoenkoderowy. Są to dodatkowe lokalne modele historii istniejącego detektora; RF-DETR oraz FastGAN pozostają zamrożone.

- **Extended DMD (EDMD):** do standaryzowanych ośmiu współrzędnych narzędzia dołączyć wszystkie kwadraty i iloczyny parami. Daje to 44 funkcje bez stałej; przesunięcie dopasować osobno. Dopasować regularizowaną macierz przejścia w tej rozszerzonej przestrzeni. Przy prognozie wielokrokowej propagować rozszerzony stan liniowo; odczytywać pierwotne współrzędne z części liniowej. Nie podstawiać prawdziwego stanu między krokami. Skalowanie słownika wyznaczyć tylko z fit i zapisać wraz z modelem. Macierz nie jest uczona przez autoenkoder.
- **Autoenkoder Koopmana:** niewielki encoder MLP 8→32→16, liniowe przejście 16→16 z przesunięciem oraz decoder 16→32→8. Uczyć wspólnie rekonstrukcji wejścia, zgodności przejścia w przestrzeni kodu oraz odtwarzania przyszłych współrzędnych. Uwzględnić prognozę wielokrokową tylko dla sekwencji mieszczących się w fit. Sam błąd rekonstrukcji nie wystarcza. Zabezpieczyć przed stałym kodem przez pomiar rekonstrukcji i wariancji kodu.
- **Ograniczony pilot:** jeden seed 42; Adam, learning rate 0.001, weight decay 0.0001, batch 256, maksymalnie 2000 aktualizacji modelu pomocniczego. Horyzonty straty 1 i 5 interwałów; składniki rekonstrukcji, zgodności latentnej i prognozy ze stałą wagą 1 po standaryzacji. Zapisać rzeczywistą konfigurację i czas. To niewielki lokalny trening pomocniczy na zapisanych liczbach, nie nowa kampania treningowa detektora. Nie rozszerzać automatycznie siatki architektur ani seedów.

Przed uruchomieniem zapisać rozłączne, chronologiczne zakresy fit, wewnętrznej walidacji i końcowego sprawdzenia; żadne okno treningowe nie może przekroczyć granicy fit. Wewnętrzna walidacja wybiera checkpoint autoenkodera, końcowy fragment nie steruje uczeniem. Wszystkie metody porównać na identycznych początkach prognoz i horyzoncie 100 aktualizacji RF-DETR, w oryginalnej wspólnej przestrzeni ośmiu współrzędnych narzędzia. Raportować także brak zmian, trend liniowy i poprzedni liniowy operator na tych samych podziałach. Zachować osobny pierwotny wynik bramki.

Zapisać błędy poza próbą, niestabilność wielokrokową, liczbę parametrów, macierze i widma oraz wagi encodera/decodera. Nie utożsamiać osi nieliniowego kodu z kierunkami tożsamości. Nie wybierać wyniku na końcowym holdout i nie ogłaszać go niezależnym potwierdzeniem zwycięzcy. Warianty mają sprawdzić hipotezę, że liniowy opis jest zbyt ubogi; lepszy wynik nie jest gwarantowany.

## Pokaz FastGAN także po nietrafnej prognozie — jawna zmiana zakresu

Użytkownik wyraźnie zlecił wygenerowanie i pokazanie obrazu również wtedy, gdy Koopman nie pokona kontroli. Ten pokaz wykonujemy po lokalnej analizie niezależnie od wartości accepted. Nie zmieniamy wyniku naukowej bramki ani nie oznaczamy jej jako zaliczonej.

1. Użyć istniejącego wytrenowanego FastGAN-a oraz zamrożonego checkpointu RF-DETR. Wybrać deterministycznie pierwszy obecny wariant original z grupy fit jako kotwicę; zapisać jej identyfikator. Dla końcowego checkpointu wyliczyć obecny kod i cel: obecny kod plus przyrost przewidywany na horyzoncie 100 aktualizacji. Model prognozy i jego wynik walidacji mają być jawnie wskazane. Prognoza z operatora dopasowanego do całej historii dotyczy nieobserwowanej przyszłości.
2. Co najmniej jedna próba dla operatora liniowego, opcjonalnie po jednej dla EDMD i autoenkodera: maksymalnie trzy cele. Wyniki wszystkich metod sprowadzić do wspólnych współrzędnych narzędzia. Nie wybierać prognozy wyłącznie dlatego, że daje ładniejszy obraz.
3. Lokalne sterowanie wejściem FastGAN: 40 kroków optymalizacji, dwa restarty, stały seed. Zmieniamy tylko liczby wejściowe generatora; wagi detektora i generatora są zamrożone. Kontrolować zgodność odczytu obecnego kodu z zapisanym latentem. Korzystać z generator_cli.py/generowanie.py po dostosowaniu ścieżek i zapisu wyników do tego eksperymentu.
4. Zapisuj obraz początkowy, najlepszy skończony wynik oraz końcowy RGB i maskę również wtedy, gdy cel nie został osiągnięty albo kontrola jakości nie przeszła. Osobno zapisz cel, osiągnięty kod, błąd dopasowania przed/po, wielkość przyrostu, przebieg optymalizacji, seed, wersje wag i wynik bramki Koopmana. Nie zastępuj wygenerowanego obrazu oryginałem ani nie przedstawiaj nieudanej próby jako sukcesu.
5. Nie przekazuj NaN/Inf do gradientu; zapisz błąd prognozy, pokaż ostatni poprawny obraz lub obraz startowy jako nieoptymalizowaną kontrolę. Nie przycinaj celu bez jawnego zapisania takiej dodatkowej zmiany. Przy ograniczeniu pamięci zmniejsz batch i zwolnij własne modele między etapami; nie zatrzymuj cudzych procesów.
6. Pokaż użytkownikowi obraz lokalnym odnośnikiem/preview oraz krótko napisz, czy przybliża zadany latent i czy prognoza Koopmana była trafna. To wizualizacja żądanego stanu, nie dowód, że obraz ujawnia brak wiedzy lub poprawi trening. Pokaz nie uruchamia dobierania klatek ani dotrenowania detektora.
