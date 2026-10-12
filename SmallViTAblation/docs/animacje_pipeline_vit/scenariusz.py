"""Scenariusz dydaktyczny: rzeczywisty kod ViT, pomiary i jawne ilustracje.
Uruchomienie zapisuje storyboard.json, używany przez syntezę mowy i ManimGL.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAPTERS = []

def chapter(slug, title, slides):
    CHAPTERS.append(dict(id=len(CHAPTERS)+1, slug=slug, title=title, slides=[
        dict(kind=x[0], title=x[1], caption=x[2], narration=x[3], mode=x[4] if len(x)>4 else 'SCHEMAT METODY') for x in slides]))

chapter('01_obraz_i_tokeny', 'Od obrazu do 600 cząstek', [
('overview','Co chcemy zrozumieć?', 'Obraz → tokeny → informacja o narzędziu → dynamika uczenia → dobór klatek',
 'Ten cykl pokazuje twój rzeczywisty pipeline małego transformera wizyjnego. Najpierw wyjaśnimy, czym są cząstki w latencie. Następnie pokażemy maski, odczyt tożsamości i ruch cech podczas uczenia. Na końcu połączymy to z prognozą i dobieraniem prawdziwych klatek. Schematy i zmierzone dane będą oznaczone osobno.'),
('frame','Pełna klatka na wejściu', 'Szerokość 480 px, wysokość 320 px. Trzy kanały RGB.',
 'Do tego małego modelu trafia pełna klatka zmniejszona do czterystu osiemdziesięciu na trzysta dwadzieścia pikseli. Pokazujemy rzeczywisty obraz ze zbioru Cataract. Rozmiar wejścia nie jest rozmiarem badanego latentu. Najpierw obraz przechodzi przez konwolucyjną część sieci.', 'RZECZYWISTY OBRAZ'),
('grid','Siatka miejsc obserwacji', '30 kolumn × 20 wierszy = 600 tokenów. Odstęp siatki: 16 px.',
 'Cztery konwolucje, każda z krokiem dwa, zmniejszają rozdzielczość szesnastokrotnie. Otrzymujemy trzydzieści kolumn i dwadzieścia wierszy. To sześćset tokenów. W naszej interpretacji każdy token jest jedną cząstką. Linie pokazują przypisanie miejsca, ale konwolucyjne pola widzenia nakładają się na siebie.'),
('dimensions','Cząstka jest wektorem', '600 × 192 w blokach; 600 × 768 w badanej warstwie MLP.',
 'Każdy token ma sto dziewięćdziesiąt dwie cechy w głównym strumieniu transformera. Wewnątrz ostatniego bloku warstwa MLP rozszerza go do siedmiuset sześćdziesięciu ośmiu aktywacji. Odczytujemy je po funkcji GELU. To sześćset wektorów, każdy o siedmiuset sześćdziesięciu ośmiu współrzędnych. Nie siedemset sześćdziesiąt osiem cząstek.'),
('architecture','Rzeczywista architektura', '2 926 493 parametrów • 6 bloków • 3 głowy uwagi w bloku',
 'Ten model ma około dwóch przecinek dziewięćdziesięciu trzech miliona parametrów. Zawiera sześć bloków transformera i trzy głowy uwagi w każdym bloku. Na końcu gęsta głowa przewiduje położenie środka narzędzia i jego ramkę. Odczyt latentu wykonujemy wcześniej, wewnątrz ostatniego bloku. Nie zmieniamy przez to dekodera.', 'PARAMETRY Z KODU')])

chapter('02_pozycja_i_uwaga', 'Pozycja, uwaga i mieszanie informacji', [
('position','Stały adres tokena', 'Cechy obrazu + stały kod pozycji sinus–cosinus',
 'Każde miejsce siatki dostaje stały wektor pozycji, obliczany z sinusów i cosinusów. Dodajemy go do cech obrazu przed blokami transformera. Nie jest uczonym parametrem. Ten sam adres dostaje ten sam kod. Treść tokena może jednak zmieniać się wraz z obrazem.'),
('attention','Skąd token bierze kontekst?', 'Q = TWq, K = TWk, V = TWv; uwaga = softmax(QKᵀ / √d).',
 'Z tokenów powstają zapytania, klucze i wartości. Podobieństwo zapytania do kluczy określa wagi uwagi. Token otrzymuje ważoną mieszaninę informacji z innych miejsc. Dlatego token leżący na cystotomie może również zawierać informację o oku, drugim narzędziu i oświetleniu. Pokazane połączenia ilustrują mechanizm, nie są zmierzoną mapą uwagi.'),
('reset','Nowa klatka: nowy odczyt', 'Brak stanu tokenów przenoszonego między klatkami w tym ViT.',
 'Gdy przychodzi nowa klatka, tokeny są liczone od nowa. Sieć nie przechowuje poprzednich cząstek jako swojego stanu czasowego. Przy zamrożonych wagach identyczny obraz w trybie oceny daje identyczną reprezentację. Podobne obrazy często dają podobne cechy, ale duża zmiana lokalnej treści może mocno przesunąć token.'),
('two_times','Dwie różne osie czasu', 'Film: zmienia się obraz. Uczenie: zmienia się sieć, sonda jest stała.',
 'Trzeba rozdzielić dwie sytuacje. W pierwszej oglądamy kolejne klatki jednego filmu przy stałej sieci. W drugiej przepuszczamy ten sam obraz przez sieć po kolejnych aktualizacjach wag. Nasz eksperyment przewidywania latentu dotyczy tej drugiej sytuacji. Nie mylmy ruchu narzędzia w filmie z uczeniem reprezentacji.')])

chapter('03_maska_jeden_piksel', 'Maska wskazuje cząstki narzędzia', [
('mask','Od pikseli do etykiet tokenów', 'Maska segmentacji jest w pikselach; token odpowiada polu siatki.',
 'W zbiorze mamy obrys cystotomu. Najpierw tworzymy z niego maskę pikseli. Potem dopasowujemy maskę do rozmiaru wejściowej klatki i nakładamy siatkę tokenów. Maska nie identyfikuje automatycznie semantycznych neuronów. Wskazuje miejsca obrazu przecinające narzędzie.', 'RZECZYWISTA MASKA'),
('one_pixel','Nowa reguła: choć jeden piksel', 'mᵢ = 1, gdy suma maski w polu 16 × 16 jest większa od zera.',
 'Zgodnie z twoim pomysłem wystarczy choć jeden piksel maski w danym polu. Wtedy zachowujemy cały token jako związany z narzędziem. Nie wymagamy już jednej czwartej pola. Przy zmniejszaniu maski zachowujemy także cienkie fragmenty, które zwykłe próbkowanie mogłoby zgubić.'),
('mask_compare','Co przybywa po zmianie progu?', 'Zielone: ≥25%. Pomarańczowe: dodatkowe tokeny od 1 piksela.',
 'Zielone pola przekraczają stary próg dwudziestu pięciu procent. Pomarańczowe to dodatkowe pola wybrane nową regułą. W dwunastu pełnych klatkach nowy próg wybrał od siedmiu do trzydziestu jeden tokenów narzędzia. Nadal zapisujemy wszystkie sześćset tokenów, także te leżące poza maską.', 'POMIAR: 12 PEŁNYCH KLATEK'),
('mask_weights','Etykieta nie oznacza czystej cechy', 'Zachowujemy udział pokrycia i oddalone tło jako kontrolę.',
 'Token z jednym pikselem narzędzia może w większości opisywać otoczenie. Dlatego oprócz etykiety zapisujemy dokładny udział pokrycia. Osobno wybieramy tło oddalone od maski. Dzięki temu możemy sprawdzić, czy efekt pochodzi z narzędzia, z jego brzegu, czy ze wspólnej zmiany całej sceny.')])

chapter('04_geometria_latentu', 'Jak czytać chmurę 768-wymiarową', [
('cloud','Obraz jako chmura punktów', 'Jeden punkt = jeden token. Kolor określa maska, nie odkryty koncept.',
 'Wektor tokena możemy potraktować jako punkt w przestrzeni cech. Cały obraz tworzy chmurę sześciuset punktów. Kolorem zaznaczamy tokeny należące do maski. Na ekranie widzimy tylko rzut na dwie osie. Bliskość na takim rysunku nie gwarantuje bliskości w pełnej przestrzeni.', 'RZUT RZECZYWISTYCH TOKENÓW'),
('directions','Punkt i kierunek to różne rzeczy', 'Aktywacja kierunku: aᵢ = wᵀ zᵢ. Token nie musi leżeć na osi.',
 'Kierunek w latencie działa jak oś pomiarowa. Cząstka może mieć dużą składową wzdłuż tej osi i jednocześnie wiele innych składowych. Nie musi leżeć dokładnie na prostej. Dodatni rzut może wskazywać narzędzie, ale znaczenie kierunku trzeba sprawdzić na danych.'),
('subspace','Tożsamość może wymagać wielu osi', 'W = [w₁, …, wₖ], współrzędne konceptu: aᵢ = Wᵀ zᵢ.',
 'Zamiast jednej osi możemy zachować podprzestrzeń wielu kierunków. Jedne cechy mogą opisywać obecność cystotomu, inne jego kształt, obrót lub części. Sam duży wymiar albo skupisko nie dowodzi tożsamości. Potrzebujemy kontroli tła, innych narzędzi i obrazów niewidzianych przy dopasowaniu.'),
('invariance','Co ma pozostać wspólne?', 'Ten sam cystotom: inne położenie, obrót, długość i tło.',
 'Dobra reprezentacja tożsamości powinna zachowywać informację o cystotomie mimo zmian wyglądu. Nie wymagamy, żeby cały wektor był nieruchomy. Chcemy, żeby rozpoznawalna składowa narzędzia przetrwała obrót, przesunięcie i zmianę tła. Pozostałe współrzędne mogą opisywać te różnice.')])

chapter('05_lda_i_odczyt', 'Co naprawdę wyznacza LDA', [
('pooling','Dotychczasowa pętla: średnia tokenów', 'Maska → średni wektor narzędzia i średni wektor tła.',
 'W poprzedniej pętli maska wybierała tokeny, ale potem je uśrednialiśmy. Otrzymywaliśmy jeden wektor narzędzia na obraz i drugi wektor tła. Ten zabieg ułatwiał obliczenia, lecz usuwał informację o osobnych częściach i różnicach ich ruchu. Sondy były wycinkami o stu dwudziestu jeden tokenach.'),
('lda','Kierunek rozdzielający dwie grupy', 'w ∝ (Σ + regularyzacja)⁻¹ (μnarzędzie − μtło)',
 'LDA bierze różnicę średnich narzędzia i tła oraz uwzględnia rozrzut cech. Szuka takiej kombinacji współrzędnych, która dobrze rozdziela obie grupy. W kodzie użyto stabilizowanego oszacowania kowariancji. To coś więcej niż samo odejmowanie dwóch obrazów lub dwóch średnich.'),
('score','Jedna liczba na sondę', 's(t) = wᵀ [(z(t) − μtła) / σtła]',
 'Na końcu rozgrzewki zamrażamy kierunek oraz skalowanie cech. Rzut wektora na tę oś daje siłę aktywacji. Dla każdego obrazu zapisujemy, jak ta liczba zmienia się podczas uczenia. Dwuklasowe LDA daje jeden kierunek z konstrukcji metody. Nie oznacza to, że całe narzędzie ma tylko jeden wymiar.'),
('causal','Odczytywalność i wykorzystanie', 'Informacja w tokenie ≠ dowód, że detektor z niej korzysta.',
 'Jeżeli z kierunku da się odczytać narzędzie, reprezentacja zawiera odpowiednią informację. Żeby wykazać, że detektor jej używa, trzeba wykonać interwencję: usunąć lub podmienić składową, a potem sprawdzić zmianę detekcji. Porównaniem powinna być równie silna zmiana w losowym kierunku.')])

chapter('06_ruch_osobnych_tokenow', 'Nowy pomiar: żadnych średnich cząstek', [
('experiment','Stałe obrazy, kolejne stany sieci', 'Z[t, obraz, token, cecha]; osobne 600 × 768 na każdą klatkę.',
 'Nowy pomiar zachowuje osobno wszystkie tokeny pełnej klatki. Ten sam obraz trafia do sieci przy kolejnych checkpointach. Maska i indeksy siatki pozostają stałe. Dzięki temu różnica reprezentacji tego samego tokena opisuje zmianę odpowiedzi sieci podczas uczenia.'),
('trajectories','Rzeczywiste przemieszczenia', 'Δzᵢ = zᵢ(t₂) − zᵢ(t₁). Pośrednie punkty animacji są interpolacją.',
 'W lokalnym pomiarze dostępne były dwa stany krótkiej próby: krok sześćdziesiąt i sto dwadzieścia. Pokazujemy rzeczywiste końce przemieszczeń tokenów. Płynny ruch pomiędzy końcami służy wizualizacji i nie jest nagraną trajektorią. Z dwóch punktów nie wyznaczymy wiarygodnie pamięci ani przyspieszenia.', 'POMIAR 60 → 120; INTERPOLACJA WIZUALNA'),
('background_drift','Odejmujemy wspólną zmianę tła', 'vᵢ = Δzᵢ / Δt; rᵢ = vᵢ − referencyjny dryf tła.',
 'Cała sieć może zmieniać się wspólnie. Dlatego obok surowych przemieszczeń badamy ruch po odjęciu referencyjnej zmiany tła. Przy nierównych odstępach dzielimy przez liczbę aktualizacji. Na odłożonych obrazach referencja pochodzi wyłącznie z filmów użytych do dopasowania.'),
('dense','Gęsta historia: właściwy cel pomiaru', 'Co 10 kroków: 101 stanów. Co krok: 1001 stanów do t = 1000.',
 'Docelowo potrzebujemy gęstej historii tych samych sześciuset tokenów. Sto jeden stanów daje zapis co dziesięć aktualizacji do kroku tysiąc. Zapis co każdą aktualizację daje tysiąc jeden stanów. Osobny eksperyment gęstej historii ma taki plan. Ten film nie deklaruje jego ukończenia ani nie zastępuje brakujących danych.', 'PLAN GĘSTEGO POMIARU; STATUS NIEZWERYFIKOWANY')])

chapter('07_kierunki_z_ruchu', 'Jak wyznaczamy wiele kierunków', [
('svd','Macierz ruchów tokenów narzędzia', 'R = [r₁ᵀ; …; rₙᵀ], R = UΣVᵀ. Kolumny V są osiami ruchu.',
 'Zbieramy osobne wektory ruchu tokenów wskazanych maską. Rozkład SVD szuka kierunków, wzdłuż których te wektory mają najwięcej energii. Każdy obraz i przedział dostają równy łączny udział w dopasowaniu. Nie zastępujemy wcześniej wszystkich tokenów jednym wektorem.'),
('spectrum','Energia ruchu i liczba kierunków', '16 osi opisuje 90% energii w tym wstępnym pomiarze.',
 'Na danych uczących naszego krótkiego pomiaru szesnaście osi opisuje dziewięćdziesiąt procent energii skorygowanego ruchu. To liczba opisująca ruch w tym konkretnym zbiorze. Nie wolno nazywać jej liczbą wymiarów tożsamości cystotomu. SVD może przechwycić także kontekst i zmiany nieswoiste.', 'WSTĘPNY POMIAR: 60 → 120'),
('holdout','Sprawdzamy inne filmy', '8 filmów: kierunki. 4 odłożone filmy: ocena narzędzie / tło.',
 'Kierunki wyznaczamy na ośmiu filmach, a oceniamy na czterech innych. Nie dzielimy losowo tokenów tego samego obrazu między dopasowanie i test. Porównujemy rzut ruchu narzędzia i tła. Taka ocena sprawdza przenoszenie między obrazami, nie prognozę przyszłych kroków uczenia.'),
('results','Wynik nie daje jeszcze tożsamości', 'Średni kierunek: AUC 0,669. 16 osi: 0,611; losowe: 0,608.',
 'Średni podpisany kierunek ruchu daje wstępny sygnał rozróżniający narzędzie od tła. Natomiast szesnaście głównych osi wypada prawie tak samo jak losowa podprzestrzeń tej wielkości. To nie jest przekonujący dowód odkrycia tożsamości. Wynik pochodzi z jednego krótkiego przebiegu i czterech odłożonych obrazów.', 'WYNIK ZAPISANY; BEZ TWIERDZENIA PRZYCZYNOWEGO')])

chapter('08_historia_bez_usredniania', 'Sto pomiarów czy dwadzieścia średnich?', [
('history','Jak zbieraliśmy historię?', '1000 aktualizacji ≈ 21 epok; pomiary w krokach 0, 10, …, 1000.',
 'Rozgrzewka trwała tysiąc aktualizacji, co przy tej bazie i paczce odpowiada około dwudziestu jeden epokom. Stałe sondy mierzyliśmy przed uczeniem i co dziesięć aktualizacji. Dawało to sto jeden pomiarów. Pomiar aktywacji nie musi oznaczać zapisania całego checkpointu wag.', 'RZECZYWISTA HISTORIA ODCZYTU LDA'),
('averaging','Co robiła średnia z pięciu?', '100 surowych pomiarów → 20 średnich; odstęp rośnie z 10 do 50 kroków.',
 'Dotychczasowy kod odrzucał pomiar początkowy i uśredniał pięć kolejnych zapisów. Dostawał dwadzieścia punktów na sondę. To filtr wygładzający, a nie wymóg Koopmana. Może ograniczyć wahania, ale może także usunąć szybkie reakcje na paczki danych.', 'DANE: SUROWE I ŚREDNIE'),
('aliasing','Jakie informacje mogą zniknąć?', 'Wzrost i spadek wewnątrz bloku mogą dać tę samą średnią.',
 'Dwie zupełnie różne ścieżki mogą mieć tę samą średnią. Jeżeli chcemy odtworzyć szybką dynamikę, wygładzenie ją ogranicza. Jeśli celem jest wolny trend, średnia może być użyteczna. Trzeba jednak mówić jasno, jaki proces modelujemy i w jakiej skali czasu.'),
('fair_compare','Uczciwe porównanie wariantów', 'Te same przyszłe cele, horyzont w aktualizacjach i kontrola czasu pamięci.',
 'Wersję surową należy porównać z wygładzoną na tym samym przyszłym odcinku. Cztery opóźnienia przy zapisie co dziesięć kroków oznaczają inną pamięć niż cztery opóźnienia co pięćdziesiąt. Dlatego kontrolujemy zarówno liczbę parametrów, jak i fizyczny czas historii. Podstawowym odniesieniem jest brak zmiany.')])

chapter('09_koopman_krok_po_kroku', 'Koopman: dopasowanie i prognoza', [
('koopman','Idea operatora Koopmana', 'Dla obserwacji ψ: Kψ(x) = ψ(F(x)). W kodzie: skończony model liniowy.',
 'Operator Koopmana opisuje, jak obserwowane wielkości zmieniają się pod działaniem dynamiki. Zamiast modelować bezpośrednio wszystkie szczegóły układu, szukamy wygodnych obserwacji. W naszej pętli używamy ograniczonego modelu liniowego historii przyrostów. To przybliżenie, nie dowód znalezienia dokładnych niezmienników.'),
('lags','Cztery poprzednie zmiany', 'Δs(t+1) = a₀Δs(t) + a₁Δs(t−1) + a₂Δs(t−2) + a₃Δs(t−3).',
 'Stanem w dotychczasowej wersji była jedna liczba na sondę: siła odczytu LDA. Liczymy jej kolejne przyrosty i układamy cztery poprzednie zmiany jako wejście. Następna zmiana jest celem. Wszystkie współczynniki dopasowujemy wyłącznie na dostępnej przeszłości, z regularyzacją stabilizującą rozwiązanie.'),
('fit','Jak powstają współczynniki?', 'X: przeszłe zmiany. Y: następne zmiany. Minimalizacja błędu + kara.',
 'Każdy wiersz macierzy wejściowej zawiera historię przed danym krokiem. Odpowiadający mu cel jest zmianą w następnym kroku. Regresja dobiera współczynniki tak, aby odtworzyć cele z przeszłych danych. W starej pętli współczynniki były wspólne dla sond. Osobne dopasowanie na jednym obrazie to odrębny wariant.'),
('rollout','Prognoza składana w czasie', 'Przewidź przyrost → dodaj do stanu → wpisz do historii → powtórz.',
 'Tutaj pokazujemy obliczony przykład dla jednej sondy, na stu pomiarach bez uśredniania. Dopasowanie kończy się w kroku tysiąc. Potem przewidujemy pięćset aktualizacji, korzystając z własnych poprzednich prognoz. Szara linia pokazuje rzeczywisty późniejszy odczyt. Błąd porównujemy z utrzymaniem ostatniej wartości. Ten jeden przykład nie jest dowodem przewagi metody.', 'RZECZYWISTY ODCZYT; PROGNOZA JEDNEJ SONDY'),
('koopman_limit','Co właściwie zostało przewidziane?', 'Prognoza odczytu ≠ odkrycie kierunku tożsamości ≠ poprawa detektora.',
 'Koopman w tej wersji nie wybierał cząstek narzędzia i nie odkrywał kierunku LDA. Maska wskazywała tokeny, LDA definiowała odczyt, a model dynamiki przewidywał jego zmianę. To trzy osobne operacje. Każda wymaga własnej oceny. Dobra prognoza nie dowodzi jeszcze użyteczności doboru klatek.')])

chapter('10_langevin_i_pamiec', 'Langevin: dryf, pamięć i reszta', [
('langevin','Co dodaje wariant Langevina?', 'Δs(t+1) = k·s(t) + Σ bⱼΔs(t−j) + csondy + reszta.',
 'Wariant nazwany w kodzie Langevinem dodaje obecny poziom aktywacji i osobną stałą dla sondy. Pozostałe składniki opisują zależność od wcześniejszych przyrostów. To może uchwycić tendencję do powrotu w okolice pewnego poziomu, ale stabilność takiego powrotu trzeba sprawdzić, a nie zakładać.'),
('forces','Interpretacja składników', 'Dryf: stan obecny. Pamięć: historia. Reszta: niewyjaśniona zmiana.',
 'Strzałki pokazują sumowanie wkładów. Jedna zależy od obecnego stanu, kolejne od historii. Różnica między przewidzianą i rzeczywistą zmianą jest resztą. Samo nazwanie jej szumem nie dowodzi, że jest biała, gaussowska lub niezależna. To ilustracja rachunku, nie zmierzona siła fizyczna.'),
('memory','Jak sprawdzać pamięć i szum?', 'Autokorelacja reszt, stabilność, test przyszłości, porównanie bez pamięci.',
 'Jeżeli reszty pozostają zależne w czasie, model nie uchwycił całej struktury. Sprawdzamy korelacje reszt, stabilność prognozy i porównanie z modelem bez pamięci. Skończona regresja z opóźnieniami nie jest automatycznie pełnym uogólnionym równaniem Langevina z potwierdzoną relacją fluktuacji i dyssypacji.'),
('selection_model','Który model wybierała pętla?', 'Ukryj ostatnie 500 aktualizacji → porównaj błędy → dopasuj zwycięzcę.',
 'Stara pętla porównywała oba modele na ukrytej końcówce dostępnej historii. Potem lepszy wariant dopasowywała ponownie na całej przeszłości i prognozowała następny odcinek. Wygranie z drugim modelem nie oznacza jeszcze wygrania z brakiem zmiany. To odniesienie trzeba pokazać osobno.')])

chapter('11_sila_paczki', 'Nowsze równanie pełnego latentu', [
('full_latent','Od jednej liczby do 768 współrzędnych', 'Δz(t+1) ≈ 0,83·Δz(t) + 0,98·S(t) + reszta.',
 'Nowszy eksperyment opisywał zmianę całego średniego wektora narzędzia, a nie tylko jego rzutu LDA. Proste równanie łączyło poprzednią zmianę z wpływem bieżącej paczki danych. Nadal był to średni wektor sondy, nie sześćset osobnych trajektorii. Współczynniki dopasowano na wcześniejszym odcinku uczenia.'),
('optimizer','Skąd bierze się siła paczki?', 'S(t) = z(θ + część aktualizacji od bieżącego gradientu) − z(θ).',
 'Najpierw liczymy gradient bieżącej paczki. Następnie wydzielamy jego wkład w aktualizację AdamW. Tymczasowo podstawiamy przesunięte wagi i ponownie odczytujemy stałe sondy. Różnica latentu jest siłą paczki w tym eksperymencie. To wymaga dodatkowych przejść sieci i znajomości bieżącego gradientu.'),
('explanation','85 procent: czego dotyczy wynik?', 'Około 85% redukcji błędu kwadratowego względem braku zmiany.',
 'W dziesięciu przebiegach proste równanie usuwało w medianie około osiemdziesięciu pięciu procent błędu przyrostu względem braku zmiany. Nie jest to osiemdziesiąt pięć procent trafnych detekcji. Ponieważ użyto gradientu paczki tworzącej krok, wynik wyjaśnia jej skutek. Nie jest prognozą z samej wcześniejszej historii.', 'WYNIK: WYJAŚNIENIE BIEŻĄCEGO KROKU'),
('no_leak','Jak wygląda czysta prognoza?', 'Przed wyborem przyszłej paczki nie wolno używać jej gradientu.',
 'W prognozie przyszłych kroków trzeba odciąć informacje, których jeszcze nie mamy. Wariant oparty na pędzie optymalizatora też wymaga kontroli: użyty mianownik AdamW zawierał gradient bieżącej paczki. Bez ponownego testu wersji pozbawionej tego wkładu nie traktujemy wyniku jako czystej prognozy.')])

chapter('12_dobor_klatek_i_wnioski', 'Od kierunków do lepszego treningu', [
('retrieval','Historyczna pętla z generatorem', 'Prognoza odczytu → sterowanie FastGAN → podobne prawdziwe klatki.',
 'W dotychczasowej pętli wybieraliśmy sondy o dużej przewidywanej zmianie odczytu. FastGAN odtwarzał obraz kotwicy, a potem zmienialiśmy jego wejście tak, aby przesunąć odczyt w zadanym kierunku. Wygenerowany obraz służył do wyszukiwania podobnych prawdziwych klatek przez cechy DINO. Detektor uczono na wybranych prawdziwych danych.'),
('diversity','Nowość musi być użyteczna', 'Ocena błędu + kierunki braków + różnorodność + rozdział filmów.',
 'Duży ruch lub zaskoczenie modelu nie oznacza automatycznie dobrej klatki treningowej. Może pochodzić z refleksu, rozmycia albo innego obiektu. Kandydatów trzeba powiązać z brakami detektora i zadbać o różnorodność. Sąsiednie niemal identyczne klatki nie powinny zużywać całego budżetu.'),
('controls','Co musi rozstrzygnąć eksperyment?', 'Losowo, sam odczyt, sam ruch, ruch + tożsamość; ten sam budżet.',
 'Ostatecznym testem jest uczenie detektora na jednakowej liczbie klatek i przy porównywalnym koszcie. Potrzebujemy losowego doboru oraz wariantów rozdzielających wpływ odczytu, ruchu i różnorodności. Wynik mierzymy na odłożonych operacjach. Wcześniejsze ramię current też korzystało z prognozy do wyboru kotwic, więc nie usuwało całego wpływu prognozy.'),
('evidence','Co wiemy, a co pozostaje hipotezą?', 'Maski i odczyty: zmierzone. Tożsamość z ruchu: kandydaci. Zysk treningu: osobny test.',
 'Mamy rzeczywisty model, maski, zapisane kierunki odczytu i pierwszy pomiar ruchu pojedynczych tokenów. Nie mamy jeszcze dowodu, że dominujące kierunki ruchu są pełną tożsamością cystotomu. Nie wolno także utożsamiać poprawnej prognozy z poprawą detekcji. Każdy z tych kroków ma w tym cyklu własną kontrolę.'),
('end','Cały pipeline w jednym obrazie', 'Obserwuj → wyznacz kierunki → sprawdź → prognozuj → dobierz → porównaj trening.',
 'Najważniejsza zmiana w twoim pomyśle polega na zachowaniu pojedynczych cząstek i całej ich historii. Maska od jednego piksela wskazuje kandydatów związanych z narzędziem. Kierunki wyznaczamy z danych, ich znaczenie sprawdzamy na innych obrazach, a dopiero później używamy dynamiki do wyboru klatek. Kod, źródła, napisy i wszystkie rozdziały są dołączone do filmu.')])

if __name__ == '__main__':
    (ROOT/'storyboard.json').write_text(json.dumps(CHAPTERS,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Scenariusz cyklu: cząstki ViT, tożsamość, Koopman i Langevin','',
           'Ilustracje nie są danymi. Płynna interpolacja między checkpointami nie jest zmierzoną trajektorią.']
    for c in CHAPTERS:
        lines += ['',f"## {c['id']:02d}. {c['title']}"]
        for s in c['slides']:
            lines += ['',f"### {s['title']}",f"Status: {s['mode']}",s['caption'],s['narration']]
    (ROOT/'SCENARIUSZ.md').write_text('\n\n'.join(lines),encoding='utf-8')
    print(len(CHAPTERS), 'rozdzialow;',sum(len(c['slides']) for c in CHAPTERS),'scen')
