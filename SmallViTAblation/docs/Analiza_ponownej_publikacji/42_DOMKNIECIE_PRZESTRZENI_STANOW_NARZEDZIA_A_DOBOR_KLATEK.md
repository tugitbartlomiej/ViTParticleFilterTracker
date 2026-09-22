# 42. Domknięcie przestrzeni stanów narzędzia a dobór klatek: analiza teoretyczna

Data: 2026-09-22. Status: **analiza teoretyczna, bez nowych pomiarów i bez treningu.** Odpowiada na pytanie autora:
„jeżeli w latencie udało się odseparować wektory odpowiedzialne za położenie, tożsamość, przesunięcie i tło, to czy da
się z tego zbudować pipeline doboru klatek; czy da się zmierzyć, ile brakuje do domknięcia modelu narzędzia i jego
przestrzeni stanu; czy da się sterować teorią Mori-Zwanziga, Koopmanem i Langevinem, jądrem pamięci i szumem, i tak
szukać próbek poprawiających model".

Rozwija dokument [38](38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md) (sekcje 4i do 4l: cząstki jako
tokeny, widmo bez symetryzacji, ślad po przejściu) i [39](39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md) (etap 1, ramiona K5).
Równolegle tego samego dnia inna sesja napisała [40](40_TOZSAMOSC_NARZEDZIA_I_ROLA_ROZSZERZONEGO_LANGEVINA.md)
i [41](41_CZASTKI_TOZSAMOSCI_NARZEDZIA_I_ROWNANIE_LANGEVINA.md). Tamte dokumenty odpowiadają na pytanie „które cząstki
niosą tożsamość" i „pamięć jest ceną rzutowania". Ten dokument ich nie powtarza: bierze rozkład latentu jako dany
i pyta, co z niego wynika dla DOBORU KLATEK oraz jak zmierzyć deficyt domknięcia liczbą, bez etykiet.

## 0. Czego ten dokument NIE twierdzi

- Nie twierdzi, że którakolwiek z opisanych reguł doboru poprawi detektor. Żadnej z nich nie trenowano.
- Nie twierdzi, że latent ma przestrzeń stanów narzędzia. Twierdzi, że da się zmierzyć, jak daleko jej do domknięcia,
  i podaje trzy różne, nierównoważne miary tego dystansu.
- Nie wprowadza nowych bytów do równania grantowego. Używa członów, które już w nim są: operatora, jądra pamięci,
  wymuszenia i szumu.
- Liczby cytowane niżej pochodzą z zapisanych raportów tej gałęzi. Nie liczono ich na nowo do tego dokumentu.

## 1. Co naprawdę zostało rozdzielone (stan pomiarowy, punkt wyjścia)

Rozdzielenie, o którym mówi pytanie, jest częściowe i warto je nazwać dokładnie.

| Składowa | Czym jest u nas operacyjnie | Stan |
|---|---|---|
| tożsamość | osiem liczb z wolnych modów operatora translacyjnego, uśrednione po tokenach narzędzia | zmierzona; odczytuje wydłużenie ramki w 86 procentach |
| położenie | środek ramki detektora, z głowy detekcyjnej | dane wprost, nie trzeba go wyciągać z tokenów |
| przesunięcie | krok operatora: ten sam fragment narzędzia w innym miejscu kadru | to jest definicja kroku, nie osobna współrzędna |
| tło | tokeny wewnątrz ramki, które nie leżą na metalu | NIE rozdzielone; wchodzą do kodu razem z narzędziem |

Trzy pomiary, które trzeba mieć przed oczami przy całej dalszej analizie:

1. Trzy rodziny modów razem (trwałe, przemienne, obrotowe) odróżniają token narzędzia od tokenu tła z wynikiem 0,893,
   same trwałe 0,834 (dokument 38, sekcja 4k). Rozdzielenie jest więc realne, ale nie ostre.
2. Kod tożsamości odczytuje wydłużenie ramki w 86 procentach, a ZWROT narzędzia, czyli po której stronie jest
   końcówka, w zerowym stopniu. To jest konkretna, zlokalizowana dziura w przestrzeni stanów (sekcja 4 niżej).
3. Od 58 do 74 procent zmienności kodu to różnice MIĘDZY filmami, nie między stanami narzędzia. Kod jest więc
   w większości odciskiem operacji i kamery, a stan narzędzia jest w nim mniejszościowy.

Dodatkowo: niezmienniki liczone po czasie (klatka po klatce na wycinku ramki) pokrywają się z niezmiennikami
liczonymi po przesunięciu w pierwszych czterech kierunkach, kosinusy główne 0,95, 0,90, 0,87 i 0,83. Dwie różne drogi
trafiają w te same wektory latentu, co jest argumentem, że te kierunki są własnością reprezentacji, a nie artefaktem
jednej procedury. Dalsze kierunki się rozchodzą (0,54 i niżej).

## 2. Ramy formalne: działanie grupy i rozkład latentu

Niech `z` będzie opisem jednego tokenu-patcha (u nas: profil po redukcji, 32 liczby). Przesunięcie narzędzia o wektor
`d` w kadrze indukuje odwzorowanie `S_d`, które przeprowadza opis danego fragmentu narzędzia w opis tego samego
fragmentu po ruchu. Operator Koopmana `T_d` działa nie na stanie, lecz na FUNKCJACH stanu:

    (T_d f)(z) = f(S_d z)

Funkcja `f` jest niezmiennikiem przesunięcia, gdy `T_d f = f` dla każdego `d`. Zbiór takich funkcji to algebra
niezmienników; formalnie są to funkcje na przestrzeni ilorazowej, czyli na latencie „podzielonym" przez ruch.
Funkcje, które zmieniają się w sposób związany z `d` w ustalony sposób, są składową ekwiwariantną i niosą pozę.

Stąd czysty rozkład, który porządkuje całe zadanie:

    latent tokenu  =  część niezmiennicza (tożsamość, kształt, tło, operacja)
                    (+) część ekwiwariantna (poza: położenie, kąt, skala)
                    (+) reszta (szum, błąd ramki, rozmycie ruchu)

Nasz operator jest estymowany na parach przy RÓŻNYCH `d`, po czym bierzemy mody o module wartości własnej blisko
jedynki. To jest estymator algebry niezmienników. Trzy rodziny modów z sekcji 4k mają w tym języku sens dosłowny:
mod o dodatniej części rzeczywistej to wielkość, która po przesunięciu wraca ta sama tylko słabsza; ujemna część
rzeczywista to wielkość zmieniająca znak, czyli niezmiennik dopiero po DWÓCH przesunięciach; para zespolona to
dwie wielkości obracające się o stały kąt, czyli niezmiennikiem jest dopiero ich długość.

Wniosek formalny, którego wcześniej nie zapisano: **niezmiennikami są moduły, nie same współrzędne.** Dla rodziny
przemiennej niezmiennikiem jest kwadrat wartości, dla obrotowej suma kwadratów pary. Dlatego rodziny te same w sobie
wypadały słabo (0,743 i 0,593), a razem z trwałymi dokładały dużo: użyliśmy ich jako współrzędnych liniowych,
a powinny wejść jako wielkości kwadratowe. To jest natychmiastowa, tania poprawka do przeliczenia kodu tożsamości.

## 3. Trzy różne znaczenia „domknięcia" i miara dla każdego

Pytanie „ile brakuje do domknięcia modelu narzędzia" ma trzy nierównoważne odpowiedzi. Mieszanie ich jest głównym
źródłem nieporozumień, więc rozdzielam je i każdej daję estymator liczony BEZ etykiet.

### 3a. Domknięcie grupowe: czy nasz operator w ogóle jest reprezentacją ruchu

Jeżeli `T` opisuje ruch poprawnie, to złożenie dwóch małych przesunięć musi dać to samo, co jedno duże:

    T_a ∘ T_b  =  T_(a+b)

Miara braku domknięcia to defekt złożenia, liczony na parach odłożonych (nieużytych do dopasowania):

    D1 = ‖ T_a T_b − T_(a+b) ‖  /  ‖ T_(a+b) ‖

Wersja lokalna, na klatkę, jest ciekawsza dla doboru. Dla mody `φ` o wartości własnej `λ` i pary tokenów `(x, x')`
złączonej przesunięciem:

    r(x) = | φ(x') − λ φ(x) |  /  ( |φ(x)| + ε )

`r` mówi, w którym miejscu puli deklarowana niezmienniczość się łamie. To jest sygnał zupełnie inny niż pewność
detektora i inny niż pokrycie: nie pyta „czy sieć jest pewna" ani „czy widziałem podobną klatkę", tylko „czy w tym
miejscu moja teoria stanu narzędzia się zgadza".

Co już wiemy: defekt widma spadł z 0,19 przed uczeniem do 0,05 po uczeniu, a liczba niezmienników wzrosła z dwóch do
czterech lub pięciu (dokument 38). Domknięcie grupowe ROŚNIE z treningiem bez żadnych etykiet w tej mierze, więc jest
to sensowny zegar postępu. Czy jest sensownym selektorem, nie wiadomo; przewidywanie i test w sekcji 8.

### 3b. Domknięcie wymiarowe: czy liczba stopni swobody się zgadza

Poza narzędzia w kadrze ma cztery stopnie swobody: dwa przesunięcia, obrót w płaszczyźnie obrazu i skala. Pochylenie
w głąb dokłada jeden lub dwa, zależnie od modelu. Artykulacja, czyli otwarcie narzędzia, dokłada jeden, ale w tym
zbiorze nie mamy dla niej etykiety i nie da się jej potwierdzić.

Miara: efektywny wymiar części ekwiwariantnej powinien wynosić około czterech, a nie kilkanaście. Efektywny wymiar
części niezmienniczej powinien być mały, bo w tym podzbiorze jest jedno narzędzie (cystotom, 22 680 klatek). Jeżeli
część niezmiennicza ma wysoki wymiar, to nadmiar opisuje tło i operację, a nie narzędzie, co zgadza się z pomiarem
58 do 74 procent zmienności między filmami.

Estymator bez etykiet: wymiar uczestniczący (participation ratio) widma każdej z dwóch części,

    d_eff = ( Σ s_i )² / Σ s_i²

Deficyt domknięcia wymiarowego to różnica `d_eff` części ekwiwariantnej wobec czterech oraz nadmiar `d_eff` części
niezmienniczej ponad liczbę realnych klas narzędzi. Ta miara nie wymaga ani jednej etykiety i liczy się w sekundach.

### 3c. Domknięcie Markowa: jądro pamięci jako miara tego, czego stan nie obejmuje

To jest odpowiedź, w której wchodzi Mori-Zwanzig. Jeżeli wybrane współrzędne `z` są PEŁNYM stanem, to następny krok
zależy tylko od bieżącego stanu. Rzutowanie na niepełny zestaw współrzędnych generuje dokładnie dwa dodatkowe człony:

    z(t+1) = A z(t) + Σ_{k≥1} K(k) z(t−k) + ξ(t)

`K` to jądro pamięci, `ξ` to szum. Twierdzenie o fluktuacji i dyssypacji wiąże je ze sobą: jedno nie może zniknąć bez
drugiego. Stąd interpretacja operacyjna, która jest sednem tej sekcji:

> Wielkość jądra pamięci mierzy, ile stopni swobody narzędzia latent POMIJA. Wielkość szumu mierzy, ile z tego jest
> nieusuwalne przy dowolnej liczbie etykiet.

Miara deficytu: udział wariancji przyrostu wyjaśniany przez historię ponad to, co wyjaśnia sam bieżący stan,

    Δ_pamięci = R²(stan + historia) − R²(sam stan)

liczony poza grupą dopasowania. To jest liczba „ile brakuje", której szukamy, i ma ona sens niezależny od etykiet.

Co już wiemy i co to psuje: w projekcie grantowym (ustalenie U14) jądro pamięci na cząstkach Eulera miało głębokość
JEDNEGO kroku i tłumaczyło od 5 do 6 procent przyrostu, a korelacja tempa latentu z ruchem kamery wynosiła 0,949.
U nas ramię z historią nie dało nic ponad ramię bez historii. Wniosek: przy obecnych współrzędnych deficyt pamięci
jest mały, ale nie dlatego, że stan jest domknięty, tylko dlatego, że mierzony ruch jest w większości ruchem kamery,
a nie ruchem narzędzia. Zanim `Δ_pamięci` stanie się użyteczna, trzeba odjąć ruch kamery, czyli liczyć ją na
współrzędnych niezmienniczych, a nie na surowym latencie. Tego dotąd nie robiono i to jest pierwszy nowy pomiar
wynikający z tej analizy.

## 4. Zlokalizowany brak: zwrot narzędzia ginie w ESTYMATORZE, nie w latencie

Najmocniejszy wynik teoretyczny tej analizy dotyczy konkretnej dziury w przestrzeni stanów.

Zwrot narzędzia, czyli po której stronie ramki jest końcówka, jest wielkością NIEZMIENNICZĄ względem przesunięcia:
przesuń narzędzie, a zwrot się nie zmieni. Powinien więc należeć do algebry niezmienników i dać się odczytać z kodu.
Pomiar mówi, że nie daje się odczytać wcale: od 0,1 do 0,3 procent, czyli poziom zera.

Wyjaśnienie jest czysto rachunkowe. Kod klatki liczymy jako ŚREDNIĄ po tokenach narzędzia:

    kod = (1/N) Σ_i φ(z_i)

Średnia jest symetryczna względem permutacji tokenów. Każda wielkość, która zależy od tego, po której stronie osi
narzędzia leży dany token, jest funkcją NIEPARZYSTĄ wzdłuż tej osi i uśrednianie ją kasuje z definicji. Zwrot jest
dokładnie taką wielkością. Latent może ją nieść w pełni, a mimo to nasz estymator zwróci zero.

Poprawka wynika z tego samego rachunku: obok momentu zerowego (średniej) policzyć moment pierwszy, ważony
współrzędną względną wzdłuż osi ramki,

    kod₁ = (1/N) Σ_i u_i · φ(z_i),     u_i = położenie tokenu wzdłuż osi ramki, od −1 do +1

`kod₁` zachowuje część nieparzystą i jest nadal niezmienniczy względem przesunięcia, bo `u` liczymy względem ramki,
nie względem kadru. Przewidywanie: po dołożeniu momentu pierwszego odczyt zwrotu końcówki wyskoczy z zera, a odczyt
wydłużenia się nie pogorszy. To jest test, który albo potwierdzi, albo obali całe rozumowanie, i kosztuje jeden
przebieg na gotowym nagraniu.

Ta sama logika obejmuje tło. Tokeny bierzemy z wnętrza ramki, więc kod jest mieszaniną narzędzia i tła w proporcji
zależnej od grubości narzędzia. Ponieważ kod jest średnią, mieszanina wchodzi liniowo:

    kod_zmierzony = α · kod_narzędzia + (1 − α) · kod_tła

gdzie `α` to udział tokenów narzędzia. Zmienność `α` między klatkami (cienkie narzędzie w dużej ramce wobec grubego
w małej) wchodzi więc do kodu jako fałszywa zmienność stanu. Poprawka bez etykiet: ważyć tokeny podobieństwem do
wzorca narzędzia, czyli miękką maską. Wzorzec liczy się ze średniej po wnętrzach ramek, więc nadal nie ma tu etykiet.

## 5. Od domknięcia do doboru klatek: trzy reguły i czym różnią się od już sprawdzonych

Etap 1 sprawdził dziewięć ramion i ustalił, czego NIE robić. Reszta Langevina nie pomaga (placebo równe prawdziwemu
sygnałowi), pewność detektora nie pomaga (kupiła 69 procent klatek niepewnych i nic z tego nie wynikło), pokrycie
wyglądu nie bije równomiernego czasu. Jedyny kontrast powyżej progu to pokrycie modów Koopmana, plus 0,0068. Nowa
reguła ma sens tylko wtedy, gdy celuje w coś, czego tamte ramiona nie mierzyły.

**Reguła A: defekt niezmienniczości.** Wybieraj klatki o największym `r(x)` z sekcji 3a, po rangach wewnątrz filmu.
Czym się różni: mierzy łamanie deklarowanej struktury, nie niepewność i nie odległość. Ryzyko teoretyczne, które
trzeba sprawdzić PRZED treningiem: defekt może być duży tam, gdzie ramka detektora jest zła, i wtedy reguła zapadnie
się do ramienia pewności, które już przegrało. Test: korelacja `r` z pewnością na puli. Gdy przekracza 0,6, reguła
jest zbędna.

**Reguła B: informacja o parametrach, które liczą się dla zadania.** Pilot D-optymalny zmniejszył niepewność
OPERATORA (ślad kowariancji 0,116 wobec 0,210) i jednocześnie pogorszył przewidywanie kroku poza grupami (4,0 wobec
7,3). To nie jest paradoks, tylko wskazówka: identyfikowanie operatora nie jest tym samym, co uczenie detektora.
Teoretycznie poprawny obiekt to informacja Fishera parametrów, od których zależy STRATA zadania, rozłożona na dwie
części rozkładu z sekcji 2:

    kryterium = waga_pozy · (deficyt odczytu pozy) + waga_tożsamości · (deficyt odczytu tożsamości)

Oba deficyty da się szacować sondą liniową bez etykiet ręcznych, biorąc jako cel wyjście samego detektora.
Czym się różni od D-optymalnego: tamten maksymalizuje informację o dynamice, ten o odczycie, którego używa głowa.

**Reguła C: pokrycie w przestrzeni ilorazowej, nie w przestrzeni cech.** To jest ramię K5, tyle że teraz z uzasadnieniem
formalnym: pokrywamy przestrzeń po podzieleniu przez ruch, więc dwie klatki różniące się tylko położeniem narzędzia
są dla nas TĄ SAMĄ próbką i nie kupujemy jej dwa razy. Etap 1 tego nie testował, bo żadne jego ramię nie różniło się
od losu w tej przestrzeni (mediana pokrycia od 0,84 do 0,87 dla wszystkich, K5 schodzi z największą luką z 2,9 do 1,3).

Wspólna uwaga do wszystkich trzech: jednostką doboru nie musi być klatka. Skoro deficyt domknięcia jest funkcją
OBSZARU przestrzeni stanów, naturalną jednostką jest obszar, a klatki są tylko jego próbkami. To zmienia pytanie
z „którą klatkę oznaczyć" na „w którym rejonie stanu jestem niedomknięty i ile klatek stamtąd potrzebuję".

## 6. Sterowanie: dobór jako wymuszenie, i co ogranicza zysk

W równaniu grantowym wymuszenie `W u` jest tym, co wchodzi do układu z zewnątrz. Dla dynamiki latentu w czasie filmu
wymuszeniem jest ruch kamery. Dla dynamiki latentu w czasie TRENINGU wymuszeniem jest zbiór etykiet: to jedyna
wielkość, którą sterujemy. Dobór klatek jest więc zadaniem sterowania, a nie zadaniem statystyki, i ma trzy konsekwencje.

**Po pierwsze, osiągalność.** Nie każdy stan latentu jest osiągalny przy danym wymuszeniu. Dla układu liniowego mówi
o tym gramian sterowalności; kierunki, w których jest on mały, są nieosiągalne niezależnie od liczby próbek. Jeżeli
brakująca współrzędna (na przykład zwrot narzędzia) jest w takim kierunku, ŻADNA reguła doboru jej nie domknie
i trzeba zmienić architekturę albo stratę, nie dane.

**Po drugie, sufit z szumu.** Twierdzenie o fluktuacji i dyssypacji wiąże jądro pamięci z szumem. Praktycznie: część
błędu pochodząca z szumu (rozmycie ruchu, błędy adnotacji, klatki bez widocznego narzędzia) jest nieusuwalna przy
dowolnym doborze. Nasz pomiar tego sufitu istnieje: chybienia w szybkim ruchu to około 32 procent w KAŻDYM ramieniu,
a usuwalne jest tylko 23 procent chybień małej sieci. To są liczby, o które nie warto się bić doborem.

**Po trzecie, sprzężenie zwrotne.** Gdy dobór zależy od bieżącego modelu, a model od doboru, mamy pętlę zamkniętą
i klasyczny problem obciążenia identyfikacji w pętli: dane przestają być reprezentatywne dla rozkładu, bo model sam
wybiera, czego się uczy. Lekarstwo z teorii sterowania jest znane i tanie: dodać składnik wzbudzenia, czyli stałą
domieszkę klatek losowych albo równomiernych w czasie. Ramię K5b, które oddaje 20 procent budżetu na klatki bez kodu,
jest dokładnie takim wzbudzeniem, tyle że postawionym intuicyjnie. Teraz ma uzasadnienie.

## 7. Sufit: ile teoria pozwala ugrać przy 1500 etykietach

Zbierając trzy liczby: usuwalne jest 23 procent chybień, najlepszy sygnał trafia w usuwalne z precyzją 7 procent
w górnym decylu, a najlepszy zmierzony kontrast między ramionami to 0,0068 AP. Każda reguła doboru operuje wewnątrz
tych 23 procent, bo reszty etykiety nie naprawią. Dlatego realistyczne oczekiwanie wobec reguł A, B i C to pojedyncze
tysięczne AP, a nie skok jakościowy. To nie jest argument, żeby ich nie testować, bo przy trzech ziarnach i dwóch
budżetach taki efekt da się wykazać. To jest argument, żeby nie obiecywać więcej.

Główny wniosek teoretyczny tego dokumentu brzmi zatem tak: **struktura niezmiennicza jest znacznie mocniejsza jako
człon STRATY niż jako reguła DOBORU.** Dobór przesuwa etykiety wewnątrz usuwalnych 23 procent. Strata wymuszająca
niezmienniczość (ta sama wielkość ma wyjść tak samo dla tego samego narzędzia w dwóch położeniach) działa na WSZYSTKICH
klatkach, także nieoznaczonych, więc nie jest ograniczona pułapem usuwalności. Ten sam operator, to samo widmo, ta
sama teoria, ale wstawiona w inne miejsce potoku uczenia.

## 8. Przewidywania falsyfikowalne, od najtańszego

Każde z poniższych jest bez etykiet ręcznych i bez treningu detektora. Kolejność oddaje stosunek wartości do kosztu.

1. **Moment pierwszy odzyskuje zwrot narzędzia.** Po dołożeniu `kod₁` z sekcji 4 odczyt zwrotu końcówki wzrośnie
   z zera do co najmniej kilkunastu procent, a odczyt wydłużenia nie spadnie. Obalenie: zwrot dalej na poziomie zera,
   co znaczyłoby, że latent naprawdę go nie niesie.
2. **Niezmienniki kwadratowe biją liniowe.** Użycie kwadratu wartości dla rodziny przemiennej i sumy kwadratów pary
   dla obrotowej podniesie odróżnienie narzędzia od tła ponad 0,893. Obalenie: bez zmiany albo gorzej.
3. **Miękka maska usuwa fałszywą zmienność.** Ważenie tokenów podobieństwem do wzorca zmniejszy udział zmienności
   między filmami poniżej 58 procent. Obalenie: udział nie drgnie, co znaczyłoby, że to nie tło, tylko kamera.
4. **Deficyt pamięci policzony na współrzędnych niezmienniczych jest większy niż na surowym latencie.** Bo z tych
   pierwszych odjęto ruch kamery, który był głównym źródłem gładkości. Obalenie: `Δ_pamięci` nadal bliskie zeru,
   co znaczyłoby, że stan jest domknięty i szukanie brakujących współrzędnych nie ma sensu.
5. **Defekt niezmienniczości nie jest przebraną pewnością.** Korelacja `r` z pewnością detektora poniżej 0,6.
   Obalenie: powyżej, i wtedy reguła A odpada bez treningu, tak jak odpadł D-optymalny.
6. **Domknięcie wymiarowe rośnie z treningiem.** Efektywny wymiar części ekwiwariantnej zbliża się do czterech między
   wagami startowymi a końcowymi. Obalenie: rośnie albo stoi, co podważyłoby całą narrację o krystalizacji.

Dopiero gdy przynajmniej pierwsze trzy wyjdą, ma sens trenowanie ramienia opartego na regułach A, B lub C, bo
inaczej trenowalibyśmy kod, o którym wiadomo, że gubi połowę informacji.

## 9. Kontrargumenty

- Cały rozkład opiera się na założeniu, że ruch narzędzia w kadrze jest przesunięciem. Narzędzie obraca się i zmienia
  skalę, więc prawdziwa grupa jest większa, a nasz operator uśrednia po niej. Część tego, co nazywamy defektem
  domknięcia, może być po prostu obrotem, którego nie modelujemy.
- Niezmienniki liczone po czasie miały moduł 0,989 przy podłodze 0,966 dla potasowanego czasu. Prawie cała trwałość
  bierze się z bezwładu latentu, nie z fizyki narzędzia. Ta sama pułapka wywróciła wcześniejsze wnioski w projekcie
  grantowym i trzeba ją raportować przy każdej liczbie typu „moduł blisko jedynki".
- Strażnik determinizmu rejestratora nie przechodzi (0,131 jednostki współrzędnej). Dopóki nie przejdzie, różnice
  rzędu kilku setnych w miarach na pojedynczych tokenach są nieodróżnialne od szumu numerycznego.
- Teza z sekcji 7, że strata bije dobór, jest wnioskiem z trzech liczb zmierzonych na jednym zbiorze i jednym
  detektorze. Na większym budżecie etykiet albo innym zbiorze proporcje mogą wyglądać inaczej.
- Nie ma etykiety stanu narzędzia (otwarte, zamknięte, kąt), więc „prawidłowa przestrzeń stanów" jest u nas
  rekonstruowana z ramki i końcówki. Wszystkie odczyty stanu są przez to odczytami geometrii ramki.

## 10. Data gaps

- Brak etykiety artykulacji i kąta narzędzia; bez nich domknięcia wymiarowego nie da się potwierdzić, tylko oszacować.
- Brak obrysu narzędzia; miękka maska jest jedynym zamiennikiem i sama wymaga sprawdzenia.
- Nie policzono `Δ_pamięci` na współrzędnych niezmienniczych ani gramianu sterowalności dla osi treningu.
- Nie wiadomo, czy niezmienniki policzone na jednym checkpoincie przenoszą się na następny; wszystkie ramiona
  zakładają, że kod z L0 opisuje stan także po douczeniu.
- Nie porównano głębokiego rozkładu Koopmana (autoenkoder z operatorem liniowym w podniesionej przestrzeni)
  z wersją liniową; pierwsze cztery kierunki obu dotychczasowych dróg pokrywają się powyżej 0,83, więc test ma sens
  dopiero dla kierunków dalszych.
