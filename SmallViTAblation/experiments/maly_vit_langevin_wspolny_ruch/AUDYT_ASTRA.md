# Audyt Astra: wspólny ruch cząsteczek, otoczenia i składowych pomocniczych

Data: 2026-10-02. Status: **FINAL — audyt matematyki, kodu, wyników liczbowych i końcowej interpretacji zakończony bez błędu blokującego. Wynik pozostaje eksploracyjny; nie uzyskano ogólnej przewagi nowego modelu ani skutecznego transferu długiej prognozy.**

Własność audytora obejmuje wyłącznie ten plik. Audytor nie zmienia kodu, parametrów, planu ani wyników wykonawcy i nie uruchamia dodatkowych dopasowań lub forwardów. Stosuje zasady dowodowe procedury `audit-project-correctness`; miejsce zapisu ograniczono do pliku wyraźnie wskazanego w zleceniu, bez zmian osobnego rejestru Grant/Research/Audyty. Wszystkie poniższe uwagi o przyszłym zachowaniu są warunkami weryfikacji, dopóki nie zostaną potwierdzone kodem i artefaktami.

Poprzedni audyt: [Langevin w kierunkach U500](../maly_vit_langevin_kierunki_tozsamosci/AUDYT_ASTRA.md). Nowy eksperyment korzysta z tego samego biegu i wcześniej oglądanej historii; zmiana modelu nie tworzy świeżego, niezależnego testu potwierdzającego.

## Rozliczenie wcześniejszego zakresu

| Ustalenie poprzednika | Dowód i bieżące znaczenie |
|---|---|
| Sonda 2 ma źródłową rolę fit/train | Zachować prawidłową rolę; nowy podział jest czasowy. |
| U1000 przecieka przyszłość dla wcześniejszych originów | Nowa prognoza korzysta z zamrożonego U500; U1000 pozostaje wyłącznie retrospektywne. |
| Poprzedni rollout autonomiczny i skale fizyczne poprawne | Nie przenosi to automatycznie weryfikacji na nowy coupled rollout; potrzebna ponowna kontrola nowych ścieżek. |
| 81.59% znaku po filtrze nie oznacza pewności online | Próg może pochodzić z fit, ale odsiew według rzeczywistej zmiany wymaga poznania celu. |
| Tokeny są współzależne, jedna trajektoria daje mało bloków | Bootstrap wspólnie po originach, bez niezależnego losowania tokenów; CI pozostają opisowe. |
| Brak faktycznych batch-ID/augmentacji tego biegu | Otoczenie i składowe pomocnicze są obserwacjami stanu, nie zapisanymi wejściami sterującymi treningiem. |
| Zbliżony wynik prognozy także na tle | Sukces nowego modelu nie dowodzi sam w sobie semantycznej wyłączności ruchu narzędzia. |

## A-01. Domknięcie prognozowanej średniej foreground

**Status dowodowy: POTWIERDZONE algebraicznie; implementacja jeszcze nieweryfikowana. Priorytet P1, pewność wysoka.**

Niech s_i będzie zakotwiczonym stanem cząsteczki, a g=sum_i w_i s_i przy stałych, znormalizowanych wagach coverage. Jeżeli wszystkie cząsteczki mają wspólne macierze oraz równanie

```text
Δs_i = b + A s_i + C_fg g + C_bg h
       + sum_l [K_l Δs_i(t-l) + G_l Δg(t-l) + H_l Δh(t-l)],
```

to średnia spełnia dokładnie

```text
Δg = b + (A+C_fg) g + C_bg h
     + sum_l [(K_l+G_l) Δg(t-l) + H_l Δh(t-l)].
```

Średnia g nie jest dodatkowym niezależnym pomiarem ani dowolnym stanem. W każdym rollout musi zostać wyznaczona z aktualnych prognoz wszystkich s_i i tych samych wag. Oddzielnie dopasowana, niespójna prognoza g złamałaby tę tożsamość. Aktualizacje foreground i h muszą korzystać z jednego snapshotu czasu t, a dopiero potem równocześnie przejść do t+dt.

Dla odchylenia d_i=s_i−g wszystkie wspólne człony znikają:

```text
Δd_i = A d_i + sum_l K_l Δd_i(t-l).
```

Zaproponowany model opisuje wspólny ruch i lokalne odchylenia. Nie dopuszcza, aby to samo otoczenie dodawało różny przyrost różnym tokenom. Nie identyfikuje ogólnych parowych sił ani zależnych od cząsteczki interakcji. Ta własność nie blokuje pierwszego ograniczonego testu; ogranicza interpretację jego wyniku.

W regresji tokenowej s_i i g nie są z definicji dokładnie współliniowe, jeżeli odchylenia d_i mają dostateczną zmienność. Przy słabych lub niskorangowych odchyleniach rozdzielenie A i C_fg może być źle określone. W regresji samej średniej identyfikowany jest tylko A+C_fg. Ridge wybiera regularizowaną reprezentację współczynników; nie usuwa przyczynowej nieidentyfikowalności. Sprawdzić rangę i uwarunkowanie rzeczywistego design oraz brak zdublowanych bloków globalnych cech.

## A-02. Dostępność czasowa kierunków pomocniczych

**Status dowodowy: RYZYKO do weryfikacji w implementacji. Priorytet P1.**

U500 pozostaje stałe. Pomocnicze V ma być ortonormalne i ortogonalne do U500, przy czym warto sprawdzić numerycznie zarówno VᵀV, jak i U500ᵀV. Redukcja residualnych przyrostów może korzystać wyłącznie z danych zakończonych najpóźniej w fit=700. PCA obejmująca przyrosty 501..700 jest dostępna przy pierwszym originie validation=700. PLS/SVD crosscov z residualnego przyrostu t do Δq(t+1), t=500..699, jest również dostępne: ostatni target to 700, a pierwszy potrzebuje znanej historii 499..500.

Centrowanie, wagi i ewentualne skale crosscov muszą pochodzić z tego samego prefiksu. Po wyznaczeniu V powinno być zamrożone podczas refitu operatorów do 799 oraz podczas transferu do sondy 8. Dobór wariantu PCA/PLS i rangi może korzystać z validation zgodnie z wcześniej ustaloną siatką, bez wyboru po wyniku testowym.

Sama projekcja na V nie jest dowodem, że uzyskane współrzędne reprezentują tożsamość narzędzia. Są to pomocnicze kierunki ruchu lub predykcji poza U500, których użyteczność ma rozstrzygnąć ablation.

## A-03. Autonomiczny rollout i uczciwe porównanie

**Status dowodowy: RYZYKO do weryfikacji w implementacji. Priorytet P1.**

Prognozować trzeba wszystkie własne współrzędne q i r, stan otoczenia oraz ich historie przyrostów. Przyszłe średnie i cechy pomocnicze muszą pochodzić z własnych prognoz. Kontrola prefix/poison powinna obejmować dane wejściowe przed tworzeniem r i globalnych średnich; kontrola już przygotowanej tablicy cech nie wykryje przecieku powstałego podczas jej konstruowania.

MSE i znak wszystkich modeli należy liczyć w tych samych fizycznych współrzędnych q, na tych samych tokenach, wagach i originach. Rozszerzony stan nie może sztucznie poprawiać metryki przez zmianę mianownika lub ocenę łatwiejszych współrzędnych. Poprzedni exact U8 GLE stanowi kontrolę zgodności; ewentualne nowe ponowne strojenie tej kontroli należy nazwać osobnym wariantem.

Probe 8 po wyborze parametrów jest transferem między obrazami tego samego biegu. Nie jest drugim ziarnem treningu. Liczba tokenów może się zmienić, ale współczynniki operatora mają pozostać wspólne, a średnie muszą mieć ponownie znormalizowane wagi. Źródłowe tokeny poza maską mogą zawierać inne obiekty i nie są gwarantowanym czystym tłem.

## Stan weryfikacji

Przekazano wykonawcy i rootowi powyższe ograniczenia architektury przed ukończeniem kodu. Nie wykryto matematycznej konieczności porzucenia zaproponowanego modelu. Weryfikacja kodu, zapisanych wyborów validation, autonomicznych testów oraz końcowych metryk pozostaje do wykonania.

## A-04. Kontrola zmiany standaryzacji i siatki

**Status dowodowy: POTWIERDZONE w kodzie; zgłoszono przed testem. Priorytet P1 dla atrybucji korzyści sprzężenia, pewność wysoka.**

`model.py:8–16` standaryzuje każdą kolumnę regresji osobnym odchyleniem standardowym, również osobno dla kolejnych lagów. Poprzedni `gle.py:8–12`, zachowany w `metryki_bazowe.py`, stosował wspólny wektor skali przyrostów dla wszystkich lagów. Nowa siatka ridge także różni się od poprzedniej. Porównanie samego coupled U8 ze starym GLE zawiera zatem więcej niż jedną zmianę.

Do rozdzielenia tych zmian potrzebny jest nowy lokalny U8, `rank=0, coupled=False`, dobrany na tej samej walidacji, siatce i implementacji co coupled U8. Stary exact GLE nadal jest poprawną kontrolą historycznej skuteczności, lecz sam nie izoluje wpływu sprzężenia. Uwagi przekazano workerowi i rootowi przed ukończeniem etapu validation; realizacja tej ablation pozostaje do sprawdzenia.

**Aktualizacja przed testem:** `analiza.py:8` zawiera wymagany `localU8` z `aux=0, coupled=False`, obok `coupledU8` i pozostałych rodzin. `:43–58` stosuje tę samą nową siatkę, ważenie oraz kryterium wyboru. Root zmienił też plan z wariantu opcjonalnego na wymagany. Warunek konstrukcyjny został wdrożony i sprawdzony statycznie; wynik wykonania zostanie rozliczony przy odczycie walidacji i metryk.

## Odczyt kodu konstrukcji przed walidacją

Przeczytano `PLAN.md`, `przygotuj.py`, `model.py`, `common.py` i kopię `metryki_bazowe.py`. W odczytanej wersji `model.py:19–30` buduje znormalizowaną czasowo i tokenowo regresję foreground oraz oddzielną regresję kontekstu. W każdym operatorze poziomy i pamięć są dopasowane wspólnie. Fizyczne współczynniki w konwencji wierszowej spełniają `B=diag(1/s_x) B_standard diag(s_y)` i `b=mean_y−mean_x B`; kontrola `:13–14` jest algebraicznie poprawna.

`model.py:40–60` wylicza przyrosty foreground i kontekstu przed podmianą stanów, a średnią foreground tworzy ponownie z własnych przewidywanych tokenów. Przyszła pamięć jest uzupełniana przewidzianymi przyrostami. Ten odczyt nie wykazał teacher forcing ani niespójnego oddzielnego stanu foreground. `:62–77` analizuje poprawną dekompozycję na odchylenia tokenów oraz układ średniej i kontekstu; promienie widmowe dotyczą dopasowanego modelu, nie stacjonarności prawdziwego procesu treningu.

`przygotuj.py:18–23` używa dokładnie residualnych przyrostów PCA 501..700 i par PLS z predyktorami 500..699 oraz celami 501..700. Wagi są wspólne dla czasu i coverage tokenów; centrowanie wykorzystuje wyłącznie te pary. `:25–30` ortogonalizuje V względem U, sprawdza oba warunki ortonormalności i zapisuje U/V, centra oraz widmo. Pozostałe stany są jedynie rzutowane ustaloną bazą, co samo w sobie nie narusza dostępności czasowej.

## Odczyt pipeline i przygotowanych danych

`DANE_PROJEKCJE.npz` ma 10041177 bajtów, a jego SHA zgadza się z `DANE.json`. Probe 2 to 31 tokenów maski i 545 tokenów źródłowego tła. Probe 8 to `cataract_lmm/tracking/TR_0134_S1_P03/00000341`, rola `holdout`, ale split źródłowy nadal `train`: 7 tokenów maski i 584 tokeny źródłowego tła. Wyniku transferu nie należy nazywać niezależnym testem na obrazie spoza treningu detektora.

Każda tablica s ma 1001 stanów, liczbę tokenów właściwą dla sondy i 16 współrzędnych; kontekst ma shape 1001×16. PCA/PLS korzystają z tych samych pierwszych ośmiu q, uzupełnionych o osiem współrzędnych V. Numeryczne UᵀV ma maksimum około 6.0e-17 dla PCA i 8.2e-17 dla PLS; błąd VᵀV względem I nie przekracza 1.2e-15. `metryki_bazowe.py` jest tekstowo identyczną kopią poprzedniego `gle.py`; różnica SHA samych plików wynika z zakończeń linii, nie zmiany funkcji.

W `analiza.py:43–68` każda rodzina jest dopasowana do target700, wybierana z walidacji i następnie refitowana do799/dt1 lub790/dt10. Ranga i metoda V są elementem rodziny. Wybór globalnego zwycięzcy może pozostać przy starym GLE, jeżeli jego uprzednio zapisany wynik walidacyjny jest najlepszy. Osobny wybór krótkiego horyzontu nie zastępuje głównego. Macierze i SHA projekcji są zapisywane przed wywołaniem etapu testowego.

`analiza.py:77–103` ocenia obie sondy dopiero po wyborze na sondzie2. Metryka zawsze obejmuje tylko pierwsze osiem q. Progi znaku i majority baseline są wspólne dla wszystkich metod i pochodzą ze starego fit na sondzie2, także podczas transferu8. Bootstrap losuje wspólne originy; porównanie z oldGLE jest sparowane. Prognoza nie zmienia operatorów na sondzie8, lecz używa jej własnego znanego anchora500.

Kontrola `:87–92` zatruwa q, r i kontekst oraz usuwa sufiks przed wykonaniem rollout; średnie foreground powstają dopiero wewnątrz rollout. Projekcje s/bg zostały wcześniej policzone punktowo przez stałą bazę, bez filtrowania w czasie. Łącznie z audytem `przygotuj.py` obejmuje to nową ścieżkę preprocessingu. Sama ta kontrola nadal nie sprawdza fitu V — jego granica <=700 wynika z osobno przeczytanego kodu i metadanych.

Przed testem poproszono jeszcze o bezpośrednie odtworzenie starego q/GLE wobec zapisanych projekcji i prognoz poprzednika. Ponadto porównanie dt1 z dt10 ma korzystać ze wspólnych 16 originów co10, a nie mieszać główne 151 originów dt1 z 16 originami dt10. Można to uzyskać z zapisanych prognoz bez ponownego dopasowania.

Oddzielny operator kontekstu ma 200 wierszy fit dla dt1 i 20 dla dt10, następnie 299/29 po reficie. Liczba tokenów foreground nie zwiększa tej liczby. Przykładowo D16 i L8 daje 288 cech bg, przy maksymalnej randze wycentrowanego design 19 dla fit dt10. Regularyzacja umożliwia prognozę, ale nie czyni pełnej macierzy bg jednoznacznie identyfikowalną z danych. Wykonawcy przekazano wymóg osobnego raportowania target count, rank, feature count i condition dla fg/bg.

## Zamknięcie audytu przedtestowego

Odczytano `ZGODNOSC_STARY_GLE.json`: nowe q sondy2 są dokładnie równe poprzednim, maxabs=0. Odtworzenie starych wybranych macierzy względem zapisanych prognoz dt1/H1,H10,H50 i dt10/H10,H50 także ma maxabs=0. To potwierdza porównywalność celu i poprawność użycia poprzedniej walidacji jako zamrożonej kontroli.

`analiza.py:100–103` zapisuje dodatkowe porównanie wszystkich modeli dt1 na 16 originach co10, dla obu sond, przez wybór pozycji 0,10,...150 z już wykonanych prognoz. Nie wykonuje nowego fitu i nie zmienia wyboru modelu.

Weryfikacja SHA przed testem: oba pliki macierzy oraz dane zgadzają się z zapisami `WYBOR_DT*.json`. Przeczytana walidacja zawiera 200 nowych konfiguracji dt1 i 250 dt10, wszystkie z wynikiem finite. Siatka jest duża względem 50/5 originów walidacyjnych, które ponadto mają nakładające się cele; wynik selekcji jest eksploracyjny, bez dowodu stabilności wyboru w innych biegach.

| dt | Główny wybór z walidacji | L / ridge | Mean skill validation | Osobny wybór najkrótszego H |
|---:|---|---|---:|---|
|1|localPLS8, stan16D|2 / 1|28.896695%|localPCA4, L2 / ridge0.01, H1 skill63.852139%|
|10|coupledPLS4, stan12D|0 / 10|6.851077%|localPLS4, L1 / ridge1, H10 skill8.698171%|

Główny dt1 został wybrany bez sprzężenia globalnego, natomiast główny dt10 bez jawnych lagów przyrostów. Trzeba zachować te cechy w opisie wyników, zamiast z góry przypisywać poprawę wszystkim dodanym składnikom architektury.

Zamrożone SHA macierzy dt1: `732bbefd36b555868be9eabaeae314a19900ee27059f34579e032ec2a06d7e6c`.

Zamrożone SHA macierzy dt10: `715c97fab39f2b30aada1e35b887d8c44ee1d662d3c3d264d7097992ad5d76bd`.

SHA danych projekcji: `8a91e50e5dc8f41a7ce23ced7def795b5c1a1941bd6337d65a8950408e16c845`.

W odczytanych macierzach główny dt1 ma rank48/48, condition7.22645, błąd fizycznej postaci3.62e-16 i promień modelu0.999264. Główny dt10 ma fg rank36/36, condition1.86833, bg rank24/24, condition1.65491, błędy fizycznej postaci poniżej3.5e-16 i promienie odchyleń/układu średnich0.998324/0.998908. Są to wartości po reficie, a nie dowód pełnej identyfikowalności fizycznego układu.

Audytor przekazał wykonawcy i rootowi zamknięcie audytu przedtestowego przed uruchomieniem testu. Dalszy etap powinien wyłącznie ocenić zamrożone modele, bez zmiany siatki, kierunków lub parametrów na podstawie testu.

## Zweryfikowane wyniki testu i integralność

Po wykonaniu testu ponownie porównano SHA obu macierzy i danych z zapisami wyboru oraz powyższą migawką przedtestową: zgodność zachowana. `AUDYT_PREFIX_DT1.json` i `AUDYT_PREFIX_DT10.json` zawierają po 44 poprawne kontrole, razem 88, obejmujące 11 nowych modeli, dwie sondy i originy800/950. Każda daje dokładną zgodność prognozy po usunięciu albo zatruciu przyszłych q/r/bg. Wszystkie ocenione prognozy są skończone. Wyniki tego samego starego GLE na sondzie2 odtwarzają poprzedni eksperyment.

| Sonda | dt | H | Wybrany model: skill | Stary GLE: skill | Wybrany: znak wszystkich q |
|---:|---:|---:|---:|---:|---:|
|2|1|1|49.6823%|60.2382%|75.8712%|
|2|1|10|14.8845%|9.1137%|63.8680%|
|2|1|50|0.1821%|−5.0288%|65.4077%|
|2|10|10|15.5845%|16.1179%|64.3586%|
|2|10|20|21.2716%|20.1236%|64.0457%|
|2|10|50|21.0701%|13.2763%|64.2633%|
|8|1|1|30.5224%|57.4740%|70.9258%|
|8|1|10|−5.1327%|−29.7214%|56.6998%|
|8|1|50|−32.4522%|−128.1920%|58.1796%|
|8|10|10|−0.8443%|−4.4602%|55.4458%|
|8|10|20|−6.5894%|−9.6922%|56.3419%|
|8|10|50|−31.9329%|−56.9867%|50.9358%|

Główna prognoza dt1 nie poprawia wyniku jednostajnie. Na H1 ma MSE0.003745671 wobec0.002959884 starego GLE: błąd rośnie o26.5479%, a opisowy sparowany przedział wzrostu wynosi21.0110–34.8096%. Znak wszystkich współrzędnych spada o2.7402 punktu procentowego. H10 poprawia MSE o6.3494%, H50 o4.9613% względem starego GLE, ale H50 pozostaje praktycznie na poziomie persystencji. Dodatni kontrast do starego modelu nie jest tym samym co duża poprawa względem braku ruchu.

Równa średnia skill H1/H10/H50 na teście sondy2, czyli to samo kryterium agregacji co przy głównym wyborze, wynosi21.582950% dla localPLS8 i21.441067% dla starego GLE. Różnica to jedynie+0.141883 punktu procentowego. Poprawa kryterium na walidacji nie przeniosła się jako mocna ogólna przewaga w tym odłożonym odcinku.

Osobny model wybrany na walidacji dla H1, localPCA4 L2/ridge0.01, osiąga na sondzie2 H1 skill59.9506% i znak77.8729%. Nie przewyższa starego głównego ani starego best-H1; sam nie jest poprawą pierwszego kroku. Jego filtrowany znak80.8131% pozostaje metryką warunkową, przy gorszym znaku wszystkich składowych.

Główny dt10 poprawia H50 punktowo o7.7939 punktu procentowego skillu względem starego GLE. Sparowany opisowy CI tej różnicy wynosi[−0.9500,+16.9003] punktu procentowego i obejmuje zero. Równoważna redukcja MSE względem starego wynosi8.9870%, z CI[−1.3773,+15.6336]%. Nie wykazano rozstrzygniętej przewagi przy tak małej liczbie bloków.

## Ablacje oraz transfer

Wymagana kontrola nowego localU8 została wykonana. Przy dt1 daje niemal identyczny wynik jak stary GLE: H1 skill60.2384%, H10 9.1117%, H50−5.0275%. Zmiana standaryzacji nie wyjaśnia więc zaobserwowanych różnic w tej konkretnej wybranej konfiguracji, choć potrzebna była jako kontrola metodologiczna.

Na sondzie2 coupledU8 w tej samej nowej procedurze wyboru daje H1/H10/H50 skill49.8103/19.0796/10.5789%, wobec60.2384/9.1117/−5.0275% localU8. Wskazuje to na kompromis między krótkim i dłuższym horyzontem w tych dopasowanych rodzinach. Hiperparametry rodzin były wybierane osobno na walidacji; nie jest to eksperyment przyczynowy ani porównanie przy sztucznie identycznych wszystkich współczynnikach.

Przy dt10 na sondzie2 coupledU8 uzyskuje H10/H20/H50 12.4472/17.9430/20.0530%, wobec5.8546/6.8681/7.3578% nowego localU8. Ten efekt nie przenosi się na sondę8: H50 coupledU8−42.7873%, localU8−0.2871%. Nie ma podstaw do stwierdzenia uniwersalnej przewagi sprzężenia średnich.

Główny localPLS8 poprawia długi transfer względem starego GLE, lecz na sondzie8 przegrywa z persystencją dla H10, H20 i H50. Główny coupledPLS4 dt10 także przegrywa z persystencją na wszystkich trzech badanych horyzontach tej sondy. Porównanie ze słabym poprzednim modelem nie uzasadnia określenia transferu jako skutecznej długiej prognozy. Dodatkowo na H1 sondy8 prosta ostatnia prędkość ma skill60.5821%, więcej niż stare i nowe wybrane GLE.

Nie wolno podmieniać zwycięzcy po odczycie dodatnich H50 innych rodzin. Głównym dt1 pozostaje localPLS8, głównym dt10 coupledPLS4. Wyniki pozostałych, uprzednio zadeklarowanych rodzin są diagnostyką ablation i nie stanowią niezależnej walidacji nowego wyboru.

Na wspólnych 16 originach co10 główny dt1 ma dla sondy2 H10/H20/H50 skill18.7335/13.6667/0.4417%, a dt10 15.5845/21.2716/21.0701%. Zatem dt10 nie jest lepszy na każdym horyzoncie nawet w tym obrazie. W transferze8 odpowiednie dt1 wyniki to−3.4806/−7.9241/−29.5662%, a dt10−0.8443/−6.5894/−31.9329%; oba modele zawodzą względem persystencji na dłuższym ruchu.

## Nowe reszty i wspólny ruch

Odczytano `reszty.py` oraz `SZUM_DT1/DT10.json`. Diagnostyka została wykonana po wyborze, bez strojenia: one-step residuale dla ciągłych targetów801..1000 (200 obserwacji dt1) i810..1000 co10 (20 obserwacji dt10), osobno dla głównego nowego i starego modelu oraz obu sond. Przy każdym originie używana jest znana rzeczywista historia; to właściwa diagnostyka jednokrokowej reszty, odrębna od autonomicznego błędu H50.

W dt1 localPLS8 nominalny Ljung–Box p<0.05 występuje we wszystkich256/256 seriach sondy2 i64/64 seriach sondy8. Odpowiednie wartości starego GLE wynoszą134/256 i50/64. Jarque–Bera dla nowego modelu odrzuca nominalnie46/256 i22/64. Te liczby obejmują tokeny i wspólną średnią; szeregi są współzależne, surowe p nie mają korekty wielokrotności ani korekty dopasowanych parametrów. Wynik nie uprawnia do twierdzenia, że nowe pomocnicze stany wybieliły szum.

Energia wspólnej reszty sondy2 pozostaje duża:71.7156% dla nowego dt1 i72.8337% dla starego. W dt10 odpowiednio58.0975% i59.9900%. Tylko20 obserwacji dt10 mocno ogranicza interpretację asymptotycznych testów. Gaussowskość i czasową biel należy nadal rozdzielać; korelacja między tokenami nie jest sama w sobie testem czasowej bieli.

Root zapisał dodatkową diagnostykę wyłącznie fit501..700 w `RUCH_WSPOLNY_FIT.json`:82.1019% energii przyrostów q w średniej foreground,17.8981% w ważonych odchyleniach; błąd tożsamości rozkładu1.63e-15. Jest to rozkład niecentrowanej energii, a nie trafność prognozy. Jego matematyczna podstawa to `sum_i w_i||Δq_i||² = ||sum_i w_iΔq_i||² + sum_i w_i||Δq_i−Δg||²`. Cosine średnich przyrostów fg/bg0.73846 jest miarą współruchu, nie dowodem oddziaływania lub dostępności dodatkowej informacji prognostycznej.

## Werdykt końcowy i rozliczenie uwag

Przeczytano końcowe `WYNIKI.md`, `README.md` oraz `PODSUMOWANIE.md`. Liczby i zasadnicza interpretacja są zgodne z zapisanymi artefaktami. Po uwadze audytora poprawiono jednostki kolumny różnic skillu na punkty procentowe, ujednolicono tabelę rangi i doprecyzowano znaczenie L0. Root dopisał normalizację `w_i=coverage_i/sum_j coverage_j` w rozkładzie energii i osobny akapit o braku wybielenia reszt. Żadna korekta redakcyjna nie zmieniała macierzy, dopasowania ani wyników testu.

| Zakres | Końcowy stan dowodowy |
|---|---|
| A-01: średnia jako funkcja tokenów i synchroniczny update | Potwierdzone w kodzie; prefix/poison obejmuje nowe stany i kontekst. Ograniczenie do wspólnego addytywnego ruchu pozostaje jawne. |
| A-02: czasowy fit V i zamrożenie baz | Potwierdzone indeksami kodu, metadanymi, ortogonalnością i niezmiennym SHA danych. Nie wykryto użycia walidacji/testu do fitu V. |
| A-03: autonomia i porównywalność | Potwierdzone 88 kontrolami, odtworzeniem oldGLE, wspólnymi progami q i wspólnymi originami co10. |
| A-04: kontrola nowej standaryzacji | Naprawa konstrukcji porównania potwierdzona: nowy localU8 wykonano w tej samej siatce, zachowując także historyczny oldGLE. |
| Transfer między obrazami i identyfikacja fizyczna | Ograniczenia naukowe, nie naprawiony problem ogólny: długi transfer pozostaje gorszy od persystencji; nie wykazano fizycznych sił ani białego szumu. |

Końcowe hashe siedmiu plików kodu i obu plików metryk są zgodne z `WYNIKI.json`. Małe artefakty zajmują około18.8 MB, poniżej limitu150 MB; odczyt wolnego miejsca przy zamknięciu przekraczał wymagany1 GB. Audytor nie kasował ani nie przenosił plików. Nie uruchamiał ponownych fitów, forwardów, treningów ani dodatkowych modeli. Bez commitów.

Wniosek końcowy: badanie poprawnie realizuje ograniczony wspólny model dynamiczny i jego ablations. Wybrany dt1 poświęca jakość H1 na rzecz części dłuższych horyzontów; średnie testowe kryterium H1/H10/H50 pozostaje prawie bez zmiany. Wybrany dt10 ma punktową korzyść H50, ale opisowy przedział kontrastu obejmuje zero. Oba wybrane modele nie zapewniają skutecznej długiej prognozy sondy8. Obecny raport zachowuje te ograniczenia i nie uzasadnia zastąpienia poprzedniego modelu jednym uniwersalnym nowym równaniem.
