# Niezależny przegląd metodologii: dryf Koopmana, pamięć GLE i obrazy

2026-10-01. Przegląd przez gpt-6-astra, xhigh. Zakres: wyłącznie `PLAN.md`, `MODEL.md`, `wyniki/krotki_horyzont/WYNIKI.md`, `kod/krotki_horyzont.py` oraz `zrodla/LITERATURA.md`. Nie uruchamiano obliczeń eksperymentalnych ani nie sprawdzano surowych danych, loadera czy publikacji poza dostarczonym opisem literatury. Poniżej osiem priorytetów następnego pilota.

## 1. Zachować wynik, ograniczyć interpretację

Pilot H1 daje na s48 80,98% zgodności znaku i 64,96% redukcji MSE względem braku zmiany. Wobec `old_langevin20` przewaga wynosi odpowiednio 1,12 i 2,56 punktu procentowego. To interesujący wynik krótkiej prognozy zamrożonego skalara LDA. Nie identyfikuje semantycznej tożsamości, operatora pełnego ViT ani mechanizmu przyczynowego. Przy H10 wybrany model ma 65,07% zgodności znaku. H1 oznacza jedną aktualizację; nie jest prognozą następnego checkpointu wag zapisywanego co 500 aktualizacji.

Kod dopasowuje osobną regresję `s[t+H]-s[t]` dla każdego H. Używa wielomianów, opóźnień i historycznych N/P; nie dopasowuje przejścia między wektorami obserwabli EDMD. Określenie „lifting Koopmana” opisuje tutaj motywację rodziny cech. Rozdzielenie wyuczonego operatora dryfu, pamięci i wejścia jest dopiero następnym zadaniem. Odpowiednik bez pamięci jest ponadto znacznie uboższy informacyjnie, więc jego porażka nie dowodzi szczególnej przewagi teorii GLE nad autoregresją.

s47 służy rozwojowi i doborowi, a s48 było wielokrotnie oglądane. Cała dalsza praca na tych biegach pozostaje eksploracyjna. 180 sond jest współzależnych i podlega tym samym aktualizacjom. Nie stanowią 180 niezależnych replikacji; nie należy liczyć przedziałów niepewności z liczby wszystkich komórek czas × sonda.

## 2. Ustalić jedną oś czasu i rozdzielić zawartość obrazu od skutku aktualizacji

Niech `u_k` oznacza opis źródłowej paczki powodującej przejście `s_k -> s_(k+1)`. Informacja dostępna przy prognozie po zakończeniu kroku t wynosi

\[
\mathcal F_t=\sigma\{s_{0:t},u_{0:t-1},\text{historyczna telemetria},\text{polityka znana w }t\}.
\]

Retrospektywna identyfikacja przejścia k może używać pary `(s_k,u_k,s_(k+1))`, gdy `k+1` nie przekracza odcięcia uczenia. Prognoza z t nie może używać rzeczywistego `u_t`. Zmienna nazwana w kodzie `N[t]` jest legalna tylko wtedy, gdy loader rzeczywiście przesunął do niej źródłowy wiersz `t-1`. Przejrzany test prefiksu obcina tablice **po** `load()`: sprawdza lokalne indeksowanie predyktora, ale nie wykryje przecieku wcześniej wytworzonego przez loader, kalibrację ani cache. Flagi `historical_only=True` nie zastępują tej weryfikacji. Potrzebny jest test podmiany surowego sufiksu i ponownego zbudowania cech: stany od `t+1`, paczki od `t`.

Wejście obrazowe powinno obejmować mały, jawny opis zawartości historycznych paczek i wykonanych augmentacji. Oddzielnie traktować stratę, normę gradientu, N/P i zmierzone pchnięcia: są zależnymi od sieci wynikami przetwarzania/aktualizacji. Korzystanie z ich przeszłości jest dozwolone, ale przewaga takiego ramienia nie dowodzi wartości samych obrazów. `paczka_lat` pochodzi ze zmieniającego się encodera, więc nie jest stałą miarą treści obrazu. Należy sprawdzić zmienność skali i sens współrzędnych między biegami; deskryptor z zamrożonego encodera byłby osobnym wariantem, jeśli jego dane już są lokalne.

Paczka jest wspólna dla sond, natomiast jej wpływ może zależeć od stanu sondy. Sama addytywna składowa `B u_k` przewiduje wspólny efekt; małe, z góry wybrane interakcje `q_k × u_k` mogą sprawdzić tę zależność. Większa elastyczność wymaga osobnej ablacji, nie automatycznego przyjęcia.

## 3. Zbudować rozdzielny model jednokrokowy

Rekomendowana pierwsza realizacja to liniowa dynamika w ustalonej przestrzeni obserwabli z jawnym stanem pamięci. Dla jednej sondy, ze wspólnymi parametrami między sondami:

\[
\begin{aligned}
q_k&=\psi(r_k),\\
q_{k+1}&=Aq_k+C_m m_k+B u_k+c+e_k+\epsilon_{k+1},\\
m_{k+1}&=R m_k+Lq_k+M u_k,\\
e_{k+1}&=F e_k+\omega_{k+1},\\
y_k&=C_yq_k+\nu_k.
\end{aligned}
\]

`r_k` zawiera bieżący obserwowany stan; na początek `s_k` w zamrożonej skali. Warto włączyć `s_k` jawnie do `q_k`, aby `C_y` było liniowym odczytem. Dryf to `A`, pamięć to `m`, wejście obrazowe to `u`, a `e` jest opcjonalną kolorową resztą. Eliminacja pamięci daje dla `k>=1`

\[
C_m m_k=C_mR^k m_0+
\sum_{j=1}^{k}C_mR^{j-1}Lq_{k-j}+
\sum_{j=1}^{k}C_mR^{j-1}M u_{k-j}.
\]

W ten sposób pojawia się zarówno jądro stanu, jak i pamięć wejścia. Nie ma przesłanki, by całe opóźnione oddziaływanie obrazów zastępować natychmiastowym `B u_k`.

W pierwszym pilocie ustalić `R,L` jako rejestr opóźnień, zamiast uczyć dowolną ukrytą pamięć. Zacząć od samej pamięci stanu (`M=0`); pamięć wejścia dodać dopiero jako kontrolowane rozszerzenie. Ustalić, że dryf używa bieżącego `q_k`, a pamięć opóźnień od `k-1`. Jeżeli bieżący przyrost jest częścią stanu, nie dodawać go ponownie jako osobnej kolumny pamięci. Obecne `x*v`, `v²` i opóźnienia nie zapewniają interpretowalnego podziału współczynników.

Wycentrowanie względem ruchomej średniej 500 kroków jest przyczynowe przy starcie prognozy, lecz wymaga także reguły aktualizacji średniej podczas rollout'u. Można prowadzić bufor historii i dopisywać przewidywane stany albo zastosować zamrożone centrowanie. Nie wolno później uzupełniać tej średniej rzeczywistymi przyszłymi stanami.

## 4. Porównać EDMD z learned liftingiem na równych wejściach

Dla ustalonego słownika, np. `psi(s)=(s,s²)`, wariant bez pamięci i wejścia estymuje

\[
\widehat A=\arg\min_A\sum_k\|\psi(r_{k+1})-A\psi(r_k)\|^2+\lambda\|A\|_F^2.
\]

To konstrukcja EDMD w przyjętej orientacji wektorów; rozszerzenie o wejście i opóźnienia wymaga ich jawnego umieszczenia w regresji. Stałą można dodać do słownika albo jako wyraz wolny. Prognoza kilku kroków używa tego samego operatora, a nie osobnej regresji dla każdego H. Sprawdzać błąd odczytu stanu, domknięcia obserwabli i stabilność propagacji. Mały błąd jednokrokowy nie gwarantuje użytecznego rollout'u.

Dopuszczalna alternatywa to osobne operatory skończonego horyzontu `K_H`. Przy zamrożonej referencji `a=m_t` dopasowuje się pary `(s_t,a) -> (s_(t+H),a)` i **cały** cel liftingu `psi(s_(t+H)-a)`, nie tylko skalar endpointu. Termin: „finite-lag EDMD przy zamrożonej referencji”. Pominiecie a w regresorach zakłada, że względne współrzędne wystarczają; zależność dynamiki od absolutnego poziomu może temu przeczyć. Takich operatorów nie wolno bez sprawdzenia utożsamić z potęgami `K_1` ani uznać za jedno autonomiczne równanie. Dopasowanie dryfu, a następnie pamięci do jego reszt wprowadza umowną kolejność projekcji: dryf może już pochłaniać część wpływu pamięci. Do interpretacji bloków preferować wspólne dopasowanie; dla dwuetapowego predyktora użyć chronologicznych reszt poza dopasowaniem i jawnie ograniczyć interpretację mechanistyczną.

EDMD **nie jest autoenkoderem**. Learned lifting zastępuje ustaloną `psi` uczoną mapą `psi_theta`; autoenkoder Koopmana dodatkowo uczy dekoder `D_phi`. Sensowne kryterium łączy rekonstrukcję `D_phi(psi_theta(r_k))`, błąd przejścia w przestrzeni liftingu oraz wielokrokowy błąd obserwowanego wyjścia. Sama rekonstrukcja nie wymusza liniowej dynamiki, a sam błąd latentny dopuszcza zapadnięcie reprezentacji. Normalizacja, encoder, dekoder i współczynniki strat powstają wyłącznie na części rozwojowej.

Do skalara LDA nie potrzeba silnego autoenkodera. Mała sieć ucząca kilka funkcji skalara jest sensowną kontrolą bazy wielomianowej, ale nie odzyska utraconych wymiarów ViT. Porównywać podobną liczbę parametrów i te same informacje wejściowe. Dodanie historii wyłącznie encoderowi zmieniłoby jednocześnie klasę stanu i metodę liftingu.

## 5. Rozdzielić marginalną prognozę od obserwacji online

Poprawny cel bez przyszłych paczek to

\[
\widehat y_{t+H|t}=\mathbb E[y_{t+H}\mid\mathcal F_t],
\]

czyli propagacja rozkładu stanu, przyszłych wejść i zakłóceń. Model przyszłego wejścia musi wynikać z historii i znanej polityki. Losowanie obrazów IID nie czyni embeddingów z uczącej się sieci IID. W modelu liniowym można propagować warunkowe średnie wejść; kowariancje i zależność stan–wejście nadal wpływają na niepewność. Przy interakcjach lub nieliniowym dekoderze podstawienie średniego wejścia i średniego stanu zwykle nie odtwarza średniej prognozy; użyć małej liczby wspólnych próbek. Nie podstawiać zapisanych przyszłych paczek, ich embeddingów ani gradientów.

Dla powyższej realizacji stan filtru to `xi=(q,m,e)`. Filtr Kalmana może oszacować pamięć i kolorową resztę z pomiarów dostępnych do t, z macierzami przejścia i pomiaru wynikającymi z podanych równań. Jego optymalność wymaga odpowiednich założeń liniowości oraz modelu błędów; rozszerzona reprezentacja i empiryczne kowariancje same ich nie dowodzą. Jeśli pomiar s jest praktycznie dokładny, filtr powinien przede wszystkim estymować ukryte składniki; arbitralne wygładzanie s może opóźniać sygnał. Kowariancji pomiaru nie ustalać wyłącznie tak, by uzyskać lepszy wynik.

Po starcie t obserwator propaguje stan bez pomiarów `s_(t+1:t+H)`. Asymilacja kolejnych rzeczywistych stanów jest legalna dla nowych prognoz kroczących, ale zmienia zadanie wobec jednego swobodnego rollout'u. EKF jest opcją przy jawnie różniczkowalnej nieliniowej dynamice/pomiarze, UKF przy małym stanie i uzasadnionej aproksymacji rozkładu. Żaden z nich nie rozwiązuje braku identyfikowalności; UKF dla setek opóźnionych współrzędnych nie jest dobrym pierwszym pilotem CPU.

Tańszy obserwator reszty można zdefiniować osobno dla każdego H: ukryte `eta_j^(H)` indeksować czasem **końca** etykiety, z pomiarem `e_r^(H)=eta_(r+H)^(H)+nu_(r+H)`. W chwili t pomiar `e_(t-H)^(H)` aktualizuje wtedy `eta_t^(H)`, po czym AR1 przewiduje `eta_(t+H)^(H)`. Jest to poprawna konwencja opóźnienia, o ile błędy powstały przyczynowo. Nakładanie horyzontów może jednak skorelować szum pomiarowy; biały szum filtru jest założeniem do sprawdzenia, nie skutkiem tej zmiany indeksów.

## 6. Sprawdzać identyfikowalność konkretnej realizacji

Ridge stabilizuje estymację, ale nie rozdziela mechanizmów, gdy stan, pamięć i wejście tłumaczą te same zmiany. Ocenić SVD i liczbę uwarunkowania macierzy regresorów, stabilność współczynników między blokami czasu oraz przyrostową przewagę bloków cech. Bogactwo identyfikatorów paczek nie gwarantuje persistent excitation w przestrzeni ich agregatów. Powielenie wspólnego opisu paczki przez 180 sond nie tworzy 180 nowych wymuszeń.

Dla poszerzonej liniowej realizacji `(T,G,H_y)` raportować rangi numeryczne

\[
\mathcal O=[H_y^\top,(H_yT)^\top,\ldots,(H_yT^{n-1})^\top]^\top,
\qquad
\mathcal C=[G,TG,\ldots,T^{n-1}G],
\]

z tolerancją i widmem wartości osobliwych. Formalnie pełna ranga może współistnieć z bardzo słabą obserwowalnością w szumie. Dowolna uczona realizacja ukryta jest ponadto niejednoznaczna względem zmiany bazy. Sterowalność macierzy zakłada możliwość zadawania wejść, podczas gdy deskryptory rzeczywistych obrazów zajmują ograniczony zbiór. Historyczny embedding zależny od wag nie jest niezależnym pokrętłem sterowania.

Nawet pełne rangi nie dowodzą obserwowalności pełnych wag, momentów AdamW i buforów ViT z LDA ani przyczynowego wpływu paczki. Częściowe obserwacje i polityka doboru danych mogą powodować konfuzję. Dodatni wynik predykcyjny uzasadnia model wejście–wyjście w badanym zakresie; przypisanie efektu przyczynowego wymagałoby dodatkowych założeń lub kontrolowanych interwencji.

## 7. Testować odzyskiwalny sygnał w resztach z właściwym opóźnieniem

Reszta bezpośredniej prognozy H,

\[
e_r^{(H)}=s_{r+H}-\widehat s_{r+H|r},
\]

jest dostępna dopiero w `r+H`. Przy prognozie z t najnowsza dojrzała reszta ma początek `t-H`. Użycie `e_(t-1)^(H)` dla H>1 przecieka przez etykietę. Model korekcji powinien uczyć się błędów historycznych prognoz wygenerowanych przed ujawnieniem ich celów, np. w chronologicznym cross-fittingu; reszty z dopasowania na tych samych etykietach są nadmiernie optymistyczne.

Hipotezę ukrytego sygnału potwierdza przyrostowa poprawa na późniejszych blokach wobec tego samego modelu bez korekcji, po uwzględnieniu etapu doboru. Korelacja reszty z przeszłym stanem wskazuje kandydat na niedomknięcie, ale sama nie wykazuje użytecznej poprawy. Histogram gaussowski nie oznacza białości; białość nie oznacza automatycznie niezależności. Oddzielnie badać ACF wielu opóźnień, związki z historią wejść/stanu, zmiany wariancji i pokrycie przedziałów. Reszty obserwowane co 5 kroków mają ACF dla odstępu 5, nie 1; dla H>5 nakładają się cele. Agregowanie sond może maskować różne procesy resztowe. FDT nie wynika z AdamW ani z dopasowania GLE.

## 8. Ograniczyć następny pilot CPU i z góry ustalić decyzje

Użyć już dostępnych nagrań, bez powtarzania zakończonego krótkiego pilota. Zachować dotychczasowe odcięcia etykiet; nowe transformacje i modele wejścia uczyć wyłącznie na odpowiednim prefiksie s47. Wstępnie ograniczyć słownik do 2–4 obserwabli, opis obrazu do 4–8 cech wybranych na treningu, pamięć do 5 lub 20 kroków, ridge do trzech wartości. H1 i H10 wystarczą do pierwszej decyzji; pozostałe istniejące horyzonty ocenić dopiero po zamrożeniu wariantu. Zapisać limit czasu CPU i zakończyć etap po jego osiągnięciu.

Porównać: istniejące kontrole, dryf EDMD, dryf z pamięcią, ten sam model z obrazowym u, osobną kontrolę z N/P oraz jeden wariant z przyczynową korekcją reszt. Pozwala to przypisać zysk konkretnej zmianie. Learned lifting jest małą, limitowaną kontrolą dopiero po działającej wersji ustalonej; np. jedna mała warstwa, wymiar do 4 i maksymalnie 50 epok CPU, bez dużej siatki architektur. Rezygnacja z tej kontroli po wyczerpaniu budżetu nie uprawnia do wniosku o jej nieskuteczności.

Wybrać jeden model według zapisanej miary na blokowej walidacji s47. Raportować także przegrane ablacje, znak/balanced accuracy, MAE/RMSE, redukcję MSE, koszt, niestabilność rollout'u i kalibrację. Walidacja użyta do wyboru modelu nie jest niezależnym zbiorem kalibracji przedziałów. Próg 80% oraz największy **sprawdzony** H spełniający próg podawać osobno dla znaku i redukcji MSE; nie interpolować niezmierzonego horyzontu ani zakładać monotoniczności. Każdy wynik s48 nadal opisać jako eksploracyjny. Potwierdzenie wymaga nowego odłożonego biegu po zamrożeniu procedury; ten przegląd nie zaleca jego uruchamiania teraz.

**Przyszły cel wielowymiarowy:** ustalić zamrożoną podstawę tożsamości i prognozować cały wektor jej współrzędnych oraz ranking zmieniających się składowych. Osobne kalibracje LDA dla ziaren nie gwarantują tej samej semantycznej osi; orientację i porównywalność trzeba zdefiniować. Potrzebne będą kontrole narzędzia, tła, augmentacji i podprzestrzeni losowych o tym samym wymiarze. Obserwowalność skalara ani prognoza jego znaku nie rozstrzygają tych kwestii. Jest to plan dalszej pracy, bez obietnicy odzyskania pełnego latentu lub semantycznej tożsamości.
