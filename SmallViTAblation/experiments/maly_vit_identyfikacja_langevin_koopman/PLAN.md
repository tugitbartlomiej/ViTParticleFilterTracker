# Identyfikacja Langevina i Koopmana z historyczną paczką

Utworzono: 2026-10-01 (Europe/Warsaw). Protokół zapisano przed pierwszym testem.
Ostatnia modyfikacja: 2026-10-01 15:28:17 (Europe/Warsaw).

## Pytanie i granice

Czy bezpośrednia prognoza zmiany zamrożonej siły LDA na 500 aktualizacji poprawia się po wygładzeniu pamięci oraz dołączeniu deskryptorów i zmierzonych wpływów wykonanych paczek? Wszystkie wejścia mają pochodzić z aktualizacji zakończonych do chwili t. Wynik negatywny i przewaga kontroli są pełnoprawnymi wynikami. Nie zmieniamy starego kodu i nie uruchamiamy treningu ani Edenu.

## Dostępne dane i przyczynowość

Lokalne `maly_vit_rownanie_latentu/nagranie_s47_sily` i `nagranie_s48_sily`: 3001 stanów sond po aktualizacjach 0..3000, identyfikatory rzeczywistych paczek, odbicia, kontrast/jasność, latent paczki przed krokiem, lr, strata i norma gradientu. Pchnięcia paczki/pamięci zapisano od kroku 900 dla s47 i 2000 dla s48. Wiersz paczki k i jej pchnięcie prowadzą ze stanu k do k+1. W prognozie t wolno więc użyć wyłącznie wierszy paczek k<t. Informacja nowa[t] jest zakazana, nowa[t-1] jest historyczna.

Mu, sd i w kalibrujemy z sond w chwilach 960,970,980,990,1000; pierwsze prognozy nie występują przed 1000. Kalibracja jest oddzielna dla ziarna, zamrożona i taka sama dla ramion. Dane sond nie są deskryptorami paczek. Deskryptory pochodzą wyłącznie z paczka_lat i odpowiadających im metadanych. Do prognozy nie wczytujemy przyszłych checkpointów ani gradientów. Duże tablice czytamy przez memmap; nie kopiujemy wielogigabajtowych danych. Wolne F: około 23,5 GB w audycie; budżet nowych artefaktów poniżej 100 MB.

## Podział przed testami

Pilot wykorzystuje s47 do rozwoju: trening do końca etykiety 1800, walidacja w chwilach 1800..2000 co 10 aktualizacji (cele do 2500). Dopasowanie bezpośrednie H=500 zaczyna od stanów 1000 i bierze tylko pary r+H<=1800. Po wyborze cech/ridge na walidacji refit używa s47 i etykiet <=2500. Żadna wartość testowa s48 nie wybiera parametrów.

Test transferu na s48: chwile 2100..2500 co 10, cele 2600..3000. Wszystkie modele porównujemy na tych samych chwilach; dolna granica 2100 zapewnia pełne 100 kroków historycznych wpływów s48. Test późnego czasu s47: jeden punkt 2500->3000. Główny punkt bez nakładania to t=2500 dla każdego ziarna; 41 nakładających się okien s48 stanowi diagnostykę, a nie 41 replikacji. Dwa ziarna nie dają wiarygodnego potwierdzenia populacyjnego. Model transferu jest wcześniej zbudowany na biegu s47; kalibracja LDA s48 z jego rozgrzewki jest jedyną kalibracją biegu testowego. Nie jest to zsynchronizowana prognoza między równolegle trenującymi sieciami.

Dodatkowe horyzonty 10,50,100 aktualizacji mają ten sam podział końców etykiet i własny dobór na walidacji. H=1 można sprawdzić w osobnej diagnostyce, jeśli rozdzielczość co 10 nie wystarcza. Nie mieszamy krótkich horyzontów z wynikiem H=500.

## Ramiona i równe wejścia

Kontrole: brak zmiany, stały dryf z ostatnich 100 aktualizacji, prosty bezpośredni AR; odtworzenie GLE z `prognoza_gle.py` na zapisach co 10 aktualizacji oraz starych rekurencyjnych Koopmana/Langevina z bloków 50 aktualizacji. Wszystkie ocenione na końcowych stanach, także gdy kontrola uczy się na blokach.

Nowe rodziny bezpośrednie: kompaktowe jądro (prędkości uśrednione na 10,20,50,100,200 krokach, odchylenie od poziomu); to samo z deskryptorem paczki; to samo z wpływem paczki; to samo z oboma; to samo z dodatkową pamięcią AdamW; prosty lifting kwadratowy jako kontrola nieliniowego domknięcia. Wpływy historyczne uśredniamy przez 10,50,100 kroków. Stany i deskryptory dostępne do t dla wszystkich odpowiednich ramion; żadne przyszłe u nie są znane. Prędkość latentu nie jest pędem AdamW. Wyższe różnice nie tworzą nowej informacji, mogą powiększać szum.

Pooled ridge ma współczynniki wspólne dla sond; sondy są zależne. Siatka ridge dla nowych rodzin: 0.001,0.1,10; okna poziomu 500 lub 1000. Wybór minimalizuje MSE walidacji. Kontrola GLE zachowuje oryginalne parametry p=20 zapisów, okno=50 zapisów, ridge=0.001. Baseline nie wymaga dodatniego kappa. Lifting jest aproksymacją, nie obietnicą dokładnej skończonej linearyzacji.

## Co obala i jak mierzymy

MAE, RMSE, umiejętność 100*(1-MSE/MSE_brak_zmiany), znak wszystkich sond oraz wcześniejsza konwencja połowy największych zmian (ta druga maska tylko w ocenie). Raportujemy każde ramię, także przegrane. Najlepszy wariant wybieramy na walidacji i zamrażamy przed testem. Przewaga wybrana po wynikach s48 nie jest sukcesem.

Obalenie poprawy: wybrany model przegrywa z GLE lub prostą kontrolą na H=500; dodatkowe wejście nie daje poprawy względem odpowiednika stanu; zysk występuje wyłącznie w walidacji. Wynik dodatni pilotażu nie zastępuje wieloziarnowej kampanii. Podłoga zachowuje historię i cel: te same ramiona bez dodatkowego u, brak zmiany oraz AR. Nie tasujemy całych trajektorii.

Diagnostyka reszt oddziela skośność/kurtozę, autokorelację, korelację z historycznym stanem/wejściem i zmiany wariancji. Nakładające się błędy H=500 mają mechaniczną zależność: nie używamy ich ACF jako dowodu kolorowego jednego kroku. Dodatkowo diagnozujemy błędy H=10 z niepokrywających się przejść. Historyczne błędy walidacji kalibrują orientacyjne przedziały; testowe błędy tylko oceniają pokrycie. Brak gwarancji IID/exchangeability.

## Weryfikacja i artefakty

Test prefiksu: wejścia/model dają identyczną prognozę z prefiksu i po podmianie całego sufiksu t+1.. (stanów) oraz t.. (paczek). Test przesunięcia u ma potwierdzić, że wpływ kroku t nie trafia do prognozy w t. Krotność indeksowania i granice dojrzałych par są kontrolowane automatycznie.

Kod, konfiguracja, cache małych cech, audyt, predykcje, wszystkie wyniki i raport pozostają w tym katalogu. Literatura i wyprowadzenie pamięci z eliminacji ukrytego stanu dostają osobne pliki. Lean/SymPy sprawdzają tożsamości algebraiczne, nie trafność prognozy ViT. Obserwowalność i sterowalność rozpatrujemy dla konkretnej skończonej realizacji; nie deklarujemy tych własności dla pełnej sieci bez pomiaru/rangi. Nie stosujemy równowagowego FDT do AdamW.

## Etap drugi, protokół po pierwszym pilotażu

Zapisano przed drugim testem 2026-10-01 14:24 Europe/Warsaw. Pierwszy pilot jest w `wyniki/pilot_v1/`. Jego wyniki testowe zostały już obejrzane: dalszy etap jest eksploracyjny i nie odzyskuje statusu niezależnego potwierdzenia.

Przyczyna rozszerzenia: rodzina kompaktowa przegrywa z oryginalnym jądrem także na walidacji. Sprawdzamy oryginalną bazę 20 opóźnień, z oddzielnymi dodatkami deskryptorów, wpływu i pamięci, oraz strojenie samego ridge. Siatka ridge: 0.0001,0.001,0.01,0.1,1,10. Nowe rozszerzenia używają tej samej konwencji bez wyrazu wolnego co GLE. Cechy stałe na treningu dostają jednostkową skalę zamiast dzielenia przez niemal zerowe odchylenie. Wybór nadal wyłącznie na s47 walidacji, nie na s48; pozostały podział nie zmienia się. Raportujemy pierwszy i drugi etap oraz wszystkie przegrane warianty.

## Aktualizacja 2026-10-01 14:53:54: bliższy zapis i składowe tożsamości

Priorytet użytkownika: prognoza jednego bliższego zapisu/checkpointu, bez upierania się przy 500 aktualizacjach. Każdy horyzont podawać w liczbie aktualizacji i interwale telemetrii: główna kampania miała sondy co10 i wagi po500, a istniejące nagrania siły mają stany co1. H1 nie jest tym samym zadaniem co następne wagi po500.

Cel80procent pozostaje celem do sprawdzenia, nie osiągnięciem ani gwarancją. Rozdzielić80procent zgodności znaku na wszystkich sondach, zbilansowaną trafność oraz80procent redukcji MSE względem braku zmiany; żadna z tych miar nie zastępuje pozostałych. Maska największej połowy zmian jest wyłącznie pomocnicza.

Lokalny pilot H1/3/5/10/20/50 na istniejących s47/s48: trening s47 z końcem etykiety<=1800, walidacja startów1800..2000 co5, refit po wyborze do2500, transfer diagnostyczny s48 starty2100..2950 co5. Model wybrać wyłącznie na walidacji. Dane s48 już obejrzano wcześniej, więc cały pilot jest eksploracyjny. Zachować poprzednie wyniki H500; nowe artefakty w podkatalogu `wyniki/krotki_horyzont/`. Kontrole: brak zmiany, ostatni przyrost/dryf, większość znaku z przeszłości, prosty AR/GLE. Modele: lokalny dryf/lifting Koopmana w obserwowanym stanie + jądro pamięci GLE + historyczne wpływy paczek i AdamW. Nie obiecywać globalnej dokładnej linearyzacji. Każda paczka powodująca cel pozostaje nieznana; pole nowa[t] prowadzące t->t+1 nie jest wejściem w t. Wykorzystać wyłącznie nowa[t-1] i wcześniejsze oraz prognozowany wpływ z historii. Prędkość latentu nie jest pędem optymalizatora.

Raportować MAE/RMSE/redukcję MSE, znak wszystkich sond, większość z przeszłości, zbilansowaną trafność, kalibrację przedziałów oraz korelacje/autokorelacje i wariancję reszt. Dotychczasowe wyniki warunkowe z już znaną paczką kolejnego kroku w `maly_vit_horyzont_rownania_krok` i `maly_vit_prognoza_tozsamosci_z_historii` nie spełniają aktualnego kryterium prognozy. Kierunek LDA pozostaje stały; test s nie jest prognozą pełnego wektora768.

Osobny przyszły etap: ustalić, które wektory/składowe reprezentacji tożsamości będą się zmieniać, aby dopełniać reprezentację. Zdefiniować wielowymiarowy stan w jawnej podprzestrzeni tożsamości, jej wymiar, podstawę, współrzędne i cel dopełniania. Podprzestrzeń oraz próg zmiany wybierać tylko z przeszłości/treningu; przy każdym porównaniu podstawa zamrożona i wspólna. Przewidywać zmiany współrzędnych i ranking składowych do zmiany, osobno błąd wektorowy, znaki i wykrycie aktywnych składowych. Nie utożsamiać bazowych neuronów, modów PCA i semantycznych kierunków. Wybór podprzestrzeni wymaga kontroli swoistości względem innych narzędzi, tła, odbić, krawędzi i losowych podprzestrzeni o tym samym wymiarze; wskazane kontrolowane interwencje na narzędziu przy wspólnym tle oraz na tle przy wspólnym narzędziu. Trafna prognoza ruchu sama nie dowodzi tożsamości semantycznej.

Etap wielowymiarowy na dziś tylko zaplanowany: bez pobierania22GB, nowych treningów lub Edenu. Pilot s jest diagnostyką modelowania czasowego, nie odpowiedzią na pytanie o wszystkie wektory.

Rozszerzenie krótkiego pilota przed kolejną estymacją: porównać samo lokalne przybliżenie Koopmana (x lub lifting x,x², bez jądra przyrostów), Koopman + GLE + historyczne wejścia, dawny bezpośredni Langevin20 bez wyrazu wolnego z ridge1e-6 oraz brak zmiany. Te same podziały i okna, nadal wybór wyłącznie na walidacji. Zachować pierwszy krótki pilot w wyniki/krotki_horyzont/pilot_v1; całe badanie eksploracyjne. Potwierdzenie wymaga świeżego odłożonego przebiegu.

### Wynik krótkiego pilota, 2026-10-01 15:00:30 Europe/Warsaw

Na H1 (jedna aktualizacja, nie500) walidacja s47 wybrała lokalny lifting Koopmana + jądroGLE20 + historyczne wpływy paczki/AdamW, ridge0.001. Test eksploracyjny s48:80,98procent znaków na wszystkich sondach,80,98procent zbilansowanej trafności,64,96procent redukcjiMSE; większość znaku z przeszłości49,73procent. Dawny bezpośredni Langevin20 na tych samych oknach:79,86procent znaków i62,40procent redukcjiMSE. Sam Koopman z liftingiem stanu daje tylko0,42procent redukcjiMSE na H1; przewagi nie ma z definicji. Późny s47 dla wybranego modelu:79,24procent znaków i59,50procent redukcjiMSE. Cel80 osiągnięty wyłącznie w mierze znaku na jednym biegu eksploracyjnym, nie w redukcjiMSE ani jako potwierdzenie między ziarnami.

Dla kolejnego zapisu sond co10 aktualizacji: znak65,07procent, redukcjaMSE22,13procent. Wyniki H3/5/20/50 i wszystkie kontrole w osobnym `wyniki/krotki_horyzont/METRYKI.json`; opis `wyniki/krotki_horyzont/WYNIKI.md`. Rzeczywiste paczki powodujące cele nie są wejściami. Dotychczasowe wynikiH500 zachowano. Do potwierdzenia potrzeba świeżego odłożonego przebiegu po zamrożeniu modelu, konkretnego horyzontu zapisu i definicji celu80. Etap podprzestrzeni wielowymiarowej tożsamości pozostaje planem.

## Rozszerzenie 2026-10-01 15:04:08: Koopmanowski dryf, obrazowe sterowanie i obserwator

Priorytetem jest GLE z dryfem estymowanym w przestrzeni Koopmana, jądrem pamięci oraz osobnym sterowaniem. Porównać EDMD z ustalonym liftingiem i learned lifting/autoenkoder Koopmana. EDMD samo nie jest autoenkoderem. Uczenie podprzestrzeni, normalizacja i dopasowanie modelu tylko na przeszłości; nie zakładać dokładnej skończenie wymiarowej globalnej linearyzacji nieliniowego treningu.

Sterowanie ma jawnie opisywać cechy obrazów faktycznie wykonanych historycznych paczek. Opis/embedding paczki nie jest embeddingiem sondy, a jej wpływ na aktualizację nie jest samą treścią obrazu. Stan obejmuje historię reprezentacji, ewentualne uzasadnione przyrosty i pamięć optymalizatora. Zdefiniować obserwowany i ukryty stan oraz zgodność czasu. W plikach aktualizacja k prowadzi s[k]->s[k+1]; paczka k jest dostępna dla prognozy dopiero po jej wykonaniu. Osobno raportować identyfikację wejście-stan-wyjście z historycznych danych i marginalną prognozę bez rzeczywistych przyszłych wejść. Przyszłe wymuszenia tylko modelowane z historii i znanej polityki.

Zbadać identyfikację nieliniową, lokalne osadzanie Koopmana, obserwowalność konkretnego poszerzonego modelu, obserwatora stanu oraz reakcję na wejścia. Rangi i wzbudzenie sprawdzać dla modelu; nie ogłaszać tych własności dla pełnego ViT na podstawie samego embeddingu. Porównać filtr Kalmana w poszerzonym stanie z pamięcią/skorelowaną resztą. EKF/UKF jako opcjonalne rozszerzenie modelu nieliniowego, z jawną dynamiką i modelem pomiaru. Hipoteza: część reszty zawiera niewydzielony sygnał. Potwierdzi ją wyłącznie przyczynowa poprawa poza dopasowaniem; nie przyjmować możliwości odzyskania prawdziwego białego szumu. Gaussowskość, białość, autokorelacje, korelacje z wejściem i heteroskedastyczność oceniać oddzielnie.

Dalsza literatura automatyki/robotyki/Koopmana/Mori-Zwanziga: paper-search i źródła pierwotne. Undermind/deep research tylko przy rzeczywistej dostępności; brak funkcji raportować, nie zastępować go udawaną usługą. Niezależny przegląd metod przez gpt-6-astra xhigh, w osobnym pliku, bez kolizji i cofania zmian innych.

Raportować krzywe znak/balanced accuracy/MAE/RMSE/redukcjaMSE vs H i maksymalny zmierzony H utrzymujący jawnie zadany próg80. Nie gwarantować monotonicznej trafności. Kryterium80 znaku nie jest80 redukcjiMSE. s48 pozostaje eksploracyjne po obejrzeniu; świeży odłożony przebieg konieczny do potwierdzenia przewagi. Kolejny etap zapisuje odrębne artefakty, nie nadpisuje wcześniejszych pilotów.

Budżet: ograniczone lokalne obliczenia CPU na już dostępnych danych; bez nowych Eden jobs, pobierania22GB, dużego treningu, commitów i zmian dawnych eksperymentów. Przed ewentualnym lokalnym GPU wykonać audyt dostępności/konfliktów; mały autoenkoder można badać na CPU w ustalonym limicie epok. Dotychczasowego krótkiego pilota nie powtarzać.

## Obowiązująca konwencja cząsteczki i układów latentnych, 2026-10-01 15:11:41 Europe/Warsaw

To terminologia uzgodniona przez użytkownika, nie dowód semantycznej tożsamości ani zmiana istniejącej implementacji.

- Cząsteczka: reprezentacja jednego tokenu obrazu z_i(t) w wspólnej przestrzeni R^768.
- Globalny układ latentny: wspólna przestrzeń współrzędnych, nie uśredniony wektor obrazu lub zbioru tokenów.
- 768 makrowektorów: wspólne wektory bazowe e_j tej przestrzeni.
- Mikrowektory cząsteczki: jej składowe z_ij(t)e_j, których suma tworzy z_i(t).
- Mikro układ cząsteczki: opis składowych/współrzędnych jednej cząsteczki względem wspólnej bazy. Nie oznacza osobnej wyuczonej bazy ani niezależnego zbioru wektorów przechowywanego w modelu.
- 600 cząsteczek na obraz wynika z tokenizacji konkretnego ViT przy używanej rozdzielczości. 768 to wymiar reprezentacji w mierzonym miejscu sieci; liczby nie są uniwersalnymi parametrami każdej tokenizacji.
- Ruch: zmiana z_i pomiędzy stanami treningu przy tym samym obrazie i tokenie. Mikroruch: zmiana odpowiadających składowych z_ij(t)e_j. Zmiana obrazu/augmentacji lub indeksu tokenu wymaga osobnego dopasowania, nie jest automatycznie ruchem tej samej cząsteczki.

Pooling tokenów i zamrożony rzut LDA s są osobnymi agregatami/obserwablami. Nie zastępować trajektorii cząsteczek ich średnią bez jawnego oznaczenia agregacji. Dotychczasowe lokalne piloty s prognozują skalar z uśrednionego latentu narzędzia, nie600 pełnych trajektorii ani768 mikroruchów każdej cząsteczki. Etap tych trajektorii pozostaje osobnym planem na dostępnych, jawnie wskazanych danych.

### Wynik odrębnego pilota EDMD/learned lifting/KF, 2026-10-01 15:28:17 Europe/Warsaw

Zakończono lokalny CPU pilot H1/3/5/10/20/50 z prawdziwym dopasowaniem całego liftingu przez K_H, learned encoder/decoder18epok oraz osobną pamięcią, PCA8 cech rzeczywistych obrazów paczek i przyczynowym filtrem reszty. Wewnętrzny podział chronologiczny: dryf do1400, reszty do1800; po wyborze refit odpowiednio2000/2500. S48 nadal eksploracyjne. Wyniki walidacyjnie wybranych modeli nie przewyższają wcześniejszego direct modelu; żaden wybranyH nie osiągnął80 w znaku ani redukcjiMSE. H1: fixedEDMD+pamięć+obrazy+wpływy+KF daje−53,08procent redukcjiMSE i57,06procent znaku; H10 fixedEDMD+pamięć20,19procent i64,04procent; H20 learnedlifting24,70procent i67,06procent. Wszystkie negatywy i ramiona bezKF zachowano.

PCA obrazów wykazuje przesunięcie rozkładu: mediana normy w8 standaryzowanych współrzędnych2,73 dla treningu s47,7,40 dla s48. To diagnostyka, nie dowód jedynej przyczyny. Reprezentacje pochodzą z osobno uczonych sieci; zgodność semantyczna współrzędnych między biegami nie jest udowodniona. Następny ograniczony test powinien użyć wspólnego zamrożonego encodera obrazów albo dopasowania reprezentacji wyłącznie z przeszłości, z kontrolą prostszych deskryptorów.

KF filtruje jedną AR1-resztę dołączoną do predyktora opóźnionego; nie jest pełnym filtrem600 cząsteczek lub wszystkich wag. Używa tylko zakończonej etykiety z początku t−H. EKF/UKF i pełna rozszerzona realizacja nadal wymagają osobnego protokołu; nie twierdzić, że zostały przetestowane. Independent Astra review rozróżnia direct regresję odEDMD oraz mechanistyczną identyfikację od umownej kolejności projekcji.

Raport, operatory, cechy/PCA, małe wagiAE, odtworzone modele, diagnoza i krzywe: `experiments/maly_vit_identyfikacja_langevin_koopman/wyniki/koopman_obserwator/`. PrzeglądAstry `PRZEGLAD_ASTRA.md`; literatura `zrodla/LITERATURA_OBSERWATOR.md` i `SOURCES_OBSERWATOR.json` (7źródeł). Undermind użyto do rzeczywiście dostępnych lookup/metadanych, bez deepresearchjobs. Bez nowego treninguViT, GPU/Edenu, pobierania22GB i commitów. Świeży test konieczny przed potwierdzeniem.

## Zmiana priorytetu: wspólna identyfikacja controlled Koopman-GLE, 2026-10-01 16:21:29 Europe/Warsaw

Obowiązujące nowe polecenie: filtr Kalmana/obserwator KF zostaje odłożony i nie wchodzi do kolejnego etapu. Istniejące wyniki KF pozostają historycznymi negatywami. Najpierw identyfikujemy wspólnie macierze dryfu w wybranej przestrzeni obserwabli, jądra pamięci stanu i historycznych wejść oraz resztę. Nie dopasowujemy autonomicznego dryfu i nie doklejamy później członu B u. Liniowa ewolucja w liftingu jest aproksymacją konkretnego modelu, nie założeniem dokładnej liniowości latentu ani globalnej linearyzacji ViT.

Źródłowa paczka u_k wywołuje S[k] -> S[k+1]. W chwili prognozy S[t] dostępne są wykonane paczki do u_(t-1). Ich cechy obrazowe pozostają jawne i odrębne od sond oraz pomiarów wpływu/gradientu. Zidentyfikować osobno kontrolowany model na ukończonych parach historycznych i marginalną prognozę bez rzeczywistych przyszłych u. W marginalnym wielohoryzontowym modelu wspólne regresory obejmują lifting bieżącego stanu, historię stanu i cechy wykonanych paczek oraz ich pamięć; wszystkie bloki estymujemy w jednym regularizowanym problemie. To operator projekcji wielohoryzontowej, nie dowód reakcji na interwencję.

Nowy odrębny etap porówna joint controlled lifting z pamięcią i wejściami, joint model bez obrazów, kontrolę autonomiczną oraz zapisane dawny GLE/direct i EDMD bez KF. Protokół ustalić przed testami: ten sam cel i H1/3/5/10/20/50, chronologiczny train/validation, wybór na walidacji, s48 eksploracyjne. Sprawdzić rangę, uwarunkowanie, współliniowość i niejednoznaczność rozdziału dryfu/pamięci/wejść. Nie ogłaszać identyfikowalności mechanizmów fizycznych na podstawie ridge ani liczby próbek sond.

Astra xhigh ponownie dopracowuje równanie i indeksowanie; Lean/SymPy weryfikują właściwe tożsamości eliminacji stanu ukrytego i osobnych jąder stanu/wejść. Jawne założenia i zakres dowodu algebraicznego nie oznaczają gwarancji prognozy. Białość, gaussowskość, zależność reszt od historii i heteroskedastyczność diagnozujemy oddzielnie; bez narzucania równowagowego FDT dla AdamW. Metryki znaku, balanced accuracy i wielkości/błędu pozostają oddzielne, próg80 celem wymagającym doprecyzowania.

Uruchomić celowany Undermind Deep Search i lekturę właściwych pełnych źródeł, z koordynacją research wag, bez nieograniczonej serii. Lokalny pilot CPU na istniejących danych; bez ViT treningu, Edenu, commitów i prawdziwych przyszłych klatek. Osobne artefakty etapu joint_controlled.

### Wynik etapu wspólnej identyfikacji bez KF, 2026-10-01 16:35:25 Europe/Warsaw

Zakończono odrębny joint controlled Koopman-GLE pilot CPU25,54s: cały lifting(z,z²),20 opóźnień jego przyrostów, pamięć wykonanych obrazów w rozłącznych oknach1/4/15 i oddzielne wpływyN/P; wszystkie bloki dopasowane JOINT. Train s47<=1800, val1800..2000, refit<=2500; s48 nadal eksploracyjne. Wybrane modele H1/3/5/10/20/50: redukcjaMSE64,12/40,38/25,67/21,86/22,95/29,74procent; znak80,92/72,84/67,20/65,19/64,73/68,17procent. Znak i balanced>=80 tylkoH1, nie redukcjaMSE. Dawny directH1 pozostaje nieznacznie lepszy64,96/80,98; brak wykazanej przewagi. Samo obrazowe wejścieH1 daje62,43/79,77 wobec samej pamięci62,31/79,85. Do80,92 doprowadziło ramię także z historycznymi pomiarami wpływu, nie same obrazy.

Oddzielny joint controlled one-step identyfikowano na ukończonych historycznych parach; przy prognozie wszystkie przyszłe obrazy zastąpiono modelem średniej20 wykonanych paczek. WynikH1:62,50redukcjiMSE i79,93znaku. Nie korzysta z rzeczywistych przyszłych klatek.36 kontroli prefiksowych przechodzi. S48 i późny fragment s47 nie stanowią świeżego niezależnego potwierdzenia. Poziom80 jest nadal celem o oddzielnych miarach.

Lean7twierdzeń nadInt i SymPy eliminacji macierzowej2x2 oraz jointblockridge przechodzą. Założona realizacja daje osobne jądra BE^jC orazBE^jF; nieznanyD*u_t nie znika przez eliminację. Dowód algebraiczny nie dowodzi trafności ani dokładnego liftingu ViT. Astra xhigh: PRZEGLAD_ASTRA_STEROWANIE.md. Celowany Undermind DeepSearch zakończył119wyników z abstraktów; przeczytano4pełnePDF przez narzędzie (Lin/Lu, Lin/Tian/Perez/Livescu, Uy/Peherstorfer, Otto/Peitz/Rowley), sprawdzono daty publikacji i ograniczenia. Raport zrodla/LITERATURA_JOINT.md rozdziela zakres odczytu i zastosowanie doAdamW. Istotne przyszłe rozszerzenie to interakcja stanu i wejścia, ponieważ addytywneBu nie opisze każdej dynamiki sterowanej.

Wyniki i krzywe: experiments/maly_vit_identyfikacja_langevin_koopman/wyniki/joint_controlled/WYNIKI.md; modele, prognozy, metryki, diagnostyka reszt/wejść i próg80 zapisane osobno. KFpozostaje odłożony, starych wyników nie usunięto. Bez treninguViT/GPU/Eden/commitów.
