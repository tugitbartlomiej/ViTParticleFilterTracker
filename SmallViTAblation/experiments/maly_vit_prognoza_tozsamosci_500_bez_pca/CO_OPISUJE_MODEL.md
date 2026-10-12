# Co przewiduje model i jak sprawdzamy próg 85 procent

Data: 2026-10-02. Ten dokument opisuje konstrukcję ukończonej próby. Wyniki liczbowe są w WYNIKI.md i PODSUMOWANIE.md.

## Model rzeczywiście wybrany na walidacji

Porównano 99 konfiguracji. Główne kryterium wybrało powrót do historycznej średniej, z oknem W=200 oraz czasem tau=200 aktualizacji. Dodatkowo przed testem zachowano wariant wybrany według MSE: W=50 i tau=50. Oba zachowują wszystkie 128 współrzędnych:

```text
mu_i(t) = (1/W) sum_{k=0..W−1} z_i,J(t−k)
zhat_i(t+H) = z_i,J(t) + [1−exp(−H/tau)] [mu_i(t)−z_i,J(t)]
```

Mu jest wyznaczone w chwili startu i zamrożone na całą prognozę. To średni dryf powrotu do równowagi; nie potwierdza rozkładu szumu procesu Ornsteina–Uhlenbecka. Żaden z 16 wariantów wybranych przed testem nie osiągnął 85 procent trafności znaku średnio w grupach dopasowania, transferu lub wszystkich obrazów, dla żadnego H od 1 do 500.

## Stan cząsteczki

Śledzimy ten sam token tego samego obrazu w kolejnych stanach uczonej sieci. Cząsteczka ma w źródłowym zapisie 768 współrzędnych. W tym doświadczeniu zachowujemy 128 wskazanych wcześniej **oryginalnych kanałów**. Nie obracamy bazy i nie wykonujemy PCA. Każda wybrana współrzędna nadal odpowiada temu samemu makrowektorowi, czyli oryginalnej osi sieci.

Zmieniają się współrzędne cząsteczki w tych osiach. Nie obracamy ani ponownie nie identyfikujemy osi na każdym checkpointcie. Wcześniejsza identyfikacja powyżej90 procent dotyczyła łącznego rozpoznawania par oryginał–LaMa w znanym obszarze narzędzia. Nie dowodzi, że każdy z 128 kanałów sam rozpoznaje narzędzie albo że te kanały są jego wyłączną przyczynową reprezentacją.

## Koopman i pamięć Langevina

Dla cząsteczki i zapisujemy `x_i(t)=z_i,J(t)−z_i,J(0)` oraz `v_i(t)=x_i(t)−x_i(t−1)`. Odjęcie stanu0 tylko ustala początek współrzędnych i nie korzysta z przyszłości. Proponowana średnia prognoza ma postać:

```text
vhat_i(t+1) = b + A x_i(t) + sum_l K_l v_i(t−l) + część_wspólna(t)
xhat_i(t+1) = x_i(t) + vhat_i(t+1)
```

Macierz przejścia chwilowego to `I+A`. Wariant bez pamięci jest afiniczną regresją Koopmana z obserwablami będącymi oryginalnymi współrzędnymi. Wariant z pamięcią dodaje poprzednie przyrosty albo kilka wykładniczych śladów tych przyrostów. Wspólne dopasowanie estymuje A, K i b tylko z przeszłości.

```text
m_r(t) = lambda_r m_r(t−1) + (1−lambda_r) v(t)
lambda_r = 2^(−1/half_life_r)
```

Po rozpoczęciu prognozy kolejne v i m pochodzą wyłącznie z przewidywanych stanów. Dla wariantu sprzężonego dodatkowymi stanami są coverage-średnia cząsteczek narzędzia i średnia cząsteczek tła. Ich przyszły ruch również trzeba przewidywać. Nie podajemy późniejszych rzeczywistych stanów tła jako znanych wymuszeń.

To skończony model regresyjny inspirowany GLE i projekcją Wienera. Samo dopasowanie lagów nie dowodzi odtworzenia dokładnego operatora Mori–Zwanziga. Nierozwiązany składnik pozostaje resztą prognozy. W tym porównaniu oceniamy deterministyczną prognozę średnią; nie dodajemy losowego szumu, żeby przypadkowo poprawić procent trafień. Pełny skalibrowany model stochastyczny wymaga osobnej oceny rozkładu reszt i pokrycia przedziałów.

## Kalman i obserwator stanu

Obserwator szacuje położenie i prędkość każdego z 128 kanałów na podstawie przeszłych odczytów. W chwili rozpoczęcia prognozy znamy dokładny zapis reprezentacji, dlatego rozpoczynamy ruch od tego zapisu. Używamy oszacowanej prędkości i przewidujemy jej wygasanie:

```text
zhat_i(t+H) = z_i(t) + vhat_i(t) sum_{h=1..H} rho^h
```

Po t nie ma pomiarów korygujących filtr. Nie jest to odszumianie całej znanej trajektorii. Parametr szumu pomiarowego stanowi założenie obserwatora: rzeczywisty zapis latentów jest odczytem sieci, a nie pomiarem fizycznego czujnika o zmierzonej wariancji szumu.

W sprzężonym obserwatorze każdy kanał dzielimy na składnik związany ze średnim ruchem tła i pozostałą część. Współczynnik tego związku pochodzi z historii fit. Oba składniki są przewidywane autonomicznie i sumowane z powrotem w oryginalnej współrzędnej.

## Co oznacza powodzenie

Główna miara sprawdza, czy przewidziano właściwy znak **łącznej zmiany** od startu do wybranego horyzontu. Wynik85 procent oznacza zgodny znak dla ważonego odsetka współrzędnych cząsteczek. Nie oznacza85 procent rozpoznanych narzędzi ani85 procent idealnie odtworzonych wektorów.

Równolegle mierzymy błąd wielkości zmiany względem założenia braku ruchu. Pokazujemy normę prawdziwego i przewidywanego ruchu oraz cosinus. Prognoza bardzo małego ruchu może trafiać znak, ale nie odtwarzać dynamiki; dlatego sam procent nie wystarcza.

Historia ma 1001 stanów. Po ustaleniu kanałów na kroku500 istnieje tylko jeden start umożliwiający pełne500 kroków prognozy. Dwanaście obrazów pokazuje transfer między przypadkami, lecz wszystkie obserwują ten sam bieg uczenia. Sukces lub porażka w tym biegu nie jest jeszcze niezależnym potwierdzeniem dla nowych ziaren.
