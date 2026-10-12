# Dobór klatek z dynamiki pojedynczych tokenów

Rewizja przed pierwszym treningiem, 2026-09-22. Użytkownik doprecyzował: cząstka = jeden token patcha. Szkic z poolingiem ROI nie był uruchomiony. Właściwy przepływ zaczyna się od token_ekstrakcja.py; wcześniejsze moduły pozostają pomocnikami lub szkicem.

## Pytanie i dwa zegary

Czy wybór klatek według poprawy zadaniowego odczytu z historii tokenów przez Koopmana i pełniejsze GLE poprawia uczenie detektora na 1500 obrazach względem zwykłego wyboru 1500 i 3000 obrazów?

Cząstka: jeden patch ostatniej warstwy ViT przed projektorem detektora. Stała projekcja Gaussa zmniejsza tylko kanały do 24, bez uśredniania patchy. Trajektoria filmowa śledzi stałe miejsce siatki. Narzędzie przechodzi między patchami: opis Eulera, bez twierdzenia o śledzeniu tej samej części narzędzia.

Zegar filmu: osiem wcześniejszych obserwacji z krokiem pięciu klatek przewiduje kolejny token. Na tej osi pracują Koopman i GLE. Zegar treningu: ten sam obraz i patch odczytywane z checkpointów po epokach 2, 4, 6, 8, 10, bez augmentacji. Nie utożsamiamy kroków SGD z czasem filmu. Pierwszy pilot zamraża wybór przed treningiem; nie testuje jeszcze wymiany danych w każdej epoce. Ekstrakcję i ranking można ponowić z dowolnego checkpointu.

## Dostęp do informacji

Aktualny detektor: wspólny checkpoint L0 uczony na 500 klatkach. Kandydaci: te same 10 637 klatek train co w etapie 1. Kontekst: 57 073 gęste klatki train, 118 klipów, oryginalne piksele. Nie traktujemy nierównych odstępów rzadkiej puli jako równych.

Trzy foldy po operacjach. Skale, odczyt tokenu, Koopman i GLE dopasowane poza ocenianymi grupami. Readout korzysta wyłącznie z etykiet L0 spoza foldu. Prostokąt GT daje słabą etykietę narzędzie/tło, nie maskę segmentacji. GT kandydatów służy stratom rankingu: to selekcja oznaczonego zbioru, nie aktywne uczenie bez etykiet. Backbone był wcześniej uczony na całym L0, jawnie wspólnym dla ramion.

## Równanie i wynik rankingu

Koopman: EDMD w liftingu [z,cos(Rz+b), 1]. GLE: z_next=z+D+JP+W+Z+N. D: dryf, prędkość i przyspieszenie. JP: uwaga z zanikiem i oscylacją. W: przyczynowa historia metadanych detektora. Z: powrót do ostatniego tokenu odczytanego jako narzędzie, nie potwierdzona obecność fizyczna. N: deterministyczny filtr wcześniejszych reszt bazowych. To adaptacja pełnej struktury z Grant, a nie sam AR(8).

Prognoza nie widzi tokenu ani metadanych celu. Port zawiera też własną skalę ruchu patcha sigma z model_s.py (clamp 0.5..50), uczone gamma dla D/JP/W, osobne skale prędkości i przyspieszenia wyłącznie z FIT. Z kotwicy i N nie są dodatkowo mnożone przez sigma. Warunkowy skok przyszłego zdarzenia jest wyłączony w trybie strict. GLE: 12 stałych epok, AdamW1e-3, weight decay 1e-4, clip 1, hidden 64, heads 2, residual_rank 8, maksymalnie 20 000 trajektorii patchy na fold. Predyktor to stała średnia 0.5 Koopman+0.5 GLE, bez strojenia na AP.

Każda klatka: maksymalnie 16 patchy prostokąta i 16 tła, deterministycznie próbkowanych. e_obs to zbalansowana BCE odczytu z bieżących tokenów; e_dyn to ta sama strata tokenów przewidzianych. S=max(0,e_obs-e_dyn). Dopiero straty pojedynczych patchy są agregowane do klatki. To hipoteza użyteczności treningowej, nie udowodniona miara braku wiedzy czy optymalnego gradientu.

## Kontrole

| Ramię | Obrazy | Co sprawdza |
|---|---:|---|
| random |1500|zwykły dobór przy tym samym budżecie|
| readout_loss |1500|czy wystarcza sam błąd bez dynamiki|
| koopman_langevin |1500|zadaniowy zysk z dynamiki tokenów|
| placebo |1500|znaczenie przypisania zysku do konkretnej klatki|
| random_large |3000|porównanie z większym zwykłym zbiorem|

Wspólne pierwsze 500 L0. Random 1500 jest podzbiorem random 3000. Placebo przestawia względną poprawę max(0, 1-e_dyn/e_obs) wewnątrz operacji i czterech przedziałów trudności, wyłącznie między klatkami mającymi historię, potem mnoży przez oryginalne e_obs. Zachowuje dostępność historii, kontekst i przybliżoną trudność; nie jest testem wszystkich własności czasu. Brak okna daje score 0, niedobór dodatnich score jest uzupełniany losowo i raportowany. Wspólny minimalny odstęp 5 klatek, brak coverage i kwot scen.

## Kryteria przed wynikiem

Pięć treningów seed 42, 10 epok, COCO, ten sam RF-DETR Nano i receptura. Główny AP@[.50:.95]: zwykłe wagi ostatniej epoki, niezależna ocena na pełnej walidacji. Kryterium praktyczne:+0.004 AP względem readout_loss oraz dodatni kontrast z placebo. Raportujemy również oba losowe wybory. Brak spełnienia kryterium oznacza brak wykazanej korzyści według tej bramki w tym pilocie.

Tempo: krzywe walidacji, pole pod krzywą na wspólnym zakresie liczby ekspozycji obrazów i czas treningu. Walidacja w logach może wybierać max zwykłych/EMA, więc nie mieszamy jej z niezależnym końcowym AP zwykłych wag. Koszt ekstrakcji, selektora i wspólnego L0 oddzielnie; krótszy trening nie dowodzi przyspieszenia całego procesu.

Checkpointy co 2 epoki: obsługiwana opcja RF-DETR1.10. Odczyt foreground/background dopasowany na L0, oceniany na stałej sondzie walidacyjnej. CKA mierzy zmianę reprezentacji, nie poprawność semantyczną. Sonda nie wpływa na ranking ani hiperparametry.

Jedno ziarno jest eksploracyjne. Walidacja była oglądana w etapie 1. Mocniejszy dowód wymaga co najmniej 3 sparowanych ziaren i końcowej oceny na nietkniętym teście. Pilot nie rozstrzyga osobnego wkładu każdego członu GLE; do tego potrzebne ablacje pamięci, wymuszeń, zdarzeń i reszt.

Adaptacja skilla F:/Studia/MCHTR/Grant/Projekt/.agents/skills/eksperyment/SKILL.md: osobny katalog, wspólne dane, plan przed pomiarem, zapis również wyniku negatywnego. Grant jest źródłem równania i pozostaje niezmieniony.

Korekta redakcyjna po uruchomieniu, przed końcowym AP: słowo „obala” zastąpiono opisem niespełnienia bramki, ponieważ pojedynczy seed nie wystarcza do ogólnego obalenia użyteczności. Kryteria, liczności i procedura pozostały takie same; wersja sprzed treningu jest zachowana w archiwum źródeł na Edenie.
