# Wyniki identyfikacji Langevina i Koopmana

Aktualizacja: 2026-10-01T14:23:37.775300+02:00 (Europe/Warsaw).

Nie wykazano lepszego równania prognozy na 500 aktualizacji z historycznych paczek. Walidacja nowych wariantów wybrała samo jądro GLE bez dodatkowego wejścia. Strojenie ridge zmieniło wynik transferu s48 z 22,64 do 22,65 procent redukcji MSE, co jest praktycznie tą samą prognozą. Wynik zależy mocno od biegu; w głównym punkcie późnego czasu s47 ten GLE jest gorszy od braku zmiany.

## Aparat i podział

Dwa istniejące nagrania s47/s48, po 3001 stanów i 180 sond. Kalibracja LDA z 960..1000; wykonane historyczne paczki, ich reprezentacje i wpływy włączane z opóźnieniem jednego kroku. Odczyt dużych plików przez memmap; nowy trening ani Eden nie były uruchamiane. Trening s47: etykiety do1800; walidacja s47: starty1800..2000 co10; refit po wyborze: etykiety do2500. Transfer s48: starty2100..2500 co10 i cele do3000. Późny test s47: tylko2500->3000.

Pierwszy pilot (16 sekund CPU) zachowano w `wyniki/pilot_v1/`. W nim najlepszy nowy model z paczkami miał 15,49 procent na s48 wobec 22,64 procent GLE. Po obejrzeniu tego testu przeprowadzono jawnie eksploracyjny etap drugi: rozszerzono oryginalną bazę jądra i siatkę ridge, zachowując wybór tylko na walidacji. Etap drugi trwał około13 sekund. Te dane nie są już świeżym testem potwierdzającym.

Każde porównanie dotyczy tego samego celu końcowego s(t+H)-s(t). Stare rekurencyjne rodziny dopasowują średnie blokowe, a ich prognozy absolutne oceniamy tutaj na końcowym stanie. Nie odtwarzamy tym zmiany w średniej przyszłego bloku. Wynik jest oceną tych predyktorów dla wspólnego zadania końcowego.

## H=500: wszystkie ramiona

Umiejętność to 100*(1-MSE_model/MSE_brak_zmiany), a nie procent trafnych znaków. Kolumna s48 obejmuje 41 zależnych, nakładających się okien diagnostycznych. Główny punkt bez nakładania to t2500 dla każdego biegu.

| Ramię | Walidacja s47 [%] | Test s48 okna [%] | Główny s48 t2500 [%] | Główny s47 t2500 [%] | Znak s48 [%] |
|---|---:|---:|---:|---:|---:|
| ar | -27.96 | 18.43 | 18.11 | -4.94 | 65.9 |
| gle_compact | -21.69 | 22.33 | 16.72 | -6.29 | 66.8 |
| gle_descriptor | -17.48 | 14.91 | 26.25 | 3.04 | 64.1 |
| gle_effect | -22.56 | 22.10 | 14.93 | -5.90 | 66.6 |
| gle_descriptor_effect | -22.27 | 14.30 | -6.24 | 5.74 | 62.2 |
| gle_descriptor_effect_ped | -19.74 | 15.49 | -4.02 | 6.27 | 63.1 |
| lifting_quadratic | -23.12 | 22.29 | 16.73 | -5.86 | 66.8 |
| gle_kernel_ridge | 21.59 | 22.65 | 21.55 | -34.83 | 67.0 |
| gle_kernel_descriptor | -14.58 | -7.84 | 23.44 | -0.81 | 51.9 |
| gle_kernel_effect | 7.92 | 18.91 | 14.33 | -2.19 | 66.2 |
| gle_kernel_descriptor_effect | -23.39 | -6.90 | 22.93 | 1.95 | 52.8 |
| gle_kernel_descriptor_effect_ped | -11.84 | 0.24 | 24.88 | 1.45 | 56.7 |
| legacy_gle | 21.52 | 22.64 | 21.56 | -34.83 | 67.0 |
| zero | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| drift | -2010.35 | -1817.51 | -3120.15 | -2111.92 | 37.7 |
| legacy_koopman | 23.51 | 20.90 | 2.69 | -10.43 | 67.2 |
| legacy_langevin | 2.15 | -3.84 | 27.78 | -24.17 | 62.8 |

Wybór spośród nowych rodzin na walidacji: `gle_kernel_ridge`; wybór spośród wszystkich kontroli i modeli: `legacy_koopman`. Nie wybrano modelu o najlepszym wyniku na s48. Zmiana GLE ridge0.001 na0.0001 nie daje istotnej użytkowo poprawy. Sam wpływ paczki dodany do oryginalnej bazy ma na s48 18,91 procent wobec 22,65 procent tej samej bazy bez wejścia. Deskryptory z oboma wpływami mają 0,24 procent. Współczynniki poszczególnych cech nie są interpretowane przyczynowo.

## Horyzonty i zamrożony wybór

| H [aktualizacje] | Wybrany nowy wariant | Umiejętność s48 [%] | Oryginalny GLE [%] | Znak wybranego [%] |
|---:|---|---:|---:|---:|
| 10 | gle_kernel_effect | 19.98 | 20.20 | 63.3 |
| 50 | gle_kernel_ridge | 33.13 | 33.13 | 69.6 |
| 100 | gle_kernel_ridge | 38.18 | 38.29 | 71.6 |
| 500 | gle_kernel_ridge | 22.65 | 22.64 | 67.0 |

Na H10 walidacja wybrała jądro z historycznym wpływem paczki. Nie jest to dowód przydatności tego wejścia na H500. RMSE, MAE, znak wszystkich sond i znak połowy największych rzeczywistych zmian są w METRYKI.json; maska największych zmian służy tylko do oceny.

![Porównanie prognoz](wyniki/WYKRES.png)

## Reszty, ranga i kalibracja

Pełna baza samego jądra ma rangę21/21 i kondycję około8; baza jądra z opisem i oboma wpływami ma rangę52/57 i kondycję około1,36e19. Część deskryptorów jest stała lub zależna. Ridge stabilizuje predykcję, ale nie dowodzi jednoznacznej identyfikacji dynamiki ani wpływu sterowania. Brak interwencji na polityce i tylko jeden bieg uczący ograniczają wzbudzenie.

Dla H10, `gle_kernel_effect`, s48: skośność -0.18, kurtoza nadmiarowa 1.10, ACF lag1 -0.06, korelacja reszty z historycznym stanem -0.11, z wpływem paczki -0.02. Wariancje pierwszej/drugiej połowy: 0.51/0.524. Orientacyjny przedział90 pokrywa 92.6 procent testowych wyników, szerokość 2.58 jednostek s.

Dla H500, `gle_kernel_ridge`, s48: skośność -0.55, kurtoza nadmiarowa 1.29, ACF lag1 0.75, korelacja reszty z historycznym stanem -0.00, z wpływem paczki -0.02. Wariancje pierwszej/drugiej połowy: 1.31/1.53. Orientacyjny przedział90 pokrywa 85.0 procent testowych wyników, szerokość 3.79 jednostek s.

ACF dla H500 nie jest testem białości jednokrokowego szumu: okna silnie się nakładają. H10 ma rozstawienie startów10, więc cele nie nakładają się, lecz sondy i stany wciąż są zależne. Kształt rozkładu, zależności i wariancję oceniamy osobno; nie stwierdzamy gaussowskości ani białości. Przedziały wyznaczono z błędów historycznej walidacji, bez testowych reszt w wejściu. Pokrycie dalekie od90 pokazuje ograniczoną kalibrację przy przeniesieniu między biegami.

## Sprawdzenia matematyczne i przyczynowość

Wszystkie nowe predyktory przeszły test równości prognozy z prefiksu i z pełnego pliku oraz po podmianie sufiksu stanów/wejść. To obejmuje odczyt tylko ukończonego wpływu t-1, nie wpływu aktualizacji t. Dodatkowy test dojrzałych etykiet i niezmienności dopasowania po zatruciu sufiksu znajduje się w `wyniki/TESTY.json`.

Lean: cztery twierdzenia o skalarnej pamięci nad Int, kompilacja exit0 bez sorry/admit i nowych aksjomatów. SymPy:28 par dokładnych kontroli pamięci, Gaussian AR z niezerową kowariancją i dwa przykłady rang. To weryfikacja algebraiczna, a nie formalny dowód modelu ViT lub jego trafności. Pełny macierzowy wzór ma wyprowadzenie indukcyjne i kontrole SymPy; zakres Lean jest węższy. Szczegóły w matematyka/MATEMATYKA.md.

## Czego pilot nie dowodzi i co dalej

Nie potwierdza poprawy H500, domknięcia latentu, obserwowalności/sterowalności pełnej sieci, identyfikacji skutku dowolnej przyszłej paczki ani relacji FDT dla AdamW. Nie ustala nieosiągalnej granicy prognozy. Negatywny wynik tej siatki nie wyklucza innych reprezentacji paczek i modeli ich przyszłej dystrybucji.

Biegi mają stały zbiór i tylko3000 aktualizacji. Dwa ziarna są jedynymi replikacjami;180 sond i41 okien nie zwiększają ich liczby. Sondy obejmują wspólne obrazy i warianty; nie jest to test nowych filmów ani ogólnego rozpoznawania narzędzia. LDA każdego biegu jest osobną kalibracją z jego rozgrzewki. Historia s47 z tej samej fazy już nie jest odłożonym źródłem przy dowolnym kolejnym strojeniu.

Rozsądny następny etap to dłuższe wieloziarnowe nagrania z różnymi, z góry zadanymi politykami paczek oraz jawny model ich warunkowego przyszłego wpływu i reszt. Najpierw potrzebne są kilka niezależnych okien H500 po treningu/walidacji i rejestracja tych samych wejść. Nie uruchomiono takiej kampanii, nie wdrożono nowego równania do pętli SD i nie wykonano commitów.

## Artefakty

`MODEL.md`, `CONFIG.json`, `kod/dane.py`, `kod/pilot.py`, `kod/raport.py`, `kod/testy.py`, `wyniki/AUDYT.json`, `wyniki/METRYKI.json`, `wyniki/PROGNOZY.npz`, `wyniki/MODELE.npz`, `wyniki/TESTY.json`, `wyniki/WYKRES.png`, `wyniki/pilot_v1/`, `zrodla/` i `matematyka/`.


Aktualizacja 2026-10-01 16:35:25 Europe/Warsaw: [etap joint controlled bezKF](wyniki/joint_controlled/WYNIKI.md), [Astra sterowanie](PRZEGLAD_ASTRA_STEROWANIE.md), [Lean/SymPy](matematyka/CONTROLLED_MEMORY.md), [pełne źródła i Undermind DeepSearch](zrodla/LITERATURA_JOINT.md). H1s48 znak80,92procent i redukcjaMSE64,12procent; brak przewagi nad dawnymdirect. s48eksploracyjne, samych obrazów nie utożsamiać z wpływamiN/P.
