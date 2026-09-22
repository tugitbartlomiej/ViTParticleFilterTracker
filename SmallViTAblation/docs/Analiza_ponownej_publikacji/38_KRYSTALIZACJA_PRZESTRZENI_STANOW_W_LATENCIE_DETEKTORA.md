# 38. Krystalizacja przestrzeni stanów narzędzia w latencie detektora: Koopman i Langevin na dwóch zegarach

**Aktualizacja zakresu:** [plan 39](39_PLAN_SELEKCJI_KLATEK_RFDETR_I_ROZWOJU_JEPA.md) określa aktualny eksperyment selekcji już oznaczonych klatek do standardowego RF-DETR oraz kryteria jego oceny. Poniższe wyniki pozostają eksploracyjne z ograniczeniami sekcji 4e; dawne progi selekcji i szerokie twierdzenia o nowości nie zastępują nowego protokołu. Zapis o działających zadaniach jest historycznym statusem autora dokumentu, nie potwierdzeniem bieżącego stanu klastra.

Data: 2026-09-21. Status: **literatura przejrzana, kod gotowy, pierwszy pomiar (oś budżetu) wykonany, seria po epokach biegnie na Edenie (zadanie 1809133).**
Rozwija dokument [36](36_KOOPMAN_LANGEVIN_TOZSAMOSC_NARZEDZIA_I_DOBOR_KLATEK.md). Równolegle powstał plan
[37](37_PLAN_EKSPERYMENTU_DIAGNOSTYKA_LATENTU_W_WIDZENIU.md) (inna sesja, ten sam dzień); ten dokument realizuje pomiarowo jego
hipotezę H1 na latencie DETEKTORA i dokłada oś epok, której plan 37 nie ma. Przy rozbieżności kryteriów wiąże plan 37.
Kod: `pipeline/koopman_langevin/{clip_package,detector_latent,epoch_dynamics}.py`,
`eden/run_detector_latent.slurm`, `eden/run_epoch_series.slurm`, opcja trenera `--keep-epoch-checkpoints`.

## 0. Pytanie

Czy operatorem Koopmana i rozszerzonym równaniem Langevina da się (a) pokazać, że sieć w trakcie uczenia
buduje w latencie przestrzeń stanów narzędzia (położenie, rozmiar, oś, prędkość), (b) mierzyć to co kilka epok
na tych samych klatkach traktowanych jak cząstki, (c) na tej podstawie dobierać klatki do uczenia.

## 1. Sprostowanie, od którego zależy cała reszta

RF-DETR widzi JEDNĄ klatkę. Wektor latentu jednej klatki nie może zawierać prędkości narzędzia, poza śladem
rozmycia ruchu. Prędkość i prawo ruchu żyją w CIĄGU latentów kolejnych klatek. To nie osłabia pomysłu, tylko
ustawia go poprawnie:

- latent jednej klatki = położenie, rozmiar, oś, wygląd (to sieć może wykrystalizować ucząc się na ramkach),
- trajektoria latentu = prędkość i dynamika (to wyciąga dopiero Koopman i Langevin dopasowane do ciągu).

Dla modelu wideo (V-JEPA, docelowa architektura JEPA) prędkość może siedzieć już w jednym latencie, bo wejściem
jest kilka klatek. Literatura to potwierdza (niżej), więc ten sam pomiar przeniesiony na JEPA ma inną hipotezę
zerową: tam `single` ma prawo czytać prędkość, tu nie.

Dlatego każdy odczyt jest liczony w trzech ramionach: `single` (jedna klatka), `pair` (klatka plus różnica
z poprzednią) i `box_pair` (to samo z samych przewidzianych ramek). Jeśli prędkość z `pair` nie przebija
`box_pair`, to latent nie wnosi nic ponad ramkę i „mod prędkości" jest trywialną różnicą skończoną położenia.

## 2. Co mówi literatura (wyszukiwanie Undermind plus weryfikacja Scite, 2026-09-21)

| Wątek | Najbliższe prace | Wniosek dla nas |
|---|---|---|
| Stan fizyczny w latencie | Garrido i in. 2025 (V-JEPA, 10.48550/arXiv.2502.11831); Joseph i in. 2026 (10.48550/arXiv.2602.07050); Musa i in. 2026 (arXiv 2609.01551); Zhan i in. 2023 (10.52202/079017-1377) | W modelach WIDEO prędkość i kierunek są odczytywalne; reprezentacja jest rozproszona, nie „jeden kierunek = jedna wielkość". W modelach jednoklatkowych pokazano geometrię, nie prędkość. Musa częściowo przeczy Garrido (lekkie sondy blisko losu). |
| Koopman na wideo i w JEPA | Ruiz-Morales i in. 2025 (10.48550/arXiv.2511.09783); Comas, Sznaier, Camps 2021 (arXiv 2110.00547); Grosek i Kutz 2014 (arXiv 1404.7592); Erichson i in. 2016 (10.1007/s11554-016-0655-2) | Wszyscy uczą sieć RAZEM z operatorem albo pracują w pikselach. Nikt nie dopasował Koopmana do latentu gotowego detektora. Ruiz-Morales: minimum straty JEPA to funkcje własne Koopmana o wartości 1 (niezmienniki), tylko dane syntetyczne. |
| Koopman i Langevin dla uczenia | Redman i in. 2024 (10.48550/arXiv.2302.09160); Kim i in. 2026 (10.48550/arXiv.2602.22988); Kunin i in. 2023 (10.1162/neco_a_01626); Simon i in. 2023 (10.48550/arXiv.2303.15438) | Koopman opisuje WAGI w czasie treningu, Langevin opisuje SGD w przestrzeni wag. Diagnostyki, która śledzi operator dynamiki LATENTU DANYCH między checkpointami, nie znaleziono. Simon: uczenie SSL idzie schodkami, więc co 10-20 epok może ominąć stopnie. |
| Dobór danych | Mindermann i in. 2022, RHO-loss (10.48550/arXiv.2206.07137); Hu i in. 2021 (arXiv 2107.07741); Karamcheti i in. 2021 (10.18653/v1/2021.acl-long.564); Mittal i in. 2025 (10.1007/s11263-025-02372-z); Zolfaghari Bengar i in. 2019 (10.1109/ICCVW.2019.00120); Kim i in. 2026, Swift Sampling (10.48550/arXiv.2605.22678) | Rozdział błędu na usuwalny i nieusuwalny to koncepcyjnie RHO-loss. Wybór według straty kupuje szum (Hu) i zbiorowe odstające przykłady (Karamcheti). Przy augmentacji zysk z aktywnego doboru prawie znika (Mittal). Swift Sampling liczy zaskoczenie z prędkości i przyspieszenia cech, ale do inferencji, nie do etykietowania. |
| Mori-Zwanzig z danych | Lin i in. 2021 (10.1137/21m1401759); Liu, Huang i in. 2025, MEMnets (10.1038/s43588-025-00815-8) | Jądro pamięci jako kryterium reprezentacji istnieje dla dynamiki molekularnej i jako cel uczenia. Jako audyt gotowej sieci wizyjnej nie znaleziono. |

**Co jest nowe:** operator Koopmana i równanie z pamięcią dopasowane do latentu danych rzeczywistych
i śledzone między checkpointami jako miara krystalizacji; test, czy wolne mody niosą kinematykę narzędzia;
dobór klatek według usuwalnej części reszty w wideo chirurgicznym. **Uczciwie:** nowość to połączenie znanych
elementów i zastosowanie; każdy składnik z osobna istnieje.

## 3. Pomiar (dwa zegary, te same cząstki)

Cząstka = jedna klatka. Paczka: 16 klipów (12 walidacyjnych, 4 treningowe, zero testowych), po 300 klatek
co 3 klatki, razem 4800, z torami GT. `detector_latent.py` zapisuje dla każdego checkpointu trzy wektory na
klatkę: stan zapytania dekodera o najwyższym wyniku (`query`, z niego sieć czyta ramkę), średnią mapy cech
w przewidzianej ramce (`box_pool`) i średnią z całej klatki (`global_pool`). Etykiet nie czyta.

`epoch_dynamics.py` liczy na każdym checkpoincie:

| Wielkość | Co znaczy |
|---|---|
| `probe` single, pair, box_pair | ile położenia, rozmiaru, osi i prędkości da się odczytać liniowo, na innych operacjach niż dopasowanie |
| `langevin` | jaką część następnego kroku latentu przewiduje równanie; zysk z jądra pamięci; korelacja reszt; podłoga z przetasowanymi krokami |
| `koopman` | ile stanu fizycznego niosą najwolniejsze funkcje własne, przeciw tej samej liczbie głównych kierunków PCA |
| `participation_ratio` | ilu kierunków latent naprawdę używa |
| `detection_iou` | sama krzywa uczenia na tych klatkach |

i między kolejnymi checkpointami: przesunięcie każdej cząstki po najlepszym obrocie (Procrustes) oraz CKA.
Przesunięcia per klatka zapisuje do `*.displacement.npz`: to surowiec dla doboru klatek po zegarze treningu
(klatka, której cząstka wciąż się rusza pod koniec uczenia, nie jest jeszcze „osadzona").

## 4. Kryteria zapisane PRZED wynikiem

Hipoteza krystalizacji uznana za potwierdzoną, jeśli na osi treningu (COCO -> kolejne epoki):

1. `probe.position.single` i `probe.axis.single` dla `query` rosną razem z `detection_iou`;
2. `probe.velocity.single` zostaje blisko zera (poniżej 0,15) na każdym checkpoincie, a `probe.velocity.pair`
   rośnie i na końcu przebija `box_pair` o co najmniej 0,05;
3. `langevin.skill_with_memory` rośnie i jest wyraźnie nad podłogą z przetasowania (co najmniej 10 punktów);
4. `koopman.carried.position.slow_modes` przebija `leading_pca` na wytrenowanym modelu, a nie na COCO.

Hipoteza obalona w całości, jeśli punkty 2 i 4 nie zachodzą: wtedy latent niesie tylko to, co ramka, a Koopman
nie wyróżnia kierunków fizycznych lepiej niż zwykłe PCA. Wynik częściowy raportować wprost.

Dobór klatek ma osobne kryterium z dokumentu 36: trzy ziarna nad średnią `uniform_time` 0,6832 i przewaga
średnia powyżej 0,004 przy budżecie 1500.

## 4a. Wynik pierwszego pomiaru: oś budżetu (2026-09-21, zadanie Eden 1809131)

Cztery wersje wag na tych samych 4800 klatkach: COCO (przed uczeniem) oraz `uniform_time` ziarno 42 po 500, 1500
i 2500 klatkach (`best.pth`). Plik: `results/koopman_langevin_epochs/REPORT_budget_axis.json`. To NIE jest oś epok.

| Wagi | IoU | położenie `query` / `box_pool` | rozmiar | oś | Langevin z pamięcią (podłoga) | zysk pamięci | położenie: wolne mody Koopmana / PCA (`box_pool`) |
|---|---|---|---|---|---|---|---|
| COCO | 0,088 | -0,01 / -0,10 | 0,00 | -0,06 | 31,3 (-1,5) | +4,6 | -0,32 / -0,03 |
| 500 | 0,833 | 0,68 / 0,86 | 0,78 | 0,68 | 22,5 (-1,6) | +4,1 | 0,84 / 0,04 |
| 1500 | 0,839 | 0,76 / 0,83 | 0,80 | 0,73 | 21,5 (-1,6) | +3,6 | 0,80 / 0,17 |
| 2500 | 0,847 | 0,69 / 0,83 | 0,82 | 0,62 | 20,7 (-1,7) | +4,1 | 0,87 / 0,19 |

Ruch cząstek między wagami (po najlepszym obrocie): COCO -> 500 przesunięcie 1,36 i CKA 0,07 dla `query`;
500 -> 1500 i 1500 -> 2500 przesunięcie 0,25 i CKA około 0,88. `global_pool` prawie stoi (CKA 0,85 do 0,96).

Odczyt względem kryteriów z sekcji 4:

1. **Spełnione.** Przed uczeniem położenia, rozmiaru i osi nie da się odczytać (R2 około 0), po uczeniu R2 0,6 do 0,86.
   Uczenie przebudowuje latent narzędzia, a latent całej sceny zostawia w spokoju.
2. **Niespełnione i w tej postaci nierozstrzygalne.** Prędkości nie czyta żadne ramię: jedna klatka 0,00, para 0,00 do
   0,03, same ramki poniżej zera. Powód leży w celu, nie w latencie: 1 procent klatek niesie 58 procent energii
   prędkości (skoki adnotacji, powrót narzędzia), a nawet z prawdziwych torów GT para daje tylko 0,39 (po obcięciu
   skoków 0,18). Na kroku 3 klatek prędkość narzędzia to głównie drganie. Cel `velocity_trimmed` i sufit z GT
   dodano PO pierwszym odczycie; to odstępstwo od kryteriów, jawnie.
3. **Niespełnione, kierunek odwrotny.** Przewidywalność kroku latentu SPADA z uczeniem (31 -> 21 procent), przy
   podłodze około -1,5, więc to realna dynamika. Latent COCO opisuje wolno zmienną scenę; latent po uczeniu idzie
   za narzędziem, a ruch narzędzia jest przewidywalny w 22 do 29 procent (dokument 36, tory GT). Latent dziedziczy
   przewidywalność obiektu, który śledzi. Kryterium 3 było źle postawione: miarą krystalizacji nie jest wzrost
   przewidywalności, tylko jej zbieżność do przewidywalności toru fizycznego.
4. **Spełnione.** Sześć najwolniejszych funkcji własnych Koopmana niesie położenie z R2 0,73 do 0,87, a ta sama
   liczba głównych kierunków PCA 0,04 do 0,45. Na COCO obie wartości są około zera. Dla rozmiaru przewagi nie ma.

Oś budżetu: między 500 a 2500 klatek prawie nic się nie zmienia (IoU 0,833 -> 0,847), zgodnie z kampanią statyczną.

## 4b. Gdzie dobór klatek ma co poprawiać: jakość detektora według prędkości narzędzia

Te same klatki walidacyjne (3458, 12 klipów), podział według prędkości środka ramki GT na krok 3 klatek.
Udział chybień = ramka o IoU poniżej 0,5.

| Wagi | wolne 50 procent | 50 do 80 | 80 do 95 | najszybsze 5 procent (powyżej 14 px na krok) |
|---|---|---|---|---|
| 500 klatek | 3,5 procent | 3,7 | 5,0 | 13,9 |
| 1500 klatek | 4,5 | 4,1 | 6,0 | 13,3 |
| 2500 klatek | 2,9 | 3,5 | 5,2 | 14,5 |

Detektor chybia w szybkim ruchu cztery do pięciu razy częściej niż w wolnym, a pięć razy więcej zwykłych,
równo rozłożonych klatek tego NIE naprawia. Średnie mAP tego nie widzi, bo szybkie klatki to 5 procent zbioru.
Zastrzeżenie: część najszybszych klatek to skoki adnotacji i powrót narzędzia w kadr, więc nie całe 14 procent
jest do odzyskania. Wniosek dla doboru: miarą sukcesu ma być udział chybień w najszybszych 5 i 20 procentach
klatek, obok mAP. Wariant `langevin_surprise` kupuje właśnie takie klatki (28 razy częściej niż losowanie,
dokument 36), warianty `veto` je odrzucają, więc test 12 biegów (zadanie Eden 1809135, `eden/run_langevin_selection.slurm`)
rozstrzyga kierunek. Kryterium dopisane przed wynikiem: `langevin_surprise` obniża udział chybień w najszybszych
5 procentach o co najmniej 3 punkty procentowe względem `uniform_time` na wszystkich trzech ziarnach, bez spadku mAP
poniżej 0,6832 minus 0,004.

## 4c. Prędkość jako stan ukryty: okno czasu i filtr Langevina (2026-09-21)

Kod: `latent_velocity.py`, `kalman_state.py`. Wyniki: `REPORT_velocity.json`, `REPORT_kalman_*.json`.
Miarą informacji jest korelacja z nachyleniem prawdziwego toru (okno +-2 kroki, 1 procent skoków odcięty).

**Okno czasu.** Nachylenie odczytu położenia z latentu `box_pool` po +-1, 2, 4, 8 krokach: korelacja 0,33 / 0,46 /
0,57 / 0,62 (wagi 2500); z samych ramek detektora 0,55 / 0,68 / 0,79 / 0,87; na wagach COCO około 0,03. Prędkość
JEST w ciągu latentów, ale dopiero na dłuższej podstawie czasu; różnica dwóch klatek tonie w szumie odczytu.

**Filtr.** Model na oś obrazu: `p' = p + v`, `v' = f_v v + a`, `a' = f_a a + w`, `y = p + n`, czyli rozszerzone równanie
Langevina z szumem pomiaru; przyspieszenie to kolorowe wymuszenie (ręka chirurga, której nie widać). Cztery liczby
dopasowane największą wiarygodnością na INNYCH operacjach. Pierwsza, gaussowska wersja zawiodła (korelacja 0,10 do
0,31, gorzej niż zwykła różnica): innowacje mają nadwyżkową kurtozę 18 do 65, więc dopasowanie uciekło w „brak
bezwładności, szum 18 px". Wersja odporna (strata Hubera, obcięcie innowacji na 2,5 odchylenia, Masreliez):

| Wagi | Źródło położenia | różnica +-2 (R2) | filtr przyczynowy (R2) | wygładzacz (R2) | szum odczytu |
|---|---|---|---|---|---|
| COCO | oba | 0,00 | 0,00 | 0,01 | 38 do 77 px |
| 500 | latent `box_pool` | 0,37 (-2,89) | 0,17 (+0,01) | 0,42 (-0,61) | 12,0 px |
| 500 | ramka detektora | 0,54 (-2,07) | 0,49 (+0,22) | 0,64 (-0,63) | 2,1 px |
| 2500 | latent `box_pool` | 0,46 (-1,88) | 0,31 (+0,04) | 0,48 (-0,52) | 11,4 px |
| 2500 | ramka detektora | 0,68 (-0,53) | 0,67 (+0,44) | 0,76 (+0,22) | 3,2 px |

Odczyt: (1) filtr zamienia zaszumione położenie w użyteczną prędkość (R2 dodatnie tam, gdzie różnica ma ujemne);
(2) jakość tej prędkości rośnie z uczeniem detektora (0,49 -> 0,67), więc nadaje się na miarę krystalizacji stanu;
(3) latent przegrywa z własną ramką detektora, bo liniowy odczyt położenia z latentu ma błąd 11 px, a ramka 3 px;
na tym etapie latent NIE wnosi o ruchu nic ponad ramkę; (4) innowacje nie są białe (korelacja 0,17 do 0,20) i mają
bardzo grube ogony (kurtoza 25 i 132): wymuszenie przychodzi rzadkimi skokami, jak zakłada HAVOK (Brunton i in. 2017);
(5) rozmiar innowacji wskazuje chybienia detektora z AUC 0,60 do 0,68, a pewność samego detektora z AUC 0,81 do 0,83,
więc jako samodzielny wykrywacz błędów innowacja PRZEGRYWA z pewnością detektora.

## 4d. Autoenkoder Koopmana: co literatura pozwala obiecać (przegląd 2026-09-21, Undermind plus Scite)

- Lusch, Kutz, Brunton 2018 (10.1038/s41467-018-07210-0): bez ograniczeń sieć „did not converge to a unique and
  interpretable solution". Dowolne odwracalne T daje równoważny model, więc osie „położenie, prędkość, przyspieszenie"
  same się nie wyłonią; trzeba je wpisać w obserwable (stan jawnie w słowniku). Brunton i in. 2016
  (10.1371/journal.pone.0150171): dokładne domknięcie ze stanem w słowniku istnieje tylko przy jednym punkcie stałym.
- Proctor, Brunton, Kutz 2018 (10.1137/16M1062296): wymuszenie zewnętrzne „psuje" mody autonomicznego Koopmana;
  przy nieznanym wejściu zostaje operator stochastyczny, czyli przewidywana jest wartość oczekiwana. Wanner i Mezić 2022
  (10.1137/21m1414425): zwykły DMD jest obciążony przy zaszumionych obserwablach, a to nasz przypadek.
- Becker i in. 2019, Recurrent Kalman Networks (arXiv 1905.07357): „impossible to obtain velocities from a single
  image"; prędkość trafia do pamięciowej części latentu uzupełnianej filtrem. To samo robi tutaj `kalman_state.py`.
- HAVOK, Brunton i in. 2017 (10.1038/s41467-017-00030-8) i Hirsh i in. 2021 (10.1098/rspa.2021.0097): model liniowy
  w opóźnieniach plus rzadkie wymuszenie; współrzędne opóźnień to kolejne pochodne toru, czyli szukany stan.
- Nie znaleziono pracy, która robi to na latencie zamrożonego detektora narzędzi chirurgicznych.

Konstrukcja do zbudowania: pierwsze współrzędne to zamrożona liniowa głowica położenia, rozmiaru i osi; reszta to
uczone obserwable na oknie opóźnień; macierz K blokowo-łańcuchowa; nieznane wejście szacowane z innowacji; pamięć
Mori-Zwanziga dopiero na resztach po filtrze. **Próg do pobicia:** filtr z tej sekcji (korelacja 0,67, R2 0,44 na
ramkach; 0,31 i 0,04 na latencie). Autoenkoder ma sens tylko wtedy, gdy z LATENTU wyciągnie więcej niż filtr z ramek.

## 4e. Stosunek do planu 37 (po przeczytaniu całości, 2026-09-21)

Plan 37 jest ostrzejszy od pomiarów z tego dokumentu i przy badaniu potwierdzającym wiąże. Pomiary stąd mają
status PILOTA EKSPLORACYJNEGO. Zarzuty planu, które trafiają w kod użyty tutaj:

- **A1, trafne.** `LatentLangevin` standaryzuje latent i skalę przyrostów na CAŁEJ puli przed podziałem na grupy
  (`latent_langevin.py`, linie 81 i 87). To łagodny przeciek; liczby „Langevin 21 do 31 procent" wymagają powtórzenia
  po przeniesieniu skalowania do foldów. PCA w `epoch_dynamics.pca` też jest liczone na wszystkich klatkach.
- **A3, trafne.** `reducible_gain` dzieli przez błąd bieżącego celu, więc jest wynikiem retrospektywnym, nie prognozą.
- **Sekcja 7.1, trafne.** „Zysk pamięci +4 punkty" z sekcji 4a to różnica między historią 3 i 8 kroków. Model krótki
  już używa prędkości i przyspieszenia, więc NIE jest kontrolą „bez historii". Właściwy kontrast to M4 kontra M2 z planu.
- **Sekcja 8.1, trafne.** `detector_latent.py` bierze zapytanie dekodera o najwyższym wyniku; numer zapytania nie jest
  trwałą tożsamością. Dlatego wnioski opierać na `box_pool`, a `query` traktować jako wrażliwe na przeskoki.
- **Sekcja 5.1, trafne wstecz.** Pomiary torów z dokumentu 36 czytały wszystkie 167 grup, w tym testowe (same tory GT).
  Paczka z tego dokumentu testowych klipów nie zawiera.
- **Sekcja 9.** Test 12 biegów (zadanie 1809135) NIE jest etapem D planu: nie ma ramion S3 (niepewność detektora)
  ani S4 (model bez historii), a wynik jest retrospektywny. Może pokazać kierunek (kupować czy odrzucać zaskoczenia),
  nie może przypisać zysku historii ani równaniu Langevina.

Czego plan 37 nie ma, a ten pilot wnosi jako wejście do niego:

1. Latent całej sceny (R1 planu) nie niesie stanu narzędzia wcale (R2 około 0 przed i po uczeniu), a latent obszaru
   narzędzia niesie go dopiero po uczeniu detektora. Warto od razu liczyć R2/R3 na latencie DETEKTORA, nie tylko DINO.
2. Oś epok (zegar treningu) i przesunięcie cząstek między wagami.
3. Prędkość jako stan ukryty z filtru odpornego na skoki; innowacje mają kurtozę 25 do 132, więc testy reszt z A5
   muszą zakładać grube ogony, a modele M0 do M5 stratę odporną, inaczej pojedyncze skoki zdominują skill.
4. Warstwa oceny według prędkości narzędzia: 14 procent chybień w najszybszych 5 procentach klatek wobec 3 procent
   w wolnych. Średnie AP tego nie pokaże; dopisać tę warstwę do miar etapu D.
5. Wstępny sygnał dla H2: sama innowacja wskazuje chybienia z AUC 0,60 do 0,68, pewność detektora 0,81 do 0,83.
   Kontrast C3 kontra C1 będzie trudny do wygrania.

## 4f. Oś epok: jeden trening z wagami co 3 epoki (zadanie Eden 1809133, 2026-09-21)

`uniform_time`, 1500 klatek, ziarno 42, 30 epok; latent `box_pool` na tych samych 4800 klatkach.
Plik: `REPORT_epoch_axis.json`.

| Epoka | IoU | położenie | oś | wolne mody Koopmana / PCA | Langevin | przesunięcie cząstek od poprzedniego punktu (CKA) | AP walidacji |
|---|---|---|---|---|---|---|---|
| COCO | 0,088 | -0,10 | -0,03 | -0,32 / -0,03 | 29,6 | | |
| 2 | 0,832 | 0,87 | 0,65 | 0,83 / 0,10 | 22,4 | 1,29 (0,09) | 0,665 |
| 5 | 0,841 | 0,85 | 0,67 | 0,81 / 0,20 | 21,9 | 0,13 (0,94) | 0,687 |
| 14 | 0,834 | 0,78 | 0,64 | 0,77 / 0,20 | 22,2 | 0,12 (0,93) | |
| 29 | 0,836 | 0,86 | 0,73 | 0,84 / 0,24 | 22,2 | 0,09 (0,95) | 0,656 |

Odczyt: (1) cała przebudowa latentu dzieje się w pierwszych trzech epokach; od epoki 2 odczyt położenia, osi,
przewaga modów Koopmana nad PCA i przewidywalność kroku stoją w miejscu. Punkty co 10 do 20 epok niczego by nie
pokazały; żeby zobaczyć samo układanie się stanu, trzeba zapisywać wagi wewnątrz epok 0 do 3 (co kilkadziesiąt kroków).
(2) AP walidacji rośnie do epoki 5 (0,687), a potem spada do 0,656 w epoce 29: przy 1500 klatkach 30 epok to
przeuczenie. Cząstki dalej dryfują (około 0,1 na 3 epoki) bez żadnej poprawy odczytu stanu, więc ten dryf nie jest
nauką o narzędziu. (3) Dla miary „szybkość uczenia" z dokumentu 39 liczą się więc epoki 0 do 5; późniejsze różnice
między ramionami mierzą głównie odporność na przeuczenie. Ocena `best_val` i `last_epoch` muszą być raportowane obie.

## 4g. Widmo Koopmana latentu detektora: niezmienniki, podział statyczne/dynamiczne, ile da się zlinearyzować (2026-09-22)

Kod: `pipeline/koopman_langevin/koopman_spectrum_axis.py`; wynik `REPORT_spectrum_axis.json`. Latent `box_pool` po PCA do 24
wymiarów, podniesiony jednym opóźnieniem, operator liniowy dopasowany na klipach treningowych, oceniany na testowych.
Klasy wartości własnych: niezmiennik (|lambda| > 0,97, kąt < 5 stopni), wolne rzeczywiste, obracające (kąt >= 5 stopni), szybkie.

| Wagi | niezmienniki | wolne | obracające | szybkie | niezmienniki: wariancja między filmami | niezmienniki: R2 ze stanu narzędzia | niezmienniki: defekt poza próbą |
|---|---|---|---|---|---|---|---|
| COCO | 2 | 3 | 31 | 12 | 0,81 | -0,08 | 0,19 |
| epoka 2 (b1500) | 4 | 7 | 29 | 8 | 0,81 | 0,59 | 0,05 |
| 500 klatek | 4 | 6 | 26 | 12 | 0,78 | 0,58 | 0,05 |
| 2500 klatek | 5 | 6 | 26 | 11 | 0,74 | 0,29 | 0,09 |
| epoka 29 (b1500) | 3 | 7 | 28 | 10 | 0,77 | 0,29 | 0,04 |

Odczyt: (1) niezmienniki to tożsamość OPERACJI (74 do 81 procent wariancji między filmami) i po uczeniu także stan
narzędzia (R2 0,58 tuż po starcie uczenia, 0,29 przy dłuższym treningu). (2) **Defekt niezmienników poza próbą spada
z 0,19 przed uczeniem do 0,05 po uczeniu**: to, co sieć uznaje za stałe, staje się naprawdę stałe z klatki na klatkę.
To jest mierzalna „krystalizacja" bez etykiet i kandydat na miarę do śledzenia co epokę. (3) Mody obracające mają okres
około 7 klatek i zerowy związek ze stanem narzędzia: to drganie, nie dynamika narzędzia. Podział statyczne/dynamiczne
jest więc: niezmienniki = tożsamość plus stan, reszta = szum o krótkim okresie. (4) Uwaga: 24 wymiary i 4800 klatek
dają 48 wartości własnych z próby o ograniczonej długości; liczby modów są przybliżone (dokument 38, kontrargument o EDMD).

**Ile części nieliniowej da się zamienić w liniową** (umiejętność jednokrokowa poza próbą, latent 2500 klatek):

| wymiar | liniowy, 1 opóźnienie | liniowy, 8 opóźnień (najlepsza regularyzacja) | nieliniowy z historią (RFF + 8 kroków) |
|---|---|---|---|
| 8 | 7,7 | 17,0 | 21,4 |
| 16 | 2,0 | 5,2 | 21,0 |
| 24 | -1,1 | -2,5 | 20,9 |
| 32 | -10,4 | -10,3 | 21,0 |
| 64 | -17,3 | -19,1 | 20,4 |

Liniowy operator we współrzędnych samego latentu działa tylko przy 8 wymiarach i 8 opóźnieniach (17 procent) i psuje się
z wymiarem; model nieliniowy trzyma 21 procent niezależnie od wymiaru, a tor fizyczny narzędzia ma 22 do 29 procent.
Autoenkoder Koopmana (podniesienie uczone) miałby do domknięcia lukę 4 do 5 punktów przy małym wymiarze i całą
przy dużym; sufit jest fizyczny (ręka chirurga), nie słownikowy. Sam autoenkoder nie został jeszcze uczony.

## 5. Kontrargumenty

- Reprezentacja fizyki w sieciach jest rozproszona (Joseph i in.); pojedynczy mod równy jednej wielkości jest
  mało prawdopodobny. Dlatego mierzona jest podprzestrzeń kilku modów, nie jeden mod.
- EDMD na krótkich ciągach w wielu wymiarach produkuje fałszywą strukturę. Dlatego jest podłoga z permutacji
  i redukcja PCA do 24 wymiarów przed dopasowaniem.
- Zapytanie dekodera o najwyższym wyniku może przeskakiwać między slotami z klatki na klatkę. To wnosi sztuczny
  szum do trajektorii `query`; `box_pool` i `global_pool` są na to odporne i służą jako kontrola.
- CKA bywa zawodne jako miara podobieństwa (Davari i in. 2022, znany tylko tytuł); dlatego obok jest Procrustes.
- Trzy poziomy budżetu (500, 1500, 2500 klatek) to nie jest oś epok. Oś epok daje dopiero `run_epoch_series.slurm`.
- Zysk z doboru danych przy augmentacji bywa znikomy (Mittal 2025); kampania statyczna już to pokazała
  (wszystkie selektory w pół punktu od losowania).

## 6. Data gaps

- Seria checkpointów po epokach (co 3 epoki, `uniform_time` b1500 ziarno 42) biegnie: zadanie Eden 1809133; odczyt trafi do `results/koopman_langevin_epochs/`.
- Prędkość wymaga dłuższej podstawy czasu (wygładzenie toru po 5 do 9 krokach) albo gęstszych klatek; na kroku 3 klatek sufit z GT to 0,18.
- Brak pomiaru na modelu wideo i na JEPA; hipoteza zerowa dla `single` jest tam inna.
- Brak linii bazowej RHO-loss dla doboru klatek; recenzent jej zażąda.
- Niezmiennik punktu obrotu (nacięcie) wyszedł słabo na torach GT (dokument 36); w latencie niesprawdzony.
- Prace oznaczone „znany tylko tytuł" w przeglądzie wymagają przeczytania przed cytowaniem.
