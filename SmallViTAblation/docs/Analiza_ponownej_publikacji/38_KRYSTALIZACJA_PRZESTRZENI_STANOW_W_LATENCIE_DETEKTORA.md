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

## 4h. Człony z teorii sterowania: co literatura pozwala dopisać do równania (przegląd 2026-09-22, Undermind plus Scite)

Pytanie autora: wymuszenie to ruch kamery (tak w testach DiagnosticTool, gdzie kamera OU jest sterowana i znana co
do liczby); w chirurgii kamera mikroskopu stoi, a wejściem jest ręka chirurga, nieobserwowane sterowanie. Co jeszcze
z teorii sterowania pasuje do równania i do doboru danych?

| Wątek | Prace (zweryfikowane w Scite, bez not redakcyjnych) | Wniosek |
|---|---|---|
| Koopman z wejściem | Proctor, Brunton, Kutz 2018 (10.1137/16M1062296); Korda, Mezić 2018 (10.1016/j.automatica.2018.03.046); Iacob, Tóth, Schoukens 2022 (arXiv 2207.12132); Bruder i in. 2021 (10.1109/LRA.2021.3068117); Shang i in. 2026 (arXiv 2602.14537) | Dokładne podniesienie daje stałe `A`, ale macierz wejścia `B(z, u)` zależną od stanu: właściwy człon jest BILINIOWY `sum_i u_i N_i z`, nie `W u`. Przy nieobserwowanym `u` identyfikowalny jest tylko iloczyn `B u`, więc bez pomiaru `u` człon wymuszenia zlewa się z resztą. |
| Wymuszenie nieznane | HAVOK, Brunton i in. 2017 (10.1038/s41467-017-00030-8); Dylewsky i in. 2022 (10.1103/physreve.105.015312); Yu, Chakravorty 2015 (10.1016/j.automatica.2015.10.013); Abooshahab i in. 2022 (10.1016/j.automatica.2021.110017) | Dwie drogi: wymuszenie jako reszta modelu opóźnieniowego (HAVOK) albo nieznane wejście jako proces AR w stanie rozszerzonym `[z; u]` z warunkiem rangi `CB`. |
| Obserwowalność | Yeung, Liu, Hodas 2018 (10.23919/ACC.2018.8431738); Mesbahi i in. 2020 (10.1016/j.automatica.2020.109353); Park, Hwang 2024 (10.1109/CDC56724.2024.10886269) | Gramian obserwowalności Koopmana `W_o = sum_k (A^k)^T C^T C A^k` mówi, które kierunki stanu podniesionego widzi wyjście; u nas wyjściem `y = C z` jest głowa detektora. Kierunki o małej wartości własnej `W_o` to te, które latent zmienia, a detekcja nie widzi. Symetrie dynamiki dają kierunki nieobserwowalne. |
| Projekt eksperymentu | Abraham, Murphey 2019 (10.1109/TRO.2019.2923880); Ott, Kochenderfer, Boyd 2024 (arXiv 2409.13088); Wu 2026 (arXiv 2605.09545, 2605.17966); Shang, Cortés, Zheng 2024 (10.1109/LCSYS.2024.3522594) | Informacja Fishera o operatorze, T- i D-optymalność, certyfikaty jakości danych (pokrycie stanu, niedegeneracja cech, widmo regresji), trwałe pobudzenie dla układu podniesionego. Nikt tego nie zrobił dla wideo ani reprezentacji sieci. Ostrzeżenie Wu: wynik zadania końcowego nie jest monotoniczny w żadnym certyfikacie; You i in. 2026 (arXiv 2607.10362): błąd jednokrokowy operatora nie mierzy użyteczności modelu latentnego. |
| Rzadkie szarpnięcia ręki | Gollee i in. 2017 (10.1113/JP274288); Susilaradeya i in. 2019 (10.7554/eLife.40145); Navas, Stark 1968; Chen, Zhou, Gao 2023 (10.1016/j.dsp.2023.104003) | Ręka steruje PRZERYWANIE: otwarta pętla plus korekty wyzwalane progiem błędu, z refrakcją 200 do 500 ms, submovements 2 do 3 na sekundę. To tłumaczy kurtozę innowacji 25 do 132 z sekcji 4c. Estymator: odporny filtr wielomodelowy ze Studentem-t (IMM), HAVOK. Zastrzeżenie: chirurg reaguje na to, co widzi, więc identyfikacja w pętli jest obciążona (Ljung, Forssell 1999, 10.1109/9.802946). |
| Dane jako sterowanie | Lessard, Zhang, Zhu 2018 (arXiv 1810.06175); E, Han, Li 2018 (10.1007/s40687-018-0172-y) | „Dobierz przykłady, żeby stan uczącego przeszedł z A do B najmniejszym kosztem" istnieje dosłownie (sequential machine teaching, zasada Pontriagina), ale rozwiązane tylko dla uczącego gradientowego z kwadratową stratą i znanym celem. Dla detektora cel jest nieznany, więc kryterium degeneruje do przyrostu informacji. |

Człony do dopisania (postać, identyfikacja, kontrola):

1. **Człon biliniowy** `sum_i u_{t,i} N_i z_t` zamiast `W u_t`; `u_t` musi być choć częściowo zmierzone, naturalne pełnomocnictwo to
   przemieszczenie końcówki narzędzia z samego detektora. Kontrola: permutacja `u_t` w czasie musi wyzerować `N_i`.
2. **Człon skokowy** `sum_k a_k 1[t = tau_k]` z wyzwalaniem progowym i refrakcją; identyfikacja przez odporny IMM,
   histogram odstępów między zdarzeniami. Kontrola: surogat gaussowski o tej samej wariancji nie odtworzy kurtozy.
   To jest obrazowy odpowiednik członu zdarzeń `Z` z równania grantowego (dokument `tft_langevin_czlon_zdarzen`).
3. **Stan rozszerzony** `[z; u]`, `u_{t+1} = Phi u_t + w_t`; warunek rangi `CB`, `B` z członu 1, inaczej błędne koło.
4. **Sprzężenie zwrotne** `u_t = -K (z_t - z_ref) + w_t`; identyfikacja metodą pośrednią z instrumentem zewnętrznym
   (faza operacji). Kontrola: korelacja krzyżowa reszt z ich przeszłością znika po dodaniu członu.

Nowe kryterium doboru klatek z teorii sterowania: **D-optymalne (dźwignia)**. Dla regresji operatora kowariancja
estymaty jest proporcjonalna do `(Psi_S Psi_S^T + lambda I)^-1`, więc klatka o największej dźwigni

    s_t = psi_t^T (Psi_S Psi_S^T + lambda I)^-1 psi_t

najbardziej zmniejsza niepewność operatora; zachłannie `argmax s_t` to maksymalizacja `log det`. Wybiera klatki
z RZADKO odwiedzanych rejonów przestrzeni trajektorii, nie klatki z dużą resztą. Test bez treningu (zapisany przed
wynikiem): przy równym budżecie ślad kowariancji operatora ma spadać monotonicznie dla doboru D-optymalnego, a nie
dla resztowego; błąd jednokrokowy poza grupami nie może rosnąć. Kod: `pipeline/koopman_langevin/leverage_selection.py`.
Czy pomaga DETEKTOROWI, rozstrzygnie dopiero ramię treningowe etapu 2.

**Wynik pilota D-optymalnego (2026-09-22, bez treningu; `REPORT_leverage.json`).** Pula 10 637 klatek, z podniesionym
stanem 6874; 39 ze 117 grup operacji odłożonych dla każdego ramienia; L0 = 335 klatek z podniesionym stanem; dobrane 1000.

| Ramię | log det | ślad kowariancji operatora (mniej = pewniej) | umiejętność jednokrokowa operatora na odłożonych grupach |
|---|---|---|---|
| równo w czasie | 300,4 | 0,210 | 7,3 |
| losowanie | 300,4 | 0,209 | 7,4 |
| reszta Langevina (górne) | 314,8 | 0,126 | 4,8 |
| D-optymalne (dźwignia) | 320,7 | **0,116** | **4,0** |

Przewidywanie sprawdziło się w połowie: dobór D-optymalny obniża niepewność operatora najbardziej i monotonicznie
(0,42 -> 0,27 -> 0,18 -> 0,12 przy 100, 250, 500, 1000 klatkach). Ale dobór resztowy obniża ją prawie tak samo, bo klatki
o dużej reszcie są zarazem klatkami o dużej dźwigni (część wspólna obu wyborów 53 procent, z równym w czasie 16 procent).
I co ważniejsze: operator dopasowany na klatkach D-optymalnych przewiduje przejścia na NOWYCH operacjach GORZEJ
(4,0 wobec 7,3 dla równego wyboru), bo pewność co do operatora kupiono na przypadkach skrajnych, które nie opisują
typowego kroku. To jest dokładnie ostrzeżenie Wu 2026 i You 2026: certyfikat informacji nie jest monotoniczny
w użyteczności. Kryterium D-optymalne zlewa się więc z kryterium zaskoczenia, które w etapie 1 nie pomogło (K1, K3),
i NIE zasługuje na ramię treningowe w tej postaci. Wersja do rozważenia: dźwignia liczona w rozkładzie typowym
(A-optymalność ważona gęstością), czyli pokrycie, do którego stage 1 już doszedł od strony Koopmana (K4).

## 4i. Cząstki jako tokeny-patche (konwencja DiagnosticTool projektu grantowego), 2026-09-22

Na życzenie autora cząstka to jeden PATCH obrazu, nie cała klatka: `patch_particles.py` (rejestrator według
`Research/DiagnosticTool/test/02_rejestrator.md`) i `patch_analysis.py` (testy 03 i D65 z DiagnosticTool). Rusztowanie
odczytane z wag końcowej normy backbone DINOv2 w RF-DETR (jądro = `beta`, długości = `|gamma|`, K = 384), stan cząstki
= profil `(z - beta)/|gamma|`, siatka 24 x 24 = 576 cząstek na klatkę, 4800 klatek, baza PCA r = 32 dopasowana raz
na 512 klatkach rozłożonych po całym nagraniu (udział wariancji 66 do 69 procent). Reżim: inferencja, model
diagnostyczny, skala czasu nadkrok, wejście w rozkładzie uczącym. Zadania Eden 1809374 (zapis, 10 minut, 5 wersji
wag po około 700 MB) i 1809381 (analiza). Wynik: `REPORT_patch_particles.json`.

**Strażnicy.** Kierunek martwy: udział 7e-9 (T3c zdany; w teorii 9e-6 przy podłodze 4e-2). Promień profilu: 1,000
dla każdej cząstki, czyli człon `eps` normy jest tu pomijalny i stopień swobody „głośności" (T24c) numerycznie NIE
istnieje. **Determinizm (T35) NIE zdany**: 50 klatek zakodowanych drugi raz różni się o 0,11 do 0,22 w współrzędnych
zredukowanych, czyli 7 do 14 procent odchylenia współrzędnej. Przyczyna do ustalenia (autocast i jądra SDPA zależne od
składu partii w `predict`); każde porównanie klatka-klatka ma przez to podłogę szumu rzędu 0,1 odchylenia.

| Wagi | baseny: IoU z maską narzędzia (podłoga q95 0,414; klipów nad podłogą z 6) | lambda transl. (max / ósma; podłoga) | cos podprzestrzeni transl. vs czasowa | tożsamość narzędzia AUC: wolne 8 / szybkie 8 / PCA 8 / pełne 32 |
|---|---|---|---|---|
| COCO | 0,482 (3) | 1,01 / 0,27 (0,01) | 0,51 | 0,776 / 0,808 / 0,867 / 0,894 |
| 500 klatek | 0,522 (5) | 0,80 / 0,41 (0,04) | 0,55 | 0,848 / 0,667 / 0,889 / 0,911 |
| epoka 5 (b1500) | 0,563 (5) | 0,83 / 0,43 (0,04) | 0,55 | 0,851 / 0,806 / 0,905 / 0,921 |
| 2500 klatek | 0,537 (5) | 0,81 / 0,42 (0,04) | 0,56 | 0,899 / 0,789 / 0,913 / 0,924 |
| epoka 29 (b1500) | 0,555 (5) | 0,78 / 0,40 (0,04) | 0,48 | 0,869 / 0,767 / 0,902 / 0,918 |

Odczyt:

1. **Test basenów (03) zdany po uczeniu, niepewny przed.** Skupienia cząstek jednego klipu (k = 96, sam kierunek profilu)
   pokrywają maskę narzędzia z IoU 0,52 do 0,56 wobec podłogi gładkich podziałów losowych 0,41; przed uczeniem 0,48
   i tylko 3 klipy z 6 nad podłogą. Narzędzie jest cienkie (3,6 procent klatki), stąd IoU dużo niższe niż 0,81 dla
   niedźwiedzia z DAVIS w DiagnosticTool (U2). Geometria zanurzenia koduje narzędzie, a uczenie ją wyostrza.
2. **Operator translacyjny (D65) na tokenach narzędzia.** Pary = ten sam obszar narzędzia w dwóch miejscach kadru,
   wskazany WŁASNĄ ramką detektora (bez etykiet). Uwaga: dla COCO ramka pokrywa 35 procent klatki (detektor nie zna
   narzędzia), więc pary są parami przypadkowych tokenów i wiersz COCO nie jest porównywalny; po uczeniu ramka to 3,6
   procent klatki. Po uczeniu najwolniejszy mod ma lambda 0,8 przy podłodze z zerwanych par 0,04: niezmienniki
   przesunięcia istnieją i są odróżnialne od losu.
3. **Tożsamość narzędzia w niezmiennikach przesunięcia: pojawia się z uczeniem.** Osiem wolnych modów translacyjnych
   odróżnia patch narzędzia od tła z AUC 0,78 przed uczeniem i 0,90 po 2500 klatkach, gdy osiem modów szybkich spada
   z 0,81 do 0,79. To jest efekt D65 („wolne mody niosą tożsamość"), ale tu dopiero WYUCZONY, i słabszy niż osiem
   głównych kierunków PCA (0,913) oraz pełne 32 wymiary (0,924). Niezmienniki Koopmana są więc dobrym opisem tego,
   co sieć zrobiła z tożsamością narzędzia, a nie lepszym kodem niż wariancja.
4. Podprzestrzenie wolne operatora translacyjnego i czasowego mają średni cosinus około 0,5: „to samo w innym
   miejscu" i „to samo w następnej klatce" to różne kierunki latentu, zgodnie z opisem Eulera (T15a): w zegarze
   filmu przez cząstkę przepływa scena, nie narzędzie.
5. Rejestry (T6a): brak miejsc siatki z anomalnym promieniem (promień stały), więc test rejestrów wymaga normy
   pre-LN wprost, której ten rejestrator nie zapisuje; do zrobienia.

## 4j. Z latentu do klatek: dekodowanie niezmienników i wyszukiwanie „dobrze domkniętych" klatek (2026-09-22)

Pomysł autora: znaleźć klatki, które poprawnie zakodowały niezmienniki Koopmana, zdekodować z latentu cechy obrazu
potrzebne do domknięcia tożsamości narzędzia i szukać podobnych klatek w zbiorze uczącym. Kod:
`pipeline/koopman_langevin/invariant_decode.py`; wynik `REPORT_decode_b2500.json`, rysunki `dekodowanie_b2500_*.png`.

**Kod tożsamości i domknięcie.** Na cząstkach-patchach detektora (2500 klatek) osiem wolnych modów operatora
translacyjnego to kod tożsamości. Każda klatka dostaje średni kod swoich patchy narzędzia (z własnej ramki, bez etykiet),
prototyp = średni kod klatek pewnych z klipów treningowych, domknięcie = odległość kodu klatki od prototypu.
Zmierzone na walidacji (etykiety tylko do oceny): domknięcie wskazuje chybienia detektora z AUC 0,625, pewność
detektora 0,810; na małym detektorze (500 klatek) domknięcie wskazuje błąd USUWALNY z AUC 0,533 (los), pewność 0,899.
Osiem najlepiej domkniętych klatek walidacji ma IoU 0,87 do 0,93, osiem najgorzej 0,72 do 0,92: domknięcie porządkuje
klatki według KLIPU i sceny (ostre narzędzie na niebieskiej tęczówce kontra rozmyte przy brzegu), nie według tego,
czego sieci brakuje.

**Dekodowanie do pikseli.** Liniowy dekoder z pełnego profilu tokenu (384 liczby) do jego 16 x 16 x 3 pikseli, dopasowany
na klipach treningowych, oceniony na walidacyjnych: R2 0,54 na wszystkich patchach, ale UJEMNE na patchach narzędzia
(-0,69): liniowo odtwarza się tło (tęczówka, kolor), narzędzia nie. Ten sam dekoder na tokenie rzutowanym na osiem
wolnych modów daje R2 0,09, na osiem szybkich 0,09, na osiem głównych kierunków PCA 0,26. Rysunek
`dekodowanie_b2500_klatka.png` (oryginał, pełny latent, wolne, szybkie, PCA): z niezmienników nie da się liniowo
odczytać wyglądu, zostaje jednolita plama z ledwie widoczną tęczówką. Kod tożsamości nie jest więc cechą obrazu,
którą dałoby się „odszumić" i wyszukać po wyglądzie; to kod semantyczny, do którego z pikseli prowadzi tylko cała sieć.

Wniosek: „proces odwrotny do enkodera" w wersji liniowej nie działa dla narzędzia; wersja nieliniowa to osobny
trenowany dekoder (jak `DEKODER` w DiagnosticTool dla V-JEPA), który pokaże, co sieć wierzy, ale nie da kryterium
wyboru klatek innego niż podobieństwo w samym latencie. Żywotna postać pomysłu: wyszukiwać klatki podobne w KODZIE
TOŻSAMOŚCI (nie w pikselach) i pokrywać ten kod w puli, czyli ramię „pokrycie kodu tożsamości narzędzia" liczone
na tokenach narzędzia z aktualnego detektora, a nie na odcisku sceny jak K4. To wymaga tokenów detektora L0 na
puli 10 637 klatek (zadanie Eden) i ramienia treningowego etapu 2 (zatwierdzenie autora).

Ciąg dalszy (2026-09-22, dokument 39, sekcja 5f): kod tożsamości policzono dla całej puli 10 637 klatek pod detektorem L0 (zadania Eden 1809407 i 1809408) i zbudowano cztery ramiona doboru K5 (k-centrum w kodzie, plus klatki bez kodu, najbliżej i najdalej prototypu). Żadne ramię etapu 1 nie różni się od losu w tej przestrzeni; K5 zmniejsza największą lukę pokrycia o połowę. Trening tych ramion czeka na zatwierdzenie.

## 4k. Widmo operatora translacyjnego bez symetryzacji: część rzeczywista dodatnia, ujemna i urojona (2026-09-22)

Uwaga autora: kod tożsamości liczony był operatorem PO symetryzacji (`generalised_koopman`), więc jego wartości
własne są z definicji rzeczywiste i cała część obrotowa przepadała. `widmo_tozsamosci.py` liczy ten sam operator
(token narzędzia w jednym położeniu przechodzi w ten sam token po przesunięciu narzędzia) BEZ symetryzacji i dzieli
mody na trzy rodziny. Nagranie L0 na 16 gęstych klipach, 34 165 par, etykiety WYŁĄCZNIE do oceny, podział po filmach.

| Rodzina modów | Ile | Odróżnienie narzędzia od tła (AUC) | Przeżycie przesunięcia | Wydłużenie ramki | Wielkość ramki |
|---|---|---|---|---|---|
| trwałe: Re większe od zera, bez obrotu | 16 | 0,834 | 0,61 | 71,3 procent | -1,0 |
| przemienne: Re mniejsze od zera (wartość zmienia znak po przesunięciu) | 6 | 0,743 | -0,08 | -17,5 | -24,3 |
| obrotowe: Im różne od zera (para wartości obraca się o stały kąt) | 10 | 0,593 | 0,35 | -39,8 | -0,3 |
| trwałe + obrotowe | 12 | 0,853 | | 71,8 | 18,9 |
| trwałe + przemienne | 12 | 0,888 | | 79,0 | -12,8 |
| wszystkie trzy rodziny | 18 | **0,893** | | **79,6** | **30,1** |

Odczyt:

1. **Osobno liczą się tylko mody trwałe**, ale RAZEM pozostałe dwie rodziny dokładają wyraźnie: odróżnienie
   narzędzia od tła rośnie z 0,834 do 0,893, wydłużenie ramki z 71 do 80 procent, a wielkość ramki z zera
   do 30 procent. Symetryzacja kasowała więc informację, nie tylko szum.
2. **Mody przemienne (ujemna część rzeczywista) same nie przeżywają przesunięcia** (zgodność -0,08), a mimo to
   dokładają najwięcej do odróżnienia narzędzia. Zmiana znaku po przesunięciu to sygnał o strukturze narzędzia
   wzdłuż jego osi, a nie wielkość stała.
3. **Kąt obrotu modów zespolonych jest znikomy** (okres około 256 przesunięć), więc "obrót" jest tu prawie
   zdegenerowaną parą modów rzeczywistych; ich wartość leży w tym, że są parą, nie w samym obracaniu.
4. **Zwrotu narzędzia (gdzie jest końcówka) nie niesie żadna rodzina** (poniżej 2 procent wszędzie). Ograniczenie
   nie leży więc w symetryzacji, tylko w tym, że uśredniamy tokeny po całym wnętrzu ramki.

Konsekwencja dla ramienia K5: kod tożsamości należy przeliczyć na wszystkich trzech rodzinach (18 liczb zamiast 8)
przed treningiem etapu 2. Dowód: `evidence_koopman_langevin_2026-09-21/REPORT_widmo_tozsamosci.json`.

## 4l. Czy kwadrat siatki pamięta narzędzie, które przez niego przeszło (2026-09-22)

Pytanie autora, z eksperymentów z niedźwiedziem: cząstka w opisie Eulera stoi w miejscu, ale czy po przejściu
obiektu zostaje w niej ślad? `slad_po_przejsciu.py` mierzy to bez uczenia: podobieństwo tokenu do wzorca narzędzia
w kwadratach, gdzie narzędzia TERAZ nie ma, wobec kwadratów, których narzędzie nie dotknęło przez osiem pomiarów.
Kontrolą jest ten sam rachunek w przyszłość: kwadraty, w których narzędzie DOPIERO BĘDZIE. Zbiory rozłączne
(„było i nie wróci" wobec „będzie, a nie było"). Etykiety wyłącznie do oceny.

| Odstęp | Ślad po przejściu (AUC) | Zapowiedź przed przyjściem (AUC) |
|---|---|---|
| 3 klatki | 0,899 | 0,902 |
| 6 klatek | 0,885 | 0,893 |
| 9 klatek | 0,874 | 0,881 |
| 15 klatek | 0,858 | 0,868 |
| 24 klatki | 0,851 | 0,867 |

Odczyt: kwadrat, przez który narzędzie przeszło, wygląda potem „narzędziowo" bardzo wyraźnie (0,899 po trzech
klatkach), ale **dokładnie tyle samo wygląda tak, ZANIM narzędzie tam dotrze** (0,902). Efekt jest symetryczny
w czasie, więc nie jest pamięcią kwadratu, tylko sąsiedztwem toru narzędzia: pole widzenia tokenu sięga poza ramkę.
Zgadza się to z ustaleniem U14 projektu grantowego, gdzie jądro pamięci na cząstkach Eulera miało głębokość jednego
kroku i tłumaczyło 5 do 6 procent przyrostu. Dowód: `evidence_koopman_langevin_2026-09-21/REPORT_slad_po_przejsciu.json`.

## 4m. Niezmienniki wycinka narzędzia klatka po klatce (pomysł autora, 2026-09-22)

Autor: zamiast parować pojedyncze tokeny przy przesunięciu, opisać CAŁY wycinek w ramce narzędzia i pytać,
co w nim nie zmienia się z klatki na klatkę. `niezmiennik_czasowy.py` przepisuje tokeny leżące w prawdziwej
ramce na stałą siatkę 3 na 3 we współrzędnych względnych ramki (opis nie zależy więc od wielkości ani
położenia ramki), redukuje to do 48 wymiarów (90,3 procent wariancji) i dopasowuje operator
`x(t+1) = x(t) K` bez symetryzacji, osobno w każdym filmie.

Zestaw gęsty: 16 klipów, 4 698 klatek z zaznaczonym narzędziem, 4 570 par klatka do następnej (odstęp
do 4 klatek przy próbkowaniu co 3).

| Wielkość | Osiem najtrwalszych modów (moduł wartości własnej) |
|---|---|
| operator czasowy | 0,989 0,986 0,972 0,967 0,949 0,935 0,926 0,909 |
| podłoga: czas potasowany w obrębie filmu | 0,966 0,951 0,890 0,839 0,812 0,754 0,688 0,675 |
| podłoga: pary łączące różne filmy | 0,104 0,104 0,101 0,101 0,097 0,097 0,096 0,096 |

Co niosą te niezmienniki (etykiety wyłącznie do oceny, uczenie i ocena na różnych filmach): wydłużenie
prawdziwej ramki 88,1 procent, wielkość ramki 17,8, położenie narzędzia w kadrze minus 0,8. Udział
zmienności między filmami: 74 procent.

Kosinusy główne z modami operatora TRANSLACYJNEGO (na ile obie drogi znajdują te same kierunki latentu):
0,95, 0,90, 0,87, 0,83, potem 0,54, 0,46, 0,37, 0,10.

Odczyt:

1. **Obie drogi znajdują prawie te same cztery pierwsze kierunki** (kosinusy od 0,83 do 0,95), a rozchodzą
   się dopiero na dalszych. Niezmiennik tożsamości narzędzia nie jest więc artefaktem sposobu parowania.
2. **Podłoga z potasowanym czasem jest wysoka** (0,966 wobec 0,989): duża część trwałości bierze się z tego,
   że wycinek w ogóle wolno się zmienia, a nie z samej dynamiki. Nadwyżka jest realna, ale mała; to ta sama
   pułapka, którą projekt grantowy opisał jako bezwład reprezentacji.
3. **Droga czasowa czyta wydłużenie narzędzia lepiej** niż translacyjna (88 wobec 71 procent) i, tak jak ona,
   nie niesie położenia w kadrze.
4. **Trzy czwarte zmienności leży między filmami**, więc opis wycinka jest nadal mocno tożsamością operacji.

Ta sama analiza na CAŁEJ puli uczącej (10 637 klatek z zaznaczonym narzędziem, 118 filmów, para tylko gdy dwie
kolejne klatki dzieli najwyżej 6 klatek, stąd 1 809 par; pula jest próbkowana średnio co 20 klatek):

| Wielkość | Osiem najtrwalszych modów |
|---|---|
| operator czasowy | 0,964 0,964 0,937 0,923 0,911 0,894 0,863 0,857 |
| podłoga: czas potasowany | 0,841 0,792 0,779 0,750 0,675 0,675 0,629 0,583 |
| podłoga: pary między filmami | 0,069 0,069 0,069 0,069 0,066 0,066 0,065 0,065 |

Niosą: wydłużenie ramki 85,7 procent, wielkość 41,4, położenie w kadrze 44,0. Zmienność między filmami 64 procent.
Kosinusy główne z modami translacyjnymi: 0,92, 0,84, 0,67, 0,61, potem poniżej 0,55.

Na puli nadwyżka nad potasowanym czasem jest większa niż na zestawie gęstym (0,964 wobec 0,841), bo odstęp między
klatkami jest tu większy i sam bezwład nie wystarcza. Pierwsze dwa kierunki nadal pokrywają się z translacyjnymi
(0,92 i 0,84), dalsze już nie. Za to niezmienniki czasowe na puli niosą też położenie narzędzia w kadrze
(44 procent), czego wersja translacyjna z definicji nie robi. To znaczy, że opis wycinka klatka po klatce miesza
tożsamość narzędzia z jego ruchem po kadrze, a operator translacyjny je rozdziela.

Wybór siatki wycinka sprawdzony, nie przyjęty (pula, te same 1 809 par; pytanie autora, czemu 3 na 3, skoro
w ramce mieści się około 25 tokenów):

| Siatka | Najtrwalszy mod | Podłoga (czas potasowany) | Wydłużenie | Położenie w kadrze | Kosinusy z translacyjnymi (pierwsze dwa) |
|---|---|---|---|---|---|
| 2 na 2 | 0,961 | 0,817 | 85,9 procent | 37,3 | 0,90 i 0,87 |
| 3 na 3 | 0,964 | 0,841 | 85,7 | 44,0 | 0,92 i 0,84 |
| 4 na 4 | 0,965 | 0,845 | 86,1 | 43,0 | 0,91 i 0,84 |
| 5 na 5 | 0,965 | 0,840 | 85,6 | 29,4 | 0,89 i 0,77 |

Gęstsza siatka nic nie dokłada: od 2 na 2 do 5 na 5 najtrwalszy mod zmienia się o 0,004, a wydłużenie o pół punktu.
Powód jest prosty: ramka narzędzia mieści średnio około 25 tokenów, więc przy 5 na 5 na komórkę przypada jeden
token i opis zaczyna kopiować szum pojedynczego kwadratu (udział wariancji w 48 wymiarach spada z 91 do 75 procent,
a zgodność z modami translacyjnymi maleje). Przy 3 na 3 na komórkę przypada blisko trzy tokeny, co uśrednia szum,
nie gubiąc układu narzędzia wzdłuż ramki. Do etapu 2 zostaje 3 na 3, z 4 na 4 jako kontrolą.

Dowód: `evidence_koopman_langevin_2026-09-21/REPORT_niezmiennik_czasowy_gesty.json`,
`REPORT_niezmiennik_czasowy_pula.json` oraz warianty `_c2`, `_c4`, `_c5`.

## 4n. Deep Koopman na opisie ramki: nic nie dokłada, ale znajduje te same kierunki (2026-09-22)

`deep_koopman_ramki.py`: enkoder (sieć gęsta) podnosi opis wycinka do 8, 16 albo 32 liczb, w których krok klatki
ma być zwykłym mnożeniem przez jedną macierz; dekoder wraca do opisu, więc podniesienie nie może zgubić treści.
Podział po filmach, wczesne zatrzymanie na osobnych filmach, trzy ziarna. Zestaw gęsty, 4 570 par, 16 klipów.

| Model | Umiejętność nad „nic się nie zmienia" | Podłoga: uczenie na potasowanym czasie | Wydłużenie ramki | Kosinusy z modami liniowymi |
|---|---|---|---|---|
| liniowy operator (odniesienie) | -0,7 procent | | 88,1 procent | |
| deep, podniesienie 8 | -58,2 | -57,8 | -27,8 | 0,95 i 0,86 |
| deep, podniesienie 16 | -61,4 | -56,7 | 37,9 | 0,97 i 0,95 |
| deep, podniesienie 32 | -45,5 | -54,1 | 68,1 | 0,98 i 0,94 |

Odczyt:

1. **Żaden model nie bije przewidywania „nic się nie zmienia"** na nowych filmach. Liniowy wychodzi na zero
   (-0,7 procent), deep jest wyraźnie gorszy. Przy odstępie 3 klatek wycinek zmienia się tak mało, że kopiowanie
   poprzedniej klatki jest mocnym punktem odniesienia; to ta sama pułapka bezwładu, którą projekt grantowy opisał
   na latencie DINOv2 (przyrost nad trwałością równy zeru).
2. **Deep nie odróżnia się od własnej podłogi**: model uczony na parach BEZ kolejnosci czasu wypada tak samo
   (-54 do -58 procent). Nieliniowe podniesienie nie znajduje więc dynamiki, której liniowy operator by nie widział.
3. **Za to znajduje te same kierunki co operator liniowy** (kosinusy 0,98 i 0,94 przy podniesieniu 32). To odpowiada
   na pytanie autora: obie drogi trafiają w ten sam podprzestrzeń latentu, deep niczego nowego tam nie dokłada.
4. Odczyt wydłużenia narzędzia jest przy deep GORSZY (68 wobec 88 procent) i mocno zależy od rozmiaru podniesienia,
   co przy 16 klipach i trzech ziarnach wygląda na przeuczenie, nie na własność danych.

Wniosek: do etapu 2 zostaje operator liniowy (translacyjny plus czasowy). Deep Koopman wraca dopiero, gdy pary będą
liczone na dłuższych odcinkach albo z wymuszeniem (ruch kamery jako wejście), bo tam liniowy model faktycznie
ma czego nie umieć. Dowód: `evidence_koopman_langevin_2026-09-21/REPORT_deep_koopman_gesty.json`.

## 4m. Czy tożsamość siedzi w części rzeczywistej modów (hipoteza autora, sprawdzona 2026-09-22)

Pytanie autora: „tożsamość powinna być zakodowana w części rzeczywistej modów Koopmana". Sprawdzone wprost
(`czesc_rzeczywista_a_tozsamosc.py`): dla każdej rodziny modów operatora translacyjnego liczone są osobno część
rzeczywista, część urojona, moduł pary i kwadrat. Miary: odróżnienie tokenu narzędzia od tokenu tła (uczenie i ocena
na różnych filmach, prawdziwe ramki tylko do oceny), przeżycie przesunięcia i odczyt wydłużenia ramki.
Po pięć modów z rodziny, 34 165 par, nagranie L0 na 16 gęstych klipach.

| Rodzina | Postać | Odróżnienie (AUC) | Przeżycie | Wydłużenie |
|---|---|---|---|---|
| trwałe (Re większe od zera) | Re | **0,825** | 0,65 | 69,1 procent |
| trwałe | Re do kwadratu | 0,502 | 0,47 | 44,9 |
| przemienne (Re mniejsze od zera) | Re | 0,686 | -0,09 | -16,3 |
| przemienne | Re do kwadratu | **0,723** | 0,03 | -38,2 |
| obrotowe (Im różne od zera) | Re | 0,809 | 0,20 | -15,7 |
| obrotowe | Im | **0,650** | 0,20 | -17,2 |
| obrotowe | moduł pary | **0,844** | 0,14 | -24,0 |
| obrotowe | Re i Im razem | 0,794 | 0,20 | -14,0 |

Zestawy przy RÓWNEJ liczbie kolumn (żeby przewaga nie brała się z liczby cech):

| Zestaw | Kolumn | Odróżnienie | Wydłużenie |
|---|---|---|---|
| same trwałe Re | 5 | 0,825 | 69,1 procent |
| trwałe Re + obrotowe moduł | 10 | 0,879 | 68,3 |
| **same części rzeczywiste, trzy rodziny** | 15 | **0,887** | **78,3** |
| same części urojone (tylko obrotowe) | 5 | 0,650 | -17,2 |
| trwałe Re + obrotowe Re i Im | 15 | 0,886 | 73,7 |
| trwałe Re + obrotowe moduł + przemienne kwadrat | 15 | 0,882 | 66,7 |
| trwałe Re + obrotowe Re i Im + przemienne kwadrat | 20 | 0,893 | 73,1 |
| wszystko liniowo, jak dotąd | 20 | **0,902** | **79,3** |

Odczyt:

1. **Hipoteza autora potwierdzona w kierunku, ale nie jako reguła wyłączna.** W rodzinie obrotowej część rzeczywista
   niesie wyraźnie więcej tożsamości niż urojona (0,809 wobec 0,650). Sama część urojona nie jest jednak szumem:
   dołożona do części rzeczywistej podnosi odczyt wydłużenia z 68,3 na 73,7 procent.
2. **Najwięcej tożsamości w tej rodzinie niesie moduł pary** (0,844), czyli długość wektora złożonego z obu części.
   To zgadza się z rachunkiem: dla pary obrotowej niezmiennikiem przesunięcia jest dopiero długość, nie współrzędna.
3. **Ale przewidywanie z dokumentu 42, że postaci kwadratowe pobiją liniowe, jest OBALONE przy równej liczbie kolumn**
   (0,893 wobec 0,902; wydłużenie 73,1 wobec 79,3). Powód: ścisła niezmienniczość nie jest tym, czego potrzebuje
   odczyt. Faza, czyli część urojona, niesie informację związaną z pozą, a moduł ją wyrzuca. Dla doboru klatek liczy
   się użyteczność cechy, nie jej formalna niezmienniczość.
4. **Wyjątkiem jest rodzina przemienna**, gdzie kwadrat faktycznie bije wartość surową (0,723 wobec 0,686), zgodnie
   z rachunkiem: wartość zmienia znak po kroku, więc niezmiennikiem jest kwadrat.
5. **Moduł pary obrotowej przeżywa przesunięcie GORZEJ niż jej część rzeczywista** (0,14 wobec 0,20), co jest odwrotne
   do oczekiwania. To kolejny argument, że te pary nie są prawdziwym obrotem: kąt jest znikomy (okres około 256
   przesunięć, sekcja 4k), więc parowanie jest raczej numeryczne niż fizyczne.

6. **Same części rzeczywiste trzech rodzin (15 liczb) prawie dorównują pełnemu zestawowi (20 liczb):** 0,887 wobec
   0,902 przy odróżnieniu i 78,3 wobec 79,3 procent przy wydłużeniu. Część urojona dokłada więc niewiele i tylko
   jako dodatek; sama w sobie jest słaba (0,650 i odczyt wydłużenia poniżej zera). Na jedną liczbę kodu części
   rzeczywiste są wyraźnie wydajniejsze, więc hipoteza autora broni się w wersji praktycznej.

Uwaga o słowie „tożsamość" w tej sekcji: mierzone jest odróżnienie tokenu leżącego na narzędziu od tokenu tła oraz
odczyt kształtu ramki. W tym podzbiorze jest jedna klasa narzędzia, więc tożsamości w sensie „które to narzędzie"
nie da się tu zmierzyć wcale.

Wniosek praktyczny dla kodu tożsamości: zostawić postać liniową ze wszystkimi trzema rodzinami (20 liczb), a jeśli
zależy na krótkim kodzie, wziąć same części rzeczywiste (15 liczb) kosztem 0,015 na odróżnieniu; nie przechodzić
na moduły. Dowód: `evidence_koopman_langevin_2026-09-21/REPORT_czesc_rzeczywista.json`.

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
