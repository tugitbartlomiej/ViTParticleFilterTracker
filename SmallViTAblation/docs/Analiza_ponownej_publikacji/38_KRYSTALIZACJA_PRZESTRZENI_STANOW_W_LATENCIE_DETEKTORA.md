# 38. Krystalizacja przestrzeni stanów narzędzia w latencie detektora: Koopman i Langevin na dwóch zegarach

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
