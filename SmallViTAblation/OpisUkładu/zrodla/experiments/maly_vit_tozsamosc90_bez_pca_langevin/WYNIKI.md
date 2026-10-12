# Tożsamość ponad 90 procent bez PCA i Langevin natywnych kanałów

Zakończono 2026-10-02T02:24:54.349730+02:00. Wybrane **128 oryginalnych kanałów** razem rozpoznają oryginał versus usunięcie narzędzia w znanym ROI z testową film-weighted accuracy/BA **92.83%**. Cel ponad 90 procent osiągnięto punktowo. CI95 po całych filmach [88.17, 96.99]% nie ma dolnej granicy ponad 90 procent. To nie jest trafność prognozy ruchu.

Dynamika tych samych 128 współrzędnych: lokalny GLE L2/ridge1 wybrany na walidacji daje na H1 skill **47.75%**, znak wszystkich składowych **71.68%**. Stała prędkość jest lepsza na H1. Na H10/H50 wybrany model przegrywa z persystencją. Transfer na drugą sondę jest słaby. Nie wykazano jednego dobrego równania dla tych horyzontów.

## Rozpoznanie obecności: co oznacza 90 procent

Zadanie: 400 par prawdziwy obraz train / syntetyczny LaMa po usunięciu narzędzia. Stan przy checkpointcie 500, pełny raw latent 768 po ostatnim GELU. Każdy obraz reprezentuje coverage-ważona średnia tokenów ORYGINALNEJ maski narzędzia; identyczne indeksy w obu członach pary. Etykieta 1=oryginał,0=usunięcie. To rozpoznanie na poziomie obrazu/ROI, nie każdego tokenu, detekcja nieznanego ROI, klasyfikacja innych narzędzi ani trafność na każdym późniejszym checkpointcie.

Po filmach train/validation/test:69/23/24 filmów oraz 240/75/85 par. Każdy film ma tę samą masę, obrazy równą masę w filmie, klasy0/1 równe. Primary BA przy progu 0.5 jest równoważna weighted accuracy. Zbiór był oglądany w wcześniejszych eksperymentach, więc ten zamrożony algorytm ma oddzielny test, ale nie jest nietkniętym niezależnym potwierdzeniem.

Źródło pooled400 zostało zapisane PRZED projekcją U w wcześniejszym pomiarze; odczytano wyłącznie tę surową tablicę 400×2×768. Ignorowano U500, eigenvalues i wszystkie 8D probe/scaler z poprzednika. Nie zastosowano PCA/POD/SVD, rotacji bazy ani autoenkodera do selekcji lub stanu. Nowe standaryzowanie jest diagonalne per natywny kanał, wyłącznie train i film-weighted.

Ranking train-only: abs(mean pairedΔ)/SD pairedΔ, film→obraz; remisy według numeru kanału. K z listy 1,2,4,8,16,32,64,128,256,384,512,768; L2 logistyczna C.01/.1/1/10. Walidacja wybiera najmniejsze K z listy z BA>0.90, potem największe BA, AUROC, najmniejsze C. Jest to minimum w przeszukanej liście i rankingowym prefiksie, nie dowód globalnie minimalnego zbioru kanałów.

Wybrano K 128, C 0.01, val BA =90.67%; najlepsze K 64 ma 89.31 procent. Wszystkie 48 fitów zbieżne. C/zbiór/scaler zamrożono 2026-10-02T02:04:24.336653+02:00, jednorazowy test zakończono 2026-10-02T02:08:05.348849+02:00. Po teście nie zmieniono żadnych kanałów lub hiperparametrów.

| Test obecności na k500 | Wynik |
|---|---:|
|Film-weighted accuracy = BA|92.83%|
|Sensitivity|91.97%|
|Specificity|93.68%|
|Film-weighted AUROC|97.41%|
|Nieważona accuracy|91.76%|
|95%CI BA,2000 whole-film bootstrap|[88.17, 96.99]%|

Nie twierdzimy, że wszystkie128 osi są indywidualnie konieczne/odpowiedzialne semantycznie. Wspólnie są predykcyjne w tym zadaniu. Wersja LaMa ma artefakty i nie odzyskuje prawdziwego zakrytego tła; inne narzędzia mogą pozostać. Model ViT był uczony na danych train, więc ten test nie jest niezależnym heldout detektora.

## Dokładny zestaw natywnych makrowektorów

Makrowektor to wspólny wektor bazowy e_j, cząsteczka to z_i∈R^768. W nowym stanie przechowujemy właśnie składowe z_i,J_j, bez rotacji; wspólny klasyfikator wykorzystuje je razem. Pełne współczynniki zapisano w KANALY_MODEL.npz. Numery 1-based:

```text
359, 594, 472, 512, 138, 500, 3, 14, 241, 461, 621, 369, 617, 285, 98, 738, 293, 220, 190, 315, 474, 597, 705, 735, 116, 226, 686, 189, 398, 613, 556, 676, 445, 441, 257, 244, 640, 592, 411, 166, 208, 706, 501, 70, 442, 591, 240, 286, 448, 493, 54, 2, 289, 733, 221, 737, 382, 543, 659, 433, 336, 259, 624, 570, 511, 727, 345, 630, 67, 92, 490, 326, 225, 726, 160, 672, 400, 638, 184, 465, 363, 13, 720, 103, 577, 248, 655, 524, 145, 229, 111, 515, 587, 378, 36, 419, 228, 108, 509, 144, 137, 10, 576, 566, 525, 316, 232, 365, 124, 254, 521, 568, 387, 307, 645, 647, 25, 502, 476, 179, 271, 30, 183, 523, 529, 460, 80, 602
```

Numery 0-based:
```text
358, 593, 471, 511, 137, 499, 2, 13, 240, 460, 620, 368, 616, 284, 97, 737, 292, 219, 189, 314, 473, 596, 704, 734, 115, 225, 685, 188, 397, 612, 555, 675, 444, 440, 256, 243, 639, 591, 410, 165, 207, 705, 500, 69, 441, 590, 239, 285, 447, 492, 53, 1, 288, 732, 220, 736, 381, 542, 658, 432, 335, 258, 623, 569, 510, 726, 344, 629, 66, 91, 489, 325, 224, 725, 159, 671, 399, 637, 183, 464, 362, 12, 719, 102, 576, 247, 654, 523, 144, 228, 110, 514, 586, 377, 35, 418, 227, 107, 508, 143, 136, 9, 575, 565, 524, 315, 231, 364, 123, 253, 520, 567, 386, 306, 644, 646, 24, 501, 475, 178, 270, 29, 182, 522, 528, 459, 79, 601
```

Wzór logistycznego rozpoznania: p=σ(intercept + Σ_j beta_j (pooled_z[J_j]−mu_train[J_j])/sd_train[J_j]). To granica decyzyjna na natywnych kanałach, nie nowe obrócone współrzędne stanu. coef_physical/intercept_physical są równoważnym zapisem w surowych jednostkach; test zgodności numerycznej przeszedł.

## Langevin/GLE bez PCA: całe 128 współrzędnych

Obraz główny 2: `cataract_lmm/tracking/TR_0057_S1_P03/00001431`, 31 cząsteczek. Transfer 8: `cataract_lmm/tracking/TR_0134_S1_P03/00000341`, 7 cząsteczek; źródłowa rola holdout, split train. Są to dwa stałe obrazy w tym samym biegu. Każdy ma własne anchory 500, coverage i średnią pełnych tokenów źródłowej maski background; operatory transferu nie są refitowane.

```text
s_i(t)=z_i,J(t)−z_i,J(500),  J=zamrożone128 natywnych kanałów
g_fg(t)=sum_i w_i s_i(t)
g_bg(t)=mean_background z_J(t)−mean_background z_J(500)
Delta s_i(t+dt)=b+A s_i(t)+C_fg g_fg(t)+C_bg g_bg(t)
                +sum_l [K_l Delta s_i(t−l dt)+F_l Delta g_fg(t−l dt)
                         +G_l Delta g_bg(t−l dt)]+epsilon_i(t+dt)
Delta g_bg(t+dt)=d+E_fg g_fg(t)+E_bg g_bg(t)
                +sum_l [N_l Delta g_fg(t−l dt)+M_l Delta g_bg(t−l dt)]+eta
```

Lokalny model usuwa sprzężenie/operator tła. g_fg zawsze wynika z prognoz cząsteczek. Oba przyrosty foreground/bg powstają z jednego snapshotu i aktualizowane są synchronicznie. Wszystkie future kanały i globalne średnie są autonomiczne; żadnych przyszłych obrazów, wag, paczek, stanów, gradientów lub reszt. Macierze są wspólne dla tokenów, pełne 128D. To niskorzędowe addytywne sprzężenie ze wspólnym ruchem, nie identyfikacja sił między każdą parą tokenów.

Ridge estymuje wszystkie bloki jednocześnie, coverage×równe wagi czasu foreground, równe wagi czasu dla średniej bg. Każda cecha i przyrost skalowane SD z fit; przyrosty i anchory po konwersji raw float32 do float64. Po odwróceniu skal B_raw=diag(1/s_x) B_std diag(s_y), b_raw=mean_y−mean_x B_raw. Bloki transponowane według layout_fg/layout_bg są kolumnowymi A,C,K. SD i intercept pochodzą tylko z zakończonych fitów. Epsilon_q jest progiem odsiewu znaku, nie realizacją szumu równania.

Nie ma PCA w definicji celu, pomocniczejV ani obcięcia do 8 wymiarów. Rozkład Gram użyty do rozwiązania/diagnostyki ridge nie zmienia bazy stanu. MSE jest fizycznym błędem wszystkich 128 kanałów, średnim po współrzędnych i coverage-weighted po tokenach. Nie zestawiamy tej wartości z wcześniejszym 8D celem jako tej samej miary.

Brak kompletnej faktycznej minibatch/augmentacja/gradient telemetrii, więc bez jawnego historycznego sterowania. Zmieniały się również biegnące statystyki BatchNorm. Empiryczny GLE/AR z pamięcią jest aproksymacją dryfu w obserwowanych współrzędnych; nie dowodzi dokładnego Mori–Zwanziga, globalnej linearyzacji, semantycznych sił ani FDT dla AdamW.

## Przedtestowy wybór dynamiki

Local/coupled × L0/1/2/4 × ridge .01/.1/1/10,32 konfiguracje na dt. K 128/J są już zamrożone przez rozpoznanie; prognoza nie wybiera innych osi. dt1 fit 501..700, validation origins 700..749,H1/10/50, refit <=799; test 800..950,H1/2/5/10/20/50. dt10 fit 510..700, val 700..740 co 10,H10/50, refit <=790; test 800..950 co 10,H10/20/50. Te same originy i cele między metodami. Mean skill równych horyzontów jest głównym kryterium; osobny best_short wybiera H1/H10. Test został wykonany raz po audycie Astry i zamrożeniu SHA.

Wybrane dt1 local L2/ridge1, dt10 local L2/ridge10. Best_short pokrywa się z głównym wyborem. Rodzinowe local/coupled i osobne L0 również zamrożone na walidacji; zachowano wszystkie negatywne wyniki siatki. L0 oznacza brak jawnych lagów przyrostów w rozszerzonym stanie, nie brak możliwej pamięci w samych obserwowanych kanałach po eliminacji kontekstu.

## Test 128D: błąd, znak, kosinus

| H | MSE selected | MSE persistence | Skill | Znak wszystkich 128 | Znak po odsiewie celu | Cosine zmiany |
|---:|---:|---:|---:|---:|---:|---:|
|1|0.0001151|0.0002203|47.75%|71.68%|76.35%|0.6648|
|2|0.0005212|0.0007832|33.45%|67.38%|69.65%|0.5455|
|5|0.0029628|0.0032042|7.53%|60.60%|61.35%|0.3327|
|10|0.0067234|0.0064553|-4.15%|58.73%|59.22%|0.2792|
|20|0.0110114|0.0102768|-7.15%|59.70%|60.14%|0.3084|
|50|0.0173233|0.0119224|-45.30%|57.18%|57.47%|0.2423|

Znak oznacza poprawność składowych, nie całych wektorów. Cosine dotyczy przewidzianej i prawdziwej zmiany, nie poziomu; persistence ma cosine nieokreślony. Odsiew abs(rzeczywista przyszła zmiana)>epsilon odbywa się podczas oceny po poznaniu celu; próg z fit, ale nie jest online bramką pewności. Wszystkie modele i obie sondy mają te same progi local finalfit głównej sondy.

| Model na tym samym J | H1 skill | H10 skill | H50 skill | H1 znak |
|---|---:|---:|---:|---:|
|local|47.75%|-4.15%|-45.30%|71.68%|
|coupled|34.85%|1.63%|-21.46%|68.29%|
|local_L0|-0.43%|1.28%|-14.34%|50.65%|
|coupled_L0|-3.27%|-3.33%|-31.47%|51.16%|
|velocity|55.92%|-289.63%|-4677.47%|78.67%|
|persistence|0.00%|0.00%|0.00%|0.00%|

Stała prędkość na H1 osiąga 55.92 procent skilla i 78.67 procent znaku, więcej niż wybrany GLE; później silnie ekstrapoluje błędne przyrosty. Sprzężony GLE jest mniej zły na H50, lecz nadal gorszy od braku zmiany. Nie zmieniamy modelu po wynikach testu.

## Transfer 8 i checkpoint co 10

| dt / sonda / model | H1 skill | H10 skill | H50 skill |
|---|---:|---:|---:|
|1 / 2 / selected|47.75%|-4.15%|-45.30%|
|1 / 2 / coupled|34.85%|1.63%|-21.46%|
|1 / 2 / local_L0|-0.43%|1.28%|-14.34%|
|1 / 8 / selected|-1.41%|-262.36%|-265.27%|
|1 / 8 / coupled|-17.21%|-223.23%|-308.34%|
|1 / 8 / local_L0|-8.50%|-18.19%|-162.50%|
|10 / 2 / selected|niedostępne|4.30%|4.15%|
|10 / 2 / coupled|niedostępne|-4.72%|-11.42%|
|10 / 2 / local_L0|niedostępne|7.38%|-0.61%|
|10 / 8 / selected|niedostępne|-23.01%|-70.63%|
|10 / 8 / coupled|niedostępne|-139.63%|-230.95%|
|10 / 8 / local_L0|niedostępne|-33.04%|-123.96%|

Długi transfer na sondę 8 jest gorszy od persystencji. H50 dt1 na sondzie8 ma skill −265.27%, czyli MSE 3.65 razy persystencja; to nie procent błędnych prognoz. Na H1 tej sondy velocity ma 62.53% skilla wobec −1.41% GLE. Przy dt1 nawet H1 wybranego GLE nie poprawia błędu. Przy dt10/sondzie 2 punktowy H50 skill 4.15% ma opisowy CI [−0.91,+7.89]%, więc przewaga nie jest rozstrzygnięta. Źródłowa sonda 8 nie służyła do żadnego fitu nowej dynamiki; nie jest niezależnym nowym ziarnem. Wynik rozpoznania na k500 nie dowodzi utrzymania tej trafności podczas całej późniejszej historii.

| H | dt1 selected na co 10originach | dt10 selected na tych samych originach |
|---:|---:|---:|
|10|-1.02%|4.30%|
|20|-8.62%|9.40%|
|50|-45.38%|4.15%|

## Niepewność, ranga i stabilność

CI dynamiki:300 sparowanych circular moving block bootstrap, bloki>=50 aktualizacji, wspólne originy/tokeny/model. Tylko około 3 efektywne bloki, jeden bieg: opisowa niepewność, tokeny nie są niezależnymi replikacjami. Porównania coupled versus local dotyczą NOWEGO 128D celu na tym samym J, nie dawnego 8D.

| H | CI95 selected skill | CI95 selected znak | Δskill coupled−local [pp], CI |
|---:|---|---|---|
|1|[46.47, 48.79]%|[71.03, 72.31]%|-12.90pp; [-13.69, -12.16]pp|
|10|[-17.04, 5.82]%|[57.34, 60.01]%|5.78pp; [0.20, 11.93]pp|
|50|[-59.10, -34.58]%|[55.29, 60.03]%|23.84pp; [14.44, 32.98]pp|

| dt / model / operator | Targets refit | Feature count | Gram rank | Condition ridge | Raw-scaled błąd |
|---|---:|---:|---:|---:|---:|
|1/selected/fg|299|384|384|41.3|2.85e-16|
|1/coupled/fg|299|1152|682|13.2|1.68e-16|
|1/coupled/bg|299|768|298|9.69|7.34e-17|
|10/selected/fg|29|384|384|4.31|6.45e-16|
|10/coupled/fg|29|1152|412|118|7.27e-16|
|10/coupled/bg|29|768|28|87.3|4.54e-16|

selected, dt1: radius odchyleń=1.001476, meanfg/bg=None. coupled, dt1: radius odchyleń=1.000520, meanfg/bg=1.0004459428714676. selected, dt10: radius odchyleń=1.005697, meanfg/bg=None. coupled, dt10: radius odchyleń=1.016194, meanfg/bg=1.0156697820246174.

Wybrane promienie przekraczają 1, więc nie dowiedziono asymptotycznej stabilności. Skończone H50 są policzone bez usuwania niekorzystnych rolloutów; nie stosowano projekcji stabilizującej po teście. Operator tła ma pierwotnie 200/20 próbek czasu, 299/29 po reficie i wiele cech, więc ridge nie daje jednoznacznej fizycznej identyfikacji macierzy. Powtarzane globalne cechy nie tworzą nowych obserwacji. Obserwowalności/sterowalności fizycznego procesu nie wykazano.

## Reszty: normalność oddzielnie od białości

| dt / sonda | Szeregi | LB raw p<.05 | JB raw p<.05 | Degenerate | Wspólna energia reszty |
|---|---:|---:|---:|---:|---:|
|1/2|4096|4084|1949|0|65.63%|
|1/8|1024|1024|370|0|68.49%|
|10/2|4096|448|261|0|64.89%|
|10/8|1024|185|43|0|66.02%|

Nowe reszty wybranych modeli z ciągłego one-step target 801..1000 (200 próbek) albo 810..1000 (20). Diagnoza niedomknięcia po fit, bez wpływu na wybór. ACF/Ljung–Box per token i natywny kanał w czasie, nie spłaszczone tokeny; Jarque–Bera/skew/kurtosis odrębnie. Rawp bez korekty wielokrotnych testów/fitDOF; mała próba dt10. Nie wykazano białego szumu. Brak odrzucenia normalności nie dowodzi gaussowskości, a gaussowskość nie oznacza białości. Nie narzucamy FDT.

![Rozpoznanie obecności](ROZPOZNANIE_VS_K.png)

![Znaki zmian](ZNAK_PROGNOZY_VS_H.png)

![Błąd prognozy](SKILL_PROGNOZY_VS_H.png)

![Natywne trajektorie](TRAJEKTORIE_NATYWNE.png)

## Odtwarzanie i artefakty

F:/Python/python.exe: wybor_kanalow.py --stage validation; audyt; wybor_kanalow.py --stage test (guard blokuje powtórkę); przygotuj_dynamike.py; dynamika.py --stage validation; audyt; dynamika.py --stage test (guard blokuje powtórkę); reszty.py; raport.py. Wszystko w osobnym katalogu,4 wątki BLAS, bez forward/treningu ViT lub pobierania. Wcześniejszych plików nie zmieniano. Bez commitów.

KANALY_MODEL.npz/CSV: J0/J1, ranking, wszystkie train-centry/scales/coef. WYBOR_WALIDACJA.json:48 konfiguracji obecności i freezehash; OCENA_OBECNOSCI_TEST.*: jednorazowy wynik/probabilities/filmBA; CZASY_ROZPOZNANIA.json. DYNAMIKA_NATYWNE_J.npz:31/7 pełnych trajektorii128D oraz średnie tła; MACIERZE_DT*.npz i MODELE_DT*.json: fizyczne operatory oraz layout; WYBOR_DYNAMIKA_DT*.json: pełna siatka64; METRYKI_DT*: wszystkie modele/proby/per-token/per-coordinate/CI/commonoriginy; PROGNOZY_DT*: wybrany pełny128D forecast; RESZTY/SZUM_DT*: nowe diagnostyki. AUDYT_PREFIX oraz AUDYT_ASTRA.md. PLAN/PODSUMOWANIE należą do roota.
