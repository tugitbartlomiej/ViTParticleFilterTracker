# PyKoopman NNDMD: pełne768 i końcowa tożsamośćJ128

Wykonanie zakończone 2026-10-02T13:18:22.481213+02:00. Rzeczywisty PyKoopman1.2.1 NNDMD.fit(listtrajectory), pełny encoder768→768→768, uczona macierzA768×768, decoder pełne768. J128 odczytano DOPIERO z decoded pełnych768.

Główny wynik przyH500: znakallJ128=54.498procent, RMSE_J128=0.448238, RMSE_brak_zmiany=0.233894, ratioRMSE=1.916417, skillMSE=-2.672655. Cel85procent: ciągły horyzont odH10 wynosi0aktualizacji; lista spełniających horyzontów: brak.

Dla pełnych768 przyH500 RMSE=0.440500, RMSE_brak_zmiany=0.250804, skillMSE=-2.084771. Rekonstrukcjaorigin500: RMSEfull768=0.077419, RMSEJ128=0.058715. To osobna miara od błędu przyszłej prognozy.

## Uczony operator i strata

Alearned działa w uczonej przestrzeni768 encoderu. Ta sama liczba wymiarów nie oznacza tożsamości osi z oryginalnymi kanałami ViT. Zapisujemy uczony generatorG i A=matrix_exp(G), gdzie jeden krok odpowiada10aktualizacjom źródłowego ViT. Encoder i decoder wracają do pełnych oryginalnych768; nie ma PCA ani redukcji rzędu.

Wymiar768 nie gwarantuje injektywności encoderu ani dokładnej inwersji przez decoder. Zmierzony błąd rekonstrukcji pozostaje częścią rzeczywistej prognozy ruchu. G jest parametryzacją operatora pomiędzy checkpointami co10aktualizacji; nie jest zweryfikowanym generatorem pojedynczych aktualizacji źródłowego modelu. Historyczny tytuł PLAN_NNDMD zawiera128, ale jego późniejsze jawne korekty i faktyczny kod/trening definiują768×768.

Strata oryginalnego pakietu to wielokrokowy maskedMSE prognozy plus rekonstrukcjaorigin i prawdziwych przyszłych stanów. Jawny adapter rozszerza ją o beta razy identyczne termy decodedJ128: rnn_full+rec_full+beta*(rnn_J128+rec_J128). Rzeczywiste package sieci, generator, Trainer i Adam1e-3 zostały użyte; nie przypisujemy dodatkowej wagi identity niezmienionemu API. Wszystkie cztery składowe zapisano w CSV/logach. Łącznyloss i składowe w logach są wartościami ostatniego minibatch, a walidacyjne RMSE jest dokładną agregacją coverage i równych wag obrazów.

Do próbkowania okien użyto WeightedRandomSampler: prawdopodobieństwo coverage/sumcoverage obrazu, jednakowa masa każdego obrazu i każdego dostępnego origin. Stochastyczna realizacja wag jest poprawna w oczekiwaniu, a nie równa na każdym minibatch. Normalizacja używa jednego wspólnego skalara wyliczonego wyłącznie z fitprefixu; średnianormalizacji0. RMSE raportujemy w surowych kanałach, jako sqrt(MSE PO agregacji), nie średnia RMSE obrazów.

Dla kosztu768 liczymy matrix_exp raz na minibatch i używamy go w całym rollout. MaskęMSE wektoryzujemy, zachowując torch.where, sumę i dzielnik. Testbeta0 potwierdził zgodność loss i gradientów z oryginalnym package training_step; testbeta1 potwierdził pełnyloss plusidentity. Wolny pilot15epok zachowano osobno; główna siatka od tej samej inicjalizacji ma20epok każdego kandydata.

## Wybór bez czterech końcowych obrazów

Pierwsze24fit trenowano wyłącznie0..300, sześćfit28..33 oceniono open-loop od300 do500. Kryterium: średnia pięciu rawRMSE decodedJ128 przyH10,H20,H50,H100,H200. Najlepszą epokę spośród checkpointów co5 i kandydatów wybrano przed refit/testem. Full768RMSE raportowano dodatkowo.

| look_forward | betaidentity | Najlepsza epoka | Val mean RMSEJ128 | Val mean RMSEfull768 |
|---:|---:|---:|---:|---:|
| 5 | 1 | 15 | 0.262227 | 0.240935 |
| 10 | 1 | 5 | 0.237974 | 0.224098 |
| 20 | 1 | 5 | 0.269110 | 0.245120 |
| 10 | 0 | 10 | 0.241094 | 0.223255 |

Na tej samej walidacji J128 i pięciu horyzontach średni RMSE braku zmiany wynosi 0.211199, a wybranego statycznego EDMD 0.333621. Wybrany NN ma 0.237974. Średnia RMSE po horyzontach jest kryterium wyboru, podczas gdy każdy RMSE horyzontu powstaje po agregacji MSE po obrazach i cząsteczkach. Nie utożsamiamy jej ze sqrt(MSE uśrednionego po horyzontach).

Wybrano look_forward10, beta1, epokę5. Refit30fit na0..500 od nowej inicjalizacji z tym samymseed20261002, przez stałą liczbę wybranychepok. Zamrożono 2026-10-02T13:17:21.215663+02:00. Dopiero potem nowytest czterech starychholdoutów8..11. Forecast kodujeorigin500 RAZ; dalej tylko phi_next=A*phi i decoder, bez przyszłych obserwacji, aktualizacji modelu lub re-encodingu przewidywanych stanów.

## Horyzonty głównej tożsamościJ128

| Haktualizacji | ZnakallJ128 | RMSE NN | RMSE staticEDMD | RMSE brakzmiany | RMSE reversion | SkillNN | CosineNN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 56.300% | 0.138555 | 0.123189 | 0.120948 | 0.246954 | -0.312336 | 0.275333 |
| 20 | 57.744% | 0.162711 | 0.163649 | 0.139533 | 0.259038 | -0.359814 | 0.303406 |
| 50 | 57.139% | 0.195706 | 0.218590 | 0.163220 | 0.273537 | -0.437683 | 0.235592 |
| 100 | 53.451% | 0.212525 | 0.254171 | 0.164972 | 0.286640 | -0.659596 | 0.195076 |
| 200 | 57.225% | 0.234563 | 0.248120 | 0.192491 | 0.257241 | -0.484903 | 0.330820 |
| 300 | 57.919% | 0.272977 | 0.244568 | 0.194706 | 0.245753 | -0.965601 | 0.352374 |
| 400 | 55.562% | 0.356250 | 0.275406 | 0.204109 | 0.262405 | -2.046368 | 0.220094 |
| 500 | 54.498% | 0.448238 | 0.287987 | 0.233894 | 0.280835 | -2.672655 | 0.189188 |

PrzyH500 błądnormyruchuNN=0.860815. Miary znaku obejmują wszystkie128współrzędnych ze sign(0)=0. Dodatkowo eps1e-8: znak=54.498procent, udział=1.000000; eps1e-3: znak=54.538procent, udział=0.987092. Cel85 dotyczy allJ128, nie progowanych miar i nie RMSE.

StaticEDMD porównano tylko wJ128, ponieważ jego stan miał oryginalne128kanałów. Nie tworzymy sztucznego pełnego768wyniku tego starego modelu. Persistence istnieje dlafull768/J128. Reversion przewiduje stały globalny ważony średni stan tokenów tool30fit na500 i nigdy nie pobiera przyszłych obserwacji.

## Diagnostyka zamrożonego operatora

Promień spektralny A=1.033198, norma operatorowa 2 A=1.193692, norma A^50=10.677115. Są to miary w uczonych współrzędnych. Promień spektralny sam nie wyklucza przejściowego wzrostu ani błędu decoded prognozy.

Liniowy decoder C ma numeryczny rząd 767 przy tolerancji float32=0.000373; jego kondycja2=168250. Przy odwracalnym C decoded rollout można zapisać (C A C^-1)^h R(x0), gdzie R=s C E(x0/s). Sam nonlinearencoder i kwadratowy lineardecoder nie dowodzą nieliniowej sprzężoności fizycznej dynamiki.

Reszta jednokrokowej domkniętości encoderu, RMSE względem braku zmiany w learned state: fitprefix=0.836217, końcowy holdoutfuture=1.229753. Diagnostyka używa zakodowanych prawdziwych stanów dopiero po teście, bez wpływu na wybór lub prognozę. Jej jednostki to uczone, normalizowane współrzędne, a nie surowe kanały ViT. Rekonstrukcja mierzy błąd encoder/decoder; nie jest ścisłym dolnym ograniczeniem błędu przyszłej prognozy.

J128 prognozy jest bitowo identyczny z wycinkiem zapisanej prognozy full768: True. Audyt nie zmieniał modelu.

## Granice wniosków

Wynik eksploracyjny: czteryholdouty były oglądane wcześniej, również w statycznymEDMD. Wszystkie34obrazy są w datasettrainViT; podział dotyczy identyfikacji dynamiki. Tylko25z30fit są w dokładnej bazie1500 źródłowego treningu(3old+22new), pozostałe5 są train poza tą bazą. J128 wybrano na500, więc wcześniejsza walidacja jest retrospektywna względem kanałów. To jedna wspólna trajektoria uczenia ViT, bez niezależnych replik. 20epok i jeden seed nie ustalają globalnego optimum neuralmodelu. Nie sprawdzano prognoz pomiędzy zapisami co10aktualizacji. Lepszyfit nie gwarantuje lepszej długiej prognozy ani85procent znaku.

## Artefakty

- PLAN_NNDMD.md: protokół i jawne korekty przed treningiem/wykonaniem.
- NNDMD_PREFIX500.pt: frozen encoder, pełnygenerator768, decoder768, config i normalizacja.
- NNDMD_PREFIX500_A_LEARNED_768.npy oraz GENERATOR_768.npy:768×768, learnedaxes.
- WYBOR_NNDMD.json i ZAMROZENIE_NNDMD.json: wybór/chronologia/hashe.
- NNDMD_PROGNOZA_HOLDOUT_FULL768.npy:50×4×600×768float32, wszystkie600cząsteczek.
- NNDMD_METRYKI_HOLDOUT.json, NNDMD_HORYZONTY_J128.csv, WYNIKI_NNDMD.json: wyniki50horyzontów, perimage orazfull/J.
- NNDMD_*_HISTORIA.json, nndmd_training_logs/: wszystkieepoki/kandydaty i losscomponents.
- NNDMD_RUNTIME.json, NNDMD-requirements-lock.txt i smoke: wersje i weryfikacja.

- NNDMD_AUDYT_OPERATORA.json: stabilność, reszty domkniętości i kontrola decoded J128.

- NNDMD_TRAIN_EXECUTED_SNAPSHOT.py i AMENDMENT_NNDMD_SOURCE.md: dokładny hash źródła załadowanego podczas treningu, odróżniony od źródła późniejszej ewaluacji. Niezależny audyt potwierdził identyczność AST treningu/wyboru/refit; różnica dotyczy zapisu dodatkowego pliku J128 i docstringów.

```powershell
.\.venv-pykoopman\Scripts\python.exe -m pytest -q
.\.venv-pykoopman\Scripts\python.exe nndmd_fit.py --stage train
.\.venv-pykoopman\Scripts\python.exe nndmd_fit.py --stage evaluate
.\.venv-pykoopman\Scripts\python.exe audit_nndmd.py
.\.venv-pykoopman\Scripts\python.exe report_nndmd.py
```

Skrypt odmawia ponownego strojenia w katalogu z gotowym WYNIKI_NNDMD.json. Odtworzenie wymaga świeżego katalogu z danymi, protokołem i skryptami. RLS pozostaje odłożonym planem, bez wyników. Dawne statyczne modele i dane768/J128 źródłowych eksperymentów pozostają zachowane.

Źródła pierwotne: [oficjalny NNDMD](https://pykoopman.readthedocs.io/en/master/tutorial_koopman_nndmd_examples.html), [kodv1.2.1](https://github.com/dynamicslab/pykoopman/blob/v1.2.1/src/pykoopman/regression/_nndmd.py).
