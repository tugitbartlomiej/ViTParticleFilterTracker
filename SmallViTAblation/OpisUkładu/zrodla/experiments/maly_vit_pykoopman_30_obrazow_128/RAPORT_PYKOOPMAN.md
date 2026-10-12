# Wspólna macierz Koopmana, PyKoopman: wykonany eksperyment

Zakończono: 2026-10-02T12:13:17.776866+02:00. Fit, wewnętrzna walidacja, refit i test końcowy zostały rzeczywiście wykonane.

**Cel85procent nie został osiągnięty.** Na czterech obrazach holdout zgodność znaku wszystkich128współrzędnych wynosi 57.076procent dlaH10 i 57.369procent dlaH500. Żaden ze50sprawdzonych horyzontów nie osiąga85procent; ciągły horyzont odH10 wynosi0.

PrzyH500 MSE_model=0.08293626, MSE_brak_zmiany=0.05470635, ratio=1.516026, skill=-0.516026. Model ma zatem o51.60procent większy błąd niż brak zmiany. Średnia zgodność znaku po50horyzontach: 56.344procent. To wynik negatywny tego ustalonego modelu affine, nie dowód niemożliwości innych modeli.

RMSE przyH500: model=0.28798657, brak zmiany=0.23389389, ratio=1.231270. Jednostki to surowe aktywacje oryginalnych kanałów GELU. RMSE jest pierwiastkiem z odpowiedniego MSE po agregacji coverage i równych wag obrazów, nie średnią RMSE obrazów. RMSE ruchu jest identyczne z RMSE stanu, ponieważ oba błędy różnią się tylko odjęciem tego samego origin. Cel85procent odnosi się do znaku, nie RMSE. Minimalizowanie MSE i RMSE na tych samych danych ma ten sam minimizer.

## Operator i rzeczywiste dopasowanie

A128×128 i b128 są wspólne dla wszystkich tokenów narzędzia wszystkich30obrazów. Zapisana orientacja: z_next=A@z+b dla kolumn; dla wierszy z_next=z@A.T+b. Operator obejmuje10aktualizacji treningu. [z,1] daje operator129×129. Kanały z KANALY_J128.npz pozostają oryginalne i w ustalonej kolejności.

Dopasowanie wykonał pykoopman.Koopman(Identity,EDMD(svd_rank=-1,tlsq_rank=0)). Nie użyto własnego solvera zamiast pakietu. Jawnie ważone [z,1] obu stron zachowują bias. Pełna wewnętrzna SVD nie odrzuca żadnego wymiaru; zapisana macierz w oryginalnej bazie to U@state_matrix@U.T.

Prefix0..500 obejmuje1500przejść obraz-czas i25100par tokenowych. Wagi coverage sumują się do1 na obrazie, obrazy mają równe wagi. Lambda=0.01, alpha=0.001397990491; dopisano128sztucznych par odpowiadających karze alpha*||A-I||². Bias bez tej kary. Suma wag rzeczywistych par1.

Pełny zachowany wymiar129; błąd ostatniego wiersza stałej 3.095e-15, zgodność orientacji z package.predict 4.663e-15, promień spektralnyA=0.985895, condition design=60.858. Stałą w prognozie affine ustawiamy jawnie1; zapisujemy także surowy operator pakietu.

## Wybór i zamrożenie

Pierwsze24fit: identyfikacja0..300; ostatnie6fit(indeksy28..33): open-loop300..500. Wybór według średniego ilorazu MSE po20horyzontach10..200. Cztery holdouty8..11 nie brały udziału w wyborze. Wewnętrzna walidacja również nie była dobra:

| lambda | Średni MSE ratio walidacji | Znak, eps1e-8 |
|---:|---:|---:|
| 0 | 3.329603 | 49.768% |
| 1e-06 | 3.329582 | 49.769% |
| 0.0001 | 3.327520 | 49.762% |
| 0.01 | 3.138585 | 49.352% |

Refit wszystkich30fit zakończył się zamrożeniem 2026-10-02T12:12:39.986824+02:00; test zakończył się 2026-10-02T12:13:17.776866+02:00. Prognoza500..1000 używa tylko origin500 i zamrożonego operatora. Nie ma podawania przyszłych obserwacji. Wybór modelu, zamrożenie i test są osobnymi plikami i etapami CLI.

## Holdout, horyzonty10..500

| Horyzont aktualizacji | Znak all128 | RMSE model | RMSE brak zmiany | Skill MSE | Cosine ruchu | Błąd normy |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 57.076% | 0.123189 | 0.120948 | -0.037393 | 0.279001 | 0.377307 |
| 50 | 54.068% | 0.218590 | 0.163220 | -0.793560 | 0.141100 | 0.257755 |
| 100 | 49.684% | 0.254171 | 0.164972 | -1.373744 | 0.050637 | 0.342822 |
| 200 | 56.675% | 0.248120 | 0.192491 | -0.661498 | 0.306540 | 0.226389 |
| 300 | 58.188% | 0.244568 | 0.194706 | -0.577768 | 0.400409 | 0.293937 |
| 400 | 55.783% | 0.275406 | 0.204109 | -0.820628 | 0.293879 | 0.311843 |
| 500 | 57.369% | 0.287987 | 0.233894 | -0.516026 | 0.332231 | 0.233548 |

Najniższy znak=49.684procent przyH100; najwyższy=61.226procent przyH180. DlaH500 osobne obrazy:

| Indeks obrazu | Znak all128 | Znak większościowy fit |
|---:|---:|---:|
| 8 | 54.612% | 48.979% |
| 9 | 56.374% | 48.098% |
| 10 | 60.917% | 44.468% |
| 11 | 57.573% | 51.698% |

## Jawna korekta definicji znaku po teście

Oryginalny przedtestowy protokół używał eps1e-8 i eps1e-3. Po otwarciu testu audyt przypomniał historyczny cel85 dla wszystkich128składowych z sign(0)=0. AMENDMENT_METRYKA_ZNAKU.md dokumentuje dodanie all128 jako korekty porównywalności po teście. WYNIKI.json i METRYKI_HOLDOUT.json zachowują pierwotną definicję; SIGN_ALL_COMPARABILITY.json ma wszystkie trzy. Nie dopasowano ponownie modelu, nie zmieniono wyboru lambda. Odtworzone prognozy w float32 są bitowo identyczne z wcześniej zapisanymi.

Maksymalny udział dokładnie zerowych prawdziwych składowych=0. PrzyH500 udział składowych w eps1e-8=1.00000000, w eps1e-3=0.98709229; all128=57.369procent, eps1e-8=57.369procent, eps1e-3=57.528procent.

## Macierz opisowa0..1000

Po końcowym teście osobno dopasowano MODEL_OPISOWY_0_1000.npz na wszystkich30fit i wszystkich100parach czasowych: 3000przejść obraz-czas i50200par tokenowych. Lambda odziedziczona z wcześniejszej walidacji. To macierz opisowa, która zna przyszłość względem origin500; NIE użyto jej w żadnej przedstawionej prognozie.

## Granice wniosków

Wszystkie34obrazy były w train ViT. Holdout dotyczy wyłącznie identyfikacji dynamiki i cztery obrazy były oglądane w dawnych badaniach. 22nowefit i3z8starychfit należą do dokładnej bazy1500;5starychfit pochodzi z train poza bazą. J128 wybrano na500, więc wewnętrzna walidacja300..500 jest retrospektywna względem wyboru kanałów. Jeden bieg uczenia daje skorelowane cząsteczki i pomiary, bez niezależnych replik treningu. Dysponujemy101checkpointami co10aktualizacji; nie oceniamy pojedynczych aktualizacji pomiędzy zapisami.

## Artefakty i odtworzenie

- MACIERZ_KOOPMANA_128.npy/CSV: wspólneA w oryginalnych kanałach.
- BIAS_128.npy/CSV: bias w kolejności channels0.
- MODEL_PREFIX_500.npz: A,b,operator_raw,operator_affine,basis,internal_operator,eigenvalues,singular_values,channels0.
- PROGNOZA_HOLDOUT.npz:50×4×600×128 float32; prognozy wszystkich tokenów, metryki wyłącznie narzędzia.
- SIGN_ALL_COMPARABILITY.json, METRYKI_HOLDOUT.json, HORYZONTY_HOLDOUT.csv: wszystkie horyzonty i obrazy.
- PROGNOZA_FIT_DIAGNOSTYKA.npz i METRYKI_FIT_DIAGNOSTYKA.json: diagnostyka fit, oddzielona od holdout.
- PROTOKOL_MODELU.md,WYBOR_MODELU.json,ZAMROZENIE_MODELU.json: chronologia bez retuningu.
- MODEL_OPISOWY_0_1000.npz i MODEL_OPISOWY_INFO.json: dodatkowy operator opisowy.
- RUNTIME.json, requirements-lock.txt, RUNTIME_PIP_CHECK.log, SMOKE.log: środowisko i sprawdzenia.

PyKoopman1.2.1, PyDMD0.4.1, NumPy1.26.0, SciPy1.11.2, scikit-learn1.1.3, Python3.11.8. Pełna izolowana .venv-pykoopman; brak zmian globalnego F:\Python. Przypięto setuptools80.9, ponieważ setuptools84 nie zawiera pkg_resources wymaganej przy imporcie pakietu. uv pip check:206pakietów zgodnych. Testy syntetyczne przeszły, istnieją ostrzeżenia deprecacji zależności pkg_resources/lightning.

```powershell
.\.venv-pykoopman\Scripts\python.exe -m pytest -q
.\.venv-pykoopman\Scripts\python.exe pykoopman_fit.py --stage fit
.\.venv-pykoopman\Scripts\python.exe pykoopman_fit.py --stage evaluate
.\.venv-pykoopman\Scripts\python.exe pykoopman_fit.py --stage descriptive
.\.venv-pykoopman\Scripts\python.exe report_pykoopman.py
```

Fit/evaluate celowo odmawiają ponownego strojenia w katalogu z gotowym WYNIKI.json; odtworzenie wymaga świeżego katalogu z kopiami danych, protokołu i skryptów. To zabezpiecza przed dopasowywaniem do otwartego holdoutu.

Źródła pierwotne: [EDMD v1.2.1](https://raw.githubusercontent.com/dynamicslab/pykoopman/v1.2.1/src/pykoopman/regression/_edmd.py), [oficjalny przykład par](https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmd_with_rbf.html), [release](https://github.com/dynamicslab/pykoopman/releases/tag/v1.2.1).
