# Koopman z wejściami faktycznych paczek: wynik lokalnego biegu

Analiza 3 stałych sond (2, 3, 6), wszystkich 600 tokenów i 768 współrzędnych. Zegar to pojedynczy krok AdamW. Wejściem jest zapisane pchnięcie S_nowa z rzeczywistej paczki treningowej.

Źródło: gotowy bieg H100 s47 job1823539; Z i S_nowa z tego samego biegu. Lokalny RTX3070 policzył nowe macierze Grama, redukcję i projekcje. Dopasowanie PyDMD oraz ridge odbyło się na CPU. Nie uruchomiono ponownie treningu sieci.

## Najważniejsze ustalenia

DMDc z pamięcią obniża błąd kwadratowy względem persystencji o 33.83%, 20.60% i 4.99% dla 1/5/10 kroków. Odpowiednie autonomiczne DMD z pamięcią: 7.54%, -8.38%, -3.74%. Zmierzona siła paczki wnosi informację, a poprzednia prędkość jest ważnym składnikiem stanu.

DMDc nie przewyższa jednak prostego modelu beta*v+b*S_nowa: 97.29%, 86.84%, 58.92%. Sam warunek znajomości sił daje bardzo dużo; nie należy przypisywać tej informacji odkrytym modom Koopmana.

64 mody opisują 98.98% wariancji stanów w train, ale nawet rzut rzeczywistego ruchu testowego na te mody daje skill H1=50.03%. Stała przestrzeń wyznaczona z przeszłości nie obejmuje znacznej części późniejszych kierunków ruchu. To mierzalne ograniczenie wspólnego modelu o małej randze.

Pierwszy wspólny mod ma korelacje wkładów trzech sond 0.944–0.984 w train, lecz 0.432–0.684 w teście. Dla kolejnych modów pojawiają się także zmiany znaku korelacji. Wspólna struktura jest widoczna, ale jej zgodność pomiędzy sondami nie jest stała w czasie.

Wartości własne DMDc z pamięcią dają kandydackie okresy najsłabiej tłumionych składowych: 51.8, 58.9, 75.0, 157.6 kroków. Promień spektralny wynosi 1.000572; jedna para jest lekko niestabilna. Te okresy opisują dopasowany operator. Długie okresy mają niewiele cykli w oknie uczącym i nie stanowią samodzielnego potwierdzenia trwałych oscylacji sieci.

Dla tokenów narzędzia skill DMDc z pamięcią wynosi 23.61%, 10.70%, -4.74% (H1/5/10); dla tła 34.76%, 21.34%, 5.64%. Model zbiorowy gorzej przenosi się na ruch narzędzia. Biała linia na mapach modów oznacza maskę tokenów narzędzia.

## Chronologia i znaczenie prognozy

Train: przejścia 300–699; validation 700–799; test 800–999. POD używa wyłącznie stanów 300–700, POD wejść wyłącznie sił 300–699. Ranki i regularyzacja wybrane na validation przed obliczeniem test. Wszystkie prognozy zaczynają się od prawdziwego pełnego stanu i (dla pamięci) poprzedniego stanu. Reszta poza POD zostaje zamrożona w stanie początkowym. Metryki obejmują pełne 768 współrzędnych, także tę resztę.

H1 to ocena kroku z prawdziwym stanem początkowym (teacher-forced/rolling-origin). H5/H10 są rolloutami bez kolejnych prawdziwych stanów w rekurencji, lecz używają kolejnych zapisanych sił. Siły wymagają gradientów i AdamW na prawdziwych stanach źródłowego biegu: jest to prognoza warunkowa z oracle przyszłych wymuszeń, nie samodzielna prognoza nieznanych paczek.

S_nowa jest finite-step zmianą z(theta+d_nowa,b)-z(theta,b), nie JVP. d_nowa wykorzystuje nowy gradient i mianownik AdamW całej paczki. Pomiar zapisano w hooku po kroku, ale jego definicja jest obliczalna z bieżącej paczki i stanu przed aktualizacją; nie wykorzystuje przyszłego dz. Ponowne obliczenie wejścia dla nowej paczki ma koszt gradientu i dodatkowego forward sond, którego ta analiza nie mierzy.

## Wyniki testowe

| Model | Horyzont | RMSE pełne768 | Skill vs persystencja | Cos ruchu |
|---|---:|---:|---:|---:|
|persistence|1|0.0190186|0.0000|—|
|velocity|1|0.0134383|0.5007|0.7507|
|force_direct|1|0.0168906|0.2113|0.4620|
|force_inertia|1|0.0031331|0.9729|0.9859|
|DMD_ridge|1|0.0190202|-0.0002|0.0080|
|DMD_delay|1|0.0182877|0.0754|0.2821|
|DMDc_ridge|1|0.0187494|0.0281|0.1445|
|DMDc_delay|1|0.0154706|0.3383|0.5515|
|DMDc_delay_inputlag|1|0.0155465|0.3318|0.5449|
|PyDMD|1|0.0190388|-0.0021|-0.0056|
|PyDMDc|1|0.0189277|0.0095|0.0755|
|POD_oracle_ceiling_NOT_FORECAST|1|0.0134443|0.5003|0.6875|
|persistence|5|0.0694228|0.0000|—|
|velocity|5|0.102041|-1.1604|0.2797|
|force_direct|5|0.0618617|0.2060|0.4532|
|force_inertia|5|0.0251879|0.8684|0.9324|
|DMD_ridge|5|0.0694235|-0.0000|0.0142|
|DMD_delay|5|0.0722716|-0.0838|0.1488|
|DMDc_ridge|5|0.0689539|0.0135|0.1105|
|DMDc_delay|5|0.0618599|0.2060|0.4413|
|DMDc_delay_inputlag|5|0.0625002|0.1895|0.4277|
|PyDMD|5|0.069539|-0.0033|-0.0020|
|PyDMDc|5|0.0692473|0.0051|0.0577|
|POD_oracle_ceiling_NOT_FORECAST|5|0.0518467|0.4423|0.6520|
|persistence|10|0.0975544|0.0000|—|
|velocity|10|0.202299|-3.3002|0.1401|
|force_direct|10|0.087686|0.1921|0.4633|
|force_inertia|10|0.0625263|0.5892|0.8074|
|DMD_ridge|10|0.0975703|-0.0003|0.0149|
|DMD_delay|10|0.0993643|-0.0374|0.1615|
|DMDc_ridge|10|0.0972713|0.0058|0.0984|
|DMDc_delay|10|0.0950902|0.0499|0.3317|
|DMDc_delay_inputlag|10|0.0965956|0.0196|0.3145|
|PyDMD|10|0.0979061|-0.0072|-0.0035|
|PyDMDc|10|0.0974859|0.0014|0.0398|
|POD_oracle_ceiling_NOT_FORECAST|10|0.0766235|0.3831|0.6084|

Skill = 1 − SSE_model/SSE_persystencja; wartości ujemne oznaczają pogorszenie. Cos ruchu to cosinus całego wektora przemieszczenia grupy, uśredniony po momentach startu; nie jest średnią cosinusów pojedynczych tokenów. Wyniki narzędzie/tło są w METRICS.csv.

POD_oracle_ceiling_NOT_FORECAST to diagnostyczny rzut prawdziwego przyszłego przemieszczenia na POD. Korzysta z przyszłości wyłącznie w ocenie granicy reprezentacji; nie jest prognozą ani wejściem modelu.

Najlepsza rodzina Koopmana według validation: DMDc_delay. To wybór przed testem; osobno pokazano wszystkie rodziny.
Model prosty beta*v+b*S_nowa: beta=0.859331, b=0.993422.

## Mody i ograniczenia

POD_MODES.npy zawiera wspólne przestrzenne mody o kształcie [r,3,600,768]. COHERENCE.json mierzy korelacje wkładów poszczególnych sond do tych samych modów, osobno train/test. SPECTRA.json podaje okresy i czasy zaniku/wzrostu wynikające z wartości własnych. Są to własności dopasowanego lokalnego modelu, nie dowód fizycznego prawa ani poprawy detekcji.

Jedno ziarno, trzy sondy i zakres stałego LR. Z i siły źródłowe są float16, obliczenia robocze float32, sumowanie bloków float64. Różnica kolejnych Z zawiera błąd kwantyzacji. Górna obwiednia RMS błędu różnicy wskutek zaokrągleń to 0.0101 RMS ruchu; to konserwatywna granica, nie zmierzony faktyczny błąd. Siła nowej paczki pomija osobne zmiany BN i zanik wag; pamięć optymalizatora jest tylko przybliżona opóźnieniem stanu. Podobieństwo modów oraz dobra rekonstrukcja nie dowodzą, że wybrane obrazy poprawią mAP.

## Koszt i odtworzenie

Czas lokalnego biegu: 69.6s. Peak GPU allocated: 318.3MiB; reserved: 332.0MiB.
Wersje: torch 2.10.0+cu126, PyDMD 2025.8.1, numpy 2.2.6.
Pełne parametry: RUNNING.json; hash kodu i polecenie: RUNTIME.json. MODEL_SELECTION_BEFORE_TEST.json zapisano przed testem; prepared.pkl umożliwia audyt bez GPU.

```powershell
python -u experiments/vit_dmdc_paczki_20261008/run.py --source outputs/vit_dmdc_paczki_20261008/source_eden --out outputs/vit_dmdc_paczki_20261008/full_v2 --start 300 --train-end 700 --val-end 800 --end 1000 --max-rank 64 --block 30
python experiments/vit_dmdc_paczki_20261008/report.py outputs/vit_dmdc_paczki_20261008/full_v2
```

Biblioteka i dokumentacja: [PyDMD.DMDc](https://pydmd.github.io/PyDMD/dmdc.html), [oficjalny tutorial](https://pydmd.github.io/PyDMD/tutorial7dmdc.html).
