# Wynik wykonania bibliotek temporalnych

Obrazy: 16. Rank analityczny: 16. Metryki: pełne 768D.

| Split | Faza | h | Metoda | RMSE 768D | n |
|---|---|---:|---|---:|---:|
|validation|fit|1|constant_increment|0.019520206|1312|
|validation|fit|1|deeptime_DMD|0.015618893|1312|
|validation|fit|1|persistence|0.026642772|1312|
|validation|fit|1|phase_shrink|0.025331881|1312|
|validation|fit|1|pydmd_knownB|0.012923908|1312|
|validation|fit|1|pydmd_learnedB|0.013856103|1312|
|validation|fit|5|constant_increment|0.092633015|1248|
|validation|fit|5|deeptime_DMD|0.073478498|1248|
|validation|fit|5|persistence|0.12954611|1248|
|validation|fit|5|phase_shrink|0.12319721|1248|
|validation|fit|5|pydmd_knownB|0.059561259|1248|
|validation|fit|5|pydmd_learnedB|0.064309201|1248|
|validation|fit|20|constant_increment|0.31832976|992|
|validation|fit|20|deeptime_DMD|0.2564501|992|
|validation|fit|20|persistence|0.47420926|992|
|validation|fit|20|phase_shrink|0.45217972|992|
|validation|fit|20|pydmd_knownB|0.19651582|992|
|validation|fit|20|pydmd_learnedB|0.21772681|992|
|validation|min_norm|1|constant_increment|0.018438852|1920|
|validation|min_norm|1|deeptime_DMD|0.014939571|1920|
|validation|min_norm|1|persistence|0.018630314|1920|
|validation|min_norm|1|phase_shrink|0.015399469|1920|
|validation|min_norm|1|pydmd_knownB|0.0092803418|1920|
|validation|min_norm|1|pydmd_learnedB|0.013776033|1920|
|validation|min_norm|5|constant_increment|0.08662119|1920|
|validation|min_norm|5|deeptime_DMD|0.062479679|1920|
|validation|min_norm|5|persistence|0.087653906|1920|
|validation|min_norm|5|phase_shrink|0.063710038|1920|
|validation|min_norm|5|pydmd_knownB|0.041674555|1920|
|validation|min_norm|5|pydmd_learnedB|0.044506019|1920|
|validation|min_norm|20|constant_increment|0.21811462|1920|
|validation|min_norm|20|deeptime_DMD|0.11611645|1920|
|validation|min_norm|20|persistence|0.21901157|1920|
|validation|min_norm|20|phase_shrink|0.11034706|1920|
|validation|min_norm|20|pydmd_knownB|0.086938916|1920|
|validation|min_norm|20|pydmd_learnedB|0.086976415|1920|
|test|fit|1|constant_increment|0.018333394|2240|
|test|fit|1|deeptime_DMD|0.015278457|2240|
|test|fit|1|persistence|0.024792452|2240|
|test|fit|1|phase_shrink|0.023328531|2240|
|test|fit|1|pydmd_knownB|0.012267426|2240|
|test|fit|1|pydmd_learnedB|1.6482962|2240|
|test|fit|5|constant_increment|0.087110763|2176|
|test|fit|5|deeptime_DMD|0.072332807|2176|
|test|fit|5|persistence|0.12051527|2176|
|test|fit|5|phase_shrink|0.11332967|2176|
|test|fit|5|pydmd_knownB|0.056903199|2176|
|test|fit|5|pydmd_learnedB|8.3091882|2176|
|test|fit|20|constant_increment|0.30421115|1816|
|test|fit|20|deeptime_DMD|0.25720975|1816|
|test|fit|20|persistence|0.44471269|1816|
|test|fit|20|phase_shrink|0.41925997|1816|
|test|fit|20|pydmd_knownB|0.19237309|1816|
|test|fit|20|pydmd_learnedB|35.21277|1816|
|test|min_norm|1|constant_increment|0.020076603|2560|
|test|min_norm|1|deeptime_DMD|0.015918335|2560|
|test|min_norm|1|persistence|0.02034921|2560|
|test|min_norm|1|phase_shrink|0.016194421|2560|
|test|min_norm|1|pydmd_knownB|0.010581971|2560|
|test|min_norm|1|pydmd_learnedB|0.017995534|2560|
|test|min_norm|5|constant_increment|0.099648408|2560|
|test|min_norm|5|deeptime_DMD|0.071815128|2560|
|test|min_norm|5|persistence|0.10107582|2560|
|test|min_norm|5|phase_shrink|0.072770758|2560|
|test|min_norm|5|pydmd_knownB|0.048992255|2560|
|test|min_norm|5|pydmd_learnedB|0.055025653|2560|
|test|min_norm|20|constant_increment|0.26864937|2560|
|test|min_norm|20|deeptime_DMD|0.14635785|2560|
|test|min_norm|20|persistence|0.27218486|2560|
|test|min_norm|20|phase_shrink|0.13946465|2560|
|test|min_norm|20|pydmd_knownB|0.11107804|2560|
|test|min_norm|20|pydmd_learnedB|0.11319682|2560|

To prognoza ruchu latentów, nie dowód poprawy detekcji. Kontrole DMDc i identyfikacja faz są teacher-forced. KnownB to analityczny Adam po projekcji. Nowy panel rozdziela filmy według kontraktu producenta; stary panel ma własny deterministyczny split filmów. Żaden test/validation nie dopasowuje PCA, skal, operatorów, VAMP, kmeans ani baseline. Parametry wybrano przed odczytem wyników testu.

Źródła API: [PyDMD DMDc](https://pydmd.github.io/PyDMD/dmdc.html), [deeptime DMD](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.DMD.html), [deeptime VAMP](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.VAMP.html).
