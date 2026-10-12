# Wynik wykonania bibliotek temporalnych

Obrazy: 64. Rank analityczny: 16. Metryki: pełne 768D.

| Split | Faza | h | Metoda | RMSE 768D | n |
|---|---|---:|---|---:|---:|
|validation|fit|1|constant_increment|0.015760572|13200|
|validation|fit|1|deeptime_DMD|0.015077886|13200|
|validation|fit|1|persistence|0.018979631|13200|
|validation|fit|1|phase_shrink|0.018106518|13200|
|validation|fit|1|pydmd_knownB|0.010517086|13200|
|validation|fit|1|pydmd_learnedB|0.015555124|13200|
|validation|fit|5|constant_increment|0.073017626|12944|
|validation|fit|5|deeptime_DMD|0.069975331|12944|
|validation|fit|5|persistence|0.089588556|12944|
|validation|fit|5|phase_shrink|0.08557238|12944|
|validation|fit|5|pydmd_knownB|0.047349321|12944|
|validation|fit|5|pydmd_learnedB|0.072324891|12944|
|validation|fit|20|constant_increment|0.24380173|11632|
|validation|fit|20|deeptime_DMD|0.24375597|11632|
|validation|fit|20|persistence|0.30586199|11632|
|validation|fit|20|phase_shrink|0.29687295|11632|
|validation|fit|20|pydmd_knownB|0.15123335|11632|
|validation|fit|20|pydmd_learnedB|0.25133163|11632|
|validation|min_norm|1|constant_increment|0.011235141|12800|
|validation|min_norm|1|deeptime_DMD|0.0096944485|12800|
|validation|min_norm|1|persistence|0.011462745|12800|
|validation|min_norm|1|phase_shrink|0.011123745|12800|
|validation|min_norm|1|pydmd_knownB|0.0065998808|12800|
|validation|min_norm|1|pydmd_learnedB|0.010160988|12800|
|validation|min_norm|5|constant_increment|0.053765866|12800|
|validation|min_norm|5|deeptime_DMD|0.04295374|12800|
|validation|min_norm|5|persistence|0.054969068|12800|
|validation|min_norm|5|phase_shrink|0.052366689|12800|
|validation|min_norm|5|pydmd_knownB|0.030904231|12800|
|validation|min_norm|5|pydmd_learnedB|0.044916867|12800|
|validation|min_norm|20|constant_increment|0.16392137|12800|
|validation|min_norm|20|deeptime_DMD|0.11213429|12800|
|validation|min_norm|20|persistence|0.16897458|12800|
|validation|min_norm|20|phase_shrink|0.15358356|12800|
|validation|min_norm|20|pydmd_knownB|0.082310603|12800|
|validation|min_norm|20|pydmd_learnedB|0.11747453|12800|
|test|fit|1|constant_increment|0.015929376|12400|
|test|fit|1|deeptime_DMD|0.014137738|12400|
|test|fit|1|persistence|0.018802323|12400|
|test|fit|1|phase_shrink|0.018252366|12400|
|test|fit|1|pydmd_knownB|0.010443263|12400|
|test|fit|1|pydmd_learnedB|0.01400382|12400|
|test|fit|5|constant_increment|0.073897117|12080|
|test|fit|5|deeptime_DMD|0.065106635|12080|
|test|fit|5|persistence|0.088381207|12080|
|test|fit|5|phase_shrink|0.086225847|12080|
|test|fit|5|pydmd_knownB|0.046898442|12080|
|test|fit|5|pydmd_learnedB|0.064374954|12080|
|test|fit|20|constant_increment|0.24494378|10800|
|test|fit|20|deeptime_DMD|0.22123464|10800|
|test|fit|20|persistence|0.29269308|10800|
|test|fit|20|phase_shrink|0.2941463|10800|
|test|fit|20|pydmd_knownB|0.145897|10800|
|test|fit|20|pydmd_learnedB|0.21735881|10800|
|test|min_norm|1|constant_increment|0.010502431|11520|
|test|min_norm|1|deeptime_DMD|0.0090578514|11520|
|test|min_norm|1|persistence|0.010707959|11520|
|test|min_norm|1|phase_shrink|0.010277524|11520|
|test|min_norm|1|pydmd_knownB|0.0063240752|11520|
|test|min_norm|1|pydmd_learnedB|0.0093963373|11520|
|test|min_norm|5|constant_increment|0.051835091|11520|
|test|min_norm|5|deeptime_DMD|0.041317342|11520|
|test|min_norm|5|persistence|0.05294982|11520|
|test|min_norm|5|phase_shrink|0.049811501|11520|
|test|min_norm|5|pydmd_knownB|0.030664829|11520|
|test|min_norm|5|pydmd_learnedB|0.042450703|11520|
|test|min_norm|20|constant_increment|0.16052435|11520|
|test|min_norm|20|deeptime_DMD|0.11156189|11520|
|test|min_norm|20|persistence|0.16524304|11520|
|test|min_norm|20|phase_shrink|0.14780491|11520|
|test|min_norm|20|pydmd_knownB|0.083048757|11520|
|test|min_norm|20|pydmd_learnedB|0.112975|11520|

To prognoza ruchu latentów, nie dowód poprawy detekcji. Kontrole DMDc i identyfikacja faz są teacher-forced. KnownB to analityczny Adam po projekcji. Nowy panel rozdziela filmy według kontraktu producenta; stary panel ma własny deterministyczny split filmów. Żaden test/validation nie dopasowuje PCA, skal, operatorów, VAMP, kmeans ani baseline. Parametry wybrano przed odczytem wyników testu.

Źródła API: [PyDMD DMDc](https://pydmd.github.io/PyDMD/dmdc.html), [deeptime DMD](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.DMD.html), [deeptime VAMP](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.VAMP.html).
