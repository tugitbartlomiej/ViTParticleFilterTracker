# Populacyjny PyDMD DMDc ze skalarnymi wejściami

Wykonano PyDMD.DMDc.fit na próbce train, następnie prognozy dla wszystkich1500×600 cząsteczek. Jedna wspólna para A/B. U zawiera normy g/m1/m2 z początku interwału, LR pierwszej aktualizacji, czas i bias. Nie zawiera średniego/końcowego gradientu ani przyszłego przemieszczenia.

Normy są mierzonymi amplitudami, nie pełnymi kierunkowymi gradientami. Nawet U start pochodzi z oracle zależnego od GT; przyczynowa dostępność w zapisanym czasie nie oznacza dostępności przy inference bez GT. KnownB nie jest tu zakładane. H20 frozenU zamraża również LR/czas; futureU jest jawnie oracle.

|Split|Faza|h|Metoda|RMSE768 dolna|RMSE768 górna|n|
|---|---|---:|---|---:|---:|---:|
|test|-1|5|pydmd_DMDc_startU|0.2419032|0.2419032|127200|
|test|0|5|pydmd_DMDc_startU|0.0867339|0.0867339|4737000|
|test|1|5|pydmd_DMDc_startU|0.10973663|0.10973663|12715800|
|test|-1|5|deeptime_DMD|0.25205371|0.25205371|127200|
|test|0|5|deeptime_DMD|0.092068057|0.092068057|4737000|
|test|1|5|deeptime_DMD|0.11014661|0.11014661|12715800|
|test|-1|5|persistence|0.25506024|0.25506024|127200|
|test|0|5|persistence|0.10620412|0.10620412|4737000|
|test|1|5|persistence|0.10401821|0.10401821|12715800|
|test|-1|20|pydmd_DMDc_frozen_startU|1.0212189|1.1137952|583200|
|test|0|20|pydmd_DMDc_frozen_startU|0.27322254|0.31830184|4212600|
|test|1|20|pydmd_DMDc_frozen_startU|0.25823572|0.28025592|12256800|
|test|-1|20|pydmd_DMDc_futureU_oracle|1.0084171|1.1020693|583200|
|test|0|20|pydmd_DMDc_futureU_oracle|0.28114198|0.32512508|4212600|
|test|1|20|pydmd_DMDc_futureU_oracle|0.25170142|0.27424678|12256800|
|test|-1|20|deeptime_DMD|1.0519431|1.1420315|583200|
|test|0|20|deeptime_DMD|0.30118444|0.3426041|4212600|
|test|1|20|deeptime_DMD|0.26371148|0.28530937|12256800|
|test|-1|20|persistence|1.0454379|1.1360423|583200|
|test|0|20|persistence|0.35934958|0.39471208|4212600|
|test|1|20|persistence|0.2220783|0.24733877|12256800|

Faza−1 oznacza okno przecinające zmianę celu;0fit;1min_norm. Wszystkie okna są ocenione, mieszane osobno. H5 ma znany błąd768 z dokładnością zapisu; h20 ma wyłącznie granice. Maski ról w METRICS_STRATIFIED: bit1target,bit2natural-top1,bit4konkurent;0pozostałe. Dobór rank/skali nie został zmieniony na podstawie testu. OOD, ranga i stabilność są oddzielnymi diagnostykami. Nie oceniono mAP ani nie wyeksportowano correctora.
