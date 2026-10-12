# Wynik rozszerzonego NNDMD1500 na Edenie

Wybrana epoka: 1. lrG: 0.0001. Calkowite kroki Adam: 7379. Final refit: 1457 krokow.
Fit obejmuje40 unikalnych przejsc cp50..90, czyli400 aktualizacji zrodla;31 sliding windows H10. Test10przejsc=100aktualizacji.
Test dotyczy odĹ‚oĹĽonych przyszĹ‚ych celĂłw tego fitu, tych samych1500 obrazĂłw. Historia ĹşrĂłdĹ‚owa byĹ‚a juĹĽ eksplorowana wczeĹ›niejszymi wariantami, wiÄ™c to eksploracyjny test, nie nietkniÄ™te niezaleĹĽne potwierdzenie. J128 odczytano po dekodowaniu peĹ‚nych768.

|KanaĹ‚y|H aktualizacji|RMSE|Persistence RMSE|RMSE ratio|MSE skill|Sign all|
|---|---:|---:|---:|---:|---:|---:|
|full768|10|0.11146988|0.10695018|1.0423|-0.086306|0.53269|
|full768|20|0.13249921|0.12004917|1.1037|-0.21817|0.52036|
|full768|30|0.16365307|0.14853314|1.1018|-0.21395|0.50159|
|full768|40|0.14244504|0.1262341|1.1284|-0.27333|0.51744|
|full768|50|0.16096665|0.14748719|1.0914|-0.19114|0.51806|
|full768|60|0.15159426|0.13773679|1.1006|-0.21134|0.54055|
|full768|70|0.17199462|0.14427889|1.1921|-0.4211|0.49679|
|full768|80|0.17951615|0.1458412|1.2309|-0.51512|0.48797|
|full768|90|0.17680862|0.14383312|1.2293|-0.51109|0.48866|
|full768|100|0.19177308|0.16356316|1.1725|-0.37469|0.4894|
|J128|10|0.099284508|0.092504365|1.0733|-0.15196|0.51977|
|J128|20|0.12161986|0.10918586|1.1139|-0.24073|0.51804|
|J128|30|0.13924991|0.12280937|1.1339|-0.28566|0.49946|
|J128|40|0.12755273|0.1134925|1.1239|-0.26312|0.5135|
|J128|50|0.14396252|0.12732537|1.1307|-0.27841|0.50304|
|J128|60|0.13465599|0.12332942|1.0918|-0.19211|0.53481|
|J128|70|0.14762155|0.1230343|1.1998|-0.43962|0.49904|
|J128|80|0.15722928|0.13374804|1.1756|-0.38195|0.50156|
|J128|90|0.15911239|0.13033816|1.2208|-0.49027|0.48041|
|J128|100|0.17188109|0.14314841|1.2007|-0.44173|0.47793|

Zakotwiczona prognoza z0+s*(D(A^h phi0)-D(phi0)) ma dodatkowy staly offset originu poza learned768. Encoder raz, bez przyszlych auxiliary i bez teacher forcing.
UĹĽyto sieci i propagatora PyKoopman1.2.1 z jawnym wĹ‚asnym adapterem fit/loss, zamiast niezmodyfikowanego NNDMD.fit. Decoder jest nieliniowy. Learnedstate768 kompresuje wejĹ›cie2544 i nie oznacza toĹĽsamoĹ›ci z oryginalnymi osiami768.
Adam minimalizuje lokalnie waĹĽony wielokrokowy MSE plus rekonstrukcjÄ™ i maskedauxiliary. RMSE tego samego waĹĽenia ma ten sam minimizer; globalne minimum ani cel85procent nie sÄ… gwarantowane.
RĂłwne wagi1500 obrazĂłw i wszystkich600 czÄ…steczek. Adnotacje ĹşrĂłdĹ‚owe majÄ… bbox, nie peĹ‚ne maski segmentacji. Rekonstrukcja nie stanowi Ĺ›cisĹ‚ej dolnej granicy bĹ‚Ä™du przyszĹ‚oĹ›ci.
