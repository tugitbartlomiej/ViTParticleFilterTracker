# Populacja900k — wykonana analiza

Przetworzono1500 obrazów i900000 cząsteczek. Etykiety reżimów: 89381400 interwałów; wykluczone mieszane fazy: 618600.

Fit tylko próbka train; streaming inference wszystkich danych. Nie użyto PCA3. Regimy opisują zakończony ruch, nie siłę dostępną przed tym ruchem. DMD jest autonomiczne; kontrolowane DMDc pozostaje panelem.

| Split | Faza | h | Metoda | RMSE32 | Dolna RMSE768 | Górna RMSE768 |
|---|---|---:|---|---:|---:|---:|
|validation|0|5|deeptime_velocity_DMD|0.05299713|0.05257214|0.05257214|
|validation|0|5|last_increment|0.0890693|0.05456517|0.05456517|
|validation|0|5|mean_increment|0.3391478|0.08625168|0.08625168|
|validation|0|5|persistence|0.5054374|0.1152877|0.1152877|
|validation|0|5|phase_velocity_shrink|0.08002491|0.05397809|0.05397809|
|validation|0|20|deeptime_velocity_DMD|0.3521256|0.1092188|0.2011514|
|validation|0|20|last_increment|0.6914494|0.1633503|0.2349819|
|validation|0|20|mean_increment|1.266047|0.271199|0.3195028|
|validation|0|20|persistence|1.90344|0.3971452|0.4315755|
|validation|0|20|phase_velocity_shrink|0.5651622|0.1416724|0.2204637|
|validation|1|5|deeptime_velocity_DMD|0.2062824|0.07212994|0.07212994|
|validation|1|5|last_increment|0.2350445|0.07570748|0.07570748|
|validation|1|5|mean_increment|0.6801798|0.150687|0.150687|
|validation|1|5|persistence|0.6915463|0.1528274|0.1528274|
|validation|1|5|phase_velocity_shrink|0.2425873|0.07669246|0.07669246|
|validation|1|20|deeptime_velocity_DMD|0.8675154|0.186497|0.2321754|
|validation|1|20|last_increment|1.784848|0.368999|0.3940616|
|validation|1|20|mean_increment|1.456349|0.3029793|0.3330477|
|validation|1|20|persistence|1.495648|0.3108542|0.3402273|
|validation|1|20|phase_velocity_shrink|1.061605|0.2244594|0.2636405|
|test|0|5|deeptime_velocity_DMD|0.05731723|0.05366934|0.05366934|
|test|0|5|last_increment|0.09255897|0.05568193|0.05568193|
|test|0|5|mean_increment|0.3456656|0.08787516|0.08787516|
|test|0|5|persistence|0.5105055|0.1166298|0.1166298|
|test|0|5|phase_velocity_shrink|0.08412637|0.05512166|0.05512166|
|test|0|20|deeptime_velocity_DMD|0.3616129|0.1116294|0.20427|
|test|0|20|last_increment|0.6948288|0.1647081|0.2374739|
|test|0|20|mean_increment|1.291053|0.2765201|0.3251592|
|test|0|20|persistence|1.909653|0.3986998|0.4338509|
|test|0|20|phase_velocity_shrink|0.5752255|0.1442203|0.2237512|
|test|1|5|deeptime_velocity_DMD|0.2062735|0.07527654|0.07527654|
|test|1|5|last_increment|0.2435953|0.07978807|0.07978807|
|test|1|5|mean_increment|0.7221727|0.1600758|0.1600758|
|test|1|5|persistence|0.7334131|0.1621912|0.1621912|
|test|1|5|phase_velocity_shrink|0.240897|0.07944599|0.07944599|
|test|1|20|deeptime_velocity_DMD|0.8763587|0.189413|0.2351367|
|test|1|20|last_increment|2.009909|0.414969|0.4377343|
|test|1|20|mean_increment|1.412892|0.2950504|0.3262924|
|test|1|20|persistence|1.449989|0.3024566|0.3330045|
|test|1|20|phase_velocity_shrink|1.129675|0.2388528|0.2765188|

Dla h5 obie granice są równe i dają pełny błąd768 z dokładnością zapisu. Dla h20 to granice, nie dokładny błąd768. h1 nie jest testowalne z zapisów co5. Normy/kąty768 i energia przyrostów zachowana przez PCA32 są w REGIME_SUMMARY.json. Etykiety255 są wykluczeniami. Nie wykonano oceny mAP ani eksportu correctora.
