# Wynik: jeden pełny wektor tokenu

Token 299, siatka 24×24: wiersz 12, kolumna 11 (indeksy od zera).
Model: scalar_velocity; ridge 0.0. Wybrany na wcześniejszej walidacji, bez błędów końcowych.
Wektor ma wszystkie 384 liczby. To test lokalny wybranego tokenu, nie operator całego obrazu.

| Horyzont | Metoda | RMSE | RMSE / wielkość zmiany | Cosinus przyrostu |
|---:|---|---:|---:|---:|
| 5 aktualizacji | selected | 0.00525611 | 0.3161 | 0.8849 |
| 5 aktualizacji | last_increment | 0.00540581 | 0.3251 | 0.8849 |
| 5 aktualizacji | persistence | 0.01662728 | 1.0000 | 0.0000 |
| 100 aktualizacji | selected | 0.13243375 | 0.4308 | 0.7934 |
| 100 aktualizacji | last_increment | 0.21922456 | 0.7132 | 0.7934 |
| 100 aktualizacji | persistence | 0.30740036 | 1.0000 | 0.0000 |

Spełnienie ustalonych kryteriów: False; {'vs_last': False, 'vs_persistence': True, 'direction': True, 'groups': True}; przewaga w 4/4 grup.
Protokół: PROTOCOL.json; dobór: CALIBRATION.json, SELECTION.json; błędy: ERRORS.csv; jawne macierze: MATRIX_*.npz.
Wynik jest diagnostyczny. Nie dowodzi pełnej tożsamości narzędzia ani poprawnej prognozy całej sieci.
Długi horyzont używa trzech początków, krótki 31; metod nie dobierano na długim horyzoncie.