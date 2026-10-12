# Jedno zdjęcie z interwencją J128

Obraz z puli400: indeks 29, film TR_0154_S1_P03. Checkpoint1000 i wszystkie wagi/bufory zamrożone.

| Pomiar najpewniejszej ramki | Przed | Po zmianie J128 |
|---|---:|---:|
| Confidence | 0.867869 | 0.931670 |
| IoU do etykiety | 0.856595 | 0.999339 |

Pełny latent600×768 przed/po/różnica zapisany w LATENT_PELNY.npz oraz trzech CSV. Wybrane kolumny CSV oznaczone J128. LATENT128.npz i LATENT128_WARTOSCI.csv zawierają wszystkie600×128 wartości. KANALY128_PODSUMOWANIE.csv ma statystyki każdej osi, w tym tokenów z obszaru GT. Nie istnieje jeden skalar danej osi dla całego obrazu; każda z600 cząsteczek ma własną współrzędną.

To pojedynczy korzystnie wybrany przykład z optymalizacją na jego etykiecie. Confidence nie jest dokładnością na nowych obrazach. Te same piksele bez wstawienia latentu dają wynik przed. Zakres GELU zachowany; nie dowodzi to osiągalności tego latentu z naturalnego zdjęcia.

Kontrole: odczytany z dysku pełny latent jest bitowo identyczny; pełny forward odtwarza szybki tail; pozostałe640 kanałów identyczne; hash wag/buforów zachowany.

| Kanał (indeks0) | Średnia przed | Średnia po | RMS zmiany | Średnia zmiana w ROI |
|---|---:|---:|---:|---:|
| 501 | 0.051015 | 34.812511 | 35.860703 | 32.288376 |
| 247 | 0.000613 | 32.288631 | 33.194435 | 27.657412 |
| 66 | -0.122057 | 29.407099 | 30.225676 | 26.016991 |
| 596 | 0.175089 | 26.856714 | 27.837011 | 23.487356 |
| 53 | 0.157222 | 26.055374 | 26.411222 | 23.324541 |
| 464 | 0.347901 | 25.383993 | 25.392159 | 22.583191 |
| 69 | 0.027231 | 22.437462 | 23.629696 | 18.439871 |
| 377 | 0.018596 | 21.855618 | 23.204206 | 13.477997 |
| 524 | 0.091253 | 21.201189 | 21.964283 | 22.469238 |
| 144 | -0.078922 | 17.796751 | 19.719124 | 22.511356 |
| 9 | -0.119136 | 15.370843 | 16.510523 | 19.106632 |
| 386 | 0.150301 | 13.785562 | 15.513984 | 10.424064 |
| 471 | 0.331067 | 14.713576 | 15.282258 | 10.464607 |
| 616 | -0.136000 | 12.295657 | 15.052330 | 6.111422 |
| 107 | 0.037971 | 12.155347 | 13.806310 | 15.879622 |
| 508 | 1.082506 | 13.356884 | 12.888596 | 7.626859 |
