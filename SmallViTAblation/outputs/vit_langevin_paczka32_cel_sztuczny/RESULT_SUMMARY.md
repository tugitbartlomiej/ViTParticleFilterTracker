# Paczka32: trening zakończony, brak trafienia w cel

Status complete. Ukryty worker PID37328 zakończył normalnie; nie ma już procesu tego biegu w kolejce. Zrealizowano oba ramiona po50 pełnych aktualizacji32, oryginalna strata detektora, pełna sieć i zwykły train BatchNorm. Żaden cudzy proces nie został zatrzymany. Pełny smoke32 wykazał1383.12MiB peak allocated i1660MiB reserved; próg kolejki obniżono do2816MiB z ponad1GiB zapasu. Bez mikrobatche.

Wagi i autentyczny stan AdamW k1000 pobrano z Edenu, sprawdzono zdalny SHA256 oraz identyczność wszystkich tensorów modelu z checkpointem oryginalnego celu. Wszystkie odległości dotyczą tego samego oryginalnego zdjęcia: sonda29, TR_0154_S1_P03, klatka1926, RGB480x320 i tokeny20x30. Nie porównano latentu nowej fotografii jako tej samej tożsamości.

Wybrana paczka: top32 unikalne train z sortowania całego zakończonego rankingu po naturalnym query MSE768. Pierwszy obraz TR_0006_S1_P03/001193. Wszystkie32 obrazy mają prawdziwe ramki narzędzia. Losowa kontrola32: 28 z narzędziem i4 prawidłowe obrazy negatywne. Audyt ZIP/CRC/SHA256, surowe annotation.csv i współrzędne GT zachowane.

| Stan oryginalnej sondy wobec sztucznego after | MSE128 | MSE768 | Cosinus128 | Cosinus768 |
|---|---:|---:|---:|---:|
| Brak zmiany | 85.921177 | 14.320196 | 0.106738 | 0.127023 |
| Wybrane32, krok50 | 86.034271 | 14.360467 | 0.088453 | 0.113514 |
| Losowe32, krok50 | 86.028879 | 14.350373 | 0.091720 | 0.116752 |
| Wybrane32, prognoza stałej siły na50 | 90.545516 | 16.467708 | -0.163790 | -0.074983 |

Wybrana paczka pogorszyła po50krokach MSE128 o0.1316procent i MSE768 o0.2812procent względem braku zmiany. Losowa32 pogorszyła odpowiednio0.1254 i0.2107procent. Najmniejsze rzeczywiste MSE wybranej paczki:85.928224 i14.333868, nadal ponad niezmienionym modelem. Kontrola losowa miała chwilowe minimum85.626551 i14.281360, poprawę znacznie mniejszą niż zadeklarowane10procent.

Próg osiągnięcia zadeklarowano przed biegiem: redukcja MSE o10/25/50procent, oddzielnie128 i768, horyzont50. Wszystkie hitting_time w prognozach i obu rzeczywistych ramionach są null, czyli brak trafienia do50. Nie wyznaczamy arbitralnej liczby kolejnych epok. Jeden krok jest jedną epoką stałej puli32, nie epoką całego zbioru.

Równanie oryginalne delta=S_nowa+S_ped, siły z bieżących wag i autentycznego AdamW, zostało zapisane przed treningiem. Dodatkowo użyto bez dopasowywania zapisanych współczynników0.9838/0.9844 z historycznego TOZSAMOSC_JEDEN_KROK.json. Obie prognozy nie osiągają progów. W prognozie wielokrokowej przyjęto stałą początkową siłę; nie użyto rzeczywistych przyszłych gradientów. Jej duże pogorszenie na50 jest skutkiem tego ograniczonego przybliżenia i nie jest przewidzianą liczbą epok z wiarygodnym pokryciem.

Weryfikacja pierwszego kroku oryginalnego równania na pełnych600x768: wybrana32, skill przyrostu94.6999procent wobec braku zmiany, a przy starych buforach BN99.9177procent. Losowa32 odpowiednio99.3523 i99.8230procent. To trafność przewidywania przyrostu, nie procent zbliżenia do celu. Równanie przewidziało pierwszy ruch, ale ruch wybranej paczki nie poprawił odległości do celu.

Pliki: FORCE_selected32.npz i FORCE_random32.npz, FORECAST.json/csv, FIRST_STEP_VALIDATION.json, ACTUAL_selected32/random32.json, LATENT_*_k001/k005/k010/k025/k050.npz, MODEL_*_k1050.pt, RESULT.json. Wszystkie nowe pliki tej rodziny pod SmallViTAblation. Znane niezwiązane zmiany zachowane; bez commitów.

Nie dowodzi to globalnej niemożliwości po dalszej zmianie wag. Historyczne współczynniki dopasowano do średnich tokenów narzędzia; transfer na pełne tokeny zweryfikowano tylko pierwszym krokiem w tym biegu. To eksploracja jednej sondy, bez oceny generalizacji detekcji. Protokół bez augmentacji zadeklarowano jawnie, bez zmiany architektury, zamrażania głowy i ingerencji w latent.
