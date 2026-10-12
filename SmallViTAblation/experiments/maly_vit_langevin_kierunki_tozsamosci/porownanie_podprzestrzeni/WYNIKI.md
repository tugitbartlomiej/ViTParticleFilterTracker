# Czy zmieniła się wyznaczona podprzestrzeń tożsamości?

Porównanie opisowe ośmiu kierunków z checkpointów 500 i 1000. Nie jest wejściem ani kryterium wyboru modelu prognostycznego.

Kąty główne: 27.69°, 30.56°, 31.41°, 35.54°, 42.72°, 44.10°, 58.60°, 83.95°.

Średnia kwadratów cosinusów wynosi 0.531748; norma Frobeniusa różnicy projektorów: 2.737157. Zgodne podprzestrzenie dawałyby kąty 0°, wartość 1 i odległość 0. To wskaźniki geometrii, nie procent zachowanej tożsamości narzędzia.

Makrowektory bazowe e_j pozostają stałymi osiami współrzędnych. Przesuwać mogą się cząsteczki, a wyznaczane z danych kierunki tożsamości mogą tworzyć inną podprzestrzeń. W modelu Langevina opisanym obok zamrożono kierunki z checkpointu 500. Ta decyzja oddziela ruch cząsteczek od zmiany układu pomiarowego.

![Kąty główne podprzestrzeni](KATY_PODPRZESTRZENI.png)

Metoda: QR usuwa jedynie błąd numeryczny ortonormalności; wartości osobliwe Q500ᵀ Q1000 są cosinusami kątów głównych. Wynik nie zależy od kolejności lub znaku kolumn obu baz. Pełna zgodność wyniku po rotacji baz stanowi kontrolę implementacji.
