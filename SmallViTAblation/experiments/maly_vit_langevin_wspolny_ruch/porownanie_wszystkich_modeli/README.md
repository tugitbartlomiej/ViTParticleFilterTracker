# Porównanie równań Langevina na osi procentowej

Wykresy utworzone 2026-10-02 z zapisanych wyników testowych. Nie uruchamiano ponownie dopasowania ani selekcji modeli. Główne porównanie obejmuje 15 rodzin równań i 2 osobne wybory pod najkrótszy horyzont, czyli 17 wierszy zestawienia. Część wybranych konfiguracji jest identyczna; aliasy są jawnie zapisane w MANIFEST.json.

## Jak czytać procent poprawności

Trafność znaku odpowiada na pytanie: **czy model poprawnie przewidział wzrost albo spadek współrzędnej tożsamości danej cząsteczki?** Liczymy zgodność znaku prognozowanej i rzeczywistej zmiany, uśredniając po wszystkich ośmiu współrzędnych, momentach rozpoczęcia prognozy i cząsteczkach. Wagi cząsteczek są proporcjonalne do pokrycia narzędziem i znormalizowane do sumy jeden.

Nie odrzucamy małych zmian. Nie jest to odsetek całych wektorów przewidzianych bezbłędnie, procent trafności długości wektora ani mAP detektora. Wykres główny używa pełnej skali Y od 0 do 100 procent. Pozioma przerywana linia na 80 procent to punkt odniesienia, nie osiągnięty próg. Oś X jest logarytmiczna, z etykietami rzeczywistych horyzontów.

Osobny wykres redukcji błędu pokazuje 100 × (1 − MSE_modelu/MSE_braku_zmiany). Wynik 20 procent oznacza o 20 procent mniejszy błąd kwadratowy, a wynik ujemny oznacza pogorszenie. To inna metryka niż trafność znaku.

## Wykresy

- [Wszystkie modele: trafność od 0 do 100 procent](TRAFNOSC_WSZYSTKIE.png), [wersja wektorowa SVG](TRAFNOSC_WSZYSTKIE.svg).
- [Powiększenie zakresu 40–85 procent](TRAFNOSC_WSZYSTKIE_ZOOM.png).
- [Dokładne wartości procentowe wszystkich modeli](TABELA_TRAFNOŚCI.png).
- [Redukcja błędu wielkości zmiany](REDUKCJA_BLEDU_WSZYSTKIE.png).
- [Niepewność porównania modeli wybranych na walidacji](WYBRANE_MODELE_NIEPEWNOSC.png).
- [Przeniesienie na drugi obraz bez ponownego dopasowania](TRAFNOSC_DRUGI_OBRAZ.png).
- [Dane wszystkich wykresów, z przedziałami niepewności](DANE_WYKRESOW.csv).

Każdy PNG ma również odpowiednik SVG o tej samej nazwie. Linie i znaczniki przedstawiają wynik punktowy; nie należy interpretować drobnych różnic jako pewnej przewagi. Przedziały są opisowym bootstrapem blokowym, z około trzema efektywnymi blokami tego samego treningu. Nie są przedziałami z niezależnych ziaren.

## Co obejmuje porównanie

Główne dane: sonda2, 31 cząsteczek narzędzia, osiem kierunków U500, aktualizacje testowe 800–1000, seed47. Kolumna co1 używa 151 początków prognoz800..950; kolumna co10 używa 16 początków800..950co10. Każda kolumna porównuje modele na identycznych celach. Nie porównujemy obu kolumn jako eksperymentu z identyczną liczbą prób; dla takiego porównania istnieje pole common_co10 w źródłowych metrykach.

Pierwszy rząd przedstawia starszy GLE z pamięcią L0/L1/L2/L4/L8 i osobny stary wybór pod najkrótszy horyzont. L oznacza liczbę przyrostów zachowanych w pamięci, nie liczbę wymiarów. Drugi rząd to modele lokalne: U8 oraz U8 uzupełnione o 4/8 kierunków PCA lub PLS, a także nowy wybór pod najkrótszy horyzont. Trzeci rząd to odpowiednie modele sprzężone przez wspólny ruch narzędzia i kontekstu.

Parametry wszystkich krzywych wybrano na walidacji we wcześniejszych obliczeniach. Wykres nie wybiera ponownie zwycięzcy według testu. Stary główny model to L4 przy kroku co1 i L1 przy kroku co10. Nowy główny wybór to localPLS8 przy kroku co1 i coupledPLS4 przy kroku co10; szczegóły są w raporcie nadrzędnym.

Drugi obraz: sonda8, siedem cząsteczek, ten sam bieg uczenia. Dla niego dostępne są nowe rodziny i ponownie oceniony stary główny model. Nie ma zapisanych porównywalnych wyników wszystkich dawnych długości pamięci na tym obrazie, więc ich nie dorabiamy.

Nie mieszamy tu starszych prognoz skalarnego rzutu LDA, prognoz pełnej przestrzeni768D ani wcześniejszej kampanii H500. Różnią się celem, próbami i sposobem oceny. Pominięto też H100, który używał innych początków prognoz, oraz 450 konfiguracji walidacyjnych, których nie oceniano wszystkich na teście.

## Źródła i odtworzenie

- METRYKI_DT1.json i METRYKI_DT10.json z eksperymentu maly_vit_langevin_kierunki_tozsamosci.
- METRYKI_DT1.json i METRYKI_DT10.json z eksperymentu maly_vit_langevin_wspolny_ruch.
- Generator: ../wykresy_porownanie.py, Python F:/Python/python.exe, matplotlib.
- MANIFEST.json zawiera SHA256 czterech źródeł i generatora, aliasy i definicję miary.

Generator sprawdza zgodność początków prognoz między raportami, zgodność odtworzonego poprzedniego modelu, zakresy procentów, skończoność wyników oraz równoważność wymienionych aliasów. Pełne dane źródłowe i dopasowane równania pozostają niezmienione.
