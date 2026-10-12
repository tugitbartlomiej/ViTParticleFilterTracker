# Ruch cząsteczek K1000: lokalna wizualizacja rzeczywistych zapisów

Wizualizacja opisuje zakończony eksperyment oracle, nie nowy trening. 1500 obrazów daje900000 cząsteczek w jednej bazie768D. Obrazy ani wagi nie są zmieniane.

Jedna wspólna PCA3, dopasowana do deterministycznej próbki rzeczywistych stanów0/100/500. Te same osie, środek i skala dla każdego stanu. Nie porównujemy niezależnych projekcji. Wariancja projekcji opisuje próbkę, nie gwarantuje zachowania geometrii768D.

Pełna chmura ma tylko rzeczywiste stany0/100/500. Panel16 obrazów×8 tokenów zawiera128 pełnych trajektorii co krok0..500. Nie dopisujemy pośrednich stanów900000 cząsteczek. Linie między milestone są tylko odcinkami przemieszczeń; nie nazywamy ich zmierzoną trasą.

Podłoga to naturalne położenie w kroku0; delta widoczna w PCA i norma ruchu w pełnych768D są oddzielnymi wielkościami. Globalny podgląd wyświetla deterministyczną próbkę8000/900000 punktów, pojedynczy obraz wszystkie600. Każdy widok ujawnia liczebność i rzeczywisty krok. Metryki confidence/IoU pochodzą z zapisów, nie z wyglądu chmury.

Weryfikacja: hashe transferu, skończone wartości, zgodność mapy indeksów i tokenów, wspólna PCA, brak interpolowanych danych oraz działające sterowanie w przeglądarce. Zapis lokalny projekcji i wykresów, bez nowych plików zdjęć źródłowych. Próbka projekcji i wszystkie ograniczenia muszą być zapisane w metadanych.
