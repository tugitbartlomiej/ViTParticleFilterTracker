# Najbliższe rzeczywiste zdjęcie do załączonego obrazu SD

Pytanie: które zdjęcie spośród wszystkich lokalnych obrazów Tracking ma najbliższy naturalny latent do wskazanego przez użytkownika obrazu SD?

Wzór: OBRAZ_ZAPYTANIA.png, kopia załącznika użytkownika. Nie zastępuj go nowszą generacją ani sztucznym latentem z interwencji.

Metoda: zamrożony checkpoint ViT K1000, identyczne przygotowanie 480x320 RGB dla wzoru i kandydatów, naturalne wyjście600x768, bez zmieniania aktywacji. Minimum surowego MSE całego latentu. Pomocniczo MSE128, MSE640, cosinus i pewność detektora. Wszystkie170archiwów i461547lokalnych obrazów; brak próbkowania. Najlepsze kandydatury przeliczyć osobno.

Kontrola: zachowaj wszystkie wagi i identyczny dostęp do informacji. Sprawdź sumę kontrolną wzoru, liczbę obrazów, brak zmiany wag i ponowne przeliczenie zwycięzcy. Podłoga opisowa: rozkład odległości wszystkich kandydatów dostępny w CSV. To wyszukiwanie, nie estymacja poprawy mAP; dane train/val/test mogą być użyte wspólnie do wyszukania.

Kryterium: najmniejsze MSE w objętym skanem zbiorze. Małe MSE nie gwarantuje podobieństwa postrzeganego przez człowieka; latent ViT mierzy cechy swojego modelu. Raport nie może nazywać pewności procentem poprawnych detekcji. Nie twierdzimy, że zbiór obejmuje niedostępne obrazy lub inne podzbiory Cataract-LMM.

Zapis: osobny katalog outputs/vit_sd_obraz_najblizszy. Program działa w tle, zapisuje postęp, zwycięzcę, pełny ranking, latent wzoru i trafienia, raport oraz pochodzenie.
