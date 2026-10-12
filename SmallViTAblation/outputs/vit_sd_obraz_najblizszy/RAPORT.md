# Najbliższe rzeczywiste zdjęcie do wskazanego obrazu

Przeszukano 461547 obrazów ze 170 ZIP, bez subsamplingu. Podziały: {'train': 310257, 'val': 64167, 'test': 82370, 'unknown': 4753}.

Celem jest naturalny latent starego kolorowego obrazu oka, otrzymany przez zamrożony ViT po dekodowaniu RGB i resize 480x320 INTER_AREA. Brak ingerencji w latent i wagi. Ranking: surowe MSE wszystkich 600x768 współrzędnych. Kanały J128 pochodzą z LATENT_PELNY.npz; pozostałe 640 to ich dopełnienie.

Najlepszy obraz: TR_0006_S1_P03/TR_0006_S1_P03/TR_0006_S1_P03_001193.jpg (train). MSE768=0.029590; MSE128=0.030792; MSE640=0.029350. Cosinus768=0.874824; cosinus128=0.839289; cosinus640=0.880401. Pewność detektora=60.99%.

Ranking CSV zawiera każdy obraz. Podobieństwo latentu nie gwarantuje podobieństwa wizualnego. Pewność detektora nie jest dokładnością ani mAP. Repozytorium ma znane niezacommitowane zmiany.
