# Jeden obraz z detekcją po sztucznej zmianie J128

```powershell
F:/Python/python.exe -u experiments/vit_latent128_jedno_zdjecie_90/wyznacz.py
```

Wyniki i obrazy w `outputs/vit_latent128_jedno_zdjecie_90/`. `LATENT_PELNY.npz` zawiera pełne600×768 aktywacji przed i po interwencji. `LATENT128_WARTOSCI.csv` zawiera wszystkie600×128 wartości, a `KANALY128_PODSUMOWANIE.csv` statystyki128 osi. Wybrane osie w pełnych CSV mają prefiks J128. Ramki pokazuje `DETEKCJA_PRZED_PO.png`.

Źródłowe zdjęcie daje wynik baseline. Wynik po zmianie wymaga wstawienia zapisanego latentu do zamrożonego ViT. Etykieta obrazu służy do gradientu; to przykład oracle, nie poprawa generalizacji.
