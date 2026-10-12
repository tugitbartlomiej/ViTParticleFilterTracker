# Wynik: bezpośrednia generacja m0011

## Pytanie

Jak zmienia się obraz modelu `SD_ViT_K8000_A_control_K10500`, gdy warunek przesuwa się w 10 równych krokach od sztucznego latentu `m0011` do jego oryginalnego latentu?

## Aparat

- Model: A_control, krok 10500, SHA256 `1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41`.
- Warunki: `z_k = z_sztuczny - (k/10)(z_sztuczny-z_oryginalny)`, `k=0..10`.
- Wszystkie obrazy: bezpośrednie warunkowanie pełnym tensorem 600 × 768, start od identycznego szumu Gaussa, seed 20261008, DDIM 50 `leading`, eta 0.
- Bez img2img i bez obrazu referencyjnego jako wejścia generatora.

## Wynik

Wygenerowano 11 obrazów: pełny sztuczny latent oraz 10 kolejnych odejmowań jednej dziesiątej przyrostu. Kontrola końców potwierdziła, że `k=0` jest dokładnie sztucznym latentem, a `k=10` dokładnie latentem oryginalnym.

Przyrost obejmuje 460 789 współrzędnych; jego norma wynosi 686,256, a norma jednej dziesiątej 68,626. Wszystkie obrazy są skończone i mają niezerowe odchylenie pikseli od 0,182 do 0,224.

Obraz zmienia się stopniowo. Średnia bezwzględna różnica między sąsiednimi krokami wynosi około 0,01 na początku i około 0,02 przy końcu skali. Między `k=0` i `k=10` średnia bezwzględna różnica RGB wynosi 0,0951, a RMSE 0,1353 w skali 0..1. Oznacza to mierzalną reakcję generatora na zmianę warunku przy stałym szumie.

Oba końce pozostają daleko od obrazu referencyjnego: MAE wynosi 0,3335 dla sztucznego latentu i 0,3286 dla oryginalnego latentu. Wizualnie wszystkie wyniki przedstawiają pole operacyjne oka, lecz nie odtwarzają wiernie obrazu źródłowego ani geometrii narzędzia.

## Podłoga

`k=10` jest kontrolą z oryginalnym latentem tego samego obrazu. Seed, harmonogram i początkowy szum pozostają identyczne. Dzięki temu różnice w serii pochodzą z warunku latentowego, z zastrzeżeniem numerycznej niedeterministyczności GPU.

## Zastrzeżenie

Oryginalny i sztuczny latent powstały w ViT K1000, SHA256 `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`. Generator był uczony na ViT K8000. Zgodny kształt 600 × 768 nie potwierdza zgodności osi reprezentacji. Jest to celowy test wejścia poza potwierdzonym rozkładem i nie dowodzi, że generator umie odwracać sztuczny latent K8000.

Nie liczono IoU jako miary poprawności wygenerowanego obrazu, ponieważ wygenerowana scena nie ma niezależnej etykiety, a odziedziczenie ramki źródłowej byłoby nieuzasadnione.

## Artefakty

- Galeria: `outputs/latent_cel_do_obrazow/sd_k8000_m0011_interpolacja/GALERIA.html`.
- Pasek porównawczy: `outputs/latent_cel_do_obrazow/sd_k8000_m0011_interpolacja/PASEK_11_KROKOW.jpg`.
- Metadane i liczby: `outputs/latent_cel_do_obrazow/sd_k8000_m0011_interpolacja/WYNIK.json`.
- Obraz referencyjny i 11 osobnych generacji znajdują się w tym samym katalogu.

