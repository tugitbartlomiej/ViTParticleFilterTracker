# Cel i kryterium

Przed pomiarem deklarujemy S = 100 max(0, 1 - sqrt(MSE768 / mean(target^2))). Cel to oryginalne after w LATENT_PELNY.npz, checkpoint MODEL_VIT_K1000.pt zamrożony. Próg 90 wymaga MSE <= 0.01 mean(target^2). Dolna granica aktywacji daje MSE >= 6.26027425890453, zatem S <= 34.395359237805565. Nie obcinamy celu i nie dopasowujemy skali ani ROI.

Najpierw odtwarzamy certyfikat z wag i granic. Następnie przeliczamy obrazy zapisane w wybranych wcześniejszych rodzinach przez rzeczywisty forward 600x768. Podłoga: oryginalne zdjęcie i najbliższe zdjęcie z pełnego zbioru. Warianty widzą ten sam cel i ten sam model. To dopasowanie do jednego znanego celu, nie ocena generalizacji.

Na wyraźne polecenie użytkownika wykonujemy nową ograniczoną generację lokalnym SD: jedno nowe ziarno, img2img strength 0.30, 40 kroków DDIM (12 wykonywanych), kontrola bez prowadzenia i wariant prowadzony. Cel prowadzenia: surowy pełny MSE z małą karą odchylenia od propozycji SD, trust RMS 0.05, trzy ostatnie kroki bez prowadzenia. Brak optymalizacji końcowego VAE i treningu wag. Brak poprawy surowego MSE względem kontroli obala korzyść tego prowadzenia. Niezależnie od poprawy próg 90 jest wykluczony.

MSE128, MSE640 i cosine są osobnymi miarami. Wysoki cosine można badać później jako podobieństwo kierunkowe, nie zgodność współrzędnych. Wynik certyfikatu pozostaje ograniczeniem koniecznym, nie dowodem osiągalności jego maksimum. Nie dotykamy innych zadań GPU.

Rejestr lokalny WYNIKI.json zastępuje modyfikację wspólnego rejestru, ponieważ użytkownik ograniczył zapis do dwóch katalogów tej rodziny. Plan, kod i stempel źródeł pozostają w nich.
