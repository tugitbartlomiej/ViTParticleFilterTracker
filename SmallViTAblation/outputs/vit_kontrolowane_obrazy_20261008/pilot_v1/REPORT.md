# Kontrolowane obrazy: wynik eksperymentu

Wygenerowano 176 kontrolowanych wariantów i 11 jednolitych/gradientowych kontroli. Dwanaście kotwic z adnotowanych narzędzi, cztery próbki FastGAN EMA po 50 tysiącach kroków. Pięć checkpointów: 0, 250, 500, 750, 1000. Obliczenia CPU, 126.4 s (bez dodatkowego testu przeniesienia na naturalne klatki).

Osiem wycinków dopasowało sondy. Cztery inne wycinki i cztery obrazy FastGAN sprawdziły odpowiedź. Wszystkie rzeczywiste obrazy pochodzą z treningu detektora: nie jest to jego niezależna walidacja. FastGAN RGB i maski są diagnostyczne, nie prawdziwe etykiety. SD nie został załadowany z powodu zajętych zasobów; nie jest wymagany do dokładnej zmiany czynników.

SVD szuka dominujących kierunków odpowiedzi. Wariant contrast maksymalizuje znormalizowaną odpowiedź na jeden czynnik względem pozostałych, z ustaloną regularyzacją 0.01/768 i ośmioma kierunkami. Używa tylko obrazów fit. Większa selektywność może odbywać się kosztem bardzo małego udziału całej odpowiedzi; sam kontrast nie dowodzi przyczynowego modułu.


## Sondy svd

real_cutout: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.
```
[[0.626 0.388 0.432]
 [0.458 0.5   0.381]
 [0.573 0.429 0.435]]
```

fastgan: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.
```
[[0.626 0.419 0.472]
 [0.491 0.571 0.409]
 [0.583 0.465 0.501]]
```

natural_final: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.
```
[[0.301 0.288 0.271]
 [0.263 0.385 0.267]
 [0.306 0.353 0.301]]
```

- pozycja: udział ruchu na naturalnych obrazach 0.141; korelacja z ΔIoU -0.432; efekt cofnięcia -0.02346, losowa kontrola tej samej normy -0.01871.
- kolor: udział ruchu na naturalnych obrazach 0.159; korelacja z ΔIoU -0.597; efekt cofnięcia -0.01996, losowa kontrola tej samej normy -0.01865.
- proporcje: udział ruchu na naturalnych obrazach 0.167; korelacja z ΔIoU -0.603; efekt cofnięcia -0.02014, losowa kontrola tej samej normy -0.01869.

## Sondy contrast

real_cutout: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.
```
[[0.002 0.001 0.001]
 [0.001 0.006 0.002]
 [0.001 0.002 0.002]]
```

fastgan: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.
```
[[0.002 0.001 0.001]
 [0.001 0.005 0.001]
 [0.001 0.001 0.002]]
```

natural_final: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.
```
[[0.003 0.002 0.003]
 [0.003 0.005 0.004]
 [0.004 0.005 0.005]]
```

- pozycja: udział ruchu na naturalnych obrazach 0.004; korelacja z ΔIoU -0.374; efekt cofnięcia -0.00001, losowa kontrola tej samej normy 0.00004.
- kolor: udział ruchu na naturalnych obrazach 0.007; korelacja z ΔIoU -0.674; efekt cofnięcia 0.00009, losowa kontrola tej samej normy 0.00008.
- proporcje: udział ruchu na naturalnych obrazach 0.006; korelacja z ΔIoU 0.224; efekt cofnięcia 0.00001, losowa kontrola tej samej normy 0.00008.

## Granice interpretacji

Kształt oznacza deformację proporcji narzędzia w 2D, nie całą semantykę kształtu. W kontrolowanych obrazach tło nie jest deformowane. W NATURAL_TRANSFER.csv pozycja i proporcje dotyczą całego kadru, a kolor tylko maski narzędzia; nie są to identyczne warunki. Wyniki transferu dotyczą końcowego checkpointu, pozostałe macierze są średnią z pięciu checkpointów.

Ruch mierzony na niezmienionych naturalnych klatkach. Baza ingerencji pochodzi z wcześniejszego checkpointu. Interwencja dotyczy gałęzi MLP po GELU, nie całej sieci ani rezyduum. Latent po ingerencji może wyjść poza rozkład aktywacji. IoU dotyczy ramki o najwyższej pewności; zmiana rankingu może powodować skok. Korelacje obejmują tylko 16 zależnych pomiarów na metodę/czynnik: opisowe, bez wnioskowania o istotności czy nowości naukowej.

Jednolite obrazy służą odpowiedzi na kolor/jasność i diagnostyce fałszywych wskazań. Próba obejmuje najwyżej 10 lokalnych szczytów; liczniki nie są pełnym mAP ani liczbą wszystkich możliwych ramek. Szczegóły: CONTROLS.csv.

QA.json: dokładne przesunięcie pikseli, zachowanie maski i tła przy zmianie koloru, identyczność zerowej interwencji, ortonormalność kierunków i kompletność tabel. Nie oznacza to potwierdzenia semantycznej poprawności wszystkich masek FastGAN.

Kod i plany w experiments/vit_kontrolowane_obrazy_20261008. Obrazy, manifesty i wyniki w bieżącym katalogu.

## Latenty jednolitych obrazów

CONTROL_LATENTS.csv zapisuje RMS zmiany wszystkich 600×768 współrzędnych względem szarości 127 oraz rzut na sześć sond. CONTROL_CHANNELS.csv zawiera 20 kanałów o największej zmianie dla każdego obrazu/checkpointu; są to kanały reakcji na bodziec, nie udowodnione neurony koloru. CONTROL_FEATURES.npz przechowuje średnią i odchylenie po tokenach dla wszystkich 768 kanałów. Czerń, biel i podstawowe RGB różnią się także jasnością, więc te odpowiedzi nie izolują chromatyczności tak jak maskowane perturbacje Lab.


## Wnioski z pilotażu

SVD na odłożonych rzeczywistych wycinkach: sonda pozycji przechwytuje około 62.6 procent energii własnej odpowiedzi wobec 38.8–43.2 procent dla innych zmian; sonda koloru 50.0 procent wobec 38.1–45.8 procent. Sonda proporcji przechwytuje przesunięcie mocniej niż własną deformację (57.3 wobec 43.5 procent), więc nie stanowi czystej sondy kształtu.

Na naturalnych obrazach końcowego checkpointu sonda koloru przechwytuje 38.5 procent odpowiedzi na maskowaną zmianę chromatyczności wobec około 26.3–26.7 procent przy zmianach położenia/proporcji całego kadru. Warunki transferu są inne od wycinków, a próba ma cztery obrazy.

Kierunki contrast są bardziej selektywne na wycinkach, ale zawierają poniżej jednego procenta energii odpowiedzi i ruchu. Ich średnie efekty cofnięcia na IoU są małe i bliskie losowej kontroli tej samej normy. Nie potwierdziliśmy, że one dominują w poprawie detekcji. Korelacje z ruchem przy checkpointach co 250 kroków są opisowe i nieporównywalne wprost z poprzednim próbkowaniem co 50 kroków.

Jednolite obrazy i gradienty na k1000: maksymalna pewność spośród dekodowanych szczytów wynosi od około 0.037 (zielony) do 0.430 (gradient poziomy); żaden nie przekracza 0.5. To kontrola negatywna na ustalonym progu, nie dowód braku fałszywych detekcji w ogólności.
