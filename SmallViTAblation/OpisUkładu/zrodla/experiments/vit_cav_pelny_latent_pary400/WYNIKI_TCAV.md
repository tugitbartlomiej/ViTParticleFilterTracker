# Pełny CAV i adaptowany TCAV

## Pytanie i aparat
Czy pełny kierunek oryginał versus usunięcie LaMa jest liniowo rozdzielny i wpływa dodatnio na logit centrum narzędzia? Odczyt 800 rzeczywistych obrazów, pełne 600 × 768 GELU, checkpoint oryginalny k1000. Brak poolingu, PCA, selekcji kanałów lub maskowania gradientu.

## Wynik
Test sondy: BA 0.758824, AUROC 0.858408.
TCAV: 0.082353 dodatnich pochodnych na 85 oryginałach; średnia signed -0.0011194977, średnia |S| 0.0011637845.
Film-weighted TCAV 0.081349, CI95 [0.013888888888888888, 0.1626984126984127].
30 bootstrap CAV, 30 film-permutation CAV, 30 full-random normmatched. Empiryczne two-sided p dla film-permutation 0.064516, full-random 0.032258; rozdzielczość 1/31.
Finite differences: main largest epsilon pass = True; szczegóły i float32 cancellation w FINITE_DIFFERENCE.json.
Primal-dual error 5.8e-14. Wagi i BN niezmienione: True. Checkpoint hash niezmieniony: True.
Czas rzeczywisty 108.5 sekund, device cuda, PID 15836.

## Podłoga i adaptacje
Pełne permutacje znaków etykiet: jeden znak na film treningowy, oba człony każdej pary zachowane i zawsze przeciwne. 30 losowych kierunków w całych 460800 wymiarach, normy 1. Bootstrap całych filmów. Lambda dobrana tylko na val; kontrole i bootstrap używają zamrożonej lambdy. Gram solver jest dokładnym liniowym ridge z globalnym skalarem i średnią train, bez featurewise scaling. Detektor: pre-sigmoid logit w komórce ustalonej z GT, zamiast logitu klasy w oryginalnym artykule. LaMa matched counterexamples zamiast losowych obrazów. Test t względem .5 tylko opisowy, nie dowód istotności.

## Zastrzeżenia i czego wynik nie dowodzi
460800 wymiarów przy 480 próbkach train; kierunek może kodować pozycję, tło, artefakty LaMa i różnicę formatów JPEG/PNG. Heldout dotyczy dopasowania konceptu; obrazy mogły wcześniej uczyć ViT. To nie dowód wyłącznej tożsamości narzędzia ani przyczynowej poprawy treningu. Interwencja w latencie może wychodzić poza manifold obrazów. Gradient poza jednym tokenem naturalnie zeruje się przez token-local końcówkę sieci; nie narzucono ROI. Filmowe CI i empiryczne null pozostają warunkowe względem tego zbioru, checkpointu i kontroli.

## Źródła i reprodukcja
Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_latent128_jedno_zdjecie_90\MODEL_VIT_K1000.pt`; SHA256 `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
PDF przeczytany w całości, 10 stron: `F:\Studia\Articles\Zbiory\JEPA\PDF_all\kim18d.pdf`; §3.2-3.6 i przypis 1 str.4, §4.1-4.4.
Uruchomienie: `F:/Python/python.exe -B experiments/vit_cav_pelny_latent_pary400/run.py` z SmallViTAblation. Manifest zawiera 800 hashów obrazów. CAV_FULL_600x768.npy to jednostkowa macierz, COEF_FLAT.npy to nieznormalizowane współczynniki z intercept w CAV_METADATA.json.

## Niezależny audyt artefaktów

Potwierdzono ponownie wszystkie 800 hashów obrazów, cache, checkpoint, PDF, runner i snapshot modelu. CAV ma niezerowe współczynniki na wszystkich 600 pozycjach i wszystkich 768 kanałach. Pełny bezpośredni iloczyn gradientu i CAV odtwarza zapisane sensitivity.

Dodatnia pochodna: 7 obrazów, ujemna: 78. Empiryczne p film-permutation wynosi 0.064516, więc ta kontrola nie pozwala odrzucić null przy .05. Dla full-random p=0.032258; po Bonferronim dla dwóch kontroli 0.064516. Nie ogłaszamy statystycznie potwierdzonego semantycznego konceptu. Oba wyniki są warunkowymi odniesieniami do jawnych kontroli, przy stałej lambda i preprocessingu; hipoteza wymienności etykiet nie jest gwarantowana przez dane.

AUROC film-bootstrap CI95: [0.8013071240656477, 0.921479030339084]. Wszystkie szczegóły w AUDIT.json.
