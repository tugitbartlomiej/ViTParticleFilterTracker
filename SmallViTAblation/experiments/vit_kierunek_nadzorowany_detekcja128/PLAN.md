# Wspólny kierunek korekty detekcji w 128 native kanałach

Pytanie: czy jedna uczona delta[128], wspólna dla wszystkich tokenów i obrazów, poprawia AP względem baseline i kontroli?

Podłoga zachowuje obrazy, checkpoint k1000, kanały, residual, końcówkę i dekoder. Baseline ma delta=0. Trzy losowe jednostkowe kierunki startują z normą wybranej korekty; skala obu znaków wybierana na val tym samym kryterium. GT nigdy nie wybiera tokenów podczas inferencji. Wszystkie warianty widzą tylko obraz i stałą zapisaną korektę.

Oryginały PARY.json i źródłowe bbox, bez LaMa. Historyczny split całych filmów z CAV SPLITS.json, porównanie frame_id, pair_index, hashów obrazów i skalowania bbox. Manifest przed optymalizacją. Zbiór był używany rozwojowo i mógł trenować ViT: wyłącznie pilot, bez niezależnego dowodu generalizacji.

Trzy inicjalizacje 101/202/303, dwa LR 0.01/0.05, po 200 kroków, batch 8, Adam. Zwykła strata detektora: focal objectness i 2 GIoU; jedna klasa, brak osobnego logitu klasy. Regularizator 0.001 mean(delta²), projekcja normy do 5. Punkt interwencji postGELU768 ostatniego MLP. Wagi i BN zamrożone, gałąź residual zachowana. Wybór kroku przez loss val co 50 kroków. Skale [-2,-1,-0.5,-0.25,0,0.25,0.5,1,2], wariant przez AP50:95 val, potem AP50 i loss. AP jest oceną, nie różniczkowalną loss. FREEZE.json przed testem, bez doboru na test.

AP50 i AP50:95 101-punktowe, top10 lokalnych szczytów, IoU i confidence najwyżej ocenionej ramki osobno. Sparowany bootstrap 1000 resampli całych filmów dla różnicy AP. Świeży reload modelu i DIRECTION.npz, API inferencji bez etykiet. Smoke: pełny rzeczywisty ViT kontra końcówka, źródłowa strata kontra wektoryzacja, gradient delty, 640 kanałów bitowo bez zmian, hashe wszystkich wag i BN.

Brak poprawy i przewaga losowych kontroli obalają tezę o skuteczności. Korekta to jawna interwencja, nie naturalny latent wyuczony w wagach. Nie jest to CAV, TCAV ani oracle per-image. Nie wykonujemy kierunku 600x128, historycznych affine ani anulowanych zadań. Zapis tylko we własnych dwóch katalogach, bez globalnego registry, commitów i push.
