# Kontynuacja MLP na wszystkich dostępnych oznaczonych danych TRAIN

Użytkownik polecił wczytać checkpoint ViT i MLP i dać jak najwięcej danych. Wczytujemy oryginalny zamrożony checkpoint k1000 i MLP64 z pilota. AdamW startuje ze świeżym stanem, bo pilot nie zapisał optymalizatora; wagi i bufory normalizacji MLP są zachowane. Trenujemy wyłącznie MLP.

Pełne 600×768 aktywacji postGELU wyliczane online, bez pooling, PCA, SD lub zapisywania całego cache. DINOv2 ViT-S/14 CLS384 L2 z identycznymi wagami i preprocessingiem. TRAIN to wszystkie kwalifikujące się obrazy z kanonicznych filmów train, ok.283577 na podstawie CSV. Dekodowanie, CRC i granice bbox sprawdzane online; błędy logowane. Żadne val/test nie wchodzi do aktualizacji MLP. Stałe512 val i512 test równomiernie między filmami i w czasie. Historyczny pilot miał obrazy canonicaltrain; jego85 zdjęć nie służy teraz za niezależny test.

Trzy przejścia przez TRAIN, batch8, lr0.0001, weightdecay0.01, cosine + standaryzowane MSE + InfoNCE z bankiem512 wyłącznie train. Zapisy co500 aktualizacji, ocena val co2000 i po końcu przejścia. Wybór modelu wg valtop1/top5/cosine. Early stopping dopiero po pełnym pierwszym przejściu i8 nieskutecznych walidacjach. Maksymalnie3 przejścia, a nie bezterminowy trening. Test tylko po wyborze. Czas szacowany ze zmierzonej prędkości.

Zapisujemy model_latest/model_best/model_final z optymalizatorem, pozycją danych i RNG. Wznowienie nie podmienia aktywacji. Bank negatywów jest odtwarzany po wznowieniu; nie deklarujemy bitowej identyczności wznowionego przebiegu. Pierwszy przebieg wykorzystuje całe dane, poza udokumentowanymi wadliwymi obrazami. Nie jest to trening ViT ani dowód przyspieszenia uczenia detektora. Sztuczny latent jest zapytaniem poza naturalnym rozkładem.
