# Pełny latent ViT do cech DINO przez MLP

Roboczy pilot nieliniowego mapowania dla jednego oryginalnego checkpointa k1000. Wszystkie600×768 wartości w stałej kolejności przechodzą do pierwszej warstwy gęstej. Bottleneck32 lub64 jest uczony, nie jest średnią tokenów ani PCA. Wyjście384 odpowiada L2 CLS DINOv2 ViT-S/14, identyczne wagi/preprocessing jak w pilocie liniowym.

Wykorzystujemy istniejący zweryfikowany cache400 realnych par. Train240, val75, test85; rozłączne filmy. Parametry normowania tylko train. Dwa rozmiary MLP, max800 aktualizacji, early stopping po6 nieskutecznych kontrolach co50 kroków. Cel cosine + MSE standaryzowanych różnic cech DINO + kontrastowe rozróżnianie zdjęć w galerii train. Wybór rozmiaru i checkpointa tylko val. Test po wyborze, porównanie z ridge i średnią cech. Raport train/test wykrywa przeuczenie. Świeże wczytanie na CPU sprawdza serializację.

Eksport sztucznego celu i naturalnego odpowiednika osobno. Nie zakładamy odwzorowania jeden do jednego, odwracalności ani wiarygodności poza rozkładem. Pilot nie stanowi dowodu przyspieszenia uczenia ViT.
