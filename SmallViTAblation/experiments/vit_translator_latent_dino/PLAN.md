# Bezpośredni translator ViT → DINO

Klasa: roboczy pilot. Pytanie: czy pełny latent z oryginalnego zamrożonego checkpointa k1000 pozwala przewidywać cechy DINO tego samego zdjęcia na innych filmach?

Pary: 400 istniejących oryginałów, podział 240/75/85 całymi filmami. Bez LaMa, SD i etykiet w wejściu translatora. Nie zakładamy bijekcji, dokładnego odtworzenia ani działania sztucznego celu poza rozkładem.

Model: liniowa regresja ridge w postaci dualnej, wszystkie 460800 składowych w stałej kolejności. Centrowanie i skala wyłącznie z train. DINOv2 ViT-S/14, CLS 384 L2. Selekcja regularyzacji na val, test raz po wyborze. Baseline średnia cech i kopiowanie cech najbliższego treningowego latentu ViT. Wyniki: cosine, MSE znormalizowanych cech, retrieval top1/top5 i mediana pozycji właściwego obrazu w odłożonej galerii. Kontrola ponownego wczytania modelu. Oddzielny eksport translacji pierwotnego sztucznego celu.

Pierwszy pilot nie dowodzi przyspieszenia treningu. Przed użyciem FAISS sprawdzić identyczne wagi i preprocessing. Rozszerzenie na 35000 wymaga wydajniejszej parametryzacji niż dualny model 240 kotwic i nowego badania.
