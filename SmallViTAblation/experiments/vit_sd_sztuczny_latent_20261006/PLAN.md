# Jedno SD z szumu sterowane pierwotnym sztucznym latentem

Lokalny EMA UNet020000, VAE mse, pusty embedding tekstu i zerowe maski. Czysty niezależny szum; żadne zdjęcie wejściowe ani znalezione zdjęcie z FAISS nie jest używane. Oryginalny zamrożony ViT k1000 odpowiada checkpointowi, z którego wyznaczono cel i translator MLP. SD dostaje korektę gradientową błędu pełnego600×768 celu;128 kanałów i640 pozostałych normalizowane oddzielnie. MLP/DINO nie są używane w tym wariancie.

64 DDIM, pierwsze20 i ostatnie3 kroki bez kierowania. Późne ograniczone korekty predicted-clean, ochrona prior/trust region. Brak dodatkowej optymalizacji po DDIM i brak podmiany aktywacji ViT. Jedna próba nie dowodzi jakości ani osiągalności celu. Oryginalny sztuczny cel może przekraczać zakres GELU oryginalnego checkpointa.

Na końcu naturalny forward zapisanego PNG: MSE128/640/768, cosine128, umowna zgodność wartości128 i osobno confidence. Bez GT nie podajemy mAP/IoU wygenerowanego zdjęcia. Nie zapisujemy latentów obrazów ani szumu, tylko hashe, obraz, podglądy, metadane i skalary.
