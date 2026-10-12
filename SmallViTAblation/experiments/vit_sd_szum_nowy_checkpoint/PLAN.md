# Jedna generacja z czystego szumu

Skill eksperyment, deklaracja lokalna w WYNIKI.json. Zakres zapisu wyłącznie experiments/vit_sd_szum_nowy_checkpoint i outputs/vit_sd_szum_nowy_checkpoint. Bez wspólnego rejestru, commitów i zmian innych rodzin.

Pytanie: jak bliski oryginalnemu after 600x768 będzie naturalny latent jednego nowego obrazu SD prowadzonego nowym, zamrożonym checkpointem ViT? Przekazywanie latentu do modelu zamiast pomiaru forward unieważnia próbę. Wynik S128 poniżej 90 obala osiągnięcie progu w tym biegu, nie dowodzi niemożliwości. Granica starego checkpointu nie obowiązuje nowych wag.

Warunek startu: READY_FOR_SD.json, MODEL_best.pt o zgodnym SHA256, deklaracja zakończonego świeżego reloadu. Niezależny reload i porównanie naturalnego latentu zdjęcia treningowego do finalizowanego eksportu. Zdjęcie jest wyłącznie kontrolą poprawności checkpointu, nie wejściem generatora.

Start: niezależny standardowy szum Gaussa 1x4x48x72, seed 202610057719. SD lokalny ema_020000, pusty tekst i trzy zerowe maski. Jedno ramię, 80 kroków DDIM, eta scheduler 0. Brak img2img, VAE.encode, zdjęć w samplerze lub treningu SD/ViT. Prowadzenie od kroku 25 do 77, dwie aktualizacje na krok, trust RMS 0.05 względem czystej propozycji SD. Line search 0.018, 0.006, 0.002. Ostatnie trzy kroki bez gradientu. Bez swobodnej optymalizacji po DDIM.

Energia zadeklarowana przed biegiem: MSE128/mean(target128^2) + 0.1 MSE640/mean(target640^2) + 0.01 mean((clean-anchor)^2)/0.05^2. J128 prowadzi dopasowanie, K640 zachowuje pozostałą część oryginalnego celu. Akceptacja wymaga spadku energii i niepogorszenia MSE128. Wszystkie współrzędne i tokeny, bez ROI lub dopasowania skali.

Główny wynik S128 = 100 max(0,1-sqrt(MSE128/mean(target128^2))). Raportujemy także MSE640, MSE768, S768, cosine128, norm ratio128. Pewność detektora jest drugorzędna i nie jest accuracy. Brak GT dla generowanego obrazu. Kontrola: source plain forward nowym modelem, nie dodatkowy bieg SD. Jedna próba znanego celu, bez twierdzeń o generalizacji. Nie ma gwarancji 90 ani wyniku zdjęcia treningowego.
