# 10000 losowych zdjęć, bez zapisywania latentów

Użytkownik ograniczył aktualny bieg do10000 losowych zdjęć. Przerwany bieg całego zbioru ma zapisany checkpoint po500 aktualizacjach i4000 ekspozycjach. Wczytujemy z niego wagi MLP i AdamW. Zachowujemy oryginalny zamrożony checkpoint ViT k1000 i DINOv2 ViT-S/14 CLS384.

Jednorazowy, deterministyczny uniform reservoir sample bez zwracania z283577 oznaczonych kandydatów TRAIN, seed20261005. Podzbiór10000 jest stały na trzy przejścia. Val512/test512 to te same kanoniczne odłożone filmy. Dekodowanie i granice GT sprawdzane online; wadliwe zdjęcia logowane i pomijane, faktyczna liczba poprawnych zdjęć może być mniejsza od10000.

Każdy forward daje pełny600×768 latent tylko w RAM/VRAM i od razu podaje go do MLP. Nie zapisujemy per-image latentów ViT, tablic cech DINO ani predykcji cech. Pliki na dysku to metadane, logi, skalary wyników i checkpointy modelu z optymalizatorem. Checkpoint zawiera wagi oraz bufory normalizacji, nie tablice aktywacji obrazów.

Batch8, lr0.0001, max3 przejścia. Walidacja i checkpoint co500 aktualizacji. Test po wyborze modelu na val. Wagi startowe widziały wcześniejszy pilot oraz4000 ekspozycji dużego biegu; wynik nie oznacza uczenia od zera na wyłącznie10000 zdjęciach. Bank negatywów odtwarzany po wznowieniu.
