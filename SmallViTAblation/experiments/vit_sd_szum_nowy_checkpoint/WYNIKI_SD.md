# Jeden nowy obraz SD z czystego szumu

S128=62.827646 procent; MSE128=11.952540912; MSE640=0.056493965; MSE768=2.039168456. Cosine128=0.929508880; norm ratio128=0.882977708. S768=62.557535.

Checkpoint: 841c55b4ce752863d1c25d7e89f0fca452714ee68b5693d2837e3c92f4ca2b2d. Oryginalny cel: 2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df. Seed=202610057719; 80 DDIM, 106 ograniczonych aktualizacji. Start wyłącznie szum, zerowe maski i pusty tekst. Brak img2img, treningu, podmiany aktywacji lub optymalizacji po DDIM.

Pewność detektora=0.255851 jest drugorzędna, bez GT nie jest accuracy. Zdjęcie bazowe oceniono wyłącznie przy weryfikacji reloadu: S128=74.251936. Nie było wejściem SD. Podglądy to etapy tego samego biegu. Brak dodatkowego ramienia SD oznacza brak oszacowania przyczynowej korzyści prowadzenia. Jedna próba, bez dowodu generalizacji i bez gwarancji 90.

Niezależny proces weryfikacji: świeży plain forward bez hooków i hook wyłącznie odczytujący dają bitowo identyczny detektor. Pełny latent zgadza się z eksportem w tolerancji float32 CPU/GPU; największa różnica 0.000026703. Odtworzony S128=62.827649. Cel i SHA checkpointu zgodne. Szczegóły: VALIDATION.json.
