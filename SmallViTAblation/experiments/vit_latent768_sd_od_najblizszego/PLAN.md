# SD img2img z najbliższego rzeczywistego zdjęcia

Źródło: klatka1181,TR_0064_S1_P03, zwycięzca pełnego przeszukania461547 obrazów. Cel: ten sam sztuczny `after`600×768, bez zmiany ani przycinania. Wiemy, że dokładny cel wykracza poza granice zamrożonej warstwy; badamy jedynie zbliżenie.

Własny SD EMA020000, VAE sd-vae-ft-mse, pusty tekst, zerowe maski. ObrazRGB576×384 kodujemy przez średnią posteriorVAE. Dodajemy ten sam zapisany szum przy strength0.20 i0.35, DDIM60. Dla każdego poziomu kontrola bez prowadzenia oraz prowadzenie eta0.006 i0.018, dwie iteracje na krok, bez korekty w ostatnich3 krokach. Nie stosujemy końcowej swobodnej optymalizacjiVAE.

Energia prowadzenia: surowy MSE pełnego latentu względem celu, normalizowany błędem źródła; dodatkowo kara za zmianę niskoczęstotliwościowej struktury obrazu i odejście czystegoVAE od źródła. Gradient idzie przez dekoderVAE i zamrożonyViT; aktualizujemy przewidywany czysty latentVAE, nie latentViT ani wagi. Backtracking przyjmuje tylko spadki tej energii. Zachowujemy zwykły priorSD podczas wszystkich krokówDDIM.

Ocena na zapisanymPNG: preprocessingcv2INTER_AREA480×320, naturalny forwardViT, MSE768/MSE128/MSE640, cosinus, pewność i ramka; ponadto niskoczęstotliwościowy RMS względem wejścia. Główny wybór: minimumMSE768 spośród4 prowadzonych wyników z lowpassRMS≤0.10. Jeśli żaden nie spełnia warunku, zachowujemy minimumRMS jako jawnie nieudany wariant, nie ukrywamy odstępstwa. Kontrole nie są kandydatami do wyniku prowadzonego. Wszystkie obrazy i metryki są zapisywane.

Brak przewagi nad wejściem albo nad kontrolą o tym samym strength obala poprawę dopasowania w tej próbie. Mniejszy błąd sam nie dowodzi realizmu ani prawidłowej detekcji. Jeden seed nie daje twierdzeń o uczeniu. Źródło, checkpoint, cel i modeleSD mają SHA256, szum i VAE zapisane, źródła read-only. Wszystkie wyniki w outputs/vit_latent768_sd_od_najblizszego.
