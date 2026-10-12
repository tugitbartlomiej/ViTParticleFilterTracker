# Naturalny latent do zgodności85 procent

Kontynuacja MODEL_final.pt rodziny vit_wyuczenie_latentu_jedno_zdjecie_2000, Adam step2000. Wszystkie rodziny1000/2000 pozostają nietknięte. Przed pierwszym update weryfikujemy naturalny latent, mapę detekcji, agreement68.3291376, stałe energie, cztery stany Adam step2000 i lr3e-5.

Jedyny warunek sukcesu i primary stop: agreement128 = 100 max(0,1-sqrt(MSE128/mean(target128²))) >=85, potwierdzone przez nowe modele w niezależnym procesie, plain forward bez hooków zastępujących aktywacje. Cosine, confidence, IoU, GTloss, MSE640 i MSE768 są metrykami wtórnymi, bez gate sukcesu.

Podłoga: oryginalny checkpoint0 i zapisane naturalne2000; zachowany identyczny obraz29 TR_0154 klatka1926, RGB480x320 INTER_AREA, ImageNet wewnątrz oryginalnego modelu. Target after i J128 dokładnie z pierwotnego LATENT_PELNY.npz, 640 celów jest before. Target bez zmiany, rescale, clipping i normalizowania wektorów. Loss bez zmian: MSE128/stała energia target128 + 10 MSE640/stała energia target640 + 0.02 CenterNet/GIoU na GT.

Tylko cztery parametry norm2 affine i fc1 weight/bias ostatniego bloku. Downstream i wszystkie pozostałe wagi/BN zamrożone. Eval, float32, bez autocast/TF32, pełny naturalny forward. Adam przywrócony, początkowo stały3e-5. Chunk1000 oznacza punkt zapisu, nie zatrzymanie. Twardy limit50000 dalszych aktualizacji (łącznie52000).

Monitor co100/500. Plateau: wzrost najlepszego agreement w ostatnich2000 aktualizacjach <0.10 pp, najwcześniej po2000 nowych krokach. Wtedy wolno jednorazowo zmniejszyć lr z3e-5 do1e-5 dla dokładniejszego dopasowania, ze zdarzeniem w LR_EVENTS.json. Jeżeli przez następne4000 kroków po zmianie poprawa <0.02 pp, kończymy z jawnym plateau. Bez zmiany celu, architektury, trenowalnych parametrów lub objective. Błąd niefinitywności/wykonania kończy bezpiecznie i zachowuje ostatni checkpoint; negatywny wynik nie dowodzi niemożliwości.

Best = najwyższe agreement, bez kryterium detekcji. Best_live utrwalany przed progress przy każdej poprawie; latest co100, chunk checkpoint co1000; pełny optimizer, wersje, architektura, changed_parameters. Atomowy zapis checkpointów z retry, unikatowe temporary PID/czas; atomowy JSON z retry i niekrytyczny progress. Proces lock zapobiega drugiemu writerowi. RTX3070 limit22 procent, mała alokacja, przerwa0.03 s. Nie zabijamy cudzych zadań.

Final i best: naturalne latenty600x768, CSV J128, overlay, audyty wszystkich parametrów/BN, historie i porównanie0/1000/2000/final. Niezależny proces weryfikuje również optimizer steps, źródłowe SHA oraz inference tylko z pikselami. Trening widzi target i GT, ocena jest na tym samym obrazie; to overfit, bez oceny generalizacji. Wszystkie nowe pliki wyłącznie w dwóch katalogach do85, bez commitów i zapisów Grant.
