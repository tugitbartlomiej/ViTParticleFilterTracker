# Jeden obraz SD z celu J128

Prośba: wygenerować jeden obraz z latentu eksperymentu o AP50 = 0,88474795.
To demonstracja, nie nowy test jakości detekcji na zbiorze.

- Bierzemy pierwszy rekord, bez wyboru według wyniku. Checkpoint ViT k1000 i TARGETS_FULL400, nie odrzucony REFINED400.
- Cel: pełny latent 600 × 768, w nim 128 współrzędnych każdego tokenu zastąpionych zapisanymi wartościami. Pozostałe 640 pozostaje z oryginału.
- Generator: istniejący SD 1.5 Cataract-LMM EMA 20000. Start img2img z klatki źródłowej, siła 0,5, te same maski narzędzia, rogówki i źrenicy. Nie jest to dekodowanie samego wektora bez obrazu pomocniczego.
- 30 kroków DDIM, wykonanych 15 od połowy, CFG 2. W każdym kroku 3 aktualizacje przewidywanego czystego latentu SD przez gradient VAE → cały obraz → ViT. Ostatnie dwa kroki bez prowadzenia.
- Energia: MSE do celu na J128 podzielony przez MSE źródła do celu + 0,1 względnego MSE zachowania innych kanałów. Prowadzenie na pełnej klatce 480 × 320, bez wycinka i bez uśredniania tokenów.
- Punkt odniesienia: rzeczywisty obraz źródłowy, z tym samym ViT i preprocessingiem. Końcowy pomiar na zapisanym PNG uint8. Brak spadku MSE oznacza, że cel tej próby nie został osiągnięty.
- Ta próba nie porównuje sterowania z SD bez sterowania. Nie ocenia generalizacji, mAP ani tego, czy poprawa detektora przenosi się przez generator.
- Zamrożone wagi SD i ViT, gradient tylko względem latentu VAE. Zapis PNG, celu, latentu obrazu, masek, parametrów, sum kontrolnych i przebiegu.
