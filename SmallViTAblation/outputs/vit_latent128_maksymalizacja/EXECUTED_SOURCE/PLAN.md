# Silna optymalizacja latentu J128

2026-10-04. Użytkownik zlecił mocniejszą optymalizację z ostatniego checkpointu na lokalnej RTX. To osobny eksperyment; poprzedni wynik k500 i małe promienie pozostają niezmienione.

Checkpoint: k1000 (ostatni zapis tego samego biegu seed47), pełny latent600×768. Zostawiamy te same128 natywnych kanałów wybranych przy k500; nie twierdzimy, że ponownie wykazano ich optymalność przy k1000. Zamrożone wagi i bufory. Zmieniamy600×128 aktywacji osobno dla każdego obrazu, również tła;640 pozostałych kanałów pozostaje identyczne.

Cel: minimalizacja rzeczywistej straty detektora (CenterNet +2GIoU). Projekcja na zakres GELU, bez górnego promienia i bez kary za wielkość. Małe poprzednie skale pooledSD nie ograniczają aktualizacji: optymalizujemy bezpośrednio surowe wartości aktywacji. Adam, learning rates0.03/0.1/0.3 w pilocie,500 kroków. Pilot12 pierwszych obrazów train z różnych filmów, wybranych przed oceną. Wybór według średniej końcowej straty, tie mniejszy learning rate. Pełny pomiar400 obrazów,800 kroków, najlepszy stan każdego obrazu według jego straty włącznie z zerową zmianą. Wszystkie etykiety służą optymalizacji: to oracle, nie test generalizacji. Oryginalne train/val/test etykiety są tylko deskryptywnym podziałem.

Metryki: AP50, AP75, AP50:95, loss, confidence×IoU, pewność i IoU najlepszego pewnością wykrycia, udział obrazów z IoU>=0.5 i confidence>=0.9. Porównać baseline oraz interwencję. Kontrola: identyczne wielkości zmian z losowymi znakami (ujemne niewykonalne zastąpione dodatnimi). Przyrosty i cele zapisać per obraz/token/kanał, także rozkłady norm. Co100 kroków krzywa straty. Próg bliskości sukcesu: AP50>=0.95 na optymalizowanej puli, opisowy i bez gwarancji osiągnięcia. Brak poprawy wobec baseline/losowego sterowania obala skuteczność tego wariantu optymalizacji.

Kontrole: tail=pełny forward; pojedyncza interwencja hook=tail; zakres GELU; skończone gradienty; nietknięte640 kanałów; hash wszystkich wag/buforów przed i po. Zachowanie zakresu GELU jest tylko warunkiem koniecznym, nie dowodem istnienia naturalnego zdjęcia realizującego latent. Nie jest to Langevin ani generowanie.

Źródła modelu/core/straty/metryk skopiowane z vit_optymalizacja_latentu128. Dane bez ponownego kopiowania z outputs/vit_optymalizacja_latentu128/input. Wyniki w outputs/vit_latent128_maksymalizacja. Nie uruchamiać na Edenie: bieżące zlecenie wybiera lokalny RTX.
