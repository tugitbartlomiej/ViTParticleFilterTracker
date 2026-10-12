# Plan

1. Zweryfikować, że pula kontrolna ma 256 unikalnych obrazów.
2. Usunąć deterministycznie ostatnich siedem obrazów z listy wspólnej oraz slot kontrolny.
3. Wstawić osiem unikalnych kandydatów DINO, zachowując rozmiar puli 256.
4. Wykonać kontrolę i wariant kolejno na tym samym GPU od identycznego K8000.
5. Porównać przyrosty IoU, confidence i guard AP50.
6. Zachować pełne provenance, logi, wyniki i sumy checkpointów.
