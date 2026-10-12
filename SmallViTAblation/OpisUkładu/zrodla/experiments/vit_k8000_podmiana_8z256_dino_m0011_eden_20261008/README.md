# ViT K8000: podmiana 8 z 256 obrazów

Kontrolowany test wspólnego wpływu ośmiu unikalnych naturalnych obrazów znalezionych przez DINO dla interpolacji latentu m0011.

- Kontrola: 256 oryginalnych obrazów.
- Wariant: 248 tych samych obrazów i 8 kandydatów DINO.
- Start: ten sam checkpoint K8000.
- Trening: 500 aktualizacji, batch 32, AdamW, LR 2e-5.
- Obie gałęzie są wykonywane kolejno na tym samym GPU i z tym samym seedem.
- Ocena: sonda m0011 co 25 kroków oraz AP50 na 512 obrazach z rozłącznych filmów.

Efekt zestawu jest liczony jako różnica przyrostów względem kontroli.
