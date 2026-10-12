# Translator pełnego latentu ViT do cech DINO

Uruchomienie: `F:/Python/python.exe -X utf8 experiments/vit_translator_latent_dino/run.py`.

Wyniki w `outputs/vit_translator_latent_dino`. API: `Translator('TRANSLATOR.npz').predict(latent)` przyjmuje 600 × 768 albo N × 600 × 768 i zwraca N × 384 cech L2. Pilotażowy model jest związany z konkretnym checkpointem ViT i reprezentacją DINO opisaną w CONFIG.json.
