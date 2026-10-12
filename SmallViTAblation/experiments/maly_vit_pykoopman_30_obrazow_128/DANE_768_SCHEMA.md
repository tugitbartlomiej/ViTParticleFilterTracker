# Pełne reprezentacje do NNDMD

LATENT_768.npy: float32 (101,34,600,768), osie checkpoint, obraz, token row-major 20×30, oryginalny kanał 0..767. Krok checkpointów 0..1000 co10; jeden krok operatora oznacza 10 aktualizacji.

MANIFEST.json, OBRAZY.npy, MASKI.npz i KANALY_J128.npz są te same co w DANE_SCHEMA.md. Fit: 0..7 i 12..33, holdout 8..11. DANE_768.json zawiera status complete dopiero po pełnej kontroli finite i hash; consistency_records: J128 source12 na 101checkpointach i full768 source12 na 0/500/1000. Wycinek J128 jest bitowo identyczny z LATENT_J128.npy, który pozostaje niezmieniony. channels0/1 w metadata oznacza zestaw pomocniczej oceny J128; plik pełny zawiera WSZYSTKIE kanały 0..767.

Wejście encoder i wyjście decoder: pełne 768. Ostateczna decyzja użytkownika: learned state768 i A768×768, encoder/decode768. Propozycja learned128 została zastąpiona; dane nie wymagają zmiany. Wyuczone współrzędne nie są automatycznie natywnymi osiami J. Odczyt mmap do fitu wyłącznie z dozwolonego prefiksu; future nie jest wejściem rollout. Dane obejmują zmianę wag i BatchNorm. Nie są wagami ViT ani kompletną telemetrią sterowania.
