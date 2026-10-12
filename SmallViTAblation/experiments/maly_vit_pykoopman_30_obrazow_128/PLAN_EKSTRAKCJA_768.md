# Ekstrakcja pełnych reprezentacji 768D dla NNDMD

Zapis przed uruchomieniem: 2026-10-02T12:31:37.770524+02:00, Europe/Warsaw.

Jawna korekta użytkownika: encoder NNDMD otrzymuje pełne 768 kanałów każdej cząsteczki, decoder odtwarza 768; wybrane J128 pozostają oceną obecności i oceną pomocniczą dynamiki. Pełny latent oznacza reprezentacje tokenów, nie wszystkie wagi ViT. Learned latent128 nie jest listą natywnych osi J.

Zachowujemy dokładnie MANIFEST, OBRAZY, MASKI i KANALY oraz 30 fit + 4 holdout. Nie zmieniamy LATENT_J128.npy ani wcześniejszych zweryfikowanych danych. Nowy plik LATENT_768.npy: float32, (101,34,600,768), checkpointy 0..1000 co10, około 6,33 GB. Nie zapisano wcześniejszego pełnego pliku tymczasowego; ponownie wykonujemy source forward.

Architektura i hook dokładnie jak w ekstrakcja.py: bloki[-1].mlp.act, eval, inference_mode, FP32 RGB/255 i źródłowa normalizacja, bez TF32/autocast, batch4. Każdy checkpoint ładowany strict. Bufory BatchNorm sprawdzane przed/po forward; brak treningu. GPU RTX3070 uzgodniono z agentem NNDMD jako wolne. Dysk przed alokacją: 126617620480 bajtów wolnych; wymagana rezerwa co najmniej 10 GB po alokacji.

Kontrole: identyczne 12 dawnych obrazów; J128 porównane ze źródłowym LATENT_GESTY dla wszystkich101checkpointów, pełne768 dla 0/500/1000. Nowy J128 musi być bitowo zgodny z zachowanym LATENT_J128.npy. Raport faktycznych maxabs/relativeL2 i tolerancji1e-5, bez ukrywania błędu. Wszystkie wartości finite. SHA256 nowego pliku, źródeł, obrazów, masek, manifestu, kanałów i checkpointów; metadata DANE_768.json i postęp EKSTRAKCJA_768_POSTEP.json. Brak fitów NNDMD i pobierania w tym etapie.
