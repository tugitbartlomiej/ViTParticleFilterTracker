# Trening SD z pełnego latentu ViT na filmach Edenu

Zadanie SLURM **1823167**, partycja hopper, cztery H100. Pierwsze zadanie
1823166 zakończyło się przed zapisem aktualizacji z powodu typu ramek w budowie
celów detektora. Konwersję poprawiono i przetestowano skończoną stratę i gradient.
Ponowiona próba wykonała dwa kroki DDP i przeszła kontrole gradientów, zależności
od warunku oraz odtworzenia checkpointa. Pełny bieg uruchomiono w tym samym zadaniu.

## Dane i model

- 118 filmów TRAIN: 313359 klatek CSV, 286676 z wpisaną ramką cystotomu.
- 25 filmów VAL: 66374 klatek. Nie wchodzą do aktualizacji wag.
- 27 filmów bez przypisania w używanym manifeście pominięto.
- MP4 odczytywane z istniejących ZIP do BytesIO; dekodowanie PyAV w RAM.
- Oryginalny zamrożony ViT k1000 wytwarza naturalne aktywacje 600x768 online.
- Adapter 768->768->768 dla każdego tokenu i pozycje 20x30; bez poolingu/PCA.
- SD EMA020000: uczone adapter i wszystkie attn2, 45182144 parametrów.
- VAE i ViT zamrożone, pozostałe wagi SD zamrożone. Pełny latent jest jedynym
  warunkiem UNet; trzy dawne kanały masek są zerowe.
- DDPM epsilon MSE; co osiem aktualizacji pomocnicza zgodność pełnego latentu
  oraz strata detekcji względem ramki. Brak ramki wyłącza tylko stratę detekcji.
- BF16, cztery GPU, po dwa obrazy, akumulacja cztery: globalna paczka 32.
- Budżet 20000 aktualizacji, limit zadania 23h30m; regularne checkpointy z AdamW.

Nie zapisujemy klatek, wygenerowanych obrazów ani tablic latentów. Zapisujemy
kod, manifesty, metryki, dwa ostatnie checkpointy i najlepszy checkpoint walidacyjny.
Liczba dostępnych filmów nie oznacza, że wszystkie klatki zostały już użyte.

## Lokalizacja

Remote: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy`.
Postęp: `wyniki/trening/STATUS.json`, metryki `METRICS.jsonl`, checkpointy
`checkpoint_XXXXXXXX.pt`, wskazanie ostatniego `LATEST.json`.
Test: `wyniki/smoke/SMOKE.json`, lokalna kopia obok niniejszego pliku.

`generate.py` udostępnia load_trained i DDIM generate: pełny latent do obrazu
RGB jako tensor w pamięci, bez zapisu obrazu. Jakość generacji po treningu nie
jest jeszcze zmierzona. Sztuczne cele spoza naturalnego rozkładu wymagają osobnej oceny.

## Wcześniejsze zatrzymanie

Na dodatkowe polecenie użytkownika działa early_stop.py: co 15 sekund sprawdza
raport walidacji i czeka na atomowy zapis odpowiadającego checkpointa. Zachowuje
best_checkpoint.pt przez hardlink (fallback kopia), nawet jeśli trening usunie
stary numerowany checkpoint. Od kroku 2000 cztery kolejne oceny z
val_film_mean_mse ponad 3% gorszym niż najlepszy powodują utworzenie STOP.
Trening kończy aktualizację i zapisuje stan; nie zabijamy procesu w środku kroku.
Kontroler obserwuje pogorszenie walidacji, nie dowodzi jego przyczyny na podstawie
samego skalara. Dodatkowy heartbeat w aplikacji sprawdza zakończenie i awarie
co pięć minut, pozostając cicho przy normalnym postępie.
