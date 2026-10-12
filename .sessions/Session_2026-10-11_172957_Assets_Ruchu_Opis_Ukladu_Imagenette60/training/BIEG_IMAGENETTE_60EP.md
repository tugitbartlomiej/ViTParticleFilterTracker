# Bieg Eden 1826880: ViT od zera na Imagenette, 60 epok

- Katalog: `~/smallvit_runs/2026-10-10_szklo_pokretla/wyniki/klas_vit_od_zera_60ep_lr_5e-4`, log `logi/klas_vit60_1826880.out`, hopper-2 (H200).
- Dane: imagenette2-320, 9469 uczących, 3925 walidacyjnych, środkowe wycięcie 320×320, 400 tokenów na obraz.
- Model: MalyViT (łodyga + 6 bloków, dim 192, MLP 768) + Linear(10) na średniej tokenów, 2 890 402 parametry.
- Przepis: AdamW lr 5e-4, wd 0,05, paczka 32, rozgrzewka 300, stały lr, bf16, odbicie 0,5, jasność/kontrast ±20 %, obcięcie gradientu 1, ziarno 47.
- 60 epok × 296 kroków = 17 760 kroków.
- Wagi `wagi/k<krok>.pt`: co 10 do 1000, co 100 dalej, końce epok; stany `pelny/stan_e59.pt`, `stan_e60.pt`.
- Obserwator: 24 sondy, losowy rzut 128D, 1 sonda w pełnym 768D; gęsto co 1 krok 0–2188 i 16 576–17 760 (ostatnie 4 epoki), poza tym co 10.
- Analizy: okna 300–2188 i 16 576–17 760 (`ANALIZA.json`, `ANALIZA_16576_17760.json`); surowe gesty zostawione (ZOSTAW=1).
- Próba przed biegiem: 1826879 (`klas_proba_historia_wag`, 512 obrazów, 2 epoki) OK.
