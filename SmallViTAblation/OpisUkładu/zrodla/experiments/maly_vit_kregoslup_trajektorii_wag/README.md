# Kręgosłup trajektorii wag małego ViT (bez treningu, z zapisanych wag)

Plan: [PLAN.md](PLAN.md). Poprawka po wyniku: [AMENDMENT_BLADZENIE_LOSOWE.md](AMENDMENT_BLADZENIE_LOSOWE.md).
Wynik: [WYNIKI.md](WYNIKI.md). Deklaracja: [WYNIKI.json](WYNIKI.json).

Kod jest nowy. Czyta wagi i latent sond biegu `../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny` (Eden 1818732);
niczego tam nie zmienia. Nie uczy sieci i nie używa GPU.

## Uruchomienie (lokalnie, Python 3.11)

```bash
cd kod
py -3.11 kregoslup.py --wyniki ../wyniki                  # około minuty, 2,5 GB RAM; zapisuje GRAMY_WAG.npz
py -3.11 porownanie_z_bladzeniem.py --wyniki ../wyniki    # odniesienie: błądzenie losowe; sekundy
py -3.11 dryf_okna.py --wyniki ../wyniki                  # dryf z okien odległych, wagi i latent; sekundy
```

## Pliki wyników

- `wyniki/KREGOSLUP.json`: miary z planu dla całej sieci, 9 grup wag i latentu sond, od kroku 0 i od kroku 300.
- `wyniki/GRAMY_WAG.npz`: macierze iloczynów skalarnych przesunięć wag (101 × 101) dla każdej grupy; każdą miarę
  geometrii trajektorii da się policzyć z nich bez czytania 1,2 GB wag.
- `wyniki/POROWNANIE_Z_BLADZENIEM.json`, `wyniki/DRYF_OKNA.json`: analizy z poprawki.
