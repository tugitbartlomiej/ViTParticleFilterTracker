# Gęsta historia tokenów małego ViT przy różnych paczkach (i pamięci pędu)

Plan: [PLAN.md](PLAN.md). Wynik: [WYNIKI.md](WYNIKI.md). Deklaracja: [WYNIKI.json](WYNIKI.json).
Kod uczenia skopiowany z `../maly_vit_ruch_czastek_gesta_historia/kod` (bieg 1818732) z dwiema zmianami w `bieg.py`:
`--paczka` i `--beta1` (zmiany przepisu), `--tokeny-na-obraz N` (zapis N tokenów na obraz zamiast 600).

## Eden

Katalog: `~/smallvit_runs/2026-10-03_maly_vit_gesta_paczka` (kod, slurm, wyniki/<tryb>_p<paczka>[_b<beta1>]).
Wejścia 12 obrazów i maski: z katalogu biegu 1818732 (`CZASTKI_WEJSCIA`).

```bash
sbatch bieg_paczka.slurm proba 320          # 20 kroków, sprawdzenie kodu
sbatch bieg_paczka.slurm pelny 320          # 1000 kroków, ~5 min H100, 2,2 GB
sbatch bieg_paczka.slurm pelny 32 0.99      # trzeci argument: beta1 AdamW
```

## Lokalnie

```bash
cd kod
py -3.11 porownaj_paczki.py --wyniki ../wyniki/porownanie --biegi pelny_p320      # plus paczka 32 z gęstej historii
py -3.11 porownaj_paczki.py --wyniki ../wyniki/porownanie_p128 --biegi pelny_p128
py -3.11 tabela.py ../wyniki/porownanie/POROWNANIE.json ../wyniki/porownanie_p128/POROWNANIE.json
```

Pobranie danych: `scp -r eden-cluster:smallvit_runs/2026-10-03_maly_vit_gesta_paczka/wyniki/<bieg>/dane wyniki/<bieg>/`,
sumy SHA256 w `wyniki/<bieg>/SHA256_{ZDALNE,LOKALNE}.txt`.
