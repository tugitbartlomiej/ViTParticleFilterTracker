# Pokrętła szkła: temperatura (lr) i gęstość (zdjęć na parametr) dla ruchu tokenów małego ViT

Data: 2026-10-10. Pytanie: czy odciski „szkła” w ruchu tokenów (outputs/porownanie_szklo_20261010) skalują się z parametrem
sterującym tak, jak w szkle: czas relaksacji i rozmiar klatki z temperaturą (tu: szum kroku ∝ lr przy AdamW) i z gęstością
(tu: liczba zdjęć na parametr; przejście interpolacyjne / zakleszczenie, Geiger 2019, Hertz i Tyrcha 2024).

## Biegi (Eden, `~/smallvit_runs/2026-10-10_szklo_pokretla`, partycja hopper-2)

Ten sam przepis co bieg 35 000 (`kod/trening_pokretla.py` = kopia `trening35000.py` + pokrętła), ziarno 47, paczka 32,
rozgrzewka 300 kroków, gęsty obserwator na 24 sondach (pierwsze 24 z SONDY_100 dla 35 000; równo rozłożone wiersze dla
podzbiorów), zapis PCA32 co 1 krok do 2188, potem co 10; bez zapisu wag po drodze (`--bez_wag`).

| bieg | lr | zdjęć | epok | kroków | po co |
|---|---|---|---|---|---|
| lr_5e-4 | 5e-4 | 35 000 | 3 | 3282 | odniesienie z 24 sondami (przepis biegu 35 000) |
| lr_2.5e-4 | 2.5e-4 | 35 000 | 3 | 3282 | temperatura ×0,5 |
| lr_1.25e-4 | 1.25e-4 | 35 000 | 3 | 3282 | temperatura ×0,25 |
| lr_1e-3 | 1e-3 | 35 000 | 3 | 3282 | temperatura ×2 |
| gestosc_5000 | 5e-4 | 5000 (podzbiór z 35 000, rng 0) | 21 | 3297 | gęstość ×1/7 (≈ 1 ograniczenie na parametr) |
| gestosc_1500 | 5e-4 | 1500 (podzbiór, rng 0) | 70 | 3290 | gęstość ×1/23 (strona nadparametryzowana) |

```bash
sbatch pokretla.slurm <nazwa> <lr> <epoki> <manifest> <sondy_plik|-> <sondy_n> [argumenty trening_pokretla.py]
sbatch pokretla_analiza.slurm            # po zakończeniu biegów: wyniki/<bieg>/ANALIZA.json
```

Analiza (`kod/pokretla_analiza.py`, moduły `odciski.py` i `dh.py` skopiowane z outputs/porownanie_szklo_20261010): odciski
S0/S1 w oknie 300–2188, trwałość sąsiedztwa, heterogeniczność dynamiczna w układzie CR i lab, energia kroku w pełnych 768D
z norm obserwatora i jej wykładnik stygnięcia, mAP i strata. Wyniki JSON pobierane do
`outputs/porownanie_szklo_20261010/eden/wyniki/pokretla/`, rysunek `pokretla.png` (skrypt `pokretla_rysunek.py`).

## Fale 2–3 (2026-10-10 wieczór): ziarna, długie biegi, rzut losowy 128D

Ziarna 48/49 (PCA32), długie biegi 12 epok z gęstym oknem na końcu (`--gesto_od 12034`), okno po 60. epoce (`--wznow` ze stanu e60
biegu 35 000), wszystko powtórzone z losowym rzutem Gaussa 128D (`--rzut rp`, przyrostek `_rp`, 12 sond) po tym, jak kontrola rzutu
pokazała, że PCA32 z kroku 0 widzi 4–14 % energii ruchu. Rysunek zbiorczy: `pokretla_rysunek2.py` → `pokretla2.png`.

## Fala 4 (2026-10-11): „jaka to substancja” — SGD, drugi model, skan po warstwach, drugi zbiór, f32

Nowe pokrętła w `kod/trening_pokretla.py`: `--opt sgd --momentum 0.9 --wd` (temperatura ∝ lr/paczka), `--paczka`, `--model cnn`
(`kod/model_cnn.py`: ta sama łodyga, głowa i cząstka — 768 aktywacji GELU MLP na komórkę — ale mieszanie przestrzenne depthwise 5×5
zamiast uwagi, 2,29 M parametrów), `--zapis f32`, `--warstwy --warstwy_n` (rzut GELU MLP każdego z 6 bloków do `gesty/w<i>`,
`kod/obserwator_rp.py`). Drugi zbiór: `kod/trening_klasyfikacja.py` — Imagenette (10 klas, 320 px, 9469/3925 obrazów,
`~/smallvit_data/imagenette/imagenette2-320`), głowa: średnia tokenów → Linear(10), 400 tokenów na obraz.
Pełny cykl trening → analiza → kasowanie surowych gesty: `pokretla_pelny.slurm`, `klasyfikacja.slurm` (zmienne `ANALIZY`, `ZOSTAW`,
`WARSTWY` przez środowisko — opcja `--export` sbatch jest na Edenie zabijana jako „CANCELLED by 0”). Plan biegów: `zlec_produkcje.sh`
(3 tory `--dependency=singleton`):

| tor | biegi | po co |
|---|---|---|
| A | lr_{5e-4,1.25e-4,2e-3}_rp_z{48,49} (3 epoki) | trzy ziarna w rzucie 128D dla trzech temperatur |
| B | sgd_lr_{0.03,0.1,0.3}_rp (paczka 32); sgd_lr_0.1_b{16,64,128}_rp | SGD: czy klatka² ∝ lr/paczka |
| C | cnn_lr_{5e-4,1.25e-4,2e-3}_rp; warstwy_lr_5e-4_rp (8 sond, skan 4 sond × 6 warstw); klas_lr_{5e-4,1.25e-4,2e-3}; klas_cnn_lr_5e-4; klas_dlugi_lr_5e-4 (40 epok, okno 10746–11840); okno60_rp_f32 | drugi model, warstwy, drugi zbiór, precyzja zapisu |

Stan 2026-10-11 01:30: wszystkie 22 biegi ukończone i przeanalizowane (67 plików ANALIZA*.json na Edenie, kopie lokalne w
`outputs/porownanie_szklo_20261010/eden/wyniki/pokretla/`), surowe gesty skasowane po analizie; test jądra zrobiony dla lr_5e-4_rp (e02),
lr_1.25e-4_rp (e02) i dlugi_lr_5e-4_rp (e12). Wyniki i interpretacja: `outputs/porownanie_szklo_20261010/README.md` sekcje 6b i 6c.
Na Edenie zostały tylko gesty `lr_5e-4_rp` (4,2 GB) i `dlugi_lr_5e-4_rp` (7,7 GB) na potrzeby dalszych testów jądra; kwota 134/200 GB.

Analiza (`pokretla_analiza.py`, nowe pola z `dh.py`): `trwalosc_ruchliwosci_*` (Spearman rankingu ruchliwości między podoknami 200
kroków: zamrożona czy przemijająca heterogeniczność), `korelacja_krzyzowa_*` (Meiners–Quake/Moran: korelacja położeń par względem
środka klatki z opóźnieniem), `korelacja_przesuniec` (dwupunktowa w funkcji odległości), `rozbieganie_par` (turbulencja),
`rozklad_wariancji` (wspólny / zdjęcie / sąsiedztwo / własny). Test jądra: `kod/jadro_pola.py` + `jadro_pola.slurm` — przesunięcia
tokenów z Jakobianu sieci (kierunki losowe prekondycjonowane AdamW, prawdziwe kroki na paczkach, średni gradient, szum) kontra
zmierzone przyrosty (`wyniki/<bieg>/JADRO_<stan>.json`).
