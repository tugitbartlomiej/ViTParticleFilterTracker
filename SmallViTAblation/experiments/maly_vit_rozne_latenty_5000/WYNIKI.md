# Małe ViT o różnej wielkości latentu: wyniki (Eden, 2026-10-06)

Plan: [PLAN.md](PLAN.md). Zadania SLURM: próba 1823464 (7 x 20 kroków, wszystkie ukończone), bieg 1823465
(7 x 5000 kroków, wszystkie COMPLETED, 2:27 do 3:03 na ramię, H100 PCIe, 24 do 28 ms na krok).
Wyniki lokalnie w `wyniki/pelny_<ramię>/` (MODEL_<ramię>.pt i LATENT_KONTROLNY.npz poza gitem), na Edenie
w `~/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000/wyniki` (2,2 GB, 253 pliki z próbami).

## mAP@0,5 na 3000 klatkach walidacji (filmy walidacyjne, jedno ziarno 47)

| Ramię | latent | parametry | krok 1000 | krok 2500 | krok 4500 | krok 5000 (koniec) | najlepszy |
|---|---:|---:|---:|---:|---:|---:|---:|
| femto | 600 x 128 | 0,16 M | 0,145 | 0,419 | 0,537 | 0,559 | 0,559 |
| pico | 600 x 256 | 0,41 M | 0,234 | 0,520 | 0,579 | 0,585 | 0,585 |
| nano | 600 x 384 | 0,82 M | 0,394 | 0,580 | 0,582 | 0,612 | 0,612 |
| tiny | 600 x 768 | 2,93 M | 0,419 | 0,554 | 0,552 | 0,654 | 0,654 |
| small | 600 x 1536 | 11,18 M | 0,371 | 0,575 | 0,624 | 0,589 | 0,624 |
| tiny_tl8 | 8 x 768 | 2,97 M | 0,456 | 0,591 | 0,628 | 0,595 | 0,628 |
| nano_tl8 | 8 x 384 | 0,83 M | 0,443 | 0,578 | 0,594 | 0,542 | 0,608 (3500) |

Odniesienie: MODEL_VIT_K1000 aplikacji (tiny, 1000 kroków na 1500 klatkach) około 0,31 do 0,34 w gęstej
historii. Ten sam tiny na 5000 klatkach i 5000 krokach: 0,654.

## Co widać

- femto do tiny: mAP rośnie z wielkością latentu (0,559 / 0,585 / 0,612 / 0,654 na końcu). small (600 x 1536)
  nie jest lepszy od tiny (najlepszy 0,624); przy 5000 krokach większa sieć nic nie zyskuje.
- 8 tokenów zamiast 600 w drugiej połowie sieci kosztuje mało: tiny_tl8 najlepszy 0,628 wobec 0,654 tiny.
- Mapy uwagi TokenLearnera (ostatni checkpoint, 12 klatek pomiaru, narzędzie = 3,0 % patchy): w nano_tl8
  token 4 ma mapę prawie wszędzie zerową (średnio 0,01) i 15,6 razy większą masę na patchach narzędzia niż mapa
  równa; to token „narzędzia”. W tiny_tl8 mapy są szerokie (średnio 0,40 do 0,84), najmocniej skupiony token
  ma tylko 1,41 razy więcej masy na narzędziu.

## Czego to nie pokazuje

- Jedno ziarno, ocena co 500 kroków skacze o 0,03 do 0,05 (np. tiny 0,624 / 0,601 / 0,589 / 0,552 / 0,654),
  więc różnice poniżej około 0,05 między ramionami mogą być szumem; kolejność femto < pico < nano < tiny jest
  zgodna przez większość checkpointów, położenie small i ramion _tl8 nie jest pewne.
- 5000 kroków to nie zbieżność; small może potrzebować dłuższego uczenia.
- Wielkość latentu zmienia się razem z pojemnością sieci: to porównanie modeli, nie samego latentu.
- Mapy TokenLearnera policzone na 12 klatkach z jednego zestawu; token narzędzia w nano_tl8 trzeba sprawdzić
  na sekwencjach filmu (aplikacja).
