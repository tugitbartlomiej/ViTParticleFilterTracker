# Małe ViT o różnej wielkości latentu (5000 klatek, 5000 kroków)

Siedem detektorów końcówki narzędzia (cystotom, Cataract-LMM) z eksperymentu
`experiments/maly_vit_rozne_latenty_5000`, wytrenowanych na Edenie 6 października 2026 (zadanie SLURM 1823465).
Wszystkie mają tę samą budowę ogólną i ten sam przepis uczenia; różnią się tylko szerokością sieci i tym, czy
w drugiej połowie sieci obraz jest opisany 600 tokenami, czy 8 tokenami (TokenLearner). Po to, żeby oglądać
latent tego samego zadania przy różnej jego wielkości.

## Co jest czym

Latent = warstwa ukryta MLP ostatniego bloku po GELU (`bloki[-1].mlp.act`), dla jednego obrazu 480 x 320:
liczba tokenów x (4 x szerokość). mAP@0,5 na 3000 klatkach walidacji (filmy, których modele nie widziały).

| Folder | Plik | szerokość / głowy | tokeny w drugiej połowie | latent | parametry | mAP@0,5 koniec | najlepszy (krok) |
|---|---|---:|---:|---:|---:|---:|---:|
| `femto/` | `MODEL_femto.pt` | 32 / 2 | 600 | 600 x 128 | 0,16 M | 0,559 | 0,559 (5000) |
| `pico/` | `MODEL_pico.pt` | 64 / 2 | 600 | 600 x 256 | 0,41 M | 0,585 | 0,585 (5000) |
| `nano/` | `MODEL_nano.pt` | 96 / 3 | 600 | 600 x 384 | 0,82 M | 0,612 | 0,612 (5000) |
| `tiny/` | `MODEL_tiny.pt` | 192 / 3 | 600 | 600 x 768 | 2,93 M | 0,654 | 0,654 (5000) |
| `small/` | `MODEL_small.pt` | 384 / 6 | 600 | 600 x 1536 | 11,18 M | 0,589 | 0,624 (4500) |
| `tiny_tl8/` | `MODEL_tiny_tl8.pt` | 192 / 3 | 8 (po bloku 3) | 8 x 768 | 2,97 M | 0,595 | 0,628 (4500) |
| `nano_tl8/` | `MODEL_nano_tl8.pt` | 96 / 3 | 8 (po bloku 3) | 8 x 384 | 0,83 M | 0,542 | 0,608 (3500) |

- `tiny` ma dokładnie budowę `MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt` (te same klucze wag, latent
  600 x 768), ale jest uczony dłużej i na większym zbiorze (0,654 zamiast około 0,31 do 0,34). Nie jest
  zgodny z adapterem SD: SD uczono na latentach K1000.
- `femto`, `pico`, `nano`, `small`: ta sama sieć, węższa albo szersza; latent ma 600 tokenów, zmienia się tylko
  liczba kanałów na token.
- `tiny_tl8`, `nano_tl8`: po bloku 3 TokenLearner (Ryoo i in. 2021, wersja 1.1: LayerNorm, MLP z GELU, sigmoid)
  zbiera 600 tokenów w 8; bloki 4 do 6 liczą tylko na 8 tokenach; TokenFuser oddaje je na siatkę 600 patchy
  przed głową detekcji. Latent ma 8 tokenów zamiast 600. Każdy z 8 tokenów ma mapę uwagi na 600 patchach
  (`net._mapy` po `net.zapisuj_latent(True)`), która mówi, skąd token zbiera. W `nano_tl8` token 4 ma mapę
  prawie wszędzie zerową i 15,6 razy większą masę na patchach narzędzia niż mapa równa (token „narzędzia”);
  w `tiny_tl8` mapy są szerokie.

Wszystkie: wejście RGB 480 x 320 w [0, 1], łodyga konwolucyjna z krokiem 16 (siatka 20 x 30), 6 bloków, głowa
gęsta na każdy patch: logit środka narzędzia i odległości do krawędzi ramki (l, t, r, b) w jednostkach 16 px.

## Uczenie (wspólne)

Od zera, ziarno 47, paczka 32, AdamW lr 5e-4, wd 0,05, rozgrzewka 300 kroków, potem stały lr, bf16, odbicie
poziome i jasność; 5000 kroków. Zbiór: 5000 klatek = baza 1500 + 3500 z puli treningowej równo po 118 filmach
(lista w `experiments/maly_vit_rozne_latenty_5000/wyniki/pelny_<model>/dane/ZBIOR.json`).

## Ładowanie

Definicja klasy jest w `model.py` w tym folderze (kopia `experiments/maly_vit_rozne_latenty_5000/kod/model.py`;
potrzebna przy `_tl8`, bo zwykły MalyViT nie ma TokenLearnera; wymaga `timm`).

```python
import sys, torch
sys.path.insert(0, r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\assets\MalyViT_rozne_latenty_5000kl')
import model as M

ck = torch.load(r'...\MalyViT_rozne_latenty_5000kl\tiny_tl8\MODEL_tiny_tl8.pt', map_location='cpu', weights_only=False)
net = M.MalyViT(**ck['konfig'])          # konfig zapisany w pliku, np. {'dim': 192, 'glowy': 3, 'tokeny': 8, 'po_bloku': 3}
net.load_state_dict(ck['net'], strict=True); net.eval()
net.zapisuj_latent(True)
mapa, latent = net(x)                    # x: (B, 3, 320, 480); mapa (B, 20, 30, 5), latent (B, 8, 768)
mapy_uwagi = net._mapy                   # (B, 600, 8), tylko ramiona _tl8
```

Kontener: `net` (state_dict), `krok` (5000), `konfig`, `ramie`, `mAP50` (wynik końcowy). `weights_only=False`,
bo plik ma słownik konfiguracji obok wag.

## Pochodzenie i integralność

Każdy podfolder ma `PROVENANCE.json`: źródło na Edenie
(`/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000/wyniki/pelny_<model>/MODEL_<model>.pt`,
tam zostaje), SHA256 (lokalny plik zgodny z Edenem przed i po przeniesieniu), rozmiar, wynik, poprzednią lokalną
ścieżkę. Przeniesione 7 października 2026 z `experiments/maly_vit_rozne_latenty_5000/wyniki/pelny_<model>/`;
checkpointy pośrednie co 500 kroków (`wagi/`, `pelny/`) i latent kontrolny 12 klatek (`dane/LATENT_KONTROLNY.npz`)
zostają w eksperymencie. Wyniki nie były ponownie walidowane po przeniesieniu.

## Czego nie gwarantują

- Jedno ziarno; ocena między checkpointami skacze o 0,03 do 0,05, więc różnice mAP poniżej około 0,05 mogą być szumem.
- 5000 kroków to nie zbieżność (szczególnie `small`).
- Aplikacja (`app/`) obsługuje na razie tylko latent 600 x 768 (`tiny` wczyta się od razu przez `MalyViT` aplikacji;
  pozostałe wymagają tej definicji klasy i przeróbki widoków).
