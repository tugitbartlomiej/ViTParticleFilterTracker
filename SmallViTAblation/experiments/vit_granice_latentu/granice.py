"""Dokładne granice min/max 768 kanałów latentu ViT (GELU(fc1(LayerNorm(x))) ostatniego bloku) dla danego checkpointu.

LayerNorm: u = (x - mean(x)) / sqrt(var(x) + eps), więc sum(u) = 0 i ||u|| = sqrt(d) * sqrt(var/(var+eps)) <= sqrt(d).
Wyjście LN: g*u + b. Przed GELU: a_j = (W_j*g) . u + (W_j . b + c_j).
Dla u na sferze promienia sqrt(d) w hiperpłaszczyźnie sum(u)=0: max/min (W_j*g).u = +/- sqrt(d) * ||P(W_j*g)||,
P = rzut odejmujący średnią. Granica jest osiągalna dla pewnego x z R^d (supremum po wszystkich wejściach bloku).
Latent h_j = GELU(a_j); GELU ma minimum -0,16997 w a = -0,7518, poza tym rośnie.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / HERE.name
CK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
sd = torch.load(CK, map_location='cpu', weights_only=True)['net']
blok = max(int(k.split('.')[1]) for k in sd if k.startswith('bloki.'))
g, b = sd[f'bloki.{blok}.norm2.weight'].double(), sd[f'bloki.{blok}.norm2.bias'].double()
W, c = sd[f'bloki.{blok}.mlp.fc1.weight'].double(), sd[f'bloki.{blok}.mlp.fc1.bias'].double()   # 768 x 192
d = g.numel()
Wg = W * g
Pwg = Wg - Wg.mean(1, keepdim=True)
r = (d ** .5) * Pwg.norm(dim=1)
srodek = W @ b + c
a_min, a_max = srodek - r, srodek + r
gelu = lambda a: 0.5 * a * (1 + torch.erf(a / 2 ** .5))
A0 = -0.7517916
h_max = gelu(a_max)
h_min = torch.where((a_min < A0) & (a_max > A0), gelu(torch.tensor(A0, dtype=torch.float64)), torch.minimum(gelu(a_min), gelu(a_max)))
wynik = dict(checkpoint=str(CK), blok=blok, d=d, kanalow=W.shape[0],
             h_max=dict(min=float(h_max.min()), mediana=float(h_max.median()), maks=float(h_max.max())),
             h_min=dict(min=float(h_min.min()), maks=float(h_min.max())),
             a_przedzial_mediana=[float(a_min.median()), float(a_max.median())])
L = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
if L.exists():
    z = np.load(L); kan = z['channels0']; po = torch.from_numpy(z['after']).double()
    nad = po > h_max[None] + 1e-6
    wynik['sztuczny_J128'] = dict(
        wartosci_ponad_granica=int(nad.sum()), udzial_ponad_granica=float(nad.float().mean()),
        kanalow_z_przekroczeniem=int(nad.any(0).sum()), maks_przekroczenie=float((po - h_max[None]).max()),
        granica_maks_na_128=float(h_max[kan].max()), granica_mediana_na_128=float(h_max[kan].median()),
        wartosc_maks_na_128=float(po[:, kan].max()))
    ZN = OUT.parent / 'vit_konwerter_dino_sztuczny_latent' / 'ZAKRES_NATURALNY.json'
    if ZN.exists():
        wynik['naturalnie_400_zdjec_maks'] = json.loads(ZN.read_text(encoding='utf-8'))['wszystkie_768_kanalow']['maks']
np.savez(OUT / f'GRANICE_{CK.stem}.npz', h_min=h_min.float().numpy(), h_max=h_max.float().numpy(),
         a_min=a_min.float().numpy(), a_max=a_max.float().numpy())
(OUT / f'GRANICE_{CK.stem}.json').write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(wynik, indent=2, ensure_ascii=False))
