"""Zakres naturalnych wartości latentu ViT k1000 (400 losowych klatek bazy 35000) kontra sztuczny latent J128."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'vit_konwerter_dino_filmy_eden'))
from dane import przygotuj
from model import latent_vit, wczytaj_vit
OUT = ROOT / 'outputs' / HERE.name
dev = torch.device('cuda')
vit, _ = wczytaj_vit(ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt', dev)
L = np.load(ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'); kan = L['channels0']
rows = [json.loads(l) for l in (ROOT / 'outputs/vit_dino_faiss_35000/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
idx = np.random.default_rng(0).choice(len(rows), 400, replace=False)
Z = []
for j in idx:
    w = rows[j]
    with zipfile.ZipFile(w['archive']) as z:
        rgb = np.asarray(Image.open(io.BytesIO(z.read(w['member']))).convert('RGB'))
    Z.append(latent_vit(vit, przygotuj(rgb)[0][None].to(dev))[0].cpu().numpy())
Z = np.stack(Z)                                    # 400 x 600 x 768
nat = Z.reshape(-1, 768)
q = lambda a, p: float(np.percentile(a, p))
k = nat[:, kan]
po, przed = L['after'], L['before']
token_po = np.abs(po).max(1)                      # największa wartość w każdym tokenie
wynik = dict(
    klatek=len(idx), wartosci=int(nat.size),
    wszystkie_768_kanalow=dict(min=float(nat.min()), p50=q(nat, 50), p99=q(nat, 99), p999=q(nat, 99.9), maks=float(nat.max())),
    kanaly_128_naturalnie=dict(p50=q(k, 50), p99=q(k, 99), p999=q(k, 99.9), maks=float(k.max()),
                               srednia_maks_kanalu=float(k.max(0).mean())),
    kanaly_128_sztuczny=dict(srednia=float(po[:, kan].mean()), p50=q(po[:, kan], 50), p99=q(po[:, kan], 99), maks=float(po[:, kan].max())),
    kanaly_128_przed=dict(srednia=float(przed[:, kan].mean()), maks=float(przed[:, kan].max())),
    udzial_sztucznych_powyzej_naturalnego_maks=float((po[:, kan] > k.max(0)).mean()),
    norma_tokenu=dict(naturalnie_p50=q(np.linalg.norm(nat, axis=1), 50), naturalnie_maks=float(np.linalg.norm(nat, axis=1).max()),
                      sztuczny_p50=q(np.linalg.norm(po, axis=1), 50), sztuczny_maks=float(np.linalg.norm(po, axis=1).max())),
    min_gelu_teoretyczne=-0.17)
(OUT / 'ZAKRES_NATURALNY.json').write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(wynik, indent=2, ensure_ascii=False))
