"""Walidacja checkpointów ViT Model Zoo (ViT-S/16, ImageNet-1k) na zbiorze walidacyjnym Imagenette (imagenette2-320/val, 3925 zdjęć).

Imagenette to 10 klas ImageNet, więc model ocenia się bez douczania: trafność top-1 i top-5 wśród 1000 klas oraz top-1 wśród samych
10 klas Imagenette. Przy okazji, z tego samego przebiegu, temperatura bramek GELU w każdym bloku (jak część A temperatury pułapki s47):
pobudzenia a = fc1(LN(x)) dla 197 tokenów obrazu, T_laser = kT0/sd_tokenów(a) (mediana po neuronach), okno cieplne |a| < 1,2816,
zapalone a > 1,2816, zgaszone a < −1,2816, średnie a.
Uruchomienie: python -I waliduj_imagenette.py <katalog przebiegu Model Zoo> <imagenette2-320/val> <wyjście.json> [checkpoint_XX ...]"""
import json
import sys
import time
from pathlib import Path

import timm
import torch
from torchvision import datasets, transforms

# indeksy klas ImageNet-1k (porządek alfabetyczny wnid) dla 10 klas Imagenette
WNID_IDX = {'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491, 'n03028079': 497,
            'n03394916': 566, 'n03417042': 569, 'n03425413': 571, 'n03445777': 574, 'n03888257': 701}
KT0 = 1 / 1.702
OKNO = 1.2815516
f4 = lambda x: float(round(float(x), 4))


def model_z(plik):
    sd = torch.load(plik, map_location='cpu', weights_only=True)
    sd = {k.removeprefix('vit.'): v for k, v in sd.items()}
    m = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=1000)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def ocen(m, loader, idx10, dev):
    nb = len(m.blocks)
    zb = {b: dict(T=0.0, okno=0.0, zap=0.0, zg=0.0, a=0.0) for b in range(nb)}

    def hak(b):
        def f(_, __, a):
            a = a.float()                                                   # (B, 197, 1536)
            sd = a.std(1).clamp_min(1e-9)                                   # po tokenach, dla każdego neuronu
            z = zb[b]
            z['T'] += (KT0 / sd).median(1).values.sum().item()
            z['okno'] += (a.abs() < OKNO).float().mean((1, 2)).sum().item()
            z['zap'] += (a > OKNO).float().mean((1, 2)).sum().item()
            z['zg'] += (a < -OKNO).float().mean((1, 2)).sum().item()
            z['a'] += a.mean((1, 2)).sum().item()
        return f

    uchwyty = [m.blocks[b].mlp.fc1.register_forward_hook(hak(b)) for b in range(nb)]
    n = t1 = t5 = t10 = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            lg = m(x)
            yi = idx10[y]
            top5 = lg.topk(5, 1).indices
            t1 += (top5[:, 0] == yi).sum().item()
            t5 += (top5 == yi[:, None]).any(1).sum().item()
            t10 += (lg[:, idx10].argmax(1) == y).sum().item()
            n += len(y)
    for u in uchwyty:
        u.remove()
    bloki = {b: {k: f4(v / n) for k, v in z.items()} for b, z in zb.items()}
    return dict(n=n, top1_1000=f4(t1 / n), top5_1000=f4(t5 / n), top1_10=f4(t10 / n),
                bloki={str(b): dict(T_laser_med=v['T'], okno_cieplne=v['okno'], zapalone=v['zap'], zgaszone=v['zg'],
                                    srednie_a=v['a']) for b, v in bloki.items()})


def main():
    przebieg, val, wy = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    nazwy = sys.argv[4:] or sorted(p.name for p in przebieg.glob('checkpoint_*') if (p / 'checkpoints').exists())
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = datasets.ImageFolder(val, tf)
    assert ds.classes == sorted(WNID_IDX), ds.classes
    idx10 = torch.tensor([WNID_IDX[c] for c in ds.classes], device=dev)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)
    wyniki_zoo = json.load(open(przebieg / 'result.json', encoding='utf-8'))
    wynik = dict(opis=__doc__.split('\n\n')[0], przebieg=przebieg.name, zbior=str(val), n_obrazow=len(ds),
                 przetwarzanie='Resize 256, CenterCrop 224, normalizacja ImageNet', kT0=f4(KT0), okno_cieplne_a=OKNO,
                 checkpointy={})
    for nazwa in nazwy:
        t0 = time.time()
        r = ocen(model_z(przebieg / nazwa / 'checkpoints').to(dev), loader, idx10, dev)
        ep = int(nazwa.split('_')[1]) + 1
        r['epoka'] = ep
        r['val_acc_imagenet_zoo'] = f4(wyniki_zoo[ep - 1]['val_acc']) if isinstance(wyniki_zoo, list) else None
        wynik['checkpointy'][nazwa] = r
        print(f'{nazwa} (epoka {ep}): top1/1000 {r["top1_1000"]:.4f}  top5/1000 {r["top5_1000"]:.4f}  '
              f'top1/10 {r["top1_10"]:.4f}  zoo val_acc {r["val_acc_imagenet_zoo"]}  [{time.time() - t0:.0f} s]', flush=True)
        for b, v in r['bloki'].items():
            print(f'   blok {b:>2}: T {v["T_laser_med"]:.3f}  okno {v["okno_cieplne"]:.3f}  zapalone {v["zapalone"]:.3f}  '
                  f'zgaszone {v["zgaszone"]:.3f}  a {v["srednie_a"]:+.3f}', flush=True)
    wy.parent.mkdir(parents=True, exist_ok=True)
    json.dump(wynik, open(wy, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('zapisano', wy)


if __name__ == '__main__':
    main()
