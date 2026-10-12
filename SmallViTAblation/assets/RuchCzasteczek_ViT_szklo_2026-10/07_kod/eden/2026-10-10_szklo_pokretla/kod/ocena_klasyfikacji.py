"""Ocena biegów Imagenette ze stanów pelny/stan_eNN.pt: top-1, top-5, mAP (średnia po klasach precyzji średniej, one-vs-rest
na softmaksie), pewność (średni max softmax na trafieniach i pomyłkach), ECE (15 przedziałów), NLL, dokładność na klasę,
oraz to samo na zbiorze uczącym bez augmentacji (przeuczenie). Wynik: wyniki/OCENA_KLASYFIKACJI.json.
Uruchomienie (Eden): python ocena_klasyfikacji.py <katalog wyniki> <dane imagenette2-320> bieg1 bieg2 ..."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import trening_klasyfikacja as TK


def ap_klasy(p, traf):
    """Precyzja średnia (jak sklearn average_precision_score): suma po progach (R_n - R_{n-1}) * P_n."""
    kol = np.argsort(-p, kind='mergesort')
    t = traf[kol].astype(np.float64)
    if t.sum() == 0:
        return float('nan')
    tp = np.cumsum(t)
    prec = tp / np.arange(1, len(t) + 1)
    return float((prec * t).sum() / t.sum())


@torch.no_grad()
def wyjscia(net, X, paczka=128):
    net.eval()
    out = []
    for s in range(0, len(X), paczka):
        x = torch.from_numpy(X[s:s + paczka]).cuda().permute(0, 3, 1, 2).float() / 255.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        out.append(o.float().cpu())
    return torch.cat(out)


def miary(logity, y, n_klas):
    p = F.softmax(logity, 1).numpy()
    yt = torch.from_numpy(y)
    pred = p.argmax(1)
    ok = pred == y
    pew = p.max(1)
    top5 = (torch.topk(logity, 5, 1).indices == yt[:, None]).any(1).numpy()
    ap = [ap_klasy(p[:, c], y == c) for c in range(n_klas)]
    kosze = np.linspace(0, 1, 16)
    ece = 0.0
    for a, b in zip(kosze[:-1], kosze[1:]):
        m = (pew > a) & (pew <= b)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - pew[m].mean())
    return dict(top1=float(ok.mean()), top5=float(top5.mean()), mAP=float(np.nanmean(ap)), AP_klas=[round(v, 4) for v in ap],
                pewnosc_srednia=float(pew.mean()), pewnosc_trafien=float(pew[ok].mean()), pewnosc_pomylek=float(pew[~ok].mean()) if (~ok).any() else None,
                pewnosc_mediana=float(np.median(pew)), udzial_pewnosc_ponad_0_9=float((pew > 0.9).mean()),
                dokladnosc_gdy_pewnosc_ponad_0_9=float(ok[pew > 0.9].mean()) if (pew > 0.9).any() else None,
                ECE=float(ece), NLL=float(F.cross_entropy(logity, yt)),
                dokladnosc_klas=[round(float(ok[y == c].mean()), 4) for c in range(n_klas)])


def main():
    wyniki, dane, biegi = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    t0 = time.time()
    wal = TK.wczytaj_imagenette(dane, 'val')
    tr = TK.wczytaj_imagenette(dane, 'train')
    rap = dict(klasy=wal.klasy, walidacja=len(wal), uczacy=len(tr), opis=__doc__.split('\n')[0], biegi={})
    for b in biegi:
        meta = json.load(open(wyniki / b / 'dane' / 'metadata.json', encoding='utf-8'))
        model = meta['pokretla']['model']
        for st_p in sorted((wyniki / b / 'pelny').glob('stan_e[0-9][0-9].pt')):
            st = torch.load(st_p, map_location='cuda', weights_only=False)
            net = TK.Klasyfikator(model, 10, meta['pokretla'].get('jadro', 5)).cuda()
            net.load_state_dict(st['net'])
            w = dict(model=model, lr=meta['lr'], epoka=int(st_p.stem[-2:]), krok=st['krok'],
                     walidacja=miary(wyjscia(net, wal.X), wal.y, 10), uczacy=miary(wyjscia(net, tr.X), tr.y, 10))
            rap['biegi'][f'{b}_{st_p.stem}'] = w
            v, u = w['walidacja'], w['uczacy']
            print(f"{b} {st_p.stem}: wal top1 {v['top1']:.3f} top5 {v['top5']:.3f} mAP {v['mAP']:.3f} pewność {v['pewnosc_srednia']:.3f} "
                  f"(traf {v['pewnosc_trafien']:.3f}, pomyłki {v['pewnosc_pomylek']:.3f}) ECE {v['ECE']:.3f} | uczący top1 {u['top1']:.3f} "
                  f"pewność {u['pewnosc_srednia']:.3f}", flush=True)
            del net, st
            torch.cuda.empty_cache()
    with open(wyniki / 'OCENA_KLASYFIKACJI.json', 'w', encoding='utf-8') as f:
        json.dump(rap, f, ensure_ascii=False, indent=1)
    print(f'zapisano {wyniki / "OCENA_KLASYFIKACJI.json"} w {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
