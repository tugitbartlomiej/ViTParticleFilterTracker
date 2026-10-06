"""Latent samego położenia patcha: średni latent każdego z 600 tokenów po wielu różnych klatkach.

Na 3000 klatkach walidacji treść obrazu w danym miejscu się zmienia, a położenie patcha (kodowanie pozycji, łodyga
konwolucyjna z brzegiem obrazu, typowy układ sceny) zostaje. Średnia po klatkach M_i = E[z_i] dla tokenu i to więc
oszacowanie wkładu położenia; g = średnia po wszystkich tokenach i klatkach. Uwaga: M_i zawiera też to, co w zbiorze
zwykle leży w tym miejscu (np. źrenica w środku kadru), nie tylko czystą pozycję.

Trzy sposoby odjęcia (przestrzeń makrowektorów):
- średnia sekwencji: z_ti − M^seq_i + g^seq (M^seq z klatek bieżącej sekwencji; usuwa też nieruchome tło),
- średnia 3000 klatek: z_ti − M_i + g,
- rzut: z − ((z − g) U) Uᵀ, U = r głównych kierunków rozrzutu M_i (r tak, by objęły 95 % wariancji położenia);
  usuwa z każdego tokenu tylko kierunki, w których różnią się miejsca, resztę zostawia.
Wynik liczony raz na checkpoint i zapisany w outputs/latent_editor_sd_20261006/POLOZENIE_<checkpoint>.npz.
"""
from pathlib import Path

import numpy as np
import torch

from wspolne import WYNIKI

WAL = Path(__file__).resolve().parents[1] / 'experiments' / 'maly_vit_od_zera' / 'cache' / 'wal3000.npy'
PROG_RZUTU = 0.95


def _plik(ckpt):
    p = Path(ckpt)
    return WYNIKI / f'POLOZENIE_{p.parent.name}_{p.stem}_{p.stat().st_size}.npz'


def srednia_pozycji(ckpt, det=None, postep=None):
    """Słownik: M [600, 768], g [768], U [768, r], udzial (wariancja między miejscami / całkowita), r, n."""
    plik = _plik(ckpt)
    if plik.exists():
        d = np.load(plik)
        return {k: d[k] for k in d.files}
    if det is None:
        import detekcja as D
        det = D.Detektor(ckpt)
    X = np.load(WAL, mmap_mode='r')
    blok, pam = det.net.bloki[-1], {}
    h = blok.mlp.act.register_forward_hook(lambda m, a, o: pam.update(z=o.detach()))
    suma = torch.zeros(600, 768, dtype=torch.float64, device=det.dev)
    kw = torch.zeros((), dtype=torch.float64, device=det.dev)
    n = len(X)
    try:
        with torch.no_grad():
            for i in range(0, n, 50):
                x = torch.from_numpy(np.array(X[i:i + 50])).to(det.dev).permute(0, 3, 1, 2).float() / 255
                det.net(x)
                z = pam['z'].double()                                       # [B, 600, 768]
                suma += z.sum(0)
                kw += (z ** 2).sum()
                if postep:
                    postep(min(i + 50, n), n)
    finally:
        h.remove()
    M = (suma / n).float().cpu().numpy()
    g = M.mean(0)
    calk = float(kw.cpu()) / (n * 600) - float((g.astype(np.float64) ** 2).sum())   # wariancja całkowita na token
    miedzy = float(((M - g) ** 2).sum(1).mean())                                     # wariancja między miejscami
    _, S, Vt = np.linalg.svd(M - g, full_matrices=False)
    udz = S ** 2 / (S ** 2).sum()
    r = int(np.searchsorted(np.cumsum(udz), PROG_RZUTU) + 1)
    wynik = dict(M=M.astype(np.float32), g=g.astype(np.float32), U=Vt[:r].T.astype(np.float32),
                 udzial=np.float32(miedzy / calk), r=np.int32(r), n=np.int32(n), udz_kierunkow=udz[:64].astype(np.float32))
    plik.parent.mkdir(parents=True, exist_ok=True)
    np.savez(plik, **wynik)
    return wynik


if __name__ == '__main__':
    import sys, time
    import silnik
    t = time.perf_counter()
    w = srednia_pozycji(sys.argv[1] if len(sys.argv) > 1 else silnik.VIT_CHECKPOINT)
    print(f'{time.perf_counter() - t:.1f} s; klatek {int(w["n"])}; udział położenia w wariancji latentu '
          f'{100 * float(w["udzial"]):.1f} %; kierunków na 95 % wariancji położenia: {int(w["r"])}; '
          f'pierwsze 5 kierunków: {np.round(100 * w["udz_kierunkow"][:5], 1)} %')
