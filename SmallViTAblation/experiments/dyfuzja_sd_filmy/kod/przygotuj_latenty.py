#!/usr/bin/env python3
"""Klatki filmów → latenty VAE Stable Diffusion i maski układu sceny, jeden plik na film (PLAN.md).

Wejście: archiwa `<film>.zip` (Cataract-LMM 3_Object_Tracking/annotations). Wyjście w `--out`:
  latenty/<film>.npy  (N, 4, 48, 72) float16   średnia rozkładu kodera VAE × 0,18215
  maski/<film>.npy    (N, 3, 48, 72) uint8     narzędzie, rogówka, źrenica: udział pola 0–255
  indeks/<film>.json  nazwy JPEG, rozmiar natywny, co która klatka
Filmy dzielone między procesy torchrun (rank::world). Film z gotowym indeksem jest pomijany (wznowienie).
Zapis atomowy: najpierw pliki .tmp, potem zmiana nazwy; indeks zapisywany na końcu.

    torchrun --nproc_per_node=4 przygotuj_latenty.py --archiwa A --manifest M --out O --podzial train [--co 1]
    python przygotuj_latenty.py ... --limit-filmow 2        (próba lokalna)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import wspolne as W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--archiwa', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--podzial', default='train')
    ap.add_argument('--co', type=int, default=1, help='co która klatka (1 = wszystkie)')
    ap.add_argument('--vae', required=True, help='katalog lub nazwa modelu VAE (diffusers AutoencoderKL)')
    ap.add_argument('--paczka', type=int, default=16, help='klatek naraz w koderze VAE (576×384: ~1 GB karty na klatkę w bf16)')
    ap.add_argument('--watki', type=int, default=8)
    ap.add_argument('--limit-filmow', type=int, default=0)
    a = ap.parse_args()
    rank, world, lrank = W.rozproszenie()
    dev = f'cuda:{lrank}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(dev)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval().requires_grad_(False)
    out = Path(a.out)
    for d in ('latenty', 'maski', 'indeks'):
        (out / d).mkdir(parents=True, exist_ok=True)
    filmy = W.filmy_z_manifestu(a.manifest, a.podzial)[: a.limit_filmow or None]
    moje = filmy[rank::world]
    print(f'[{rank}/{world}] {len(moje)} z {len(filmy)} filmów ({a.podzial}), co {a.co}. klatka, {dev}', flush=True)
    pula = ThreadPoolExecutor(a.watki)
    t_start = time.time()
    for i, film in enumerate(moje):
        if (out / 'indeks' / f'{film}.json').exists():
            print(f'[{rank}] pominięty {film}', flush=True)
            continue
        t0 = time.time()
        with zipfile.ZipFile(Path(a.archiwa) / f'{film}.zip') as z:
            adn = W.adnotacje(z, film)
            nazwy = sorted(n.split('/')[-1] for n in z.namelist() if n.endswith('.jpg'))[:: a.co]
            L = np.empty((len(nazwy), 4, W.LWYS, W.LSZER), np.float16)
            M = np.empty((len(nazwy), len(W.OBIEKTY), W.LWYS, W.LSZER), np.uint8)
            rozm = []
            for c in range(0, len(nazwy), 512):                         # kawałkami: pamięć niezależna od długości filmu
                kaw = nazwy[c:c + 512]
                dek = list(pula.map(W.dekoduj, [z.read(f'{film}/{n}') for n in kaw]))
                rozm += [d[1] for d in dek]
                M[c:c + len(kaw)] = np.stack(list(pula.map(lambda k: W.maski(adn.get(kaw[k]), *dek[k][1]), range(len(kaw)))))
                with torch.no_grad():
                    for s in range(0, len(kaw), a.paczka):
                        x = torch.from_numpy(np.stack([d[0] for d in dek[s:s + a.paczka]])).to(dev)
                        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1)
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.startswith('cuda')):
                            lat = vae.encode(x).latent_dist.mean
                        L[c + s:c + s + len(x)] = (lat.float() * W.SKALA_VAE).cpu().numpy().astype(np.float16)
        for nazwa, arr in (('latenty', L), ('maski', M)):
            tmp = out / nazwa / f'{film}.tmp.npy'
            np.save(tmp, arr); os.replace(tmp, out / nazwa / f'{film}.npy')
        bez_adn = sum(1 for n in nazwy if n not in adn)
        indeks = dict(film=film, podzial=a.podzial, co=a.co, klatki=nazwy, rozmiar_natywny=sorted(set(map(tuple, rozm))),
                      bez_adnotacji=bez_adn, z_narzedziem=int((M[:, 0].reshape(len(M), -1).max(1) > 0).sum()),
                      latent_srednia=float(L.astype(np.float32).mean()), latent_sd=float(L.astype(np.float32).std()))
        tmp = out / 'indeks' / f'{film}.tmp.json'
        tmp.write_text(json.dumps(indeks, ensure_ascii=False), encoding='utf-8'); os.replace(tmp, out / 'indeks' / f'{film}.json')
        dt = time.time() - t0
        print(f'[{rank}] {i + 1}/{len(moje)} {film}: {len(nazwy)} klatek, {len(nazwy) / dt:.0f} kl/s, z narzędziem '
              f'{indeks["z_narzedziem"]}, bez adnotacji {bez_adn}, latent sd {indeks["latent_sd"]:.3f}', flush=True)
    print(f'[{rank}] koniec {time.time() - t_start:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
