#!/usr/bin/env python3
"""Galeria dowodow: detekcje modelu na klatkach sondy w kolejnych krokach uczenia (2026-09-23).

Klatki sondy odtwarzamy lokalnie ta sama funkcja, ktora budowala je na klastrze
(`zbuduj_sonde` z pipeline/koopman_langevin/sonda_treningu.py, to samo ziarno), z obrazow
w archiwach adnotacji Cataract-LMM. Zgodnosc sprawdzamy: identyfikatory klatek i ramki
z etykiet musza byc identyczne z zapisem sondy.

Rysunki:
  GALERIA_sondy.jpg     kilka klatek, dla kazdej wiersz "oryginal" i wiersz "narzedzie na obcym tle",
                        kolumny = kroki uczenia; zolta ramka = detekcja modelu z pewnoscia,
                        zielona cienka = ramka z etykiety
  PEWNOSC_sondy.png     pewnosc kazdej z 32 klatek krok po kroku: oryginal wobec obcego tla,
                        z mAP walidacji

    py -3.11 experiments/krystalizacja_latentu/galeria_sondy.py \\
        --sonda experiments/krystalizacja_latentu/dane/sonda_B1m_s42/sonda_B1m_random_b250_s42_sonda \\
        --telemetria experiments/krystalizacja_latentu/dane/sonda_B1m_s42/sonda_B1m_random_b250_s42/TELEMETRIA.json \\
        --adnotacje experiments/krystalizacja_latentu/dane/annotations_val.json \\
        --zipy ../Dataset/Cataract-LMM/3_Object_Tracking/annotations \\
        --out experiments/krystalizacja_latentu/wyniki/dowody
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KORZEN = Path(__file__).resolve().parents[2]


class LeniwyKatalog:
    """Udaje katalog z images/ i annotations_val.json; obrazy wyjmuje z archiwow na zadanie."""

    def __init__(self, adnotacje: Path, zipy: Path, cache: Path):
        self.adnotacje, self.zipy, self.cache = adnotacje, zipy, cache
        cache.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, nazwa: str) -> Path:
        if nazwa == 'annotations_val.json':
            return self.adnotacje
        plik = self.cache / Path(nazwa).name
        if not plik.exists():
            stem = Path(nazwa).stem                            # TR_0002_S1_P03_00000001
            film, nr = stem.rsplit('_', 1)
            with zipfile.ZipFile(self.zipy / f'{film}.zip') as z:
                plik.write_bytes(z.read(f'{film}/{film}_{int(nr):06d}.jpg'))
        return plik


def czcionka(rozmiar):
    for f in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, rozmiar)
        except OSError:
            pass
    return ImageFont.load_default()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sonda', type=Path, required=True)
    ap.add_argument('--telemetria', type=Path, required=True)
    ap.add_argument('--adnotacje', type=Path, required=True)
    ap.add_argument('--zipy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kroki', type=int, nargs='+', default=[0, 45, 56, 95, -1])
    ap.add_argument('--ile-klatek', type=int, default=6)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spec = importlib.util.spec_from_file_location('sonda_treningu', KORZEN / 'pipeline/koopman_langevin/sonda_treningu.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    katalog = LeniwyKatalog(args.adnotacje, args.zipy, args.out / '_obrazy')
    obrazy, ramki_gt, ids = mod.zbuduj_sonde(katalog, 32, 0)

    d = dict(np.load(args.sonda / 'SONDA.npz'))
    meta = json.loads((args.sonda / 'SONDA_META.json').read_text(encoding='utf-8'))
    assert ids == meta['frame_ids'], 'odtworzone klatki sondy nie zgadzaja sie z zapisem'
    assert np.allclose(ramki_gt, d['ramka_gt'], atol=1e-3), 'ramki z etykiet nie zgadzaja sie z zapisem'
    war = meta['warianty']
    V, n = len(war), len(ids)
    kroki = d['kroki']
    S = len(kroki)
    pew = d['pewnosc'].astype(float).reshape(S, n, V)
    ramka = d['ramka'].reshape(S, n, V, 4)
    io_, it = war.index('orig'), war.index('tlo_a')

    t = json.loads(args.telemetria.read_text(encoding='utf-8'))
    mp = np.array(t['val/mAP_50_95'])
    krok_naj = int(mp[mp[:, 1].argmax(), 0])

    # ---- wykres pewnosci klatka po klatce
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for j in range(n):
        ax.plot(kroki, pew[:, j, io_], color='C0', lw=.4, alpha=.35)
        ax.plot(kroki, pew[:, j, it], color='C3', lw=.4, alpha=.35)
    ax.plot(kroki, np.median(pew[:, :, io_], 1), color='C0', lw=2.5, label='ta sama klatka (mediana 32 klatek)')
    ax.plot(kroki, np.median(pew[:, :, it], 1), color='C3', lw=2.5, label='narzedzie wklejone na obce tlo (mediana)')
    ax.axvline(krok_naj, color='k', ls='--', lw=1, label=f'najlepsze mAP walidacji (krok {krok_naj})')
    bx = ax.twinx()
    bx.plot(mp[:, 0], mp[:, 1], 'ko-', ms=3, lw=1); bx.set_ylabel('mAP 0,5:0,95 na walidacji (czarne)')
    ax.set_xlabel('krok uczenia (16 klatek na krok, 16 krokow na epoke)'); ax.set_ylabel('pewnosc detekcji')
    ax.set_title('Pewnosc detektora: to samo narzedzie we wlasnym i w obcym otoczeniu, krok po kroku')
    ax.legend(loc='center right', fontsize=8)
    fig.tight_layout(); args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'PEWNOSC_sondy.png', dpi=140); plt.close(fig)

    # ---- galeria
    kroki_idx = [(S - 1 if k < 0 else int(np.argmin(np.abs(kroki - k)))) for k in args.kroki]
    s_min = int(np.argmin(np.abs(kroki - 56)))
    spadek = pew[s_min, :, it] - pew[-1, :, it]
    wybor = list(np.argsort(-spadek)[:args.ile_klatek - 2]) + list(np.argsort(spadek)[:2])
    W, H = 300, 200
    f_ = czcionka(15)
    kol = len(kroki_idx)
    plotno = Image.new('RGB', (160 + kol * W, 36 + len(wybor) * 2 * H), 'white')
    dr = ImageDraw.Draw(plotno)
    for c, s in enumerate(kroki_idx):
        opis = f'krok {kroki[s]} (epoka {d["epoki"][s]})' + (' = najlepsze mAP' if abs(kroki[s] - krok_naj) <= 1 else '')
        dr.text((160 + c * W + 6, 8), opis, fill='black', font=f_)
    for r, j in enumerate(wybor):
        for w_i, (w_, etyk) in enumerate(((io_, 'wlasne tlo'), (it, 'obce tlo'))):
            y0 = 36 + (2 * r + w_i) * H
            dr.text((6, y0 + H // 2 - 20), f'klatka {r + 1}\n{etyk}', fill='black', font=f_)
            src = obrazy[j * V + w_]
            for c, s in enumerate(kroki_idx):
                im = src.copy()
                g = ImageDraw.Draw(im)
                bx_, by_, bw_, bh_ = ramki_gt[j * V + w_]
                g.rectangle([bx_, by_, bx_ + bw_, by_ + bh_], outline=(0, 255, 0), width=2)
                px, py, pw_, ph_ = ramka[s, j, w_]
                p = pew[s, j, w_]
                kolor = (255, 220, 0) if p >= .5 else (255, 60, 60)
                g.rectangle([px, py, px + pw_, py + ph_], outline=kolor, width=5)
                im = im.resize((W - 4, H - 4))
                g2 = ImageDraw.Draw(im)
                g2.rectangle([0, 0, 118, 24], fill=(0, 0, 0))
                g2.text((4, 3), f'pewnosc {p:.2f}', fill=kolor, font=f_)
                plotno.paste(im, (160 + c * W + 2, y0 + 2))
    plotno.save(args.out / 'GALERIA_sondy.jpg', quality=90)

    raport = dict(krok_najlepszego_mAP=krok_naj, kroki_galerii=[int(kroki[s]) for s in kroki_idx],
                  klatki_galerii=[ids[j] for j in wybor],
                  pewnosc_mediana={f'krok_{int(kroki[s])}': dict(wlasne=float(np.median(pew[s, :, io_])),
                                                                obce=float(np.median(pew[s, :, it])))
                                   for s in kroki_idx},
                  klatek_z_obcym_tlem_ponizej_0_5_na_koncu=int((pew[-1, :, it] < .5).sum()),
                  klatek_z_obcym_tlem_ponizej_0_5_w_kroku_56=int((pew[s_min, :, it] < .5).sum()),
                  klatek_wlasne_tlo_ponizej_0_5_na_koncu=int((pew[-1, :, io_] < .5).sum()))
    (args.out / 'REPORT_galeria_sondy.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(raport, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
