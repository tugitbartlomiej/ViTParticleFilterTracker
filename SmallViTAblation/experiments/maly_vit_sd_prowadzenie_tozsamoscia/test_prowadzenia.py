#!/usr/bin/env python3
"""Czy obrazy Stable Diffusion prowadzone latentem tożsamości małego ViT pokrywają przewidziany latent w ≥ 90 procentach
(prośba użytkownika 2026-10-01). Bez pętli: jedna sieć po rozgrzewce, prognoza i obrazy dla kotwic.

1. Mały ViT: rozgrzewka 1000 kroków na bazie 1500 klatek (ziarno, jak w pętli), latent sond co 10 kroków.
2. Wektor tożsamości (LDA na sondach z końca rozgrzewki) i prognoza siły tożsamości na 500 kroków Koopmanem albo
   Langevinem z wersji 3 (`sila.prognozy`, model wybrany testem wstecznym) - metoda użytkownika.
3. Dla każdej kotwicy (sonda `original` z układem masek): latent narzędzia prawdziwej klatki z_a i cele
     prognoza: z* = z_a + Δs·(sd ⊙ w)  (przewidziany latent: kotwica przesunięta wzdłuż tożsamości o przewidzianą zmianę),
     +1 / −1:   z* = z_a ± 1·(sd ⊙ w)  (sprawdzenie sterowności przy dużej zmianie).
   SD bez prowadzenia i z prowadzeniem (to samo ziarno, ten sam układ masek).
4. Pokrycie na końcowym obrazie uint8 (wycinek jak w pętli):
     tożsamość  = 100·(1 − |s − s*| / |s*|)                       (siła liczona od poziomu tła, s = 0)
     latent     = 100·(1 − ‖ẑ − ẑ*‖² / ‖ẑ*‖²)                     (ẑ = (z − μ_tła)/sd, 768 liczb)
     domknięcie luki = część odległości obrazu bez prowadzenia od celu, którą prowadzenie usunęło (s i ẑ osobno).
   Do tego zmiana pikseli wobec obrazu bez prowadzenia i wspólna część list wyszukiwania DINOv2 (top-90).

    uruchom_lokalnie.cmd test_prowadzenia.py [--ziarno 47] [--kotwic 11] [--eta 0.6] [--iteracje 3]
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'kod'))
import dane as D  # noqa: E402
import prognoza as PR  # noqa: E402
import prowadzenie as PW  # noqa: E402
import rownanie as RW  # noqa: E402
import sila as SI  # noqa: E402
import trening as T  # noqa: E402
import uklady as UK  # noqa: E402
import wybor as WB  # noqa: E402

H_BLOK = 10            # 10 bloków po 50 kroków = 500 kroków, jak obrót pętli


def manifest_pelny():
    out = {}
    for l in open(D.DATA_ROOT / 'pool_manifest.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            out[r['frame_id']] = dict(frame_id=r['frame_id'], path=r['path'], native_width=int(r.get('native_width', 720)),
                                      native_height=int(r.get('native_height', 480)))
    return out


@torch.no_grad()
def latent_obrazu(u, img576, uk):
    """Latent narzędzia (768,) końcowego obrazu uint8 przez wycinek jak w pętli (cv2 INTER_AREA), fp32."""
    w, m = UK.wycinek_176(img576, uk)
    x = torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)[None].float().to(u.dev) / 255.
    a = torch.from_numpy(m.reshape(11, 16, 11, 16).mean((1, 3)).reshape(-1) >= .25).float().to(u.dev)
    net = u.net; net.eval(); net.zapisuj_latent(True)
    net(x)
    lat = net._latent.float()[0]
    net.zapisuj_latent(False)
    return ((lat * a[:, None]).sum(0) / a.sum().clamp(min=1)).cpu().numpy().astype(np.float64)


def podpis(im, t):
    im = np.ascontiguousarray(im)
    cv2.putText(im, t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(im, t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ziarno', type=int, default=47)
    ap.add_argument('--kotwic', type=int, default=11)
    ap.add_argument('--eta', type=float, default=0.15)
    ap.add_argument('--iteracje', type=int, default=2)
    ap.add_argument('--sila-startu', type=float, default=0.5, help='1 = start z czystego szumu, <1 = z zaszumionej kotwicy')
    ap.add_argument('--waga-latentu', type=float, default=1.0)
    ap.add_argument('--koniec', type=int, default=5)
    ap.add_argument('--baza', default='kadr', help='kadr = latent klatki kotwicy tym samym wycinkiem co obraz SD; sonda = latent sondy')
    ap.add_argument('--cele', default='prognoza,+1,-1')
    ap.add_argument('--tag', default='')
    a = ap.parse_args(argv)
    s = a.ziarno
    out = Path(os.environ['MALY_VIT_WYNIKI']); out.mkdir(parents=True, exist_ok=True)
    tag = f's{s}_start{a.sila_startu}_eta{a.eta}_it{a.iteracje}_wl{a.waga_latentu}_k{a.koniec}_{a.baza}{a.tag}'
    kd = out / f'test_{tag}'; (kd / 'zapytania').mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sondy = D.sondy(); obecn, opis = sondy[4], sondy[5]
    baza = D.baza()
    # ---------- 1. rozgrzewka (zapisana, żeby kolejne próby prowadzenia miały tę samą sieć)
    u = T.Uczen(s)
    st, lf = out / f'rozgrzewka_s{s}.pt', out / f'rozgrzewka_s{s}_latent.npz'
    if st.exists() and lf.exists():
        u.wczytaj(torch.load(st, weights_only=False)); L = np.load(lf)['latent']
    else:
        rng = np.random.default_rng(s)
        lat = [(0, u.latent_sond(sondy))]
        u.kroki(baza, 1000, rng, sondy=sondy, latent_out=lat)
        L = np.stack([l for _, l in lat])
        torch.save(u.stan(), st); np.savez_compressed(lf, latent=L)
    print(f's{s}: rozgrzewka gotowa ({time.time() - t0:.0f} s), krok {u.krok}', flush=True)
    # ---------- 2. wektor tożsamości i prognoza Koopman/Langevin (wersja 3)
    wek = SI.wektor(L[-5:].astype(np.float32).mean(0), obecn)
    S = SI.sila(L[:, obecn, 0], wek)
    pp = SI.prognozy(PR.bloki(S[:, :, None]), H_BLOK)
    dS = pp['prognozy'][pp['model']] - pp['teraz']                       # (180,) przewidziana zmiana na 500 kroków
    print(f'prognoza: model {pp["model"]}, test wsteczny {pp["test_wsteczny"]}, zmiana średnia {dS.mean():+.3f}, '
          f'sd {dS.std():.3f}', flush=True)
    kier = wek['w'] * wek['sd']                                           # przesunięcie z o 1 w sile tożsamości
    Z = RW.latent_narzedzia_sond(u, sondy).astype(np.float64)             # fp32 latent narzędzia 192 sond
    # ---------- 3. kotwice
    pelny = manifest_pelny()
    poz = np.where(obecn)[0]
    kotw = []
    for j, i in enumerate(poz):
        if opis[i]['wariant'] == 'original':
            r = pelny.get(opis[i]['frame_id'])
            uk = UK.uklad(r) if r else None
            if uk is not None:
                im = cv2.cvtColor(cv2.imread(str(D.DATA_ROOT / r['path'])), cv2.COLOR_BGR2RGB)
                kotw.append(dict(j=j, i=int(i), uk=uk, obraz=cv2.resize(im, (UK.SD_SZER, UK.SD_WYS), interpolation=cv2.INTER_AREA)))
    kotw = kotw[:a.kotwic]
    G = PW.Prowadzony()
    wykl = D.wykluczone_bazowe(baza.wiersze)
    ns = lambda z: (z - wek['mu']) / wek['sd']                            # noqa: E731
    sila = lambda z: float(ns(z) @ wek['w'])                              # noqa: E731
    wyn, kafle = [], []
    for b, k in enumerate(kotw):
        z_real = latent_obrazu(u, k['obraz'], k['uk'])
        z_a = z_real if a.baza == 'kadr' else Z[k['i']]
        maski = UK.maski(k['uk'])
        ziarno = 100 * s + b
        kw = dict(start=None if a.sila_startu >= 1 else k['obraz'], sila_startu=a.sila_startu, eta=a.eta, iteracje=a.iteracje,
                  koniec_prowadzenia=a.koniec, waga_latentu=a.waga_latentu)
        img_bez = G.rysuj_prowadzony(u.net, maski, k['uk'], Z[k['i']], wek, ziarno=ziarno, prowadz=False, **kw)[0]
        z_bez = latent_obrazu(u, img_bez, k['uk'])
        cele = {}
        for c in a.cele.split(','):
            d = float(dS[k['j']]) if c == 'prognoza' else float(c)
            cele[c] = (d, z_a + d * kier)
        rzad = [podpis(UK.wycinek_176(k['obraz'], k['uk'])[0], f'kotwica s={sila(z_real):.2f}'),
                podpis(UK.wycinek_176(img_bez, k['uk'])[0], f'SD bez s={sila(z_bez):.2f}')]
        pz = kd / 'zapytania'
        Image.fromarray(UK.wycinek_zapytania(img_bez, k['uk'])).save(pz / f'k{b}_bez.png')
        lista_bez = {c['frame_id'] for c in WB.zapytaj(pz / f'k{b}_bez.png', wykl, pz / f'k{b}_bez.json')}
        for c, (d, z_cel) in cele.items():
            tc = time.time()
            img, przebieg = G.rysuj_prowadzony(u.net, maski, k['uk'], z_cel, wek, ziarno=ziarno, **kw)
            z_img = latent_obrazu(u, img, k['uk'])
            s_c, s_i, s_b = sila(z_cel), sila(z_img), sila(z_bez)
            e_i, e_b, n_c = (float(((ns(z_img) - ns(z_cel)) ** 2).sum()), float(((ns(z_bez) - ns(z_cel)) ** 2).sum()),
                             float((ns(z_cel) ** 2).sum()))
            Image.fromarray(UK.wycinek_zapytania(img, k['uk'])).save(pz / f'k{b}_{c}.png')
            lista = {x['frame_id'] for x in WB.zapytaj(pz / f'k{b}_{c}.png', wykl, pz / f'k{b}_{c}.json')}
            r = dict(kotwica=b, sonda=k['i'], frame_id=k['uk']['frame_id'], cel=c, zmiana_celu=d,
                     s_kotwicy=sila(z_a), s_kotwicy_obraz=sila(z_real), s_cel=s_c, s_bez=s_b, s_prowadzony=s_i,
                     pokrycie_tozsamosci=100 * (1 - abs(s_i - s_c) / abs(s_c)),
                     pokrycie_tozsamosci_bez=100 * (1 - abs(s_b - s_c) / abs(s_c)),
                     pokrycie_latentu=100 * (1 - e_i / n_c), pokrycie_latentu_bez=100 * (1 - e_b / n_c),
                     domkniecie_luki_s=100 * (1 - abs(s_i - s_c) / max(abs(s_b - s_c), 1e-9)),
                     domkniecie_luki_z=100 * (1 - e_i / max(e_b, 1e-9)),
                     zrealizowana_zmiana=(s_i - s_b) / (s_c - s_b) if abs(s_c - s_b) > 1e-9 else None,
                     zmiana_pikseli=float(np.abs(img.astype(np.float32) - img_bez.astype(np.float32)).mean()),
                     wspolne_wyszukanie=len(lista & lista_bez) / max(len(lista | lista_bez), 1),
                     energia_start=przebieg[0]['E'] if przebieg else None, energia_koniec=przebieg[-1]['E'] if przebieg else None,
                     sekundy=time.time() - tc)
            wyn.append(r)
            rzad.append(podpis(UK.wycinek_176(img, k['uk'])[0], f'{c}: s={s_i:.2f} cel {s_c:.2f}'))
            print(f'kotwica {b} {c:8s}: s bez {s_b:.2f} -> {s_i:.2f} (cel {s_c:.2f}); pokrycie tożsamości '
                  f'{r["pokrycie_tozsamosci"]:.1f} % (bez {r["pokrycie_tozsamosci_bez"]:.1f}), latentu {r["pokrycie_latentu"]:.1f} % '
                  f'(bez {r["pokrycie_latentu_bez"]:.1f}); piksele {r["zmiana_pikseli"]:.1f}; wspólne wyszukanie '
                  f'{r["wspolne_wyszukanie"]:.2f} ({r["sekundy"]:.0f} s)', flush=True)
        kafle.append(np.concatenate(rzad, 1))
        cv2.imwrite(str(kd / 'MOZAIKA.jpg'), cv2.cvtColor(np.concatenate(kafle, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        (kd / 'WYNIKI.json').write_text(json.dumps(dict(ziarno=s, ustawienia=vars(a), model_prognozy=pp['model'],
                                                        wyniki=wyn), ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    for c in a.cele.split(','):
        rr = [r for r in wyn if r['cel'] == c]
        f = lambda k: float(np.median([r[k] for r in rr]))                 # noqa: E731
        print(f'{c:8s} mediana: pokrycie tożsamości {f("pokrycie_tozsamosci"):.1f} % (bez {f("pokrycie_tozsamosci_bez"):.1f}), '
              f'latentu {f("pokrycie_latentu"):.1f} % (bez {f("pokrycie_latentu_bez"):.1f}), domknięcie luki s '
              f'{f("domkniecie_luki_s"):.0f} %, z {f("domkniecie_luki_z"):.0f} %; ≥ 90 % obu: '
              f'{sum(r["pokrycie_tozsamosci"] >= 90 and r["pokrycie_latentu"] >= 90 for r in rr)}/{len(rr)}', flush=True)
    print(f'gotowe ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
