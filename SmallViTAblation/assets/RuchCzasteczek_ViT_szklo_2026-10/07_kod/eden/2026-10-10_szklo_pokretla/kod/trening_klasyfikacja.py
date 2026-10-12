"""DRUGI ZBIÓR I ZADANIE (2026-10-11): klasyfikacja Imagenette (10 klas ImageNet, 320 px, 9469 obrazów uczących, 3925 walidacyjnych)
tym samym małym ViT (łodyga + 6 bloków timm, dim 192, MLP 768) albo MalyCNN; głowa: średnia tokenów → Linear(10), strata CE.
Cząstki = 400 tokenów (siatka 20×20, krok 16 px) × 768 aktywacji GELU MLP ostatniego bloku, obserwator z losowym rzutem 128D
(obserwator_rp.py). Przepis jak w biegu 35 000: paczka 32, AdamW 5e-4 wd 0,05 (albo SGD), rozgrzewka 300, stały lr, bf16,
odbicie 0,5, jasność/kontrast ±20 %, obcięcie gradientu 1. Obrazy: środkowe wycięcie 320×320 (krótszy bok już 320), uint8 w RAM.
Zapis jak w trening_pokretla.py: dane/OCENY.json (acc top-1; klucz mAP50 = acc dla zgodności rysunków), dane/LOG.json,
dane/metadata.json, pelny/stan_eNN.pt (2 ostatnie, do --wznow i gęstego okna --gesto_od), wagi k0 i ostatni, gesty/.
Uruchomienie: python trening_klasyfikacja.py --wyniki OUT --dane .../imagenette2-320 --epoki 8 --lr 5e-4 --obserwuj --sondy_n 24 --rzut rp"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as M
import model_cnn as MC
import trening as T
import obserwator_gesty as OG
from obserwator_rp import ObserwatorRP

ROZMIAR = 320
MIN_GESTE_GB = 12.0
MIN_STOP_GB = 4.0


def zj(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)


def sha_pliku(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wolne_gb(sciezka):
    try:
        w = subprocess.run(['lfs', 'quota', '-p', '1299', '/mnt/evafs'], capture_output=True, text=True, timeout=30).stdout
        for linia in w.splitlines():
            pola = linia.split()
            if pola and pola[0] == '/mnt/evafs' and len(pola) >= 3:
                uzyte, kwota = int(pola[1].rstrip('*')), int(pola[2])
                return (kwota - uzyte) / 1024 ** 2
    except Exception:
        pass
    return shutil.disk_usage(sciezka).free / 1e9


# ------------------------------------------------------------------ dane
class ZbiorKlas:
    def __init__(self, X, y, klasy):
        self.X, self.y, self.klasy = X, y, klasy

    def __len__(self):
        return len(self.y)


def wczytaj_imagenette(root, split, watki=16, limit=None):
    root = Path(root) / split
    klasy = sorted(p.name for p in root.iterdir() if p.is_dir())
    pliki = [(p, c) for c, k in enumerate(klasy) for p in sorted((root / k).glob('*.JPEG'))]
    if limit:
        pliki = [pliki[i] for i in np.random.default_rng(0).choice(len(pliki), limit, replace=False)]

    def jeden(pc):
        im = cv2.imread(str(pc[0]), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(pc[0])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        s = ROZMIAR / min(h, w)
        if abs(s - 1) > 1e-6:
            im = cv2.resize(im, (max(ROZMIAR, round(w * s)), max(ROZMIAR, round(h * s))), interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
        y0, x0 = (h - ROZMIAR) // 2, (w - ROZMIAR) // 2
        return im[y0:y0 + ROZMIAR, x0:x0 + ROZMIAR]

    t = time.time()
    with ThreadPoolExecutor(watki) as ex:
        X = np.stack(list(ex.map(jeden, pliki)))
    y = np.array([c for _, c in pliki], np.int64)
    print(f'{split}: {len(y)} obrazów {X.shape[1]}×{X.shape[2]}, {len(klasy)} klas, w {time.time() - t:.0f} s', flush=True)
    return ZbiorKlas(X, y, klasy)


# ------------------------------------------------------------------ model i uczeń
class Klasyfikator(nn.Module):
    """Szkielet MalyViT/MalyCNN (łodyga, bloki, norm; głowa gęsta zastąpiona Identity) + głowa klasyfikacji na średniej tokenów."""

    def __init__(self, model='vit', n_klas=10, jadro=5, dim=192):
        super().__init__()
        self.rodzaj = model
        self.szkielet = M.MalyViT(dim=dim) if model == 'vit' else MC.MalyCNN(dim=dim, jadro=jadro)
        self.szkielet.glowa = nn.Identity()
        self.glowa = nn.Linear(dim, n_klas)

    @property
    def bloki(self):
        return self.szkielet.bloki

    def zapisuj_latent(self, tak):
        self.szkielet.zapisuj_latent(tak)

    def forward(self, x):
        s = self.szkielet
        x = (x - M.MEAN.to(x.device, x.dtype)) / M.STD.to(x.device, x.dtype)
        f = s.lodyga(x)
        B, C, h, w = f.shape
        t = f.flatten(2).transpose(1, 2)
        if self.rodzaj == 'vit':
            t = t + M.sincos_2d(h, w, C, x.device).to(f.dtype)
            for b in s.bloki:
                t = b(t)
        else:
            for b in s.bloki:
                t = b(t, h, w)
        return self.glowa(s.norm(t).mean(1)), s._latent


class UczenKlas(T.Uczen):
    def __init__(self, ziarno, a, device='cuda'):
        torch.manual_seed(ziarno)
        self.dev = device
        self.net = Klasyfikator(a.model, 10, a.jadro).to(device)
        dec = [p for n, p in self.net.named_parameters() if p.ndim >= 2]
        nodec = [p for n, p in self.net.named_parameters() if p.ndim < 2]
        if a.opt == 'adamw':
            wd = T.WD if a.wd is None else a.wd
            self.opt = torch.optim.AdamW([dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0)], lr=T.LR, betas=(0.9, 0.999))
        else:
            wd = 0.0 if a.wd is None else a.wd
            self.opt = torch.optim.SGD([dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0)], lr=T.LR, momentum=a.momentum)
        self.krok = 0
        self.log = []

    def kroki(self, zb, n, rng, co_ocena=None, ocena=None, obserwator=None, **_):
        self.net.train()
        kol = []
        t0 = time.time()
        for _j in range(n):
            if len(kol) < T.PACZKA:
                kol += rng.permutation(len(zb)).tolist()
            idx = np.array(kol[:T.PACZKA]); kol = kol[T.PACZKA:]
            odb = rng.random(T.PACZKA) < .5
            x = torch.from_numpy(zb.X[idx]).to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.
            ob = torch.from_numpy(odb).to(self.dev)
            x = torch.where(ob[:, None, None, None], x.flip(3), x)
            a_ = torch.from_numpy(rng.uniform(.8, 1.2, (T.PACZKA, 2)).astype(np.float32)).to(self.dev)
            m = x.mean((1, 2, 3), keepdim=True)
            x = ((x - m) * a_[:, 0, None, None, None] + m * a_[:, 1, None, None, None]).clamp(0, 1)
            y = torch.from_numpy(zb.y[idx]).to(self.dev)
            for g in self.opt.param_groups:
                g['lr'] = self._lr()
            patrz = obserwator is not None and obserwator.czy(_j)
            if patrz:
                obserwator.przed(self, x, y, None, None)
                self.net.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, _ = self.net(x)
            L = F.cross_entropy(o.float(), y)
            self.opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.krok += 1
            if patrz:
                obserwator.po(self)
                self.net.train()
            if self.krok % 50 == 0:
                self.log.append(dict(krok=self.krok, strata=float(L), s_na_krok=(time.time() - t0) / 50))
                t0 = time.time()
            if co_ocena and ocena is not None and self.krok % co_ocena == 0:
                ocena(self)
                self.net.train()

    @torch.no_grad()
    def dokladnosc(self, X, y, paczka=128):
        self.net.eval()
        ok = 0
        for s in range(0, len(X), paczka):
            x = torch.from_numpy(X[s:s + paczka]).to(self.dev).permute(0, 3, 1, 2).float() / 255.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, _ = self.net(x)
            ok += int((o.argmax(1).cpu().numpy() == y[s:s + paczka]).sum())
        return ok / len(X)


class ObserwatorKlas(ObserwatorRP):
    """Jak ObserwatorRP, ale strata paczki (przed/po kroku) to entropia krzyżowa; przed(u, x, y, None, None)."""

    @torch.no_grad()
    def _strata(self, net, x, y, ltrb, poz):
        net.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        return float(F.cross_entropy(o.float(), y)), 0.0, 0.0


# ------------------------------------------------------------------ główna pętla
def main():
    a = argparse.ArgumentParser()
    a.add_argument('--wyniki', required=True)
    a.add_argument('--dane', required=True, help='katalog imagenette2-320 (train/, val/)')
    a.add_argument('--epoki', type=int, default=8)
    a.add_argument('--proba', action='store_true', help='512 obrazów, 2 epoki, walidacja 200: sprawdzenie kodu')
    a.add_argument('--wznow', action='store_true')
    a.add_argument('--obserwuj', action='store_true')
    a.add_argument('--sondy_n', type=int, default=24)
    a.add_argument('--pelne', type=int, default=1)
    a.add_argument('--lr', type=float, default=5e-4)
    a.add_argument('--gesto_do', type=int, default=2188)
    a.add_argument('--co_pozniej', type=int, default=10)
    a.add_argument('--pelne_do', type=int, default=100)
    a.add_argument('--pelne_co', type=int, default=1000)
    a.add_argument('--gesto_od', type=int, default=10**9)
    a.add_argument('--ocena_co', type=int, default=500)
    a.add_argument('--ziarno', type=int, default=47)
    a.add_argument('--min_start_gb', type=float, default=8.0)
    a.add_argument('--rzut', default='rp', choices=['rp'])
    a.add_argument('--rzut_dim', type=int, default=128)
    a.add_argument('--zapis', default='f16', choices=['f16', 'f32'])
    a.add_argument('--warstwy', action='store_true')
    a.add_argument('--warstwy_n', type=int, default=6)
    a.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'])
    a.add_argument('--momentum', type=float, default=0.9)
    a.add_argument('--wd', type=float, default=None)
    a.add_argument('--paczka', type=int, default=32)
    a.add_argument('--model', default='vit', choices=['vit', 'cnn'])
    a.add_argument('--jadro', type=int, default=5)
    a = a.parse_args()
    T.LR = a.lr; T.PACZKA = a.paczka
    out = Path(a.wyniki)
    for d in ('wagi', 'pelny', 'dane'):
        (out / d).mkdir(parents=True, exist_ok=True)
    wolne = wolne_gb(out)
    print(f'wolna kwota: {wolne:.1f} GB', flush=True)
    if wolne < a.min_start_gb:
        raise SystemExit(f'za mało miejsca na start: {wolne:.1f} GB')
    zb = wczytaj_imagenette(a.dane, 'train', limit=512 if a.proba else None)
    wal = wczytaj_imagenette(a.dane, 'val', limit=200 if a.proba else None)
    epoki = 2 if a.proba else a.epoki
    na_epoke = math.ceil(len(zb) / T.PACZKA)
    konce = [e * na_epoke for e in range(1, epoki + 1)]
    lacznie = konce[-1]
    co_ocena = 20 if a.proba else a.ocena_co
    print(f'{len(zb)} obrazów, {epoki} epok, {lacznie} kroków; końce epok: {konce}', flush=True)
    u = UczenKlas(a.ziarno, a); rng = np.random.default_rng(a.ziarno)
    wd_uzyte = float(u.opt.param_groups[0]['weight_decay']); n_param = int(sum(p.numel() for p in u.net.parameters()))
    print(f'model {a.model} ({n_param} parametrów), {a.opt} lr {a.lr} wd {wd_uzyte}, paczka {T.PACZKA}', flush=True)
    oceny = []
    meta = dict(opis='Klasyfikacja Imagenette (10 klas, 320 px) małym ViT/CNN od zera; cząstki = 400 tokenów × 768 GELU MLP ostatniego bloku.',
                zadanie='klasyfikacja', zbior='imagenette2-320', przepis=f'paczka {a.paczka}, {a.opt} lr {a.lr} wd {wd_uzyte}, rozgrzewka 300, stały lr, bf16, '
                'odbicie 0.5, jasność/kontrast ±20 %, obcięcie gradientu 1; środkowe wycięcie 320×320',
                klatki=len(zb), klasy=zb.klasy, epoki=epoki, krokow=lacznie, konce_epok=konce, walidacja=len(wal), proba=a.proba, tokenow=(ROZMIAR // 16) ** 2,
                kod_sha256={p.name: sha_pliku(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                lr=a.lr, pokretla=dict(lr=a.lr, ziarno=a.ziarno, sondy_n=a.sondy_n, gesto_do=a.gesto_do, gesto_od=a.gesto_od, co_pozniej=a.co_pozniej, bez_wag=True,
                              baza_pca32_z_kroku=('wznowienie' if a.wznow else 0), rzut=a.rzut, rzut_dim=a.rzut_dim, opt=a.opt, momentum=a.momentum, wd=wd_uzyte,
                              paczka=a.paczka, model=a.model, jadro=a.jadro, parametrow=n_param, zapis=a.zapis, warstwy=a.warstwy, warstwy_n=a.warstwy_n),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, start=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.wznow:
        stany = sorted((out / 'pelny').glob('stan_e[0-9][0-9].pt'))
        if not stany:
            raise SystemExit('brak stanu do wznowienia')
        st = torch.load(stany[-1], map_location='cuda', weights_only=False)
        stary_meta = out / 'dane' / 'metadata.json'
        if stary_meta.exists():
            shutil.copy2(stary_meta, out / 'dane' / f'metadata_do_kroku_{st["krok"]:05d}.json')
        meta['wznowiono_od'] = dict(plik=stany[-1].name, krok=st['krok'])
        u.wczytaj(st); rng.bit_generator.state = st['rng']; torch.set_rng_state(st['torch_rng'].cpu())
        oceny = st.get('oceny', []); u.log = st.get('log', [])
        print(f'wznowienie od kroku {u.krok} ({stany[-1].name})', flush=True)

    def ocen(uu, epoka=None):
        acc = uu.dokladnosc(wal.X, wal.y)
        oceny.append(dict(krok=uu.krok, epoka=epoka, acc=float(acc), mAP50=float(acc), miara='top1'))
        zj(out / 'dane' / 'OCENY.json', oceny)
        print(f'krok {uu.krok}' + (f' (koniec epoki {epoka})' if epoka else '') + f': acc {acc:.4f}', flush=True)

    def zapisz_wagi(uu):
        p = out / 'wagi' / f'k{uu.krok:05d}.pt'
        if not p.exists():
            torch.save(dict(net={k: v.detach().cpu().clone() for k, v in uu.net.state_dict().items()}, krok=uu.krok), p)

    stop = {'teraz': False}

    def po_kroku(uu):
        t = uu.krok
        epoka = konce.index(t) + 1 if t in konce else None
        if epoka:
            if wolne_gb(out) < MIN_STOP_GB:
                print('KWOTA: za mało miejsca - zapisuję stan i kończę', flush=True); stop['teraz'] = True
            ocen(uu, epoka)
            st = dict(uu.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=uu.log)
            torch.save(st, out / 'pelny' / f'stan_e{epoka:02d}.pt')
            for stary in sorted((out / 'pelny').glob('stan_e*.pt'))[:-2]:
                stary.unlink()
            zj(out / 'dane' / 'LOG.json', uu.log)
            print(f'epoka {epoka}/{epoki}: krok {t}, wolna kwota {wolne_gb(out):.1f} GB, {time.strftime("%H:%M:%S")}', flush=True)
            if t == lacznie:
                zapisz_wagi(uu)
        elif t % co_ocena == 0:
            ocen(uu)
        if stop['teraz']:
            raise KeyboardInterrupt

    obs = None
    if a.obserwuj:
        if a.proba:
            wybrane = list(range(16)); pelne = wybrane[:4]
            kw = dict(gesto_do=20, co_pozniej=2, pelne_do=10, pelne_co=4, porcja=8, porcja_pelne=4, wolne_gb=lambda: wolne_gb(out))
        else:
            wybrane = [int(i) for i in np.linspace(0, len(zb) - 1, max(a.sondy_n, 1)).round()]
            pelne = wybrane[:a.pelne]
            kw = dict(gesto_do=a.gesto_do, co_pozniej=a.co_pozniej, pelne_do=a.pelne_do, pelne_co=a.pelne_co, wolne_gb=lambda: wolne_gb(out))
        obs = ObserwatorKlas(out / 'gesty', zb, wybrane, pelne, rzut_dim=a.rzut_dim, zapis=a.zapis, warstwy=a.warstwy, warstwy_n=a.warstwy_n,
                             tokenow=(ROZMIAR // 16) ** 2, **kw)
        obs.start(u)
        if u.krok:
            obs.buf['kroki'][0] = u.krok; obs.buf_p['kroki'][0] = u.krok
            if obs.buf_w is not None:
                obs.buf_w['kroki'][0] = u.krok
        obs._czy_pca = lambda k, o=obs: k <= o.gesto_do or k >= a.gesto_od or k % o.co_pozniej == 0
        meta['obserwator'] = dict(sondy=len(obs.sondy), pelne=len(obs.pelne), gesto_do=obs.gesto_do, co_pozniej=obs.co_pozniej, pelne_do=obs.pelne_do, pelne_co=obs.pelne_co)
    if u.krok == 0:
        zapisz_wagi(u); ocen(u)
    zj(out / 'dane' / 'metadata.json', meta)
    t0 = time.time()
    try:
        u.kroki(zb, lacznie - u.krok, rng, co_ocena=1, ocena=po_kroku, obserwator=obs)
    except KeyboardInterrupt:
        st = dict(u.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=u.log)
        torch.save(st, out / 'pelny' / f'stan_e99_przerwane_k{u.krok}.pt')
    if obs is not None:
        meta['obserwator'].update(obs.zakoncz())
    meta.update(koniec=time.strftime('%Y-%m-%d %H:%M:%S'), czas_s=round(time.time() - t0), ostatni_krok=u.krok, wolna_kwota_gb=round(wolne_gb(out), 1))
    zj(out / 'dane' / 'metadata.json', meta); zj(out / 'dane' / 'LOG.json', u.log)
    print(f'KONIEC: krok {u.krok}, {meta["czas_s"]} s', flush=True)


if __name__ == '__main__':
    main()
