"""DRUGI MODEL I ZBIÓR (2026-10-13): DeiT-Tiny na CIFAR-10 od zera, z tym samym zapisem co bieg s47 MałegoViT,
żeby powtórzyć analizy laserów, pułapek i koloidów na innym modelu i innych danych (zarzut recenzenta: jeden bieg, jeden model).

Przepis jak s47 (porównywalność): paczka 32, AdamW 5e-4, wd 0,05 na tensorach o wymiarze ≥ 2, rozgrzewka 300 kroków, potem
stały lr, bf16, odbicie 0,5, jasność/kontrast ±20 %, obcięcie normy gradientu `--obciecie` (1,0 jak s47; 0 = bez obcięcia,
druga próba: czy klatki/starzenie nie są artefaktem obcięcia, które w s47 działało w 100 % kroków). Norma gradientu przed
obcięciem jest logowana co krok (dane/NORMY_GRAD.npz).

Zapis (`--wyniki OUT`):
  wagi/k<krok>.pt        same wagi: co 10 kroków do 1000, potem co 100 (jak s47) i na końcu każdej epoki
  pelny/stan_eNN.pt      pełny stan (wagi + AdamW: pierwszy i drugi moment) co `--stan_co` epok i 2 ostatnie (do --wznow)
  gesty/                 obserwator RP128 (obserwator_rp.py): 64 tokeny × 768 GELU MLP ostatniego bloku dla `--sondy_n` sond
  dane/OCENY.json, LOG.json, metadata.json, NORMY_GRAD.npz
Moduły trening, model, obserwator_gesty, obserwator_rp, trening_klasyfikacja: na Edenie z PYTHONPATH (biegi s47 i pokrętła),
lokalnie z kod/z_edena (kopie).
Uruchomienie: python trening_cifar_deit.py --wyniki OUT --dane cifar-10-python.tar.gz --epoki 60 --obserwuj --historia_wag [--obciecie 0]"""
from __future__ import annotations

import argparse
import math
import pickle
import shutil
import sys
import tarfile
import time
from pathlib import Path

TU = Path(__file__).resolve().parent
if (TU / 'z_edena').is_dir():                                     # lokalnie: kopie modułów z Edena
    sys.path.insert(0, str(TU / 'z_edena'))
sys.path.insert(0, str(TU))

import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402

import trening as T                                                # noqa: E402
import trening_klasyfikacja as TK                                  # noqa: E402
from model_deit_cifar import DeitCifar                             # noqa: E402


class MalyViTCifar(TK.Klasyfikator):
    """MałyViT użytkownika bez zmian (łodyga 4 × konw. z krokiem 2 = krok 16 px, sin-cos 2D, 6 bloków timm, głowa na średniej
    tokenów jak w biegach na Imagenette). CIFAR 32×32 powiększony `skala` razy (dwuliniowo) do 128 px, więc siatka 8×8 = 64 tokeny,
    te same co w DeiT z łatką 4. Klucze state_dict jak w Klasyfikator('vit')."""

    def __init__(self, n_klas=10, skala=4):
        super().__init__('vit', n_klas)
        self.skala = skala
        self.tokenow = (32 * skala // 16) ** 2

    def forward(self, x):
        return super().forward(F.interpolate(x, scale_factor=self.skala, mode='bilinear', align_corners=False))


# ------------------------------------------------------------------ dane
def wczytaj_cifar10(sciezka, split, limit=None):
    """CIFAR-10: X uint8 (N, 32, 32, 3), y int64, nazwy klas. Plik cifar10_uint8.npz (przygotuj_cifar10.py, z HuggingFace)
    albo oryginalne archiwum cifar-10-python.tar.gz."""
    if str(sciezka).endswith('.npz'):
        d = np.load(sciezka)
        X, y, klasy = d[f'X_{split}'], d[f'y_{split}'].astype(np.int64), [str(k) for k in d['klasy']]
    else:
        with tarfile.open(sciezka) as t:
            nazwy = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)] if split == 'train' else ['cifar-10-batches-py/test_batch']
            X, y = [], []
            for n in nazwy:
                d = pickle.load(t.extractfile(n), encoding='bytes')
                X.append(d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)); y.append(np.asarray(d[b'labels'], np.int64))
            klasy = [k.decode() for k in pickle.load(t.extractfile('cifar-10-batches-py/batches.meta'), encoding='bytes')[b'label_names']]
        X, y = np.concatenate(X), np.concatenate(y)
    X = np.ascontiguousarray(X)
    if limit:
        i = np.sort(np.random.default_rng(0).choice(len(y), limit, replace=False)); X, y = X[i], y[i]
    print(f'{split}: {len(y)} obrazów 32×32, {len(klasy)} klas', flush=True)
    return TK.ZbiorKlas(X, y, klasy)


# ------------------------------------------------------------------ uczeń
class UczenDeit(TK.UczenKlas):
    def __init__(self, ziarno, a, device='cuda'):
        torch.manual_seed(ziarno)
        self.dev = device
        self.net = (DeitCifar(n_klas=10, latka=a.latka) if a.model == 'deit' else MalyViTCifar(10, a.skala)).to(device)
        dec = [p for n, p in self.net.named_parameters() if p.ndim >= 2]
        nodec = [p for n, p in self.net.named_parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW([dict(params=dec, weight_decay=a.wd), dict(params=nodec, weight_decay=0)], lr=T.LR, betas=(0.9, 0.999))
        self.obciecie = float(a.obciecie)
        self.krok = 0
        self.log = []
        self.normy = []                                            # (krok, norma gradientu przed obcięciem)

    def kroki(self, zb, n, rng, co_ocena=None, ocena=None, obserwator=None, **_):
        """Jak UczenKlas.kroki (to samo losowanie i augmentacja), plus obcięcie opcjonalne i log normy gradientu."""
        self.net.train()
        kol = []
        t0 = time.time()
        bufor = []
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
            gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.obciecie if self.obciecie > 0 else float('inf'))
            bufor.append(gn.detach())
            self.opt.step()
            self.krok += 1
            if patrz:
                obserwator.po(self)
                self.net.train()
            if self.krok % 50 == 0:
                nb = torch.stack(bufor).float().cpu().numpy(); bufor = []
                self.normy += [(self.krok - len(nb) + 1 + i, float(v)) for i, v in enumerate(nb)]
                self.log.append(dict(krok=self.krok, strata=float(L.detach()), norma_grad_med=float(np.median(nb)), s_na_krok=(time.time() - t0) / 50))
                t0 = time.time()
            if co_ocena and ocena is not None and self.krok % co_ocena == 0:
                ocena(self)
                self.net.train()


def zapisac_wagi(t, koniec_epoki, co_po10k=100, co_po1k=100):
    """Harmonogram jak bieg s47 (806 checkpointów): co 10 kroków do 1000, co 100 dalej; --wagi_co_po10k rozrzedza po kroku 10 000,
    --wagi_co_po1k między 1000 a 10 000 (biegi powtórzeń i kontrolne: 1000 i 1000, ~200 plików)."""
    return koniec_epoki or (t <= 1000 and t % 10 == 0) or (t <= 10000 and t % co_po1k == 0) or t % co_po10k == 0


# ------------------------------------------------------------------ główna pętla
def main():
    a = argparse.ArgumentParser()
    a.add_argument('--wyniki', required=True)
    a.add_argument('--dane', required=True, help='cifar-10-python.tar.gz')
    a.add_argument('--epoki', type=int, default=60)
    a.add_argument('--proba', action='store_true', help='1024 obrazy, 2 epoki, test 500: sprawdzenie kodu')
    a.add_argument('--wznow', action='store_true')
    a.add_argument('--obserwuj', action='store_true')
    a.add_argument('--historia_wag', action='store_true')
    a.add_argument('--wagi_co_po1k', type=int, default=100, help='odstęp zapisu wag między krokiem 1000 a 10 000')
    a.add_argument('--losowe_etykiety', action='store_true', help='kontrola zerowa: etykiety uczące losowo przemieszane (ziarno --ziarno)')
    a.add_argument('--wagi_co_po10k', type=int, default=100, help='odstęp zapisu wag po kroku 10 000 (100 = jak s47, ~1000 plików × 21,5 MB)')
    a.add_argument('--sondy_n', type=int, default=24)
    a.add_argument('--pelne', type=int, default=4)
    a.add_argument('--lr', type=float, default=5e-4)
    a.add_argument('--wd', type=float, default=0.05)
    a.add_argument('--obciecie', type=float, default=1.0, help='maks. norma gradientu; 0 = bez obcięcia')
    a.add_argument('--paczka', type=int, default=32)
    a.add_argument('--latka', type=int, default=4)
    a.add_argument('--model', default='deit', choices=['deit', 'malyvit'], help='deit = DeiT-Tiny; malyvit = MałyViT (s47) na CIFAR')
    a.add_argument('--skala', type=int, default=4, help='malyvit: powiększenie 32 px → 32·skala px (4: 128 px, 64 tokeny)')
    a.add_argument('--gesto_do', type=int, default=1563, help='obserwator co krok do tego kroku (1 epoka), potem co --co_pozniej')
    a.add_argument('--co_pozniej', type=int, default=20)
    a.add_argument('--pelne_do', type=int, default=100)
    a.add_argument('--pelne_co', type=int, default=500)
    a.add_argument('--stan_co', type=int, default=5, help='pełny stan (z AdamW) co tyle epok')
    a.add_argument('--ocena_co', type=int, default=500)
    a.add_argument('--ziarno', type=int, default=47)
    a.add_argument('--min_start_gb', type=float, default=8.0)
    a.add_argument('--rzut_dim', type=int, default=128)
    a.add_argument('--zapis', default='f16', choices=['f16', 'f32'])
    a = a.parse_args()
    T.LR = a.lr; T.PACZKA = a.paczka
    out = Path(a.wyniki)
    for d in ('wagi', 'pelny', 'dane'):
        (out / d).mkdir(parents=True, exist_ok=True)
    wolne = TK.wolne_gb(out)
    print(f'wolna kwota: {wolne:.1f} GB', flush=True)
    if wolne < a.min_start_gb:
        raise SystemExit(f'za mało miejsca na start: {wolne:.1f} GB')
    zb = wczytaj_cifar10(a.dane, 'train', limit=1024 if a.proba else None)
    if a.losowe_etykiety:
        zb.y = np.random.default_rng(a.ziarno + 1000).permutation(zb.y)
        print(f'LOSOWE ETYKIETY: zgodnych z prawdziwymi {np.mean(zb.y == wczytaj_cifar10(a.dane, "train", limit=1024 if a.proba else None).y):.3f}', flush=True)
    wal = wczytaj_cifar10(a.dane, 'test', limit=500 if a.proba else None)
    epoki = 2 if a.proba else a.epoki
    na_epoke = math.ceil(len(zb) / T.PACZKA)
    konce = [e * na_epoke for e in range(1, epoki + 1)]
    lacznie = konce[-1]
    co_ocena = 20 if a.proba else a.ocena_co
    u = UczenDeit(a.ziarno, a); rng = np.random.default_rng(a.ziarno)
    n_param = int(sum(p.numel() for p in u.net.parameters()))
    nazwa = 'DeiT-Tiny' if a.model == 'deit' else 'MałyViT'
    print(f'{nazwa} CIFAR ({n_param} parametrów, {u.net.tokenow} tokenów' + (' + CLS' if a.model == 'deit' else '') + ')' + f', AdamW lr {a.lr} wd {a.wd}, paczka {T.PACZKA}, '
          f'obcięcie {a.obciecie or "brak"}; {len(zb)} obrazów, {epoki} epok, {lacznie} kroków', flush=True)
    oceny = []
    opis_m = ('DeiT-Tiny: dim 192, 12 bloków, 3 głowy, MLP 768, CLS, uczone PE, łatka 4' if a.model == 'deit' else
              f'MałyViT (s47): łodyga konw. krok 16, sin-cos 2D, dim 192, 6 bloków, 3 głowy, MLP 768, głowa na średniej tokenów; obraz ×{a.skala} dwuliniowo')
    meta = dict(opis=f'{nazwa} na CIFAR-10 od zera; cząstki = {u.net.tokenow} tokenów × 768 GELU MLP ostatniego bloku.',
                zadanie='klasyfikacja', zbior='cifar-10', model=opis_m,
                przepis=f'paczka {a.paczka}, AdamW lr {a.lr} wd {a.wd} (tensory ndim>=2), rozgrzewka 300, stały lr, bf16, odbicie 0.5, '
                f'jasność/kontrast ±20 %, obcięcie gradientu {a.obciecie or "brak"}',
                klatki=len(zb), klasy=zb.klasy, epoki=epoki, krokow=lacznie, na_epoke=na_epoke, konce_epok=konce, walidacja=len(wal),
                proba=a.proba, tokenow=u.net.tokenow, parametrow=n_param, argumenty=vars(a),
                kod_sha256={p.name: TK.sha_pliku(p) for p in sorted(TU.glob('*.py'))},
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, start=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.wznow:
        stany = sorted((out / 'pelny').glob('stan_e[0-9][0-9].pt'))
        if not stany:
            raise SystemExit('brak stanu do wznowienia')
        st = torch.load(stany[-1], map_location='cuda', weights_only=False)
        meta['wznowiono_od'] = dict(plik=stany[-1].name, krok=st['krok'])
        u.wczytaj(st); rng.bit_generator.state = st['rng']; torch.set_rng_state(st['torch_rng'].cpu())
        oceny = st.get('oceny', []); u.log = st.get('log', []); u.normy = st.get('normy', [])
        print(f'wznowienie od kroku {u.krok} ({stany[-1].name})', flush=True)

    def ocen(uu, epoka=None):
        acc = uu.dokladnosc(wal.X, wal.y)
        oceny.append(dict(krok=uu.krok, epoka=epoka, acc=float(acc), miara='top1'))
        TK.zj(out / 'dane' / 'OCENY.json', oceny)
        print(f'krok {uu.krok}' + (f' (koniec epoki {epoka})' if epoka else '') + f': acc {acc:.4f}', flush=True)

    def zapisz_wagi(uu):
        p = out / 'wagi' / f'k{uu.krok:05d}.pt'
        if not p.exists():
            torch.save(dict(net={k: v.detach().cpu().clone() for k, v in uu.net.state_dict().items()}, krok=uu.krok), p)

    def zapisz_normy(uu):
        if uu.normy:
            n = np.array(uu.normy, np.float64)
            np.savez(out / 'dane' / 'NORMY_GRAD.npz', krok=n[:, 0].astype(np.int64), norma=n[:, 1].astype(np.float32), obciecie=a.obciecie)

    stop = {'teraz': False}

    def po_kroku(uu):
        t = uu.krok
        epoka = konce.index(t) + 1 if t in konce else None
        if epoka:
            if TK.wolne_gb(out) < TK.MIN_STOP_GB:
                print('KWOTA: za mało miejsca - zapisuję stan i kończę', flush=True); stop['teraz'] = True
            ocen(uu, epoka)
            st = dict(uu.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=uu.log, normy=uu.normy)
            torch.save(st, out / 'pelny' / f'stan_e{epoka:02d}.pt')
            stany = sorted((out / 'pelny').glob('stan_e[0-9][0-9].pt'))
            for stary in stany[:-2]:                               # zostają: co --stan_co epok i 2 ostatnie
                if int(stary.stem[-2:]) % a.stan_co:
                    stary.unlink()
            TK.zj(out / 'dane' / 'LOG.json', uu.log); zapisz_normy(uu)
            print(f'epoka {epoka}/{epoki}: krok {t}, wolna kwota {TK.wolne_gb(out):.1f} GB, {time.strftime("%H:%M:%S")}', flush=True)
        elif t % co_ocena == 0:
            ocen(uu)
        if a.historia_wag and zapisac_wagi(t, bool(epoka), a.wagi_co_po10k, a.wagi_co_po1k):
            zapisz_wagi(uu)
        if stop['teraz']:
            raise KeyboardInterrupt

    obs = None
    if a.obserwuj:
        if a.proba:
            wybrane = list(range(16)); pelne = wybrane[:2]
            kw = dict(gesto_do=20, co_pozniej=2, pelne_do=10, pelne_co=4, porcja=8, porcja_pelne=4, wolne_gb=lambda: TK.wolne_gb(out))
        else:
            wybrane = [int(i) for i in np.linspace(0, len(zb) - 1, max(a.sondy_n, 1)).round()]
            pelne = wybrane[:a.pelne]
            kw = dict(gesto_do=a.gesto_do, co_pozniej=a.co_pozniej, pelne_do=a.pelne_do, pelne_co=a.pelne_co, wolne_gb=lambda: TK.wolne_gb(out))
        obs = TK.ObserwatorKlas(out / 'gesty', zb, wybrane, pelne, rzut_dim=a.rzut_dim, zapis=a.zapis, tokenow=u.net.tokenow, **kw)
        obs.start(u)
        if u.krok:
            obs.buf['kroki'][0] = u.krok; obs.buf_p['kroki'][0] = u.krok
        meta['obserwator'] = dict(sondy=len(obs.sondy), pelne=len(obs.pelne), gesto_do=obs.gesto_do, co_pozniej=obs.co_pozniej,
                                  pelne_do=obs.pelne_do, pelne_co=obs.pelne_co)
    if u.krok == 0:
        zapisz_wagi(u); ocen(u)
    TK.zj(out / 'dane' / 'metadata.json', meta)
    t0 = time.time()
    try:
        u.kroki(zb, lacznie - u.krok, rng, co_ocena=1, ocena=po_kroku, obserwator=obs)
    except KeyboardInterrupt:
        st = dict(u.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=u.log, normy=u.normy)
        torch.save(st, out / 'pelny' / f'stan_e99_przerwane_k{u.krok}.pt')
    if obs is not None:
        meta['obserwator'].update(obs.zakoncz())
    zapisz_normy(u)
    meta.update(koniec=time.strftime('%Y-%m-%d %H:%M:%S'), czas_s=round(time.time() - t0), ostatni_krok=u.krok,
                acc_koniec=oceny[-1]['acc'] if oceny else None, wolna_kwota_gb=round(TK.wolne_gb(out), 1))
    TK.zj(out / 'dane' / 'metadata.json', meta); TK.zj(out / 'dane' / 'LOG.json', u.log)
    print(f'KONIEC: krok {u.krok}, acc {meta["acc_koniec"]}, {meta["czas_s"]} s', flush=True)


if __name__ == '__main__':
    main()
