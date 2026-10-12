"""POKRĘTŁA SZKŁA (2026-10-10): kopia trening35000.py z biegu 35 000 z dodanymi pokrętłami: --lr (temperatura szumu),
manifest-podzbiór (gęstość: zdjęć na parametr), gęsty obserwator na --sondy_n sondach z --gesto_do/--co_pozniej, --bez_wag.
2026-10-11: --opt sgd/--momentum/--wd/--paczka (temperatura ∝ lr/paczka przy SGD), --model cnn (model_cnn.MalyCNN: ta sama łodyga,
głowa i cząstka, mieszanie lokalne zamiast uwagi), --zapis f32, --warstwy/--warstwy_n (skan GELU MLP każdego bloku do gesty/w<i>);
obserwator z losowym rzutem w obserwator_rp.py. Reszta przepisu bez zmian. Oryginalny opis poniżej.

Mały ViT od zera na 35 000 klatkach (wejscia/MANIFEST_35000.jsonl), 21 epok, wagi po każdej epoce i gęste kroki.

Przepis uczenia bez zmian względem biegu K1000 z gęstą historią (trening.Uczen(47), numpy default_rng(47), paczka 32,
AdamW 5e-4, rozgrzewka lr 300 kroków, potem stały lr, bf16, odbicie i jasność/kontrast). Zmienia się tylko zbiór:
35 000 klatek TRAIN Cataract-LMM Object Tracking zamiast bazy 1500. Klatki czytane prosto z archiwów zip na Edenie,
SHA-256 każdej klatki sprawdzane z manifestem. Każda klatka rozciągana do 480 × 320 (INTER_AREA), ramki skalowane
osobno w X i Y, tak jak w MANIFEST_1500 pomiaru (720×480, 720×576, 640×432, 540×540). Trzymane w RAM (uint8).

Epoka = ceil(35000 / 32) = 1094 kroki (35 008 próbek; Uczen.kroki losuje kolejne permutacje zbioru, więc paczka na
granicy epoki bierze resztę jednej permutacji i początek następnej). Koniec epoki e to krok 1094 · e, a pomiar naturalny
aplikacji z krokiem 1094 wybiera dokładnie końce epok.

Zapis (format pomiaru naturalnego aplikacji: k<krok>.pt = {'net', 'krok'}):
  wagi/k00000.pt ...   wagi sieci: krok 0, co 10 kroków do 1000, co 100 kroków dalej, koniec każdej epoki
  pelny/stan_e<EE>.pt  wagi + AdamW + stan losowania po każdej epoce (zostają 2 ostatnie, do wznowienia)
  dane/OCENY.json      mAP@0,5 na walidacji: krok 0, co 1000 kroków i koniec każdej epoki
  dane/LOG.json        strata co 50 kroków;  dane/metadata.json  opis biegu, sumy wejść, liczby
Pilnowanie kwoty: przed każdym zapisem sprawdzana jest wolna kwota projektu (lfs quota -p 1299). Poniżej
MIN_GESTE_GB gęste kroki są pomijane (zostają epoki), poniżej MIN_STOP_GB bieg zapisuje stan i kończy się.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

import dane as D
import trening as T

ZIARNO = 47
EPOKI = 21
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
    """Wolna kwota projektu 1299 na /mnt/evafs (GB); gdy lfs niedostępne: wolne miejsce systemu plików."""
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


def wiersze_35000(manifest):
    out = []
    for linia in open(manifest, encoding='utf-8'):
        r = json.loads(linia)
        assert r['split'] == 'TRAIN' and r['gt'], r['id']
        sx, sy = D.W_ / r['width'], D.H_ / r['height']          # jak MANIFEST_1500: rozciągnięcie do 480 × 320
        b = np.array([[g['bbox_xywh'][0] * sx, g['bbox_xywh'][1] * sy, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * sx,
                       (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * sy] for g in r['gt']], np.float32)
        out.append(dict(frame_id=f'cataract_lmm/tracking/{r["film"]}/{int(r["frame"]):08d}', film=r['film'],
                        archiwum=Path(r['archive'].replace(chr(92), '/')).name, member=r['member'],
                        sha=r['image_sha256'], ramki=b.reshape(-1, 4), wh=(r['width'], r['height'])))
    return out


def wczytaj_z_zipow(wiersze, katalog_zip, watki=16):
    """Klatki z archiwów (każde archiwum w jednym wątku), SHA-256 bajtów zgodne z manifestem, 480 × 320 uint8."""
    X = np.empty((len(wiersze), D.H_, D.W_, 3), np.uint8)
    grupy = {}
    for i, r in enumerate(wiersze):
        grupy.setdefault(r['archiwum'], []).append(i)

    def archiwum(nazwa):
        with zipfile.ZipFile(Path(katalog_zip) / nazwa) as z:
            for i in grupy[nazwa]:
                raw = z.read(wiersze[i]['member'])
                if hashlib.sha256(raw).hexdigest() != wiersze[i]['sha']:
                    raise ValueError(f'SHA-256 klatki niezgodne: {wiersze[i]["frame_id"]}')
                im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if im is None or im.shape[1::-1] != tuple(wiersze[i]['wh']):
                    raise ValueError(f'zła klatka {wiersze[i]["frame_id"]}: {None if im is None else im.shape}')
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                X[i] = cv2.resize(im, (D.W_, D.H_), interpolation=cv2.INTER_AREA)
        return len(grupy[nazwa])

    t = time.time()
    with ThreadPoolExecutor(watki) as ex:
        n = sum(ex.map(archiwum, sorted(grupy)))
    print(f'klatki: {n} z {len(grupy)} archiwów w {time.time() - t:.0f} s', flush=True)
    return X


def plan_zapisu(kroki_lacznie, konce_epok, proba):
    gesty = set(range(0, min(1000, kroki_lacznie) + 1, 10)) | set(range(0, kroki_lacznie + 1, 100))
    if proba:
        gesty = set(range(0, kroki_lacznie + 1, 5))
    return gesty | set(konce_epok) | {0, kroki_lacznie}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--wyniki', required=True)
    a.add_argument('--zipy', required=True, help='katalog z archiwami TR_*.zip (3_Object_Tracking/annotations)')
    a.add_argument('--manifest', required=True)
    a.add_argument('--epoki', type=int, default=EPOKI)
    a.add_argument('--proba', action='store_true', help='512 klatek, 2 epoki, walidacja 200: sprawdzenie kodu')
    a.add_argument('--wznow', action='store_true', help='wznowienie od ostatniego pelny/stan_e*.pt')
    a.add_argument('--obserwuj', action='store_true', help='gęsty zapis ruchu sond i sił co krok (obserwator_gesty.py)')
    a.add_argument('--sondy', default='', help='JSONL z polem pool_row: które wiersze manifestu są sondami (100)')
    a.add_argument('--pelne', type=int, default=8, help='ile pierwszych sond ma pełny zapis 768D')
    a.add_argument('--lr', type=float, default=5e-4, help='pokrętło temperatury: stały lr po rozgrzewce (przepis: 5e-4)')
    a.add_argument('--sondy_n', type=int, default=0, help='ile sond (z pliku --sondy pierwsze n; bez pliku: n wierszy równo rozłożonych)')
    a.add_argument('--gesto_do', type=int, default=2188, help='obserwator: zapis co 1 krok do tego kroku')
    a.add_argument('--co_pozniej', type=int, default=10, help='obserwator: potem co tyle kroków')
    a.add_argument('--pelne_do', type=int, default=100, help='obserwator: pełne 768D co 1 krok do tego kroku')
    a.add_argument('--pelne_co', type=int, default=1000, help='obserwator: potem pełne 768D co tyle kroków')
    a.add_argument('--bez_wag', action='store_true', help='nie zapisuj wag po drodze (tylko krok 0 i ostatni)')
    a.add_argument('--ocena_co', type=int, default=1000, help='mAP na walidacji co tyle kroków')
    a.add_argument('--ziarno', type=int, default=ZIARNO, help='ziarno Uczen/rng (przepis: 47)')
    a.add_argument('--gesto_od', type=int, default=10**9, help='obserwator: dodatkowo zapis co 1 krok od tego kroku (gęste okno na końcu)')
    a.add_argument('--min_start_gb', type=float, default=8.0, help='minimalna wolna kwota na start (GB)')
    a.add_argument('--rzut', default='pca32', choices=['pca32', 'rp'], help='rzut zapisu sond: PCA32 z kroku startu albo losowy rzut Gaussa (JL)')
    a.add_argument('--rzut_dim', type=int, default=128, help='wymiar losowego rzutu (zapisywany pod kluczem pca32, patrz SONDY.json)')
    a.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'], help='optymalizator: AdamW (przepis) albo SGD z pędem (temperatura ∝ lr/paczka)')
    a.add_argument('--momentum', type=float, default=0.9, help='pęd SGD')
    a.add_argument('--wd', type=float, default=None, help='weight decay (domyślnie 0.05 dla AdamW, 0 dla SGD)')
    a.add_argument('--paczka', type=int, default=32, help='rozmiar paczki (przepis: 32); zmienia liczbę kroków na epokę')
    a.add_argument('--model', default='vit', choices=['vit', 'cnn'], help='architektura: MalyViT (przepis) albo MalyCNN (model_cnn.py, mieszanie lokalne depthwise)')
    a.add_argument('--jadro', type=int, default=5, help='jądro depthwise w MalyCNN')
    a.add_argument('--zapis', default='f16', choices=['f16', 'f32'], help='precyzja zapisu rzutów i norm obserwatora (tylko --rzut rp)')
    a.add_argument('--warstwy', action='store_true', help='skan po warstwach: rzut GELU MLP każdego bloku dla --warstwy_n sond do gesty/w<i> (tylko --rzut rp)')
    a.add_argument('--warstwy_n', type=int, default=6, help='ile pierwszych sond w skanie po warstwach')
    a.add_argument('--bez_pozycji', action='store_true', help='kontrola krysztalu: zerowe kodowanie pozycji sincos (tokeny bez adresu patcha)')
    a = a.parse_args()
    T.LR = a.lr                                      # Uczen._lr czyta LR modułu przy każdym kroku
    T.PACZKA = a.paczka                              # Uczen.kroki czyta PACZKA modułu
    out = Path(a.wyniki)
    for d in ('wagi', 'pelny', 'dane'):
        (out / d).mkdir(parents=True, exist_ok=True)
    wolne = wolne_gb(out)
    print(f'wolna kwota: {wolne:.1f} GB', flush=True)
    if wolne < a.min_start_gb:
        raise SystemExit(f'za mało miejsca na start: {wolne:.1f} GB (potrzeba ≥ {a.min_start_gb:.0f} GB)')

    wiersze = wiersze_35000(a.manifest)
    if a.proba:
        wiersze = [wiersze[i] for i in np.random.default_rng(0).choice(len(wiersze), 512, replace=False)]
    wal_w, wal_X = D.walidacja()
    ids_tr = {r['frame_id'] for r in wiersze}
    wspolne = [i for i, r in enumerate(wal_w) if r['frame_id'] in ids_tr]
    if wspolne:                                      # walidacja tylko na klatkach spoza treningu
        zostaw = np.setdiff1d(np.arange(len(wal_w)), wspolne)
        wal_w, wal_X = [wal_w[i] for i in zostaw], wal_X[zostaw]
    if a.proba:
        wal_w, wal_X = wal_w[:200], wal_X[:200]
    print(f'walidacja: {len(wal_w)} klatek (usunięto {len(wspolne)} wspólnych z treningiem)', flush=True)
    X = wczytaj_z_zipow(wiersze, a.zipy)
    zb = D.Zbior(wiersze, X)

    epoki = 2 if a.proba else a.epoki
    na_epoke = math.ceil(len(zb) / T.PACZKA)          # 1094 kroki = 35 008 próbek: końce epok w równym odstępie
    konce = [e * na_epoke for e in range(1, epoki + 1)]
    lacznie = konce[-1]
    zapis = plan_zapisu(lacznie, konce, a.proba)
    co_ocena = 20 if a.proba else a.ocena_co
    print(f'{len(zb)} klatek, {epoki} epok, {lacznie} kroków; końce epok: {konce}; zapisów wag: {len(zapis)} '
          f'(~{len(zapis) * 11.7 / 1000:.1f} GB)', flush=True)

    if a.bez_pozycji:                                # kontrola kryształu: MalyViT.forward woła sincos_2d przez nazwę w module model → zerujemy
        import model as M_
        M_.sincos_2d = lambda h, w, dim, device: torch.zeros(h * w, dim, device=device)
        print('KONTROLA: kodowanie pozycji wyzerowane (--bez_pozycji)', flush=True)
    u = T.Uczen(a.ziarno); rng = np.random.default_rng(a.ziarno)
    if a.model == 'cnn':                             # ta sama łodyga, głowa i definicja cząstki; mieszanie lokalne zamiast uwagi
        import model_cnn as MC
        torch.manual_seed(a.ziarno); u.net = MC.MalyCNN(jadro=a.jadro).to(u.dev)
    if a.model == 'cnn' or a.opt != 'adamw' or a.wd is not None:
        dec = [p for n, p in u.net.named_parameters() if p.ndim >= 2]
        nodec = [p for n, p in u.net.named_parameters() if p.ndim < 2]
        if a.opt == 'adamw':
            wd = T.WD if a.wd is None else a.wd
            u.opt = torch.optim.AdamW([dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0)], lr=T.LR, betas=(0.9, 0.999))
        else:
            wd = 0.0 if a.wd is None else a.wd
            u.opt = torch.optim.SGD([dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0)], lr=T.LR, momentum=a.momentum)
    wd_uzyte = float(u.opt.param_groups[0]['weight_decay']); n_param = int(sum(p.numel() for p in u.net.parameters()))
    print(f'model {a.model} ({n_param} parametrów), {a.opt} lr {a.lr} wd {wd_uzyte}' + (f' pęd {a.momentum}' if a.opt == 'sgd' else '') + f', paczka {T.PACZKA}', flush=True)
    oceny, pominiete = [], []
    meta = dict(opis='Mały ViT od zera, 35 000 klatek TRAIN Cataract-LMM Object Tracking, 21 epok, zapis co epokę i '
                     'gęste kroki (README eksperymentu).',
                przepis='trening.Uczen(47) + numpy default_rng(47), paczka 32, AdamW 5e-4 wd 0.05, rozgrzewka 300, '
                        'stały lr, bf16, odbicie 0.5, jasność/kontrast ±20 %' + ('' if (a.opt, a.model, a.paczka) == ('adamw', 'vit', 32) else
                        f' — ZMIANA: model {a.model}, {a.opt} (pęd {a.momentum}, wd {wd_uzyte}), paczka {a.paczka}'),
                klatki=len(zb), epoki=epoki, krokow=lacznie, konce_epok=konce, kroki_zapisu=sorted(zapis),
                walidacja=len(wal_w), walidacja_usuniete_wspolne=len(wspolne), proba=a.proba,
                manifest=str(a.manifest), manifest_sha256=sha_pliku(a.manifest),
                kod_sha256={p.name: sha_pliku(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                lr=a.lr, pokretla=dict(lr=a.lr, ziarno=a.ziarno, sondy_n=a.sondy_n, gesto_do=a.gesto_do, gesto_od=a.gesto_od, co_pozniej=a.co_pozniej, bez_wag=a.bez_wag,
                              baza_pca32_z_kroku=('wznowienie' if a.wznow else 0), rzut=a.rzut, rzut_dim=a.rzut_dim, opt=a.opt, momentum=a.momentum, wd=wd_uzyte,
                              paczka=a.paczka, model=a.model, jadro=a.jadro, parametrow=n_param, zapis=a.zapis, warstwy=a.warstwy, warstwy_n=a.warstwy_n,
                              bez_pozycji=a.bez_pozycji),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, start=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.wznow:
        stany = sorted((out / 'pelny').glob('stan_e[0-9][0-9].pt'))
        if not stany:
            raise SystemExit('brak stanu do wznowienia')
        st = torch.load(stany[-1], map_location='cuda', weights_only=False)
        stary_meta = out / 'dane' / 'metadata.json'
        if stary_meta.exists():                      # opis poprzedniego odcinka zostaje obok
            shutil.copy2(stary_meta, out / 'dane' / f'metadata_do_kroku_{st["krok"]:05d}.json')
        meta['wznowiono_od'] = dict(plik=stany[-1].name, krok=st['krok'])
        u.wczytaj(st); rng.bit_generator.state = st['rng']; torch.set_rng_state(st['torch_rng'].cpu())
        oceny = st.get('oceny', []); u.log = st.get('log', [])
        print(f'wznowienie od kroku {u.krok} ({stany[-1].name}); kolejka permutacji zaczyna się od nowa', flush=True)

    def ocen(uu, epoka=None):
        ap = T.ap50(uu.wykryj(wal_X), [r['ramki'] for r in wal_w])
        oceny.append(dict(krok=uu.krok, epoka=epoka, mAP50=float(ap)))
        zj(out / 'dane' / 'OCENY.json', oceny)
        print(f'krok {uu.krok}' + (f' (koniec epoki {epoka})' if epoka else '') + f': mAP@0,5 {ap:.4f}', flush=True)

    def zapisz_wagi(uu):
        p = out / 'wagi' / f'k{uu.krok:05d}.pt'
        if not p.exists():
            torch.save(dict(net={k: v.detach().cpu().clone() for k, v in uu.net.state_dict().items()}, krok=uu.krok,
                            epoka=uu.krok * T.PACZKA / len(zb)), p)

    stop = {'teraz': False}

    def po_kroku(uu):
        t = uu.krok
        epoka = konce.index(t) + 1 if t in konce else None
        if t in zapis and (not a.bez_wag or t == lacznie):
            wolne_ = wolne_gb(out) if (t % 100 == 0 or epoka) else MIN_GESTE_GB + 1
            if wolne_ < MIN_STOP_GB:
                print(f'KWOTA: {wolne_:.1f} GB < {MIN_STOP_GB} GB - zapisuję stan i kończę', flush=True)
                stop['teraz'] = True
            elif epoka or wolne_ >= MIN_GESTE_GB:
                zapisz_wagi(uu)
            else:
                pominiete.append(t)
        if epoka:
            ocen(uu, epoka)
            st = dict(uu.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=uu.log)
            torch.save(st, out / 'pelny' / f'stan_e{epoka:02d}.pt')
            for stary in sorted((out / 'pelny').glob('stan_e*.pt'))[:-2]:
                stary.unlink()
            zj(out / 'dane' / 'LOG.json', uu.log)
            print(f'epoka {epoka}/{epoki}: krok {t}, wolna kwota {wolne_gb(out):.1f} GB, '
                  f'{time.strftime("%H:%M:%S")}', flush=True)
        elif t % co_ocena == 0:
            ocen(uu)
        if stop['teraz']:
            raise KeyboardInterrupt

    obs = None
    if a.obserwuj:
        import obserwator_gesty as OG
        from obserwator_rp import ObserwatorRP       # losowy rzut (JL), zapis f16/f32, skan po warstwach

        if a.wznow:
            print(f'obserwator po wznowieniu: baza rzutu z kroku {u.krok}, nie 0', flush=True)
        if a.proba:
            wybrane = list(range(16)); pelne = wybrane[:4]
            kw = dict(gesto_do=20, co_pozniej=2, pelne_do=10, pelne_co=4, porcja=8, porcja_pelne=4, wolne_gb=lambda: wolne_gb(out))
        else:
            if a.sondy:
                wybrane = [json.loads(l)['pool_row'] for l in open(a.sondy, encoding='utf-8') if l.strip()]
            else:
                wybrane = [int(i) for i in np.linspace(0, len(zb) - 1, max(a.sondy_n, 1)).round()]
            if a.sondy_n:
                wybrane = wybrane[:a.sondy_n]
            assert max(wybrane) < len(zb), 'sonda poza zbiorem (manifest-podzbiór?)'
            pelne = wybrane[:a.pelne]
            kw = dict(gesto_do=a.gesto_do, co_pozniej=a.co_pozniej, pelne_do=a.pelne_do, pelne_co=a.pelne_co, wolne_gb=lambda: wolne_gb(out))
        if a.rzut == 'rp':
            obs = ObserwatorRP(out / 'gesty', zb, wybrane, pelne, rzut_dim=a.rzut_dim, zapis=a.zapis, warstwy=a.warstwy, warstwy_n=a.warstwy_n, tokenow=600, **kw)
        else:
            assert not a.warstwy and a.zapis == 'f16', '--warstwy i --zapis f32 tylko z --rzut rp'
            obs = OG.Obserwator(out / 'gesty', zb, wybrane, pelne, **kw)
        obs.start(u)
        if u.krok:                                        # pierwszy zapis obserwatora ma numer kroku wznowienia, nie 0
            obs.buf['kroki'][0] = u.krok; obs.buf_p['kroki'][0] = u.krok
        obs._czy_pca = lambda k, o=obs: k <= o.gesto_do or k >= a.gesto_od or k % o.co_pozniej == 0
        meta['obserwator'] = dict(sondy=len(obs.sondy), pelne=len(obs.pelne), gesto_do=obs.gesto_do, co_pozniej=obs.co_pozniej,
                                  pelne_do=obs.pelne_do, pelne_co=obs.pelne_co, sondy_plik=a.sondy)
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
    meta.update(koniec=time.strftime('%Y-%m-%d %H:%M:%S'), czas_s=round(time.time() - t0), ostatni_krok=u.krok,
                pominiete_geste_kroki=pominiete, plikow_wag=len(list((out / 'wagi').glob('k*.pt'))),
                wolna_kwota_gb=round(wolne_gb(out), 1))
    zj(out / 'dane' / 'metadata.json', meta); zj(out / 'dane' / 'LOG.json', u.log)
    print(f'KONIEC: krok {u.krok}, {meta["plikow_wag"]} plików wag, {meta["czas_s"]} s', flush=True)


if __name__ == '__main__':
    main()
