"""Od sztucznego latentu do zdjęć uczących (PLAN.md). Etapy zapisują wyniki w outputs/latent_cel_do_obrazow/ i są
pomijane przy ponownym uruchomieniu, gdy ich plik już istnieje.

Uruchomienie z SmallViTAblation: F:/Python/python.exe experiments/latent_cel_do_obrazow/kod/eksperyment.py [etap...]
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
sys.path.insert(0, str(ROOT / 'app'))
import numpy as np
import torch

import korekta_latentu as KL

N_CELOW = 24
PUNKTY = (100, 200, 300)
N_PODMIANA = 96
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def zapisz(nazwa, x):
    KL._json(OUT / nazwa, x)


def czytaj(nazwa):
    p = OUT / nazwa
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


CK = dict(KL.checkpointy())
ST = None


def st():
    global ST
    if ST is None:
        ST = KL.Stanowisko(CK['gęsta historia k1000'])
    return ST


# ---------------------------------------------------------------------------------------------------- pula klatek
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
_PULA = {}


def pula():
    """Cechy DINO (znormalizowane), manifest i maska dozwolonych klatek: split TRAIN, z ramką narzędzia, spoza filmów
    walidacji wal3000 i spoza klatek obrazów celu."""
    if not _PULA:
        man = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        F = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32)
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        a = json.loads((ROOT / 'experiments/maly_vit_od_zera/eden/wejscia/pretrain/valid/_annotations.coco.json').read_text())
        wal = {im['file_name'].replace(chr(92), '/').split('/')[-1][:14] for im in a['images']}
        cele = set()
        c = czytaj('ETAP1_CELE.json')
        if c:
            cele = {r['frame_id'] for r in c['cele']}
        dozw = np.array([m['split'] == 'TRAIN' and bool(m.get('gt')) and m['film'] not in wal
                         and f"cataract_lmm/tracking/{m['film']}/{int(m['frame']):08d}" not in cele for m in man])
        _PULA.update(man=man, F=F, dozw=dozw, wal=wal)
        log('pula', len(man), 'dozwolone', int(dozw.sum()), 'filmy walidacji', len(wal))
    return _PULA


def wczytaj_klatki(idx):
    """Obrazy 480 x 320 i ramki xyxy (480 x 320) klatek puli o indeksach idx (odczyt z archiwów ZIP)."""
    import cv2
    from PIL import Image
    P = pula(); out = {}
    po_arch = {}
    for j in idx:
        po_arch.setdefault(P['man'][j]['archive'], []).append(j)
    for arch, js in po_arch.items():
        with zipfile.ZipFile(arch) as zf:
            for j in js:
                m = P['man'][j]
                im = np.asarray(Image.open(io.BytesIO(zf.read(m['member']))).convert('RGB'))
                h, w = im.shape[:2]
                x = cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)
                r = [[g['bbox_xywh'][0] * 480 / w, g['bbox_xywh'][1] * 320 / h, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * 480 / w,
                      (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * 320 / h] for g in m['gt']]
                out[int(j)] = (x, np.asarray(r, np.float32).reshape(-1, 4))
    return out


@torch.no_grad()
def latenty(obrazy, net=None):
    """Latent ViT k1000 [n, 600, 768] i wyjścia [n, 20, 30, 5] (eval, float32)."""
    s = st(); net = net or s.net
    Z, O = [], []
    for a in range(0, len(obrazy), 16):
        x = torch.from_numpy(np.stack(obrazy[a:a + 16])).permute(0, 3, 1, 2).float().to(s.dev) / 255
        net.zapisuj_latent(True)
        o, z = net(x)
        net.zapisuj_latent(False)
        Z.append(z.float().cpu()); O.append(o.float().cpu())
    return torch.cat(Z), torch.cat(O)


# ---------------------------------------------------------------------------------------------------- etap 1
def etap1():
    if czytaj('ETAP1_CELE.json'):
        return czytaj('ETAP1_CELE.json')
    s = st(); rows = KL.manifest()
    obr = [KL.obraz_manifestu(r) for r in rows]
    Z, O = latenty([o[0] for o in obr])
    from model_snapshot import dekoduj
    bx, sc = dekoduj(O)
    iou0 = [KL.ocena_obrazu(bx[i].numpy(), sc[i].numpy(), obr[i][1])['iou_top'] for i in range(len(rows))]
    najsl = {}
    for i, r in enumerate(rows):
        if r['film'] not in najsl or iou0[i] < iou0[najsl[r['film']]]:
            najsl[r['film']] = i
    wyb = sorted(najsl.values(), key=lambda i: iou0[i])[:N_CELOW]
    log('cele', wyb, [round(iou0[i], 3) for i in wyb])
    cele = []
    for i in wyb:
        rgb, ram = obr[i]
        spec = dict(tryb='zakresy', kanaly=np.arange(768), tokeny=0, cel='detekcja', cel_pewnosc=.95, norma='brak',
                    lam=0.0, kroki=300, lr=.05, optymalizator='adam', ziarno=0, zapis_co=50)
        w = KL.oblicz_korekte(s, rgb, ram, spec, dict(id=f'm{i:04d}', zrodlo='manifest J128', indeks=i))
        M = {m['krok']: m for m in w['metryki']}
        cele.append(dict(i=i, frame_id=rows[i]['frame_id'], film=rows[i]['film'], split=rows[i]['split'], kat=w['kat'],
                         ramki=ram.tolist(), w_bazie_uczenia=rows[i]['frame_id'] in KL.ramki_bazy_uczenia(),
                         przed={k: w['przed'][k] for k in ('strata', 'iou_top', 'pewnosc_top', 'fp')},
                         najlepszy={k: w['najlepszy'][k] for k in ('krok', 'strata', 'iou_top', 'pewnosc_top', 'fp', 'norma_dz')},
                         niewidoczna=w['najlepszy']['fc2']['udzial_niewidocznej'],
                         punkty={p: dict(iou_top=M[p]['iou_top'], pewnosc_top=M[p]['pewnosc_top'], cel=M[p]['cel'],
                                         norma_dz=M[p]['norma_dz'], fp=M[p]['fp']) for p in PUNKTY}))
        log('cel', i, cele[-1]['przed']['iou_top'], '->', {p: round(v['iou_top'] or 0, 3) for p, v in cele[-1]['punkty'].items()},
            'pewność', {p: round(v['pewnosc_top'], 3) for p, v in cele[-1]['punkty'].items()})
    out = dict(cele=cele, iou_k1000_manifest=iou0, opis='cel detekcja (pewność 0,95), zakresy, 768 kanałów, 600 tokenów, '
                                                       'Adam 0,05, 300 kroków; punkty pośrednie 100/200/300')
    zapisz('ETAP1_CELE.json', out)
    return out


def punkt(cel, p):
    """z_w [600, 768] = z + δ(p) z zapisanej trajektorii korekty oraz z bazowe."""
    kr, D = KL.wczytaj_trajektorie(cel['kat'], 'd')
    kn = np.load(Path(cel['kat']) / 'KONIEC.npz')
    z = kn['z_baza']
    return z + D[list(kr).index(p)], z


# ---------------------------------------------------------------------------------------------------- etap 2a: konwerter
def etap2a(cele):
    if czytaj('ETAP2A_KONWERTER.json'):
        return czytaj('ETAP2A_KONWERTER.json')
    import kandydaci as KD
    P = pula(); kon = KD._konwerter()
    dozw = np.flatnonzero(P['dozw'])
    wyn = []
    for c in cele['cele']:
        for p in PUNKTY:
            zw, z0 = punkt(c, p)
            with torch.no_grad():
                cls, _ = kon.predict(zw[None])
            q = np.asarray(cls.cpu(), np.float32).reshape(-1); q /= np.linalg.norm(q) + 1e-12
            sim = P['F'][dozw] @ q
            kol_d = np.argsort(-sim)[:20]
            top = dozw[kol_d]; sim_top = sim[kol_d]
            kl = wczytaj_klatki(top)
            Zc, Oc = latenty([kl[int(j)][0] for j in top])
            dw = zw - z0
            Zc = Zc.numpy()
            dist = np.linalg.norm((Zc - zw).reshape(len(top), -1), axis=1) / np.linalg.norm((z0 - zw).reshape(-1))
            cosd = ((Zc - z0).reshape(len(top), -1) @ dw.reshape(-1)) / (
                np.linalg.norm((Zc - z0).reshape(len(top), -1), axis=1) * np.linalg.norm(dw) + 1e-12)
            kol = np.argsort(dist)
            wyn.append(dict(cel=c['i'], punkt=p, kandydaci=[dict(j=int(top[k]), film=P['man'][top[k]]['film'],
                                                                 klatka=int(P['man'][top[k]]['frame']), cos_dino=float(sim_top[k]),
                                                                 odl_wzgl=float(dist[k]), cos_do_celu=float(cosd[k])) for k in kol]))
        log('konwerter', c['i'], [round(w['kandydaci'][0]['odl_wzgl'], 3) for w in wyn[-3:]])
    out = dict(wyniki=wyn, opis='odl_wzgl = ||z(klatka) − z_w|| / ||z(cel) − z_w|| (1 = tak daleko jak obraz celu); '
                                 'cos_do_celu = cos(z(klatka) − z(cel), z_w − z(cel)); kandydaci posortowani wg odl_wzgl')
    zapisz('ETAP2A_KONWERTER.json', out)
    return out


# ---------------------------------------------------------------------------------------------------- etap 2b: SD
def etap2b(cele):
    if czytaj('ETAP2B_SD.json'):
        return czytaj('ETAP2B_SD.json')
    import silnik
    import kandydaci as KD
    from model_snapshot import dekoduj
    import cv2
    trained = silnik.zaladuj_model()
    kon = KD._konwerter(); P = pula(); dozw = np.flatnonzero(P['dozw'])
    (OUT / 'sd').mkdir(exist_ok=True)
    wyn = []
    for c in cele['cele']:
        for p in (0,) + PUNKTY:
            if p == 0:
                z0 = np.load(Path(c['kat']) / 'KONIEC.npz')['z_baza']; zw = z0
            else:
                zw, z0 = punkt(c, p)
            mian = np.linalg.norm(z0 - zw) if p else np.linalg.norm(z0)
            img = silnik.generuj(trained, zw, kroki=30, ziarno=0)
            x = cv2.resize((img.transpose(1, 2, 0) * 255).astype(np.uint8), (480, 320), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / 'sd' / f'm{c["i"]:04d}_p{p:03d}.png'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
            Zs, Os = latenty([x])
            zs = Zs[0].numpy()
            bx, sc = dekoduj(Os)
            m = KL.ocena_obrazu(bx[0].numpy(), sc[0].numpy(), np.asarray(c['ramki']))
            with torch.no_grad():
                cls, _ = kon.predict(zs[None])
            q = np.asarray(cls.cpu(), np.float32).reshape(-1); q /= np.linalg.norm(q) + 1e-12
            sim = P['F'][dozw] @ q
            top = dozw[np.argsort(-sim)[:10]]
            kl = wczytaj_klatki(top)
            Zc, _ = latenty([kl[int(j)][0] for j in top]); Zc = Zc.numpy()
            dist = np.linalg.norm((Zc - zw).reshape(len(top), -1), axis=1) / max(mian, 1e-9)
            kol = np.argsort(dist)
            wyn.append(dict(cel=c['i'], punkt=p, sd_odl_wzgl=float(np.linalg.norm(zs - zw) / max(mian, 1e-9)) if p else None,
                            sd_odl_do_celu_bazowego=float(np.linalg.norm(zs - z0) / np.linalg.norm(z0)),
                            sd_iou_top_z_etykieta_celu=m['iou_top'], sd_pewnosc_top=m['pewnosc_top'],
                            kandydaci=[dict(j=int(top[k]), film=P['man'][top[k]]['film'], klatka=int(P['man'][top[k]]['frame']),
                                            odl_wzgl=float(dist[k])) for k in kol]))
        log('SD', c['i'], [(w['punkt'], round(w['sd_pewnosc_top'], 3), w['sd_iou_top_z_etykieta_celu'] and round(w['sd_iou_top_z_etykieta_celu'], 3),
                           w['sd_odl_wzgl'] and round(w['sd_odl_wzgl'], 3)) for w in wyn[-4:]])
        zapisz('ETAP2B_SD_czesciowy.json', dict(wyniki=wyn))
    out = dict(wyniki=wyn, opis='obraz SD z warunku z_w (punkt 0 = latent bazowy obrazu celu); sd_odl_wzgl = ||z(SD) − z_w|| '
                                 '/ ||z(cel) − z_w||; detekcja na obrazie SD oceniana etykietą obrazu celu (pseudo); '
                                 'kandydaci: SD -> latent -> konwerter -> klatki, posortowane wg odległości od z_w')
    zapisz('ETAP2B_SD.json', out)
    del trained; torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------------- etap 2c: pchnięcie
def etap2c(cele, kon, sd):
    if czytaj('ETAP2C_PCHNIECIE.json'):
        return czytaj('ETAP2C_PCHNIECIE.json')
    import cv2
    import galezie_uczenia as GU
    from pchniecie import Pchniecie
    from stan_uczenia import Odtworzenie
    b = GU.bieg_gestej_historii(); od = Odtworzenie(b); od.przygotuj()
    rows = KL.manifest()
    C = cele['cele']
    X = np.stack([KL.obraz_manifestu(rows[c['i']])[0] for c in C])
    maski = np.stack([KL.maski_gt(np.asarray(c['ramki']))['obiekt'] for c in C])
    P = Pchniecie(od, X, maski, [np.asarray(c['ramki']) for c in C])
    k = 990
    P.przygotuj(k, 0)
    Pl = pula(); dozw = np.flatnonzero(Pl['dozw']); rng = np.random.default_rng(5)
    kon_m = {(w['cel'], w['punkt']): w for w in kon['wyniki']}
    sd_m = {(w['cel'], w['punkt']): w for w in sd['wyniki']}
    wyn = []
    for t, c in enumerate(C):
        kn = np.load(Path(c['kat']) / 'KONIEC.npz')
        d_fin = torch.from_numpy(kn['d_najlepszy']).to(P.dev)
        g = P._grad_straty_sondy(t)
        kand = [('cel (sam obraz z etykietą)', X[t], np.asarray(c['ramki']))]
        js = []
        for p in PUNKTY:
            js.append(('konwerter', p, kon_m[(c['i'], p)]['kandydaci'][0]['j']))
            js.append(('SD -> konwerter', p, sd_m[(c['i'], p)]['kandydaci'][0]['j']))
        js += [('losowa z puli', 0, int(j)) for j in rng.choice(dozw, 3, replace=False)]
        kl = wczytaj_klatki([j for _, _, j in js])
        for zr, p, j in js:
            kand.append((f'{zr} p{p}' if p else zr, kl[j][0], kl[j][1]))
        for p in PUNKTY:
            im = cv2.cvtColor(cv2.imread(str(OUT / 'sd' / f'm{c["i"]:04d}_p{p:03d}.png')), cv2.COLOR_BGR2RGB)
            kand.append((f'obraz SD p{p} (etykieta celu)', im, np.asarray(c['ramki'])))
        for nazwa, im, ram in kand:
            S = P.pchniecie_kandydata(im, ram, t)
            cos = float((S * d_fin).sum() / (S.norm() * d_fin.norm() + 1e-12))
            wyn.append(dict(cel=c['i'], zrodlo=nazwa.split(' p')[0] if ' p' in nazwa else nazwa, nazwa=nazwa,
                            cos_do_korekty=cos, zmiana_straty=float((S * g).sum()), dlugosc=float(S.norm()),
                            rzut_na_korekte=float((S * d_fin).sum() / (d_fin.norm() ** 2 + 1e-12))))
        log('pchnięcie', c['i'], {r['nazwa']: round(r['cos_do_korekty'], 3) for r in wyn[-len(kand):]})
    zr = sorted({r['zrodlo'] for r in wyn})
    pods = {z: dict(n=sum(r['zrodlo'] == z for r in wyn), **{m: float(np.mean([r[m] for r in wyn if r['zrodlo'] == z]))
                                                             for m in ('cos_do_korekty', 'zmiana_straty', 'dlugosc', 'rzut_na_korekte')})
            for z in zr}
    out = dict(krok=k, wyniki=wyn, podsumowanie=pods,
               opis='pchnięcie jednego kroku AdamW (990 -> 991) obrazu dołożonego do paczki na latent obrazu celu (sonda); '
                    'cos_do_korekty = cos(S, δ korekty oracle przy k1000); zmiana_straty = <dL_celu/dz, S> (ujemna = pomaga)')
    zapisz('ETAP2C_PCHNIECIE.json', out)
    return out


# ---------------------------------------------------------------------------------------------------- etap 3: uczenie
def etap3(cele, kon, sd, ramiona=('A', 'E', 'B', 'C', 'D')):
    import galezie_uczenia as GU
    from stan_uczenia import _srodowisko
    from model_snapshot import dekoduj
    b = GU.bieg_gestej_historii()
    D, T, M = _srodowisko(b)
    rows = KL.manifest(); C = cele['cele']
    Xc = [KL.obraz_manifestu(rows[c['i']])[0] for c in C]; Rc = [np.asarray(c['ramki'], np.float32) for c in C]
    kn = [np.load(Path(c['kat']) / 'KONIEC.npz') for c in C]
    d_fin = np.stack([k['d_najlepszy'] for k in kn])                       # [24, 600, 768]
    baza0 = D.baza(); wal_w, wal_X = D.walidacja()
    R = np.random.default_rng(123).choice(len(baza0), N_PODMIANA, replace=False)
    Pl = pula(); dozw = np.flatnonzero(Pl['dozw'])

    def wybor(zrodlo):
        """96 unikalnych klatek: najlepsza dla każdej pary (cel, punkt), potem kolejne, aż do 96."""
        W = kon['wyniki'] if zrodlo == 'B' else [w for w in sd['wyniki'] if w['punkt'] in PUNKTY]
        uzyte, kolej = [], 0
        while len(uzyte) < N_PODMIANA and kolej < 10:
            for w in W:
                if kolej < len(w['kandydaci']):
                    j = w['kandydaci'][kolej]['j']
                    if j not in uzyte:
                        uzyte.append(j)
                if len(uzyte) >= N_PODMIANA:
                    break
            kolej += 1
        return uzyte

    def zbior(ramie):
        if ramie == 'A':
            return baza0, []
        wiersze = [dict(r) for r in baza0.wiersze]; X = baza0.X.copy()
        if ramie == 'E':
            for k, (x, r) in enumerate(zip(Xc, Rc)):
                X[R[k]] = x; wiersze[R[k]]['ramki'] = r; wiersze[R[k]]['frame_id'] = 'cel_' + C[k]['frame_id']
            return D.Zbior(wiersze, X), [C[k]['frame_id'] for k in range(len(C))]
        js = wybor(ramie) if ramie in ('B', 'D') else [int(j) for j in np.random.default_rng(7).choice(dozw, N_PODMIANA, replace=False)]
        kl = wczytaj_klatki(js)
        for k, j in enumerate(js):
            X[R[k]] = kl[j][0]; wiersze[R[k]]['ramki'] = kl[j][1]
            wiersze[R[k]]['frame_id'] = f"pula/{Pl['man'][j]['film']}/{Pl['man'][j]['frame']}"
        return D.Zbior(wiersze, X), [f"{Pl['man'][j]['film']}/{Pl['man'][j]['frame']}" for j in js]

    wyniki = czytaj('ETAP3_UCZENIE.json') or dict(ramiona={})
    for ramie in ramiona:
        if ramie in wyniki['ramiona']:
            continue
        zb, opis_podm = zbior(ramie)
        hist = []
        lat_cel = {}

        def ocena(u):
            u.net.eval()
            with torch.no_grad():
                x = torch.from_numpy(np.stack(Xc)).permute(0, 3, 1, 2).float().cuda() / 255
                u.net.zapisuj_latent(True); o, z = u.net(x); u.net.zapisuj_latent(False)
            bx, sc = dekoduj(o.float().cpu())
            m = [KL.ocena_obrazu(bx[i].numpy(), sc[i].numpy(), Rc[i]) for i in range(len(C))]
            rek = dict(krok=u.krok, iou_celow=float(np.mean([v['iou_top'] for v in m])),
                       pewnosc_celow=float(np.mean([v['pewnosc_top'] for v in m])),
                       ap50_celow=KL.ap(list(zip(bx.numpy(), sc.numpy())), Rc, .5))
            if u.krok in (500, 1000):
                rek['map50_wal3000'] = float(T.ap50(u.wykryj(wal_X), [r['ramki'] for r in wal_w]))
            if u.krok == 1000:
                lat_cel['z'] = z.float().cpu().numpy()
            hist.append(rek); u.net.train()
            log(ramie, rek)

        t = time.time()
        with GU.deterministycznie(True):
            u = T.Uczen(47); rng = np.random.default_rng(47)
            u.kroki(zb, 1000, rng, co_ocena=100, ocena=ocena)
        torch.save(dict(net=u.net.state_dict(), krok=u.krok), OUT / f'model_ramie{ramie}_k1000.pt')
        np.save(OUT / f'LATENT_CELOW_ramie{ramie}_k1000.npy', lat_cel['z'].astype(np.float16))
        wyniki['ramiona'][ramie] = dict(historia=hist, podmienione=opis_podm, pozycje_bazy=R.tolist(), czas_s=time.time() - t)
        zapisz('ETAP3_UCZENIE.json', wyniki)
    # ruch latentu celów względem ramienia A w stronę korekty oracle
    zA = np.load(OUT / 'LATENT_CELOW_ramieA_k1000.npy').astype(np.float32)
    for ramie in wyniki['ramiona']:
        if ramie == 'A':
            continue
        z = np.load(OUT / f'LATENT_CELOW_ramie{ramie}_k1000.npy').astype(np.float32)
        dz = (z - zA).reshape(len(C), -1); df = d_fin.reshape(len(C), -1)
        cos = (dz * df).sum(1) / (np.linalg.norm(dz, axis=1) * np.linalg.norm(df, axis=1) + 1e-12)
        wyniki['ramiona'][ramie]['cos_ruchu_celow_do_korekty'] = cos.tolist()
        wyniki['ramiona'][ramie]['cos_srednio'] = float(cos.mean())
    zapisz('ETAP3_UCZENIE.json', wyniki)
    return wyniki


# ---------------------------------------------------------------------------------------------------- etap 4: mała korekta
def etap4(cele, n=8, ulamki=(0.1, 0.3)):
    """Korekta z ograniczoną normą ||δ|| <= u ||z|| (ten sam cel detekcja): czy pozostaje skuteczna i czy wtedy SD
    i konwerter ją realizują (obraz SD: detekcja z etykietą celu, odległość latentu SD od z_w)."""
    if czytaj('ETAP4_MALA.json'):
        return czytaj('ETAP4_MALA.json')
    import cv2
    import silnik
    import kandydaci as KD
    from model_snapshot import dekoduj
    s = st(); rows = KL.manifest()
    wyn = []
    for c in cele['cele'][:n]:
        rgb, ram = KL.obraz_manifestu(rows[c['i']])
        z0 = np.load(Path(c['kat']) / 'KONIEC.npz')['z_baza']
        for u in ulamki:
            rho = float(u * np.linalg.norm(z0))
            spec = dict(tryb='zakresy', kanaly=np.arange(768), tokeny=0, cel='detekcja', cel_pewnosc=.95, norma='calosc',
                        rho=rho, kroki=300, lr=.05, zapis_co=100, ziarno=0)
            w = KL.oblicz_korekte(s, rgb, ram, spec, dict(id=f'm{c["i"]:04d}_mala{u}'))
            d = np.load(Path(w['kat']) / 'KONIEC.npz')['d_najlepszy']
            wyn.append(dict(cel=c['i'], ulamek=u, rho=rho, kat=w['kat'], iou=(w['przed']['iou_top'], w['najlepszy']['iou_top']),
                            pewnosc=(w['przed']['pewnosc_top'], w['najlepszy']['pewnosc_top']),
                            niewidoczna=w['najlepszy']['fc2']['udzial_niewidocznej'], zw=None))
            np.save(OUT / f'zw_m{c["i"]:04d}_u{u}.npy', (z0 + d).astype(np.float32))
            log('mała', c['i'], u, wyn[-1]['iou'], wyn[-1]['pewnosc'])
    trained = silnik.zaladuj_model()
    kon = KD._konwerter(); P = pula(); dozw = np.flatnonzero(P['dozw'])
    for r in wyn:
        zw = np.load(OUT / f'zw_m{r["cel"]:04d}_u{r["ulamek"]}.npy')
        z0 = np.load(Path(r['kat']) / 'KONIEC.npz')['z_baza']
        img = silnik.generuj(trained, zw, kroki=30, ziarno=0)
        x = cv2.resize((img.transpose(1, 2, 0) * 255).astype(np.uint8), (480, 320), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / 'sd' / f'm{r["cel"]:04d}_mala{r["ulamek"]}.png'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        Zs, Os = latenty([x]); zs = Zs[0].numpy()
        bx, sc = dekoduj(Os)
        ram = np.asarray([c for c in cele['cele'] if c['i'] == r['cel']][0]['ramki'])
        m = KL.ocena_obrazu(bx[0].numpy(), sc[0].numpy(), ram)
        dw = zw - z0
        r.update(sd_odl_wzgl=float(np.linalg.norm(zs - zw) / np.linalg.norm(dw)),
                 sd_cos_do_korekty=float(((zs - z0) * dw).sum() / (np.linalg.norm(zs - z0) * np.linalg.norm(dw) + 1e-12)),
                 sd_iou_z_etykieta_celu=m['iou_top'], sd_pewnosc=m['pewnosc_top'])
        with torch.no_grad():
            cls, _ = kon.predict(zw[None])
        q = np.asarray(cls.cpu(), np.float32).reshape(-1); q /= np.linalg.norm(q) + 1e-12
        top = dozw[np.argsort(-(P['F'][dozw] @ q))[:10]]
        kl = wczytaj_klatki(top)
        Zc, _ = latenty([kl[int(j)][0] for j in top]); Zc = Zc.numpy()
        cosk = ((Zc - z0).reshape(10, -1) @ dw.reshape(-1)) / (np.linalg.norm((Zc - z0).reshape(10, -1), axis=1) * np.linalg.norm(dw) + 1e-12)
        r.update(konwerter_cos_najlepszy=float(cosk.max()), konwerter_cos_sredni=float(cosk.mean()),
                 konwerter_filmy=[P['man'][j]['film'] for j in top[:3]])
        log('mała SD', r['cel'], r['ulamek'], round(r['sd_cos_do_korekty'], 3), r['sd_iou_z_etykieta_celu'], round(r['konwerter_cos_najlepszy'], 3))
    del trained; torch.cuda.empty_cache()
    out = dict(wyniki=wyn, opis='korekta z ||δ|| <= ułamek · ||z||; SD z z_w; cos = cos(z(obraz) − z(cel), z_w − z(cel))')
    zapisz('ETAP4_MALA.json', out)
    return out


def etap2d(cele, kon):
    """Pchnięcie w kroku 990 rozłożone względem fc2: kosinus S z częścią korekty przechodzącą przez fc2 (δ_vis) i
    niewidoczną (δ_inv), oraz zmiana wyjścia końcówki <W2 S, W2 δ>. Źródła: sam obraz celu, najlepsza klatka konwertera
    (p300), 3 losowe klatki puli."""
    if czytaj('ETAP2D_FC2.json'):
        return czytaj('ETAP2D_FC2.json')
    import galezie_uczenia as GU
    from pchniecie import Pchniecie
    from stan_uczenia import Odtworzenie
    b = GU.bieg_gestej_historii(); od = Odtworzenie(b); od.przygotuj()
    rows = KL.manifest(); C = cele['cele']
    X = np.stack([KL.obraz_manifestu(rows[c['i']])[0] for c in C])
    P = Pchniecie(od, X, np.stack([KL.maski_gt(np.asarray(c['ramki']))['obiekt'] for c in C]), [np.asarray(c['ramki']) for c in C])
    P.przygotuj(990, 0)
    Vr = st().Vr; W2 = st().blok.mlp.fc2.weight.detach()
    Pl = pula(); dozw = np.flatnonzero(Pl['dozw']); rng = np.random.default_rng(5)
    kon_m = {(w['cel'], w['punkt']): w for w in kon['wyniki']}
    wyn = []
    cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    for t, c in enumerate(C):
        d = torch.from_numpy(np.load(Path(c['kat']) / 'KONIEC.npz')['d_najlepszy']).to(P.dev)
        dv = (d @ Vr) @ Vr.T; di = d - dv
        js = [('konwerter p300', kon_m[(c['i'], 300)]['kandydaci'][0]['j'])] + [('losowa z puli', int(j)) for j in rng.choice(dozw, 3, replace=False)]
        kl = wczytaj_klatki([j for _, j in js])
        kand = [('cel (sam obraz z etykietą)', X[t], np.asarray(c['ramki']))] + [(n, kl[j][0], kl[j][1]) for n, j in js]
        for n, im, ram in kand:
            S = P.pchniecie_kandydata(im, ram, t)
            Sv = (S @ Vr) @ Vr.T
            wyn.append(dict(cel=c['i'], zrodlo=n, cos_calosc=cos(S, d), cos_widoczna=cos(S, dv), cos_niewidoczna=cos(S, di),
                            cos_wyjscia_fc2=cos(S @ W2.T, d @ W2.T), udzial_widocznej_S=float((Sv ** 2).sum() / (S ** 2).sum()),
                            udzial_widocznej_d=float((dv ** 2).sum() / (d ** 2).sum())))
        log('fc2', c['i'], {r['zrodlo'][:8]: (round(r['cos_widoczna'], 3), round(r['cos_wyjscia_fc2'], 3)) for r in wyn[-4:]})
    pods = {z: {m: float(np.mean([r[m] for r in wyn if r['zrodlo'] == z])) for m in
                ('cos_calosc', 'cos_widoczna', 'cos_niewidoczna', 'cos_wyjscia_fc2', 'udzial_widocznej_S', 'udzial_widocznej_d')}
            for z in sorted({r['zrodlo'] for r in wyn})}
    out = dict(wyniki=wyn, podsumowanie=pods, opis='S = pchnięcie latentu celu w kroku 990; δ = korekta oracle przy k1000; '
                                                   'cos_wyjscia_fc2 = cos(W2 S, W2 δ): zgodność zmiany widzianej przez końcówkę')
    zapisz('ETAP2D_FC2.json', out)
    return out


def kontrola_losowa(cele, n=20):
    """cos(z(klatka) − z(cel), δ) dla losowych klatek puli: poziom odniesienia dla klatek z konwertera."""
    if czytaj('KONTROLA_LOSOWA.json'):
        return czytaj('KONTROLA_LOSOWA.json')
    P = pula(); dozw = np.flatnonzero(P['dozw']); rng = np.random.default_rng(11)
    js = rng.choice(dozw, n, replace=False)
    kl = wczytaj_klatki(js)
    Zc, _ = latenty([kl[int(j)][0] for j in js]); Zc = Zc.numpy().reshape(n, -1)
    wyn = {}
    for c in cele['cele']:
        for p in PUNKTY:
            zw, z0 = punkt(c, p)
            dw = (zw - z0).reshape(-1); D = Zc - z0.reshape(-1)
            cos = D @ dw / (np.linalg.norm(D, axis=1) * np.linalg.norm(dw) + 1e-12)
            wyn[f'{c["i"]}_{p}'] = dict(sredni=float(cos.mean()), max=float(cos.max()))
    out = dict(wyniki=wyn, sredni=float(np.mean([v['sredni'] for v in wyn.values()])),
               max_sredni=float(np.mean([v['max'] for v in wyn.values()])))
    zapisz('KONTROLA_LOSOWA.json', out)
    return out


if __name__ == '__main__':
    etapy = sys.argv[1:] or ['1', '2a', '2b', '2c', '3']
    c = etap1()
    k = etap2a(c) if '2a' in etapy or '2c' in etapy or '3' in etapy else None
    s = etap2b(c) if '2b' in etapy or '2c' in etapy or '3' in etapy else None
    if '2c' in etapy:
        etap2c(c, k, s)
    if '3' in etapy:
        etap3(c, k, s)
    if '4' in etapy:
        etap4(c)
    if '2d' in etapy:
        etap2d(c, k)
    if 'kl' in etapy:
        kontrola_losowa(c)
    log('KONIEC')
