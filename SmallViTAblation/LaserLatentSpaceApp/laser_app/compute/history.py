"""Dane zakładki „Lasery i koloidy”: historia uczenia JEDNEGO zdjęcia przez wybrane checkpointy, w układzie z filmów 2, 12 i 16
(`outputs/porownanie_szklo_20261010/film_krysztal`, zasady w ZASADY_LASER_KOLOID.md).

Dla każdego checkpointu (forward na GPU, wagi wczytywane po kolei, strict=True, model MalyViT biegu 35 000):
  - latent z = GELU fc1 ostatniego bloku, 600 tokenów × 768 neuronów („natężenia” 768 reflektorów na 600 cząsteczkach) → Z.npy (float16, mmap);
  - fc2 ostatniego bloku (192 × 768): kierunki pchnięcia v_c; rzut P (3 × 192) = 3 główne kierunki znormowanych v_c z OSTATNIEGO checkpointu;
    d_c = P v̂_c (kierunek wiązki w sferze), położenie cząsteczki p_j = P (x̃_j − x̄) = rzut jej STANU tuż przed MLP bloku 6
    (wersja 2; w wersji 1, jak w filmie 12, było p_j = P W₂ (z_j − z̄), czyli przesunięcie od bloku, a nie stan). Przesunięcie
    cząsteczki przez MLP to dokładnie Σ_c (z_jc − z̄_c) P v_c, czyli suma wiązek;
  - atencja 6 bloków: średnia głów, bez przekątnej, ile uwagi patch DOSTAJE (definicja aplikacji i filmu 16);
  - wyjście detekcji: najpewniejsza ramka, pewność, IoU z ramką prawdziwą (gdy jest);
  - pary przeciwnych reflektorów (odsetek kanałów z partnerem cos < −0,5);
  - obroty igieł (wersja 3, film 23): prędkość kątowa v̂_c w 192D [°/100 kroków] i kierunek obrotu d_c w rzucie sfery, z checkpointów
    odległych o ≥ 100 kroków (drganie) i ≥ 2000 kroków (dryf netto). To rzeczywisty krok uczenia (gradient przez AdamW; checkpointy
    nie mają stanu optymalizatora); weight decay tylko skraca v_c, więc obrót pochodzi z gradientu.
Rozkład ruchu z filmu 2 (jak eden/szescian_prep.py): x = z − średnia tokenów w losowym rzucie 128D (ziarno 0), x0 = krok pierwszy,
centrum studni c = średnia x w oknie ±okno kroków, D = 3 główne kierunki przesunięć x − x0, s = 1 / odstęp sąsiadów w pierwszym kroku;
atomy = g + s·D(x − x0), węzły = g + s·D(c − x0), g = siatka patchy. Prokrustes 2D węzłów do g w każdym checkpoincie daje ruch globalny
(przesunięcie, obrót, skala), węzły w układzie współporuszającym (odkształcenie lokalne) i atomy wokół węzłów (drgania).
Wynik w `cache_uczenia/lasery/<skrót>/` (dane.npz + Z.npy); ten sam wybór wczytuje się od razu z dysku.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np

from laser_app.compute.common import Interrupted  # noqa: F401  (jedna klasa dla wszystkich liczeń; także jako <moduł>.Interrupted)
from laser_app.paths import CACHE_DIR, PROJECT_ROOT as ROOT

KOD_MODELU = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py'
CACHE = CACHE_DIR / 'lasery'
FILM = ROOT / 'outputs/porownanie_szklo_20261010/film_krysztal'
ZDJECIE_FILMU = FILM / 'zdjecie_train0.jpg'
H, W, TOK, NK, KOMORKA = 20, 30, 600, 768, 16
R_SFERY, R_REFLEKTOROW = 34.0, 50.0
WERSJA = 3                                   # 2: położenie cząsteczki = stan x̃ przed MLP bloku 6 (nie suma wiązek); 3: obroty igieł
NA_EPOKE = 1094
OKNO_DRGANIE, OKNO_DRYF = 100, 2000          # obrót igły v̂_c: w ≥ 100 krokach (drganie) i netto w ≥ 2000 krokach (dryf), jak film 23


def obroty(kr, VN, DN, okno):
    """Obrót igieł reflektorów między checkpointami odległymi o ≥ okno kroków (do przodu; przy końcu uczenia wstecz), jak w filmie 23.
    VN: lista K macierzy (192, 768) znormowanych kolumn fc2, DN: (K, 768, 3) kierunki d_c w rzucie sfery.
    Zwraca (K, 768, 4): xyz = kierunek obrotu d_c (rzut 3D, część prostopadła do igły), w = prędkość kątowa v̂_c w 192D [°/100 kroków]."""
    kr = np.asarray(kr, np.int64); K = len(kr)
    b = np.minimum(np.searchsorted(kr, kr + okno), K - 1)
    a = np.minimum(np.arange(K), np.searchsorted(kr, kr[b] - okno, side='right') - 1)
    W = np.zeros((K, NK, 4), np.float32)
    for i in range(K):
        A, B = VN[a[i]].astype(np.float32), VN[b[i]].astype(np.float32)
        W[i, :, 3] = np.degrees(np.arccos(np.clip((A * B).sum(0), -1, 1))) / max((kr[b[i]] - kr[a[i]]) / 100.0, 1e-9)
        t = DN[b[i]] - DN[a[i]]
        t -= (t * DN[i]).sum(1, keepdims=True) * DN[i]
        W[i, :, :3] = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return W

_model = None


def modul_modelu():
    """model.py biegu 35 000 (MalyViT, dekoduj) wczytany pod własną nazwą, bez dopisywania ścieżek do sys.path."""
    global _model
    if _model is None:
        spec = importlib.util.spec_from_file_location('maly_vit_35000_model', KOD_MODELU)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _model = m
    return _model


# ------------------------------------------------------------------ zbiór uczący biegu 35 000 (te same klatki i ramki co w treningu)
MANIFEST_UCZACY = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl'
ZIPY = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
PREFIKS_UCZACY = 'uczacy:'                   # identyfikator klatki zbioru uczącego: 'uczacy:<wiersz manifestu>'
_uczacy = None


def zbior_uczacy():
    """Wiersze MANIFEST_35000 (kolejność pliku = indeks): film, klatka, archiwum, member, sha, (w, h) i ramka xyxy w 480 × 320,
    skalowana osobno w X i Y dokładnie jak w trening35000.wiersze_35000."""
    global _uczacy
    if _uczacy is None:
        import pickle
        st = MANIFEST_UCZACY.stat()
        zapis = CACHE / f'_manifest_35000_{st.st_size}_{int(st.st_mtime)}.pkl'      # start aplikacji: 0,05 s zamiast ~1 s parsowania
        try:
            _uczacy = pickle.loads(zapis.read_bytes())
            return _uczacy
        except (OSError, pickle.UnpicklingError, EOFError):
            pass
        out = []
        with open(MANIFEST_UCZACY, encoding='utf-8') as plik:
            linie = plik.readlines()
        for linia in linie:
            r = json.loads(linia)
            sx, sy = W * KOMORKA / r['width'], H * KOMORKA / r['height']
            x, y, bw, bh = r['gt'][0]['bbox_xywh']
            out.append(dict(nr=len(out), film=r['film'], klatka=int(r['frame']), member=r['member'],
                            archiwum=r['archive'].replace(chr(92), '/').rsplit('/', 1)[-1], sha=r['image_sha256'],
                            wh=(int(r['width']), int(r['height'])), ramka=[x * sx, y * sy, (x + bw) * sx, (y + bh) * sy],
                            kategoria=r['gt'][0]['category']))
        _uczacy = out
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            zapis.write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
        except OSError:
            pass
    return _uczacy


def _wiersz_uczacy(sciezka):
    s = str(sciezka)
    return zbior_uczacy()[int(s[len(PREFIKS_UCZACY):])] if s.startswith(PREFIKS_UCZACY) else None


def czy_w_zbiorze_uczacym(sciezka):
    """Wiersz zbioru uczącego dla pliku nazwanego jak TR_0001_S1_P03_klatka_000870.jpg albo None."""
    import re
    m = re.search(r'(TR_\d+_S\d+_P\d+)\D*?(\d+)\.(?:jpg|jpeg|png)$', Path(sciezka).name, re.I)
    if not m:
        return None
    film, klatka = m.group(1), int(m.group(2))
    return next((r for r in zbior_uczacy() if r['film'] == film and r['klatka'] == klatka), None)


# ------------------------------------------------------------------ zdjęcie i ramka prawdziwa
def wczytaj_zdjecie(sciezka):
    """RGB uint8 320 × 480 (INTER_AREA jak w uczeniu) i rozmiar natywny (w, h). `sciezka` to plik albo 'uczacy:<nr>' (klatka
    zbioru uczącego czytana z archiwum zip, SHA-256 bajtów sprawdzane z manifestem, jak w trening35000.wczytaj_z_zipow)."""
    import cv2
    r = _wiersz_uczacy(sciezka)
    if r is not None:
        import zipfile
        with zipfile.ZipFile(ZIPY / r['archiwum']) as z:
            raw = z.read(r['member'])
        if hashlib.sha256(raw).hexdigest() != r['sha']:
            raise ValueError(f'SHA-256 klatki niezgodne z manifestem: {r["film"]} kl. {r["klatka"]}')
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[1::-1] != tuple(r['wh']):
            raise ValueError(f'zła klatka {r["film"]} kl. {r["klatka"]}')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(cv2.resize(im, (W * KOMORKA, H * KOMORKA), interpolation=cv2.INTER_AREA)), r['wh']
    im = cv2.imread(str(sciezka), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(sciezka)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    return np.ascontiguousarray(cv2.resize(im, (W * KOMORKA, H * KOMORKA), interpolation=cv2.INTER_AREA)), (w, h)


def ramka_prawdziwa(sciezka, rozmiar):
    """Ramka narzędzia xyxy w pikselach wejścia 480 × 320 albo None, i opis skąd. Klatka zbioru uczącego (także plik o nazwie
    takiej klatki) dostaje ramkę z manifestu, tę samą, której użył trening."""
    r = _wiersz_uczacy(sciezka)
    if r is not None:
        return list(r['ramka']), f'zbiór uczący 35 000, nr {r["nr"]}: {r["film"]} kl. {r["klatka"]} ({r["kategoria"]})'
    sciezka = Path(sciezka)
    r = czy_w_zbiorze_uczacym(sciezka)
    if r is not None and tuple(rozmiar) == tuple(r['wh']):
        return list(r['ramka']), f'plik = klatka zbioru uczącego nr {r["nr"]}: {r["film"]} kl. {r["klatka"]} ({r["kategoria"]})'
    if sciezka.resolve() == ZDJECIE_FILMU.resolve():
        p = FILM / 'atencja_train0.npz'
        if p.exists():
            return [float(v) for v in np.load(p)['gt']], 'ramka z filmów 2/16 (TRAIN 0)'
    try:
        from laser_app.compute.tool_labels import etykieta_dla_zdjecia
        r, opis = etykieta_dla_zdjecia(sciezka)
    except Exception as e:                                          # brak adnotacji/zipów: bez ramki
        return None, f'brak ramki ({type(e).__name__})'
    opis += ' — POZA zbiorem uczącym 35 000 (model nie uczył się na tej klatce)'
    if r is None:
        return None, opis
    sx, sy = W * KOMORKA / rozmiar[0], H * KOMORKA / rozmiar[1]
    return [r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy], opis


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    p = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - p
    return p / u if u > 0 else 0.0


def siatka():
    """Węzły idealne g [600, 2]: (kolumna, wiersz) patcha względem środka siatki, jednostka = 1 komórka."""
    s = np.stack([np.arange(TOK) % W, np.arange(TOK) // W], 1).astype(np.float64)
    return s - s.mean(0)


def maska_narzedzia(gt):
    if gt is None:
        return np.zeros(TOK, bool)
    cx = (np.arange(TOK) % W + 0.5) * KOMORKA; cy = (np.arange(TOK) // W + 0.5) * KOMORKA
    return (cx >= gt[0]) & (cx <= gt[2]) & (cy >= gt[1]) & (cy <= gt[3])


# ------------------------------------------------------------------ Prokrustes (jak film_krysztal/lasery_wspolne.py)
def prokrustes2d(Q, g):
    K = len(Q); par = np.zeros((K, 4)); mapy = []
    mg = g.mean(0); gc = g - mg
    for t in range(K):
        mq = Q[t].mean(0); qc = Q[t] - mq
        U_, S_, Vt = np.linalg.svd(qc.T @ gc); dd = np.sign(np.linalg.det(U_ @ Vt)); Dg = np.diag([1, dd]); R = U_ @ Dg @ Vt
        sk = float((S_ * np.diag(Dg)).sum() / max((qc ** 2).sum(), 1e-12))
        par[t] = [*(mq - mg), np.degrees(np.arctan2(R[1, 0], R[0, 0])), 1 / sk]
        mapy.append((mq, R, sk, mg))
    return par, mapy


def zastosuj(P, mapa):
    mq, R, sk, mg = mapa
    return (P - mq) @ R * sk + mg


# ------------------------------------------------------------------ obliczenie historii
def klucz(wagi, kroki, rgb, gt, okno):
    h = hashlib.sha1(json.dumps(dict(w=WERSJA, wagi=str(Path(wagi).resolve()), kroki=[int(k) for k in kroki], gt=gt,
                                     okno=int(okno))).encode())
    h.update(np.ascontiguousarray(rgb).tobytes())
    return h.hexdigest()[:16]




def klucz_przenosny(kroki, rgb, gt, okno, wagi, wersja=WERSJA):
    """Klucz historii niezależny od dysku: wersja formatu danych, dwa ostatnie człony ścieżki wag (bieg uczenia, np.
    MalyViT_35000_od_zera_s47/wagi), checkpointy, okno, ramka GT i zdjęcie. Historia policzona na innym dysku (F: zamiast W:)
    albo zaimportowana z innego komputera ma inny `klucz`, ale ten sam klucz przenośny."""
    from pathlib import PureWindowsPath
    bieg = '/'.join(PureWindowsPath(str(wagi)).parts[-2:])               # ukośniki / i \ jednakowo
    h = hashlib.sha1(json.dumps(dict(w=int(wersja), bieg=bieg, kroki=[int(k) for k in kroki], okno=int(okno),
                                     gt=None if gt is None else [round(float(v), 3) for v in gt])).encode())
    h.update(np.ascontiguousarray(np.asarray(rgb, np.uint8)).tobytes())
    return h.hexdigest()[:16]


def znajdz_przenosna(kp, katalog=None):
    """Katalog historii w cache o kluczu przenośnym kp (albo None). Najpierw szybki filtr po opis.json (liczba kroków), potem
    kroki i zdjęcie z dane.npz; wynik zapisywany w opis.json (`klucz_przenosny_v2`), żeby następnym razem wystarczył odczyt opisu."""
    for kat in sorted(Path(katalog or CACHE).iterdir()):
        if not (kat / 'dane.npz').exists() or kat.name.endswith('_tmp'):
            continue
        opis_p = kat / 'opis.json'
        try:
            opis = json.loads(opis_p.read_text(encoding='utf-8')) if opis_p.exists() else {}
        except (OSError, ValueError):
            opis = {}
        if 'klucz_przenosny_v2' not in opis:
            if 'wagi' not in opis:
                continue
            try:
                with np.load(kat / 'dane.npz', allow_pickle=False) as d:
                    opis['klucz_przenosny_v2'] = klucz_przenosny(d['kroki'], d['rgb'], opis.get('gt'), opis.get('okno', 300),
                                                                 opis['wagi'], opis.get('wersja', 1))
                if opis_p.exists():
                    opis_p.write_text(json.dumps(opis, ensure_ascii=False, indent=1), encoding='utf-8')
            except (OSError, KeyError, ValueError):
                continue
        if opis['klucz_przenosny_v2'] == kp and (kat / 'Z.npy').exists():
            return kat
    return None


def policz(wagi, kroki, rgb, gt, okno=300, postep=None, przerwij=None, urzadzenie='cuda'):
    """Liczy (albo bierze z cache) historię zdjęcia rgb [320, 480, 3] dla checkpointów `kroki` z katalogu `wagi` (k<krok>.pt).
    Zwraca katalog wyniku. postep(i, n, opis) co checkpoint; przerwij() → True kończy (wyjątek Przerwano, nic nie zostaje)."""
    import torch
    kroki = [int(k) for k in sorted(set(kroki))]
    if len(kroki) < 2:
        raise ValueError('potrzeba co najmniej dwóch checkpointów')
    out = CACHE / klucz(wagi, kroki, rgb, gt, okno)
    if (out / 'dane.npz').exists() and (out / 'Z.npy').exists():
        return out
    inny = znajdz_przenosna(klucz_przenosny(kroki, rgb, gt, okno, wagi)) if CACHE.exists() else None
    if inny is not None:                                                  # ta sama historia z inną ścieżką wag (import, inny dysk)
        return inny
    tmp = out.with_name(out.name + '_tmp')
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    M = modul_modelu()
    net = M.MalyViT().to(dev).eval().requires_grad_(False)
    pam = {}
    net.bloki[-1].mlp.act.register_forward_hook(lambda m, a, o: pam.__setitem__('z', o.detach()))
    net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))   # stan x̃ przed MLP bloku 6
    for b in range(len(net.bloki)):
        net.bloki[b].attn.register_forward_hook(lambda m, a, o, b=b: pam.__setitem__(b, a[0].detach()))
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
    sciezka = lambda k: Path(wagi) / f'k{k:05d}.pt'

    def wczytaj(k):
        st = torch.load(sciezka(k), map_location=dev, weights_only=True)
        net.load_state_dict(st.get('net', st), strict=True)

    # rzut P z ostatniego checkpointu: 3 główne kierunki znormowanych kolumn fc2 (film 12)
    wczytaj(kroki[-1])
    V = net.bloki[-1].mlp.fc2.weight.detach().double()                                   # (192, 768)
    Vn = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
    U_, _, _ = torch.linalg.svd(Vn - Vn.mean(1, keepdim=True), full_matrices=False)
    P3 = U_[:, :3].T.float()                                                              # (3, 192)
    g_r = torch.Generator().manual_seed(0)
    Rrp = (torch.randn(NK, 128, generator=g_r) / 128 ** 0.5).to(dev)                     # losowy rzut 128D (jak pomiary Edena)
    K = len(kroki)
    Z = np.lib.format.open_memmap(tmp / 'Z.npy', 'w+', np.float16, (K, TOK, NK))
    Y = np.zeros((K, TOK, 128), np.float32)
    POS = np.zeros((K, TOK, 3), np.float32); DIRS = np.zeros((K, NK, 3), np.float32)
    ATN = np.zeros((K, 6, TOK), np.float32); RAMKA = np.zeros((K, 4), np.float32); PEW = np.zeros(K, np.float32)
    PARY = np.zeros(K, np.float32)
    VN = []                                                                               # znormowane kolumny fc2 (obroty igieł)
    oko = torch.eye(NK, dtype=torch.bool, device=dev)
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    pula = ThreadPoolExecutor(4)                                      # odczyt plików z wyprzedzeniem (dysk jest wąskim gardłem)
    odczyt = lambda k: (lambda st: st.get('net', st))(torch.load(sciezka(k), map_location='cpu', weights_only=True))
    kolejka = [pula.submit(odczyt, k) for k in kroki[:6]]
    for i, k in enumerate(kroki):
        if przerwij is not None and przerwij():
            for f in kolejka:
                f.cancel()
            pula.shutdown(wait=True)
            del Z
            shutil.rmtree(tmp, ignore_errors=True)
            raise Interrupted()
        net.load_state_dict(kolejka.pop(0).result(), strict=True)
        if i + 6 < K:
            kolejka.append(pula.submit(odczyt, kroki[i + 6]))
        with torch.no_grad():
            o, _ = net(x)
            z = pam['z'][0].float()                                                       # (600, 768)
            Z[i] = z.half().cpu().numpy()
            zc = z - z.mean(0, keepdim=True)
            Y[i] = (zc @ Rrp).cpu().numpy()
            V = net.bloki[-1].mlp.fc2.weight.float()                                     # (192, 768)
            PV = P3 @ V                                                                   # (3, 768)
            xs = pam['x'][0].float()                                                      # (600, 192)
            DIRS[i] = PV.T.cpu().numpy(); POS[i] = ((xs - xs.mean(0, keepdim=True)) @ P3.T).cpu().numpy()
            Vn = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
            VN.append(Vn.half().cpu().numpy())
            cs = (Vn.T @ Vn).masked_fill(oko, 0.0)
            PARY[i] = float((cs.min(1).values < -0.5).float().mean())
            for b in range(len(net.bloki)):
                at, xin = net.bloki[b].attn, pam[b]
                B_, N_, C_ = xin.shape
                qkv = at.qkv(xin).reshape(B_, N_, 3, at.num_heads, C_ // at.num_heads).permute(2, 0, 3, 1, 4)
                A = ((qkv[0] * at.scale) @ qkv[1].transpose(-2, -1)).softmax(-1).float().mean(1)[0]
                A.fill_diagonal_(0); A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
                ATN[i, b] = A.mean(0).cpu().numpy()
            bx, sc = M.dekoduj(o.detach().float().cpu(), k=1)
            RAMKA[i] = bx[0, 0].numpy(); PEW[i] = float(sc[0, 0])
        if postep is not None:
            postep(i + 1, K, f'checkpoint {i + 1}/{K} (krok {k}), {time.time() - t0:.0f} s')
    pula.shutdown(wait=False)
    Z.flush(); del Z
    if postep is not None:
        postep(K, K, 'rozkład ruchu na trzy składowe...')
    kr = np.array(kroki, np.int64)
    g = siatka()
    # --- rozkład ruchu (film 2 / eden/szescian_prep.py)
    from scipy.spatial.distance import cdist
    x0 = Y[0]
    Dm = cdist(x0, x0); np.fill_diagonal(Dm, np.inf)
    a0 = float(np.median(np.sort(Dm, 1)[:, :4].mean(1)))
    cs = np.cumsum(Y, 0, dtype=np.float64)
    lo = np.searchsorted(kr, kr - okno, 'left'); hi = np.searchsorted(kr, kr + okno, 'right')
    C = np.empty_like(Y)
    for t in range(K):
        s = cs[hi[t] - 1] - (cs[lo[t] - 1] if lo[t] > 0 else 0)
        C[t] = s / (hi[t] - lo[t])
    del cs
    dX = (Y - x0).reshape(-1, 128)
    _, s_, Vt = np.linalg.svd(dX[::7], full_matrices=False)
    D = Vt[:3]
    skala = 1.0 / max(a0, 1e-9)
    g3 = np.concatenate([g, np.zeros((TOK, 1))], 1)
    ATOMY = (g3[None] + skala * np.einsum('tid,kd->tik', Y - x0, D)[..., [1, 2, 0]]).astype(np.float32)
    WEZLY = (g3[None] + skala * np.einsum('tid,kd->tik', C - x0, D)[..., [1, 2, 0]]).astype(np.float32)
    del Y, C
    Ql, Al = WEZLY[..., :2].astype(np.float64), ATOMY[..., :2].astype(np.float64)
    par, mapy = prokrustes2d(Ql, g)
    QC = np.stack([zastosuj(Ql[t], mapy[t]) for t in range(K)]).astype(np.float32)
    AC = np.stack([zastosuj(Al[t], mapy[t]) for t in range(K)]).astype(np.float32)
    GLB = np.stack([(g - mapy[t][3]) @ mapy[t][1].T / mapy[t][2] + mapy[t][0] for t in range(K)]).astype(np.float32)
    narz = maska_narzedzia(gt)
    LOK = np.linalg.norm(QC - g, axis=-1).astype(np.float32); VIB = np.linalg.norm(AC - QC, axis=-1).astype(np.float32)
    GLOB = np.linalg.norm(GLB - g, axis=-1).mean(1).astype(np.float32)
    IOU = np.array([iou(RAMKA[t], gt) for t in range(K)], np.float32) if gt is not None else np.full(K, np.nan, np.float32)
    # --- sfera reflektorów (film 12): stałe reflektory na spirali Fibonacciego, przypisanie węgierskie do −d_c (koniec)
    from scipy.optimize import linear_sum_assignment
    from laser_app.compute.common import kierunki, kolory_kanalow
    DIRS_N = DIRS / np.maximum(np.linalg.norm(DIRS, axis=2, keepdims=True), 1e-12)
    FIB = kierunki(False).astype(np.float64)
    r_, c_ = linear_sum_assignment(FIB @ DIRS_N[-1].T)
    SLOT = np.empty(NK, int); SLOT[c_] = r_
    RFIX = (R_REFLEKTOROW * FIB[SLOT]).astype(np.float32)
    KOL = kolory_kanalow()[:, :3][SLOT].astype(np.float32)
    R99 = np.percentile(np.linalg.norm(POS, axis=2), 99, axis=1).astype(np.float32)
    # --- obroty igieł (gdzie w układzie jest gradient): rzeczywisty krok AdamW między checkpointami, film 23
    OBR_DRG, OBR_DRYF = obroty(kr, VN, DIRS_N, OKNO_DRGANIE), obroty(kr, VN, DIRS_N, OKNO_DRYF)
    V0 = VN[0].astype(np.float32)
    OBR_KAT0 = np.array([np.median(np.degrees(np.arccos(np.clip((V0 * v.astype(np.float32)).sum(0), -1, 1)))) for v in VN], np.float32)
    del VN
    np.savez(tmp / 'dane.npz', obr_drg=OBR_DRG, obr_dryf=OBR_DRYF, obr_kat0=OBR_KAT0, kroki=kr, rgb=rgb, gt=np.array(gt if gt is not None else [np.nan] * 4, np.float32),
             narz=narz, g=g.astype(np.float32), atomy=ATOMY, wezly=WEZLY, qc=QC, ac=AC, glb=GLB, par=par.astype(np.float32),
             lok=LOK, vib=VIB, glob=GLOB, atn=ATN, ramka=RAMKA, pew=PEW, iou=IOU, pos=POS, dirs=DIRS_N.astype(np.float32),
             r99=R99, rfix=RFIX, kol=KOL, slot=SLOT, pary=PARY, a0=a0, okno=okno, udzial_D=float((s_[:3] ** 2).sum() / (s_ ** 2).sum()),
             wagi=str(wagi), na_epoke=NA_EPOKE)
    (tmp / 'opis.json').write_text(json.dumps(dict(wagi=str(wagi), kroki=len(kroki), od=kroki[0], do=kroki[-1], okno=okno, gt=gt,
                                                    czas_s=round(time.time() - t0), wersja=WERSJA,
                                                    klucz_przenosny_v2=klucz_przenosny(kroki, rgb, gt, okno, wagi)), ensure_ascii=False, indent=1),
                                   encoding='utf-8')
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


_siec = {}


def podkroki(wagi, krok, rgb, urzadzenie='cuda', baza=None, st=None, lodyga=True):
    """Jedno przejście w przód checkpointu `krok`, rozłożone na podkroki strumienia rezydualnego (głębokość = czas cząstki):
    stan 0 = łodyga + kodowanie pozycji, potem dla bloku b: stan 2b+1 = po uwadze (x + attn(LN₁ x)), stan 2b+2 = po MLP (x + mlp(LN₂ x)).
    Rozkład dokładny (timm Block, ls = Identity):
      uwaga:  Δx_i = Σ_j f_ij + b_o,   f_ij = Σ_h A^h_ij · W_o[:, h] v^h_j      (siła cząsteczki j na cząsteczkę i)
      MLP:    Δx_j = Σ_c z_jc v_c + b₂                                         (768 laserów: natężenie z_jc, kierunek v_c)
    Wszystko rzutowane tym samym rzutem P (3 główne składowe 13 stanów × 600 tokenów), więc rzuty sił sumują się do rzutu przesunięcia.
    Zwraca słownik tablic float32 (pos (13,600,3), ah (6,3,600,600), uh (6,3,600,3), bo/b2 (6,3), z (6,600,768), pv (6,768,3),
    mapa (600,), ramka, pewnosc, sprawdzenie = maks. błąd rozkładu w rzucie). st: wagi już wczytane (słownik stanu, np. z wyprzedzeniem
    w wątku); lodyga=False pomija obrazki łodygi (6 rozkładów SVD, zbędne w pomiarach)."""
    import torch
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    M = modul_modelu()
    if dev.type not in _siec:
        _siec[dev.type] = M.MalyViT().to(dev).eval().requires_grad_(False)
    net = _siec[dev.type]
    if st is None:
        st = torch.load(Path(wagi) / f'k{int(krok):05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
        x = (x - M.MEAN.to(dev)) / M.STD.to(dev)
        f = x; etapy = []
        for modul in net.lodyga:                                                            # łodyga: 4 konwolucje 3×3 z krokiem 2
            f = modul(f); etapy.append(f[0])
        B, C, h, w = f.shape
        pe = M.sincos_2d(h, w, C, dev)
        t = f.flatten(2).transpose(1, 2) + pe
        if lodyga:
            lodyga = [_rgb_pca(e.reshape(e.shape[0], -1).T, e.shape[1], e.shape[2]) for e in etapy]
            lodyga.append(_rgb_pca(pe, h, w)); lodyga.append(_rgb_pca(t[0], h, w))         # kodowanie pozycji i tokeny po jego dodaniu
        else:
            lodyga = []
        stany, AH, U, BO, B2, Z, V2 = [t[0]], [], [], [], [], [], []
        for blk in net.bloki:
            at = blk.attn
            xn = blk.norm1(t)
            qkv = at.qkv(xn).reshape(1, h * w, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = at.q_norm(qkv[0]), at.k_norm(qkv[1]), qkv[2]
            A = ((q * at.scale) @ k.transpose(-2, -1)).softmax(-1)[0]                      # (3, 600, 600)
            hd = C // at.num_heads
            Wo = at.proj.weight                                                            # (192, 192)
            u = torch.stack([v[0, g] @ Wo[:, g * hd:(g + 1) * hd].T for g in range(at.num_heads)])   # (3, 600, 192)
            t = t + blk.ls1(at(xn))
            stany.append(t[0]); AH.append(A); U.append(u); BO.append(at.proj.bias)
            pam = {}
            hak = blk.mlp.act.register_forward_hook(lambda m_, a_, o_: pam.__setitem__('z', o_))
            t = t + blk.ls2(blk.mlp(blk.norm2(t)))
            hak.remove()
            stany.append(t[0]); Z.append(pam['z'][0]); V2.append(blk.mlp.fc2.weight); B2.append(blk.mlp.fc2.bias)
        o = net.glowa(net.norm(t)).view(1, h, w, 5)
        X = torch.stack(stany)                                                             # (13, 600, 192)
        AH = torch.stack(AH); U = torch.stack(U); BO = torch.stack(BO); B2 = torch.stack(B2)
        Z = torch.stack(Z); V2 = torch.stack(V2).transpose(1, 2)                          # (6, 768, 192) kolumny fc2 = v_c
        cpu = lambda a: a.float().cpu().numpy()
        wynik = dict(krok=int(krok), ah=cpu(AH), z=cpu(Z))
        # układy: '' = surowy strumień rezydualny (widać wspólny dryf w głąb sieci), '_c' = względem środka chmury w każdym
        # stanie (x − średnia tokenów, jak rzuty filmów), '_n' = jak '_c' podzielone przez JEDNĄ liczbę na checkpoint (średnia
        # norma tokenu po 13 stanach), czyli '_c' w innej skali; każdy z własnym rzutem PCA 3D.
        # Siły w '_c' i '_n' są liczone tak jak położenia, względem średniej po tokenach: uwaga z wagami A_ij − ā_j
        # (ā_j = średnia po słuchających i), lasery z jasnościami z_jc − z̄_c, biasy znikają. Wtedy suma rysowanych sił każdej
        # cząsteczki = jej przesunięcie w tym układzie DOKŁADNIE (w wersji 1 siły zawierały składnik wspólny wszystkim tokenom,
        # którego położenia nie miały, a '_n' zmieniał skalę między stanami, co wyglądało jak ruch bez sił).
        # Uwaga: '_n' NIE jest LayerNormem (ten normuje każdy token osobno); to tylko wygodna skala.
        Xc = X - X.mean(1, keepdim=True)
        rms = Xc.norm(dim=2).mean(1).clamp_min(1e-12)                                    # (13,)
        s_ref = rms.mean()                                                               # jedna skala na checkpoint
        for suf, Xu, ku, kv in (('', X, 1.0, 1.0), ('_c', Xc, 1.0, 1.0), ('_n', Xc / s_ref, 1 / float(s_ref), 1 / float(s_ref))):
            if baza is not None:                                                           # wspólny rzut: ciągłość między checkpointami
                mu = torch.as_tensor(baza[f'mu{suf}'], device=dev, dtype=Xu.dtype)
                P = torch.as_tensor(baza[f'P{suf}'], device=dev, dtype=Xu.dtype)
            else:
                mu = Xu.reshape(-1, C).mean(0)
                _, _, Vt = torch.linalg.svd(Xu.reshape(-1, C) - mu, full_matrices=False)
                P = Vt[:3]                                                                 # (3, 192)
            wynik.setdefault('baza', {}).update({f'mu{suf}': cpu(mu), f'P{suf}': cpu(P)})
            pos = (Xu - mu) @ P.T
            bo_, b2_ = ((BO @ P.T) * ku, (B2 @ P.T) * kv) if suf == '' else (torch.zeros_like(BO[:, :3]), torch.zeros_like(B2[:, :3]))
            wynik.update({f'pos{suf}': cpu(pos), f'uh{suf}': cpu((U @ P.T) * ku), f'bo{suf}': cpu(bo_),
                          f'b2{suf}': cpu(b2_), f'pv{suf}': cpu((V2 @ P.T) * kv), f'skala{suf}': float(pos.reshape(-1, 3).abs().max())})
            # sprawdzenie rozkładu: suma rysowanych sił (+ bias w układzie surowym) = przesunięcie w podkroku
            A_ = AH if suf == '' else AH - AH.mean(2, keepdim=True)
            Z_ = Z if suf == '' else Z - Z.mean(1, keepdim=True)
            F_ = torch.einsum('bhij,bhjd->bid', A_, (U @ P.T) * ku) + bo_[:, None]
            L_ = torch.einsum('bjc,bcd->bjd', Z_, (V2 @ P.T) * kv) + b2_[:, None]
            dU, dM = pos[1::2] - pos[0:-1:2], pos[2::2] - pos[1::2]
            wynik['sprawdzenie' + suf] = max(float((F_ - dU).abs().max() / dU.abs().max().clamp_min(1e-12)),
                                             float((L_ - dM).abs().max() / dM.abs().max().clamp_min(1e-12)))
        wynik['rms'] = cpu(rms)
        # wagi uwagi i jasności względem średniej po tokenach (do sił w układach '_c' i '_n')
        wynik['ah_c'] = cpu(AH - AH.mean(2, keepdim=True)); wynik['z_c'] = cpu(Z - Z.mean(1, keepdim=True))
        wynik['xn'] = cpu(Xc / rms[:, None, None])                                     # (13, 600, 192) do rzutów płaskich
    # sfera podkroków: każdy blok ma własne 768 neuronów; reflektor c bloku b stoi na spirali Fibonacciego tam, gdzie leży najbliżej
    # za swoim kierunkiem pchnięcia −d_c (przypisanie węgierskie jak film 12), kierunki w rzucie układu '_n'
    from scipy.optimize import linear_sum_assignment
    from laser_app.compute.common import kierunki
    fib = kierunki(False).astype(np.float64)
    dn = wynik['pv_n'] / np.maximum(np.linalg.norm(wynik['pv_n'], axis=2, keepdims=True), 1e-12)
    if baza is not None and 'slot' in baza:                                   # reflektory stoją (jak film 12): przypisanie z pierwszego checkpointu
        slot = np.asarray(baza['slot'])
    else:
        slot = np.empty((6, NK), np.int64)
        for b in range(6):
            r_, c_ = linear_sum_assignment(fib @ dn[b].T)
            slot[b, c_] = r_
    wynik['baza']['slot'] = slot
    wynik.update(slot=slot, dirs_n=dn.astype(np.float32))
    with torch.no_grad():
        bx, sc = M.dekoduj(o.float().cpu(), k=1)
    wynik.update(mapa=cpu(o[0, ..., 0].sigmoid().reshape(-1)), ramka=bx[0, 0].numpy(), pewnosc=float(sc[0, 0]), lodyga=lodyga,
                 lodyga_ksztalty=[tuple(int(v) for v in e.shape) for e in etapy] + [(C, h, w), (C, h, w)])
    return wynik


def _rgb_pca(F, h, w):
    """Mapa cech [h·w, kanały] → obraz uint8 [h, w, 3]: 3 główne składowe kanałów jako R, G, B (percentyle 1–99)."""
    import torch
    F = F.float()
    F = F - F.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(F, full_matrices=False)
    Y = (F @ Vt[:3].T).cpu().numpy()
    lo, hi = np.percentile(Y, 1, 0), np.percentile(Y, 99, 0)
    return (np.clip((Y - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)


KLASY_OKA = ('tło', 'rogówka', 'źrenica', 'narzędzie')


def sciezka_lokalna(p):
    """Ścieżka zapisana w cache na innym dysku (np. F:\\...\\SmallViTAblation\\assets\\...) → ta sama ścieżka w bieżącej bazie projektu
    (ROOT); bez zmian, gdy istnieje albo nie leży w SmallViTAblation."""
    p = Path(str(p))
    try:
        if p.exists() or 'SmallViTAblation' not in p.parts:
            return p
    except OSError:                                                      # niedostępny dysk (brak uprawnień): szukamy w ROOT
        pass
    q = ROOT.joinpath(*p.parts[p.parts.index('SmallViTAblation') + 1:])
    return q if q.exists() else p


class History:
    """Wczytany wynik `policz` (tablice numpy; Z jako mmap)."""

    def __init__(self, katalog):
        self.katalog = Path(katalog)
        d = np.load(self.katalog / 'dane.npz', allow_pickle=False)
        for k in d.files:
            v = d[k]
            setattr(self, k, v.item() if v.ndim == 0 else v)
        self.Z = np.load(self.katalog / 'Z.npy', mmap_mode='r')
        if hasattr(self, 'wagi'):
            self.wagi = str(sciezka_lokalna(self.wagi))       # cache policzony na innym dysku (F:): pułapka, dolina, podkroki czytają wagi
        self.K = len(self.kroki)
        self.gt = None if np.isnan(self.gt).any() else [float(v) for v in self.gt]
        self._etykieta_obrysu()
        self.lok_m, self.vib_m = self.lok.mean(1), self.vib.mean(1)
        self.LOK99, self.VIB97 = float(np.percentile(self.lok, 99)), float(np.percentile(self.vib, 97))
        self.ustaw_etykiete()
        self.refl_pokaz = int(np.argmax(np.asarray(self.Z[-1], np.float32).var(0)))

    def _etykieta_obrysu(self):
        """Wczytuje dane obrysu (etykiety_narzedzia): udzial, srodek, oko {rogowka, zrenica}, film_klatka. narz_ramka = stara etykieta
        z ramki GT. Bez obrysu (klatka spoza Cataract-LMM, brak zipów) wszystko poza ramką jest puste (None)."""
        self.narz_ramka = np.asarray(self.narz, bool)
        self.mieszane = np.zeros(TOK, bool)
        self.udzial = self.srodek = self.oko = self.film_klatka = None
        self._kat_obrysu = []
        self.etykieta, self.tryb_etykiety, self.prog_etykiety = 'ramka GT', 'ramka', 0.5
        try:
            from laser_app.compute import tool_labels
            e = tool_labels.dla_historii(self)
        except Exception:                                                # noqa: BLE001 (brak zipów adnotacji: ramka)
            import traceback
            traceback.print_exc()
            e = None
        if e is None:
            return
        self.udzial = np.asarray(e['udzial'], np.float32)
        self.srodek = None if e.get('srodek') is None else np.asarray(e['srodek'], bool)
        self.oko = None if e.get('oko') is None else {k: np.asarray(v, np.float32) for k, v in e['oko'].items()}
        self.film_klatka = (e['film'], int(e['klatka']))
        self._kat_obrysu = list(e['kategorie'])

    def ustaw_etykiete(self, tryb='dotkniete', prog=0.5):
        """Wybiera tokeny narzędzia (self.narz, self.mieszane) z danych obrysu wg trybu z tool_labels.TRYBY ('dotkniete',
        'pokrycie' z progiem `prog`, 'srodek', 'ramka') i przelicza wszystko, co z nich wynika (narz_lok, fok). Bez obrysu albo gdy tryb
        daje < 1 łaty, zostaje ramka GT (opis w self.etykieta). Zwraca opis etykiety."""
        n_ramka = int(self.narz_ramka.sum())
        narz, mies, opis, efektywny = self.narz_ramka, np.zeros(TOK, bool), 'ramka GT', 'ramka'
        if tryb == 'ramka' and self.udzial is not None:
            opis = f'ramka GT, środek łaty w ramce: {n_ramka} łat'
        elif self.udzial is not None:
            from laser_app.compute import tool_labels
            nazwa = {'dotkniete': 'dotknięte', 'pokrycie50': 'pokrycie ≥ 50 % (jak dawniej)', 'pokrycie': f'pokrycie ≥ {prog:g}',
                     'srodek': 'środek w obrysie'}.get(tryb, tryb)
            try:
                n, m = tool_labels.maski(self.udzial, prog, tryb, srodek=self.srodek, ramka=self.narz_ramka)
            except ValueError:                                           # np. brak środków łat w starym cache
                n, m = np.zeros(TOK, bool), np.zeros(TOK, bool)
            if n.sum() >= 1:
                narz, mies, efektywny = n, m, tryb
                opis = (f'obrys {self.film_klatka[0]} kl. {self.film_klatka[1]} ({", ".join(self._kat_obrysu)}), {nazwa}: '
                        f'{int(n.sum())} łat')
            else:
                opis = f'ramka GT (obrys, {nazwa}: 0 łat): {n_ramka} łat'
        self.narz, self.mieszane = np.asarray(narz, bool), np.asarray(mies, bool)
        self.etykieta = opis
        self.tryb_etykiety = efektywny                                   # 'ramka' także przy powrocie do ramki GT
        self.prog_etykiety = float(prog)
        n = self.narz
        self.narz_lok = self.lok[:, n].mean(1) / np.maximum(self.lok[:, ~n].mean(1), 1e-9) if n.any() else np.ones(self.K)
        self.fok = int(np.flatnonzero(n)[np.argmin(np.linalg.norm(self.g[n] - self.g[n].mean(0), axis=1))]) if n.any() \
            else int(np.argmax(self.vib.mean(0)))
        return opis

    def obszary(self, prog_oka=0.5):
        """uint8 (600,): klasy KLASY_OKA (0 tło, 1 rogówka, 2 źrenica, 3 narzędzie); pierwszeństwo narzędzie (self.narz) > źrenica >
        rogówka, obszary oka = udział ≥ prog_oka. Bez danych oka tylko 0 i 3."""
        k = np.zeros(TOK, np.uint8)
        if self.oko is not None:
            k[np.asarray(self.oko['rogowka']) >= prog_oka] = 1
            k[np.asarray(self.oko['zrenica']) >= prog_oka] = 2
        k[np.asarray(self.narz, bool)] = 3
        return k

    def indeks(self, krok):
        """Ułamkowy indeks checkpointu dla kroku (interpolacja liniowa w krokach)."""
        return float(np.interp(krok, self.kroki, np.arange(self.K)))

    def krok(self, t):
        return float(np.interp(t, np.arange(self.K), self.kroki))

    def interp(self, X, t):
        i = int(np.clip(np.floor(t), 0, self.K - 1)); j = min(i + 1, self.K - 1); u = float(np.clip(t - i, 0.0, 1.0))
        return X[i] * (1 - u) + X[j] * u


# ------------------------------------------------------------------ zestaw kilku zdjęć na jednej sferze
class ImageSetZ:
    """Z kilku historii jako jedna tablica (K, 600·n, 768) float16 bez kopiowania całości: z[k] albo z[a:b] skleja tylko żądane
    checkpointy wszystkich zdjęć (tokeny zdjęcia i to i·600 … i·600+599). Rozumie ją ReflectorSphere (natężenia pary checkpointów
    i wysyłka całości na GPU porcjami)."""

    def __init__(self, hs):
        self.hs = hs
        self.shape = (hs[0].K, TOK * len(hs), NK)
        self.dtype = np.dtype(np.float16)
        self.ndim = 3

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, k):
        os_ = 0 if isinstance(k, (int, np.integer)) else 1
        return np.concatenate([np.asarray(h.Z[k], np.float16) for h in self.hs], axis=os_)


class ImageSetHistory:
    """Kilka zdjęć tej samej historii wag (te same checkpointy) jako jedna chmura 600·n koloidów dla sfery reflektorów. Rzut 3D P3 i
    reflektory zależą tylko od wag, więc są wspólne; każde zdjęcie ma własny środek chmury (położenia jak w jego historii).
    Pozostałe okienka (1–5, 11–13, R, podkroki) pokazują jedno, aktywne zdjęcie zestawu."""

    def __init__(self, hs):
        hs = list(hs)
        if not hs:
            raise ValueError('pusty zestaw zdjęć')
        h0 = hs[0]
        for h in hs[1:]:
            if len(h.kroki) != len(h0.kroki) or not np.array_equal(h.kroki, h0.kroki):
                raise ValueError(f'zdjęcia zestawu mają różne checkpointy ({Path(h.katalog).name})')
        self.hs, self.n = hs, len(hs)
        self.K, self.kroki, self.okno, self.katalog, self.wagi = h0.K, h0.kroki, h0.okno, h0.katalog, h0.wagi
        for a in ('rfix', 'kol', 'dirs', 'obr_dryf', 'obr_drg', 'obr_kat0'):                    # tylko z wag: wspólne
            if hasattr(h0, a):
                setattr(self, a, getattr(h0, a))
        self.pos = np.concatenate([np.asarray(h.pos, np.float32) for h in hs], 1)              # (K, 600·n, 3)
        self.r99 = np.max(np.stack([np.asarray(h.r99, np.float64) for h in hs]), 0)
        self.narz = np.concatenate([np.asarray(h.narz, bool) for h in hs])
        self.Z = ImageSetZ(hs)
        self.T = TOK * self.n

    def indeks(self, krok):
        return self.hs[0].indeks(krok)

    def krok(self, t):
        return self.hs[0].krok(t)
