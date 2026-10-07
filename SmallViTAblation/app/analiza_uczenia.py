"""Historia uczenia małego ViT dla zakładki „Uczenie ViT”: odnalezienie biegów z gęstym zapisem latentu,
wczytanie jednej klatki próbnej (wszystkie kroki), miary ruchu cząsteczek po krokach i detekcja z wag.

Bieg = katalog `experiments/*/wyniki/*/dane` z plikiem LATENT_GESTY.npy [kroki, klatki, 600, 768]
(latent bloki[-1].mlp.act w trybie eval, ten sam co w aplikacji), obok obrazy.npy, maski.npz,
LOG.json (strata), OCENY.json (mAP), metadata.json (numery kroków) i katalog ../wagi/kNNNN.pt.
Pierwsze wczytanie klatki czyta 1,8 GB z dużego pliku (około 20 s) i zapisuje kopię float16 w
app/cache_uczenia/, następne trwają kilka sekund.
"""
import json
import re
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
EKSPERYMENTY = APP.parent / 'experiments'
CACHE = APP / 'cache_uczenia'


def znajdz_biegi():
    """Lista biegów z pełnym latentem 600 tokenów co krok (słowniki: nazwa, dane, wagi, ksztalt)."""
    wynik = []
    for p in sorted(EKSPERYMENTY.glob('*/wyniki/*/dane/LATENT_GESTY.npy')):
        try:
            L = np.load(p, mmap_mode='r')
        except Exception:
            continue
        if L.ndim != 4 or L.shape[2] != 600 or L.shape[3] != 768:
            continue                                        # np. zapis tylko wybranych tokenów
        dane = p.parent
        nazwa = f'{dane.parents[2].name} / {dane.parent.name}'
        wynik.append(dict(nazwa=nazwa, dane=dane, wagi=dane.parent / 'wagi', ksztalt=tuple(L.shape),
                          klucz=re.sub(r'\W+', '_', f'{dane.parents[2].name}_{dane.parent.name}')))
    return wynik


def _json(p, domyslnie):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return domyslnie


def opis_biegu(bieg):
    """Obrazy próbne, maski, opisy klatek, numery kroków, strata, mAP i lista checkpointów wag."""
    d = bieg['dane']
    meta = _json(d / 'metadata.json', {})
    K = bieg['ksztalt'][0]
    kroki = np.asarray(meta.get('kroki', list(range(K))), np.int64)[:K]
    m = np.load(d / 'maski.npz')
    obrazy = np.load(d / 'obrazy.npy', mmap_mode='r')
    opisy = [f"{o.get('film', '?')} kl. {o.get('annotation_frame_number', '?')} ({o.get('split', '?')})"
             for o in _json(d / 'images.json', [])] or [f'klatka {i}' for i in range(len(obrazy))]
    log = _json(d / 'LOG.json', [])
    oceny = _json(d / 'OCENY.json', [])
    wagi = sorted(bieg['wagi'].glob('k*.pt'), key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))
    kroki_wag = np.array([int(re.sub(r'\D', '', p.stem)) for p in wagi], np.int64)
    return dict(kroki=kroki, obrazy=obrazy, opisy=opisy, narzedzie=m['tool'].astype(bool),
                tlo=m['background'].astype(bool),
                strata=(np.array([r['krok'] for r in log]), np.array([r['strata'] for r in log])),
                mAP=(np.array([r['krok'] for r in oceny]), np.array([r.get('mAP50', np.nan) for r in oceny])),
                wagi=wagi, kroki_wag=kroki_wag, ziarno=meta.get('seed'))


def wczytaj_klatke(bieg, i, postep=None):
    """Latent klatki próbnej i we wszystkich krokach: [K, 600, 768] float16 (kopia w cache_uczenia)."""
    CACHE.mkdir(exist_ok=True)
    plik = CACHE / f"{bieg['klucz']}_klatka{i:02d}.npy"
    if plik.exists():
        Z = np.load(plik, mmap_mode='r')
        if Z.shape == (bieg['ksztalt'][0], 600, 768):
            return np.ascontiguousarray(Z)
    L = np.load(bieg['dane'] / 'LATENT_GESTY.npy', mmap_mode='r')
    K = L.shape[0]
    Z = np.empty((K, 600, 768), np.float16)
    for a in range(0, K, 50):
        Z[a:a + 50] = L[a:a + 50, i]
        if postep:
            postep(min(K, a + 50), K)
    tmp = plik.with_suffix('.tmp.npy')
    np.save(tmp, Z)
    tmp.replace(plik)
    return Z


def analiza(Z, narzedzie, tlo, postep=None, cka_maks=250):
    """Miary ruchu cząsteczek po krokach uczenia, dla klatki [K, 600, 768]:
    v [K, 600]      prędkość tokenu: ||z_t - z_(t-1)|| (0 w kroku 0),
    do_konca [K, 600] odległość tokenu od jego stanu końcowego ||z_t - z_K||,
    wzgl [K]        względna zmiana całej klatki ||dZ|| / ||Z_(t-1)||,
    rzad [K]        efektywny rząd tokenów (participation ratio widma kowariancji 600 tokenów),
    separacja [K]   rozdzielenie narzędzie / tło: ||mu_n - mu_t||^2 / (śr. wariancja n + śr. wariancja t),
    norma [K]       średnia norma tokenu,
    cka_kroki [M], cka [M, M]  liniowe CKA między krokami (co któryś krok, M <= cka_maks)."""
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = len(Z)
    n_idx = torch.from_numpy(np.flatnonzero(narzedzie)).to(dev)
    t_idx = torch.from_numpy(np.flatnonzero(tlo)).to(dev)
    kon = torch.from_numpy(np.asarray(Z[-1], np.float32)).to(dev)
    co = max(1, int(np.ceil(K / cka_maks)))
    cka_kroki = np.arange(0, K, co)
    v = np.zeros((K, 600), np.float32); dk = np.zeros((K, 600), np.float32)
    wzgl = np.zeros(K, np.float32); rzad = np.zeros(K, np.float32); sep = np.zeros(K, np.float32)
    norma = np.zeros(K, np.float32)
    gramy = []
    poprz = None
    for a in range(0, K, 40):
        X = torch.from_numpy(np.asarray(Z[a:a + 40], np.float32)).to(dev)          # [b, 600, 768]
        if poprz is None:
            prev = torch.cat([X[:1], X[:-1]])
        else:
            prev = torch.cat([poprz[None], X[:-1]])
        d = X - prev
        v[a:a + len(X)] = d.norm(dim=-1).cpu().numpy()
        wzgl[a:a + len(X)] = (d.flatten(1).norm(dim=1) / prev.flatten(1).norm(dim=1).clamp_min(1e-9)).cpu().numpy()
        dk[a:a + len(X)] = (X - kon).norm(dim=-1).cpu().numpy()
        norma[a:a + len(X)] = X.norm(dim=-1).mean(1).cpu().numpy()
        Xc = X - X.mean(1, keepdim=True)
        G = Xc @ Xc.transpose(1, 2)                                                 # [b, 600, 600]
        ev = torch.linalg.eigvalsh(G.double()).clamp_min(0)
        rzad[a:a + len(X)] = (ev.sum(1) ** 2 / (ev ** 2).sum(1).clamp_min(1e-30)).float().cpu().numpy()
        if len(n_idx) > 1 and len(t_idx) > 1:
            Xn, Xt = X[:, n_idx], X[:, t_idx]
            mn, mt = Xn.mean(1), Xt.mean(1)
            wn = ((Xn - mn[:, None]) ** 2).sum(-1).mean(1); wt = ((Xt - mt[:, None]) ** 2).sum(-1).mean(1)
            sep[a:a + len(X)] = (((mn - mt) ** 2).sum(-1) / (wn + wt).clamp_min(1e-9)).cpu().numpy()
        for j in range(len(X)):
            if (a + j) % co == 0:
                g = G[j].flatten()
                gramy.append((g / g.norm().clamp_min(1e-12)).half())
        poprz = X[-1]
        if postep:
            postep(min(K, a + 40), K)
    V = torch.stack(gramy).float()
    cka = (V @ V.T).cpu().numpy()
    return dict(v=v, do_konca=dk, wzgl=wzgl, rzad=rzad, separacja=sep, norma=norma, cka_kroki=cka_kroki, cka=cka)


def detekcje(opis, rgb480, postep=None):
    """Detekcja klatki próbnej z wag każdego zapisanego checkpointu: kroki_wag [n], ramka [n, 4] (najpewniejsza),
    pewnosc [n], mapy_pew [n, 600] (pewność środka w patchu)."""
    import torch
    import detekcja as D
    det = D.Detektor(opis['wagi'][0])
    ramki, pew, mapy = [], [], []
    for k, p in enumerate(opis['wagi']):
        ck = torch.load(p, map_location='cpu', weights_only=True)
        det.net.load_state_dict(ck.get('net', ck), strict=True)
        a = det.analizuj(rgb480)
        ramki.append(a['boxes'][0]); pew.append(float(a['scores'][0]))
        mapy.append(a['mapa'][0, ..., 0].detach().float().sigmoid().cpu().numpy().reshape(-1))
        if postep:
            postep(k + 1, len(opis['wagi']))
    return dict(kroki_wag=opis['kroki_wag'], ramka=np.array(ramki, np.float32), pewnosc=np.array(pew, np.float32),
                mapy_pew=np.array(mapy, np.float32), det=det)


def oscylator(Z, od=300, do=700, okno=50):
    """Liniowy model ruchu każdej cząstki z bezwładnością (dyskretny Langevin, AR(2)):
    v_(t+1) = beta_i v_t − kappa_i (z_t − m_t), z_(t+1) = z_t + v_(t+1), m_t = średnia ostatnich `okno` kroków.
    beta_i, kappa_i [600] metodą najmniejszych kwadratów na krokach od..do (experiments/maly_vit_liniowy_ruch_czastek)."""
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = len(Z); do = min(do, K - 2); od = min(od, max(1, do // 2))
    beta = np.zeros(600, np.float32); kappa = np.zeros(600, np.float32)
    for t0 in range(0, 600, 100):
        x = torch.from_numpy(np.asarray(Z[:, t0:t0 + 100], np.float32)).to(dev)
        v = torch.cat([torch.zeros_like(x[:1]), x[1:] - x[:-1]])
        c = torch.cumsum(torch.cat([torch.zeros_like(x[:1]), x]), 0)
        t = torch.arange(K, device=dev); lo = (t - okno + 1).clamp_min(0)
        m = (c[t + 1] - c[lo]) / (t + 1 - lo).float()[:, None, None]
        a1 = v[od:do]; a2 = -(x[od:do] - m[od:do]); y = v[od + 1:do + 1]
        g11 = (a1 * a1).sum((0, 2)); g12 = (a1 * a2).sum((0, 2)); g22 = (a2 * a2).sum((0, 2))
        r1 = (a1 * y).sum((0, 2)); r2 = (a2 * y).sum((0, 2))
        det = (g11 * g22 - g12 ** 2).clamp_min(1e-12)
        beta[t0:t0 + 100] = ((g22 * r1 - g12 * r2) / det).cpu().numpy()
        kappa[t0:t0 + 100] = ((g11 * r2 - g12 * r1) / det).cpu().numpy()
    return dict(beta=beta, kappa=kappa, okno=okno, od=od, do=do)


def prognoza_oscylatora(Z, k, tok, osc, h):
    """Tor cząstki tok przez h kroków od kroku k według oscylatora, tylko ze stanów do k: [h + 1, 768]."""
    z = np.asarray(Z[k, tok], np.float32)
    v = z - np.asarray(Z[max(0, k - 1), tok], np.float32)
    m = np.asarray(Z[max(0, k - osc['okno'] + 1):k + 1, tok], np.float32).mean(0)
    b, kk = float(osc['beta'][tok]), float(osc['kappa'][tok])
    tor = [z.copy()]
    for _ in range(h):
        v = b * v - kk * (z - m); z = z + v; tor.append(z.copy())
    return np.array(tor)


def ocena_oscylatora(Z, osc, t0, hs=(1, 5, 20)):
    """Błąd prognozy oscylatora na h kroków od kroków t0..T−h (tylko stany do chwili prognozy) względem persystencji:
    {h: błąd / błąd persystencji} dla wszystkich tokenów razem (mniej = lepiej, 1 = jak „bez ruchu”)."""
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(np.asarray(Z, np.float32)).to(dev)                        # [T, 600, 768]
    T = len(X); w = osc['okno']
    c = torch.cumsum(torch.cat([torch.zeros_like(X[:1]), X]), 0)
    t = torch.arange(T, device=dev); lo = (t - w + 1).clamp_min(0)
    M = (c[t + 1] - c[lo]) / (t + 1 - lo).float()[:, None, None]
    V = torch.cat([torch.zeros_like(X[:1]), X[1:] - X[:-1]])
    b = torch.from_numpy(osc['beta']).to(dev)[None, :, None]; k = torch.from_numpy(osc['kappa']).to(dev)[None, :, None]
    wynik = {}
    for h in hs:
        ts = torch.arange(max(1, t0), T - h, device=dev)
        if len(ts) == 0:
            continue
        z, v, m = X[ts].clone(), V[ts].clone(), M[ts]
        for _ in range(h):
            v = b * v - k * (z - m); z = z + v
        e = float(((z - X[ts + h]) ** 2).sum()); ep = float(((X[ts] - X[ts + h]) ** 2).sum())
        wynik[h] = e / max(ep, 1e-30)
    del X, M, V, c
    return wynik



# ------------------------------------------------------------ jeden tor (cząstka albo średnia zbioru cząstek)
def srednia_kroczaca(x, okno):
    """Średnia ostatnich `okno` stanów (z bieżącym) w każdym kroku: [K, D] -> [K, D]."""
    x = np.asarray(x, np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x]), 0)
    t = np.arange(len(x)); lo = np.clip(t - okno + 1, 0, None)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)[:, None]


def tor_sredni(Z, maska, krok=100):
    """Średni latent cząstek z maski w każdym kroku: [K, 768] float32 (czytany porcjami, bez kopii całej klatki)."""
    maska = np.asarray(maska, bool)
    return np.concatenate([np.asarray(Z[a:a + krok][:, maska], np.float32).mean(1) for a in range(0, len(Z), krok)])


def oscylator_toru(x, od=300, do=700, okno=50):
    """beta, kappa oscylatora (jak `oscylator`) dla jednego toru x [K, D], np. średniej cząstek narzędzia."""
    x = np.asarray(x, np.float64)
    K = len(x); do = min(do, K - 2); od = min(od, max(1, do // 2))
    v = np.concatenate([np.zeros_like(x[:1]), np.diff(x, axis=0)])
    m = srednia_kroczaca(x, okno)
    a1, a2, y = v[od:do], -(x[od:do] - m[od:do]), v[od + 1:do + 1]
    g11, g12, g22 = (a1 * a1).sum(), (a1 * a2).sum(), (a2 * a2).sum()
    r1, r2 = (a1 * y).sum(), (a2 * y).sum()
    det = max(g11 * g22 - g12 ** 2, 1e-12)
    return float((g22 * r1 - g12 * r2) / det), float((g11 * r2 - g12 * r1) / det)


def prognozy_toru(x, M, starty, h, beta, kappa):
    """Tory oscylatora z wielu kroków startu naraz: [len(starty), h + 1, D]. Każdy tylko ze stanów do swojego startu
    (położenie, prędkość z ostatniego kroku, średnia kroczaca M = srednia_kroczaca(x, okno))."""
    s = np.asarray(starty, int)
    z = np.asarray(x[s], np.float64); v = z - np.asarray(x[np.maximum(s - 1, 0)], np.float64); m = M[s]
    tory = [z]
    for _ in range(h):
        v = beta * v - kappa * (z - m); z = z + v; tory.append(z)
    return np.stack(tory, 1)


def ema(x, a):
    """Wygładzanie wykładnicze wzdłuż osi 0 (przyczynowe, od pierwszego wiersza): y_t = a x_t + (1 − a) y_(t−1)."""
    x = np.asarray(x, np.float64)
    if a >= 1 or len(x) == 0:
        return x
    y = np.empty_like(x); y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def czubki_biegu(bieg):
    """Czubek narzędzia (cystotome_tip z annotation.csv) każdej klatki próbnej w pikselach obrazu 480x320; None,
    gdy brak etykiety albo archiwum adnotacji."""
    wynik = []
    for o in _json(bieg['dane'] / 'images.json', []):
        try:
            import film
            w = film.etykiety(o['film']).get(int(o['annotation_frame_number']))
            wynik.append(film.etykieta_klatki(w)['czubek'])
        except Exception:
            wynik.append(None)
    return wynik
