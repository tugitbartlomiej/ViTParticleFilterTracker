"""Stanowisko korekcji latentu: korekta z etykietą (oracle) przy zamrożonym małym ViT, zapis prawdziwej trajektorii
optymalizacji, rozkład korekty względem fc2 i ewaluator detekcji (SPECYFIKACJA_DLA_CLAUDE_CODE.md w
experiments/stanowisko_korekcji_latentu_gui).

Punkt interwencji: latent z = bloki[-1].mlp.act (po GELU, przed fc2), [600 tokenów, 768 kanałów]. Nie jest pełnym
stanem sieci. Końcówka przy zamrożonych wagach (eval, dropout wyłączony):
    t = r + fc2(z),   mapa = glowa(norm(t)),
gdzie r = wejście norm2 ostatniego bloku (gałąź residual, zapamiętana z przebiegu i zawsze jawnie podana).

Tryby interwencji (poziomy kontroli osiągalności):
  swobodne         z' = z + δ na wybranych tokenach i kanałach, bez ograniczeń (diagnostyka, bez deklaracji
                   osiągalności);
  zakresy          z' w [h_min, h_max] każdego kanału (silnik.granice_z_wag: supremum po wszystkich wejściach bloku
                   przy wagach checkpointu); ograniczenia konieczne, nie wystarczające do wspólnej osiągalności;
  dolny GELU       tylko z' >= min GELU (jak historyczna maksymalizacja); słabsze od zakresów;
  wejście MLP, gałąź     z' = GELU(fc1(LN(r + e))), residual r bez zmian (zmienia się tylko gałąź MLP);
  wejście MLP, wspólne   r' = r + e, t = r' + fc2(GELU(fc1(LN(r')))) (zmienia się też wspólne wejście residualu).
Żaden tryb nie gwarantuje istnienia zdjęcia, które daje dany stan.

Korekta z etykietą to pomiar oracle: optymalizuje aktywacje jednego obrazu z jego etykietą. Nie jest jakością
detektora bez etykiet ani dowodem generalizacji. Wynik to najlepsza znaleziona korekta przy danym celu, ograniczeniach
i budżecie, nie globalne optimum ani jedyny kierunek.

Oś czasu tej optymalizacji: time_axis = 'latent_optimization' (wagi i residual stoją, zmienia się tylko δ i momenty
optymalizatora). Ruch uczenia sieci to osobna oś 'network_training' (porownaj_z_uczeniem, galezie_uczenia.py).
"""
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import silnik  # ścieżka do model_snapshot (ta sama architektura co w maksymalizacji J128)
from model_snapshot import MalyViT, _giou, cele, dekoduj

ROOT = Path(__file__).resolve().parent.parent
WEJSCIE_J128 = ROOT / 'outputs' / 'vit_optymalizacja_latentu128' / 'input'
HISTORIA = ROOT / 'outputs' / 'vit_latent128_maksymalizacja'
WYJSCIE = Path(os.environ.get('KOREKTA_WYJSCIE', ROOT / 'outputs' / 'stanowisko_korekcji_latentu_gui'))
WAGI_GESTE = ROOT / 'experiments' / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny' / 'wagi'
BAZA_UCZENIA_COCO = ROOT / 'experiments' / 'maly_vit_od_zera' / 'eden' / 'wejscia' / 'pretrain' / 'train' / '_annotations.coco.json'
GELU_MIN = float(silnik.GELU_MIN) if hasattr(silnik, 'GELU_MIN') else -0.16997120
EWALUATOR = 'ap101_detekcja_v1'
OPIS_EWALUATORA = ('AP jednej klasy, interpolacja 101-punktowa (jak COCO), dopasowanie zachłanne po pewności do '
                   'nieużytej ramki GT przy progu IoU; wiele GT na obraz, obrazy bez obiektu (ich detekcje są fałszywe), '
                   'duplikaty liczone jako fałszywe; brak GT w całym zbiorze -> AP nieokreślone (NaN). Detekcje: do 10 '
                   'lokalnych szczytów mapy 3x3 na obraz (model_snapshot.dekoduj). AP50:95 = średnia 10 progów 0,50..0,95. '
                   'To jawny protokół tej głowy, nie pełny COCO evaluator.')

TRYBY = {
    'swobodne': 'swobodne aktywacje (diagnostyka, bez ograniczeń)',
    'zakresy': 'aktywacje w zakresach z checkpointu (konieczne, nie wystarczające)',
    'dol_gelu': 'tylko dolne ograniczenie GELU (jak historyczna maksymalizacja)',
    'wejscie_galaz': 'wejście MLP: tylko gałąź MLP (residual bez zmian)',
    'wejscie_wspolne': 'wejście MLP: wspólne wejście residualu i MLP',
}
TRYBY_LATENTU = ('swobodne', 'zakresy', 'dol_gelu')
CELE = {
    'strata': 'zmniejszenie pełnej straty detekcji (ogniskowa + 2 (1 − GIoU))',
    'lokalizacja': 'lokalizacja (ogniskowa pozytywna + 2 (1 − GIoU)) z karą za fałszywe szczyty',
    'najmniejsza': 'najmniejsza znaleziona korekta osiągająca próg jakości (siatka promieni normy)',
}
TOKENY = ['wszystkie 600', 'zaznaczone w zakładce Uczenie', 'obiekt (z GT)', 'granica obiektu (z GT)', 'tło (z GT)']
NORMY = {'brak': 'bez ograniczenia normy', 'calosc': '||δ|| całej korekty <= ρ', 'token': '||δ|| każdego tokenu <= ρ'}


# ---------------------------------------------------------------------------------------------------- pomocnicze
def sha_pliku(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def hash_stanu(net):
    h = hashlib.sha256()
    for n, x in net.state_dict().items():
        h.update(n.encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _json(p, x):
    """Zapis atomowy: najpierw plik tymczasowy, potem podmiana (przerwanie nie niszczy poprzedniej wersji)."""
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=1, ensure_ascii=False, allow_nan=True,
                              default=lambda v: v.tolist() if isinstance(v, np.ndarray) else
                              (v.item() if hasattr(v, 'item') else str(v))), encoding='utf-8')
    os.replace(tmp, p)


def _npz(p, **x):
    p = Path(p); tmp = p.with_name(p.stem + '.tmp.npz')
    np.savez(tmp, **x); os.replace(tmp, p)


def iou(b, g):
    """IoU ramek b [..., 4] z jedną ramką g [4] (xyxy)."""
    x0, y0 = np.maximum(b[..., 0], g[0]), np.maximum(b[..., 1], g[1])
    x1, y1 = np.minimum(b[..., 2], g[2]), np.minimum(b[..., 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)


def ap(detekcje, ramki_gt, prog=0.5):
    """AP jednej klasy przy progu IoU (protokół EWALUATOR). detekcje: lista (ramki [k,4], pewności [k]) na obraz;
    ramki_gt: lista tablic [n,4] (n może być 0: obraz bez obiektu)."""
    wszystkie, n_gt = [], 0
    for i, ((bx, sc), gt) in enumerate(zip(detekcje, ramki_gt)):
        n_gt += len(gt)
        wszystkie += [(float(s), i, np.asarray(b, np.float64)) for b, s in zip(bx, sc)]
    if n_gt == 0:
        return float('nan')
    wszystkie.sort(key=lambda x: -x[0])
    uzyte = [np.zeros(len(g), bool) for g in ramki_gt]
    tp = np.zeros(len(wszystkie))
    for j, (_, i, b) in enumerate(wszystkie):
        g = np.asarray(ramki_gt[i], np.float64).reshape(-1, 4)
        if len(g):
            v = np.array([iou(b, gg) for gg in g]); v[uzyte[i]] = -1
            k = int(v.argmax())
            if v[k] >= prog:
                tp[j] = 1; uzyte[i][k] = True
    if not len(tp):
        return 0.0
    ctp = np.cumsum(tp); rec = ctp / n_gt; prec = ctp / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    wynik = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        wynik += prec[k] if k < len(prec) else 0.0
    return wynik / 101


def ap_zbioru(detekcje, ramki_gt):
    progi = np.arange(.5, .951, .05)
    a = {f'{p:.2f}': ap(detekcje, ramki_gt, p) for p in progi}
    return dict(AP50=a['0.50'], AP75=a['0.75'], AP50_95=float(np.mean(list(a.values()))), progi=a, ewaluator=EWALUATOR)


def ocena_obrazu(boxes, scores, gt, tau=0.5, prog_iou=0.5):
    """Miary jednego obrazu według jawnego protokołu.
    Ramka do podglądu = najpewniejsza predykcja (wybór bez GT). Detekcje = szczyty z pewnością >= tau, dopasowanie
    zachłanne po pewności do nieużytej GT przy IoU >= prog_iou; reszta to fałszywe detekcje, nietrafione GT to
    pominięcia. Najlepsze IoU wśród 10 szczytów to miara diagnostyczna (ramka wybrana przez GT)."""
    boxes = np.asarray(boxes, np.float64); scores = np.asarray(scores, np.float64)
    gt = np.asarray(gt, np.float64).reshape(-1, 4)
    k0 = int(np.argmax(scores))
    wyn = dict(pewnosc_top=float(scores[k0]), ramka_top=boxes[k0].tolist(), tau=tau, prog_iou=prog_iou)
    M = np.array([[iou(b, g) for g in gt] for b in boxes]) if len(gt) else np.zeros((len(boxes), 0))
    wyn['iou_top'] = float(M[k0].max()) if len(gt) else None
    wyn['iou_najlepsze_diag'] = float(M.max()) if len(gt) else None
    wyn['ramka_najlepsza_diag'] = boxes[int(M.max(1).argmax())].tolist() if len(gt) else None
    uzyte = np.zeros(len(gt), bool); tp = fp = 0
    for j in np.argsort(-scores):
        if scores[j] < tau:
            break
        if len(gt):
            v = M[j].copy(); v[uzyte] = -1; k = int(v.argmax())
            if v[k] >= prog_iou:
                uzyte[k] = True; tp += 1; continue
        fp += 1
    wyn.update(tp=tp, fp=fp, pominiete=int(len(gt) - uzyte.sum()), n_gt=int(len(gt)), n_det=tp + fp)
    return wyn


def skladniki_straty(o, heat, ltrb, poz, tau_fp=0.3):
    """Składniki straty na obraz [B] (normy liczone osobno dla każdego obrazu, jak w maksymalizacji J128):
    ogniskowa pozytywna i negatywna CenterNet / liczba pozytywnych komórek, ramka = średnia ważona (1 − GIoU),
    strata = ogniskowa + 2 ramka; kara_fp = suma (p − tau_fp)_+^2 po komórkach z heat < 0,1 (fałszywe szczyty)."""
    p = o[..., 0].float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = (heat == 1).float()
    n = pos.sum((1, 2)).clamp(min=1)
    f_pos = -(p.log() * (1 - p) ** 2 * pos).sum((1, 2)) / n
    f_neg = -((1 - p).log() * p ** 2 * (1 - heat) ** 4 * (1 - pos)).sum((1, 2)) / n
    pr = o[..., 1:].float().exp()
    g = _giou(pr.reshape(-1, 4), ltrb.reshape(-1, 4)).reshape(p.shape)
    ramka = ((1 - g) * poz).sum((1, 2)) / poz.sum((1, 2)).clamp(min=1e-12)
    fp = (((p - tau_fp).clamp(min=0) ** 2) * (heat < .1).float()).sum((1, 2))
    return dict(ogniskowa_poz=f_pos, ogniskowa_neg=f_neg, ramka=ramka, strata=f_pos + f_neg + 2 * ramka, kara_fp=fp)


# ---------------------------------------------------------------------------------------------------- dane
def manifest():
    return json.loads((WEJSCIE_J128 / 'manifest.json').read_text(encoding='utf-8'))


_BAZA = None


def ramki_bazy_uczenia():
    """frame_id obrazów bazy uczenia gęstej historii (1500 klatek COCO train): obraz z bazy mógł uczyć ViT."""
    global _BAZA
    if _BAZA is None:
        try:
            a = json.loads(BAZA_UCZENIA_COCO.read_text(encoding='utf-8'))
            s = set()
            for im in a['images']:
                st = im['file_name'].replace(chr(92), '/').split('/')[-1].rsplit('.', 1)[0]
                s.add(f'cataract_lmm/tracking/{st[:14]}/{st[15:]}')
            _BAZA = s
        except Exception:
            _BAZA = set()
    return _BAZA


def obraz_manifestu(row):
    """Obraz 480 x 320 RGB uint8 (INTER_AREA jak w maksymalizacji) i ramka GT xyxy w tych pikselach."""
    import cv2
    p = WEJSCIE_J128 / row['image']
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(f'brak obrazu {p}')
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (480, 320), interpolation=cv2.INTER_AREA)
    return im, sprawdz_ramki([[row['bbox_xywh'][0], row['bbox_xywh'][1], row['bbox_xywh'][0] + row['bbox_xywh'][2],
                              row['bbox_xywh'][1] + row['bbox_xywh'][3]]])


def sprawdz_ramki(r):
    """Ramki xyxy w pikselach 480 x 320; błędna adnotacja (zerowa, odwrócona, poza obrazem) -> ValueError."""
    r = np.asarray(r, np.float32).reshape(-1, 4)
    for b in r:
        if not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1] or b[2] <= 0 or b[3] <= 0 or b[0] >= 480 or b[1] >= 320:
            raise ValueError(f'błędna adnotacja: ramka {b.tolist()} (zerowa, odwrócona albo poza obrazem 480 x 320)')
    return r


def kanaly_j128():
    c = np.load(WEJSCIE_J128 / 'channels.npz')
    return np.asarray(c['channels0'], np.int64)


def checkpointy():
    """Lista (opis, ścieżka): checkpointy gęstej historii co 10 kroków (k1000 = checkpoint maksymalizacji J128)."""
    out = []
    for p in sorted(WAGI_GESTE.glob('k*.pt')):
        out.append((f'gęsta historia {p.stem}', p))
    if not out and Path(silnik.VIT_CHECKPOINT).exists():
        out.append(('MODEL_VIT_K1000', Path(silnik.VIT_CHECKPOINT)))
    return out


def maski_gt(ramki):
    """Maski tokenów 20 x 30 z GT: obiekt (środek komórki w ramce), granica (dylatacja XOR erozja obiektu), tło (poza
    dylatacją). Wybór tokenów przez GT jest częścią interwencji oracle."""
    ys, xs = np.mgrid[0:20, 0:30]
    cx, cy = (xs + .5) * 16, (ys + .5) * 16
    ob = np.zeros((20, 30), bool)
    for x0, y0, x1, y1 in np.asarray(ramki).reshape(-1, 4):
        ob |= (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
        ix, iy = min(max(int((x0 + x1) / 32), 0), 29), min(max(int((y0 + y1) / 32), 0), 19)
        ob[iy, ix] = True
    from scipy import ndimage
    dyl = ndimage.binary_dilation(ob, np.ones((3, 3), bool)); ero = ndimage.binary_erosion(ob, np.ones((3, 3), bool))
    return dict(obiekt=ob.reshape(-1), granica=(dyl ^ ero).reshape(-1), tlo=(~dyl).reshape(-1))


# ---------------------------------------------------------------------------------------------------- model
class Stanowisko:
    """Własna kopia zamrożonego ViT (nie dzieli modelu z innymi zakładkami)."""

    def __init__(self, ckpt, dev=None):
        self.ckpt = Path(ckpt)
        if not self.ckpt.exists():
            raise FileNotFoundError(f'brak checkpointu {self.ckpt}')
        self.dev = torch.device(dev or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.sha = sha_pliku(self.ckpt)
        ck = torch.load(self.ckpt, map_location='cpu', weights_only=True)
        self.krok = ck.get('krok')
        self.net = MalyViT()
        self.net.load_state_dict(ck.get('net', ck), strict=True)
        self.net.to(self.dev).eval().requires_grad_(False)
        self.blok = self.net.bloki[-1]
        assert self.blok.mlp.act.approximate == 'none'
        self.hash0 = hash_stanu(self.net)
        lo, hi = silnik.granice_z_wag(self.ckpt)
        self.h_min = torch.tensor(lo, device=self.dev); self.h_max = torch.tensor(hi, device=self.dev)
        W2 = self.blok.mlp.fc2.weight.detach().double().cpu().numpy()          # [192, 768]
        _, S, Vt = np.linalg.svd(W2, full_matrices=True)
        tol = float(S.max() * max(W2.shape) * np.finfo(np.float32).eps)
        r = int((S > tol).sum())
        self.fc2 = dict(S=S, tol=tol, rzad=r, jadro=W2.shape[1] - r, ksztalt=list(W2.shape),
                        opis_tolerancji='σ > σ_max · 768 · eps(float32)')
        self.Vr = torch.tensor(Vt[:r].T.copy(), dtype=torch.float32, device=self.dev)     # baza przestrzeni wierszowej
        self.W2 = W2

    def fc2_podzbioru(self, kanaly):
        """Rząd i wartości osobliwe W2 ograniczonej do kolumn `kanaly` (np. J128); jej jądro nie musi być duże."""
        W = self.W2[:, np.asarray(kanaly)]
        S = np.linalg.svd(W, compute_uv=False)
        tol = float(S.max() * max(W.shape) * np.finfo(np.float32).eps)
        r = int((S > tol).sum())
        return dict(S=S, tol=tol, rzad=r, jadro=W.shape[1] - r, ksztalt=list(W.shape))

    def zgodny(self):
        return hash_stanu(self.net) == self.hash0

    # ------------------------------------------------------------ przebieg i końcówka
    @torch.no_grad()
    def przebieg(self, rgb):
        """Pełny forward obrazu uint8 [320, 480, 3]: z [600, 768], r [600, 192], o [20, 30, 5] i błąd końcówki."""
        pam = {}
        b = self.blok
        h1 = b.norm2.register_forward_pre_hook(lambda m, a: pam.update(r=a[0].detach().clone()))
        h2 = b.mlp.act.register_forward_hook(lambda m, a, o: pam.update(z=o.detach().clone()))
        try:
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(self.dev) / 255
            o, _ = self.net(x)
        finally:
            h1.remove(); h2.remove()
        o2 = self.ogon(pam['z'], pam['r'])
        return dict(z=pam['z'][0], r=pam['r'][0], o=o[0], blad_ogona=float((o - o2).abs().max()))

    def _koniec(self, t):
        return self.net.glowa(self.net.norm(t)).view(t.shape[0], 20, 30, 5)

    def ogon(self, z, r):
        b, m = self.blok, self.blok.mlp
        return self._koniec(r + b.drop_path2(b.ls2(m.drop2(m.fc2(m.norm(m.drop1(z)))))))

    def mlp_z(self, u):
        m = self.blok.mlp
        return m.act(m.fc1(self.blok.norm2(u)))

    # ------------------------------------------------------------ interwencja
    def wyjscie(self, d, baza, spec):
        """Zmieniony latent z' [1, 600, 768], residual podany końcówce [1, 600, 192] i mapa o [1, 20, 30, 5]."""
        z, r = baza['z'][None], baza['r'][None]
        T = spec['_T']
        if spec['tryb'] in TRYBY_LATENTU:
            C = spec['_C']
            D = torch.zeros_like(z[0]).index_put((T[:, None], C[None, :]), d)
            zp = z + D[None]; rr = r
        else:
            E = torch.zeros_like(r[0]).index_put((T,), d)
            u = r + E[None]
            zp = self.mlp_z(u)
            zp = torch.where(spec['_maskaT'][None, :, None], zp, z)     # tokeny spoza T: dokładnie z (jak przebieg)
            rr = r if spec['tryb'] == 'wejscie_galaz' else u
        return zp, rr, self.ogon(zp, rr)

    def wymiar(self, spec):
        nt = int(spec['_T'].numel())
        return (nt, int(spec['_C'].numel())) if spec['tryb'] in TRYBY_LATENTU else (nt, 192)

    @torch.no_grad()
    def rzutuj(self, d, baza, spec):
        """Ograniczenia w miejscu: najpierw kula normy, potem pudło (dla trybów latentu). Pudło przesuwa każdą
        współrzędną w stronę z (bo z spełnia ograniczenia), więc norma nie rośnie i wynik spełnia oba zbiory (punkt
        dopuszczalny, nie dokładny rzut euklidesowy na przecięcie)."""
        n = spec.get('norma', 'brak'); rho = float(spec.get('rho', 0) or 0)
        if n == 'calosc' and rho > 0:
            s = float(d.norm())
            if s > rho:
                d.mul_(rho / s)
        elif n == 'token' and rho > 0:
            s = d.norm(dim=1, keepdim=True).clamp(min=1e-12)
            d.mul_(torch.clamp(rho / s, max=1.0))
        if spec['tryb'] in ('zakresy', 'dol_gelu'):
            zT = baza['z'][spec['_T'][:, None], spec['_C'][None, :]]
            if spec['tryb'] == 'zakresy':
                lo, hi = self.h_min[spec['_C']][None], self.h_max[spec['_C']][None]
                d.copy_(torch.minimum(torch.maximum(zT + d, lo), hi) - zT)
            else:
                d.copy_(torch.clamp(zT + d, min=GELU_MIN) - zT)

    @torch.no_grad()
    def naruszenia(self, zp, maska):
        """Naruszenie zakresów z checkpointu przez z' na tokenach maski: suma przekroczeń, ułamek współrzędnych poza
        [h_min, h_max] i poniżej min GELU (ten sam pomiar w każdym trybie, również swobodnym)."""
        x = zp[0][maska]
        nad = (x - self.h_max).clamp(min=0); pod = (self.h_min - x).clamp(min=0)
        return dict(suma=float((nad + pod).sum()), ulamek=float(((nad > 0) | (pod > 0)).float().mean()),
                    max=float(torch.maximum(nad, pod).max()), ponizej_gelu=float((x < GELU_MIN - 1e-6).float().mean()))

    @torch.no_grad()
    def podzial_fc2(self, dz):
        """Zmiana latentu dz [600, 768] -> normy części przechodzącej przez fc2 (przestrzeń wierszowa W2) i
        niewidocznej dla fc2 (jądro W2) oraz udział sumy kwadratów; przy zerowej korekcie udział nieokreślony."""
        vis = (dz @ self.Vr) @ self.Vr.T
        inv = dz - vis
        c = float((dz ** 2).sum()); a = float((vis ** 2).sum()); b = float((inv ** 2).sum())
        tok = (inv ** 2).sum(1) / (dz ** 2).sum(1).clamp(min=1e-30)
        return dict(norma=c ** .5, przechodzaca=a ** .5, niewidoczna=b ** .5,
                    udzial_przechodzacej=a / c if c > 0 else None, udzial_niewidocznej=b / c if c > 0 else None,
                    mapa_niewidocznej=torch.where((dz ** 2).sum(1) > 0, tok, torch.full_like(tok, float('nan'))).cpu().numpy())

    # ------------------------------------------------------------ specyfikacja interwencji
    def przygotuj_spec(self, spec, ramki, zaznaczenie=None):
        spec = dict(spec)
        wyb = spec.get('tokeny', 0)
        oracle_maska = False
        if wyb == 0:
            m = np.ones(600, bool)
        elif wyb == 1:
            if zaznaczenie is None or not len(zaznaczenie):
                raise ValueError('brak zaznaczonych tokenów w zakładce „Uczenie ViT”')
            m = np.zeros(600, bool); m[np.asarray(zaznaczenie, int)] = True
        else:
            if ramki is None or not len(ramki):
                raise ValueError('maska tokenów z GT wymaga etykiety obiektu')
            m = maski_gt(ramki)[['obiekt', 'granica', 'tlo'][wyb - 2]]; oracle_maska = True
        if not m.any():
            raise ValueError('pusty zbiór tokenów interwencji')
        spec['maska_tokenow'] = m
        spec['maska_z_gt'] = oracle_maska
        kan = np.asarray(spec.get('kanaly', np.arange(768)), np.int64)
        if spec['tryb'] not in TRYBY_LATENTU:
            kan = np.arange(768)                       # zmienia się całe 768 przez fc1 i GELU
        spec['kanaly'] = kan
        spec['_T'] = torch.tensor(np.flatnonzero(m), device=self.dev)
        spec['_maskaT'] = torch.tensor(m, device=self.dev)
        spec['_C'] = torch.tensor(kan, device=self.dev)
        return spec

    # ------------------------------------------------------------ ocena stanu
    @torch.no_grad()
    def ocena(self, o, ramki, cel, tau=0.5, prog_iou=0.5, tau_fp=0.3):
        cz = skladniki_straty(o, *cel, tau_fp=tau_fp)
        bx, sc = dekoduj(o.detach())
        m = ocena_obrazu(bx[0].cpu().numpy(), sc[0].cpu().numpy(), ramki if ramki is not None else np.zeros((0, 4)),
                         tau, prog_iou)
        m.update({k: float(v[0]) for k, v in cz.items()})
        return m


def cel_z_ramek(ramki, dev):
    heat, ltrb, poz = cele([np.asarray(ramki, np.float32).reshape(-1, 4)], 20, 30)
    return heat.to(dev), ltrb.to(dev), poz.to(dev)


def funkcja_celu(cel, cz, lam_fp):
    if cel == 'lokalizacja':
        return cz['ogniskowa_poz'] + 2 * cz['ramka'] + lam_fp * cz['kara_fp']
    return cz['strata']


# ---------------------------------------------------------------------------------------------------- zapis
class Zapis:
    """Folder pomiaru: CONFIG.json, STATUS.json, METRYKI.json, trajektoria porcjami (traj/czesc_NNNN.npz z
    rzeczywistymi indeksami kroków), KONIEC.npz, WYNIK.json. Każdy plik zapisywany atomowo."""

    def __init__(self, nazwa, porcja=50):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.kat = WYJSCIE / f'{stamp}_{nazwa}'
        k = 1
        while self.kat.exists():
            self.kat = WYJSCIE / f'{stamp}_{nazwa}_{k}'; k += 1
        (self.kat / 'traj').mkdir(parents=True)
        self.porcja = porcja; self.bufor = []; self.n_czesci = 0; self.bajty = 0
        self.t0 = time.time()

    def status(self, status, **x):
        _json(self.kat / 'STATUS.json', dict(status=status, czas_s=time.time() - self.t0,
                                              pamiec_gpu_max_MiB=(torch.cuda.max_memory_allocated() / 2 ** 20
                                                                  if torch.cuda.is_available() else None), **x))

    def dodaj(self, krok, **tablice):
        self.bufor.append((krok, tablice))
        if len(self.bufor) >= self.porcja:
            self.zrzuc()

    def zrzuc(self):
        if not self.bufor:
            return
        kroki = np.array([k for k, _ in self.bufor], np.int64)
        tab = {n: np.stack([t[n] for _, t in self.bufor]) for n in self.bufor[0][1]}
        _npz(self.kat / 'traj' / f'czesc_{self.n_czesci:04d}.npz', kroki=kroki, **tab)
        self.bajty += sum(v.nbytes for v in tab.values())
        self.n_czesci += 1; self.bufor = []


def wczytaj_trajektorie(kat, nazwa='d'):
    """(kroki [n], tablica [n, ...]) z porcji traj/ w kolejności zapisu."""
    czesci = sorted(Path(kat, 'traj').glob('czesc_*.npz'))
    if not czesci:
        return np.zeros(0, np.int64), None
    ks, xs = [], []
    for p in czesci:
        with np.load(p) as f:
            ks.append(f['kroki']); xs.append(f[nazwa])
    return np.concatenate(ks), np.concatenate(xs)


def koszt_zapisu(spec_wymiar, kroki, co, z_gradientem=True):
    n = kroki // max(co, 1) + 2
    return n * int(np.prod(spec_wymiar)) * 4 * (2 if z_gradientem else 1)


# ---------------------------------------------------------------------------------------------------- optymalizacja
def _jeden_przebieg(st, baza, ramki, cel_t, spec, przerwij=None, postep=None, zapis=None, p0=0, p1=100):
    """Jedna optymalizacja δ (Adam albo SGD) z rzutowaniem na ograniczenia. Zapisuje metryki co krok i trajektorię co
    `zapis_co` kroków (δ i gradient celu w tym stanie). Zwraca słownik z najlepszym i końcowym stanem."""
    dev = st.dev
    wym = st.wymiar(spec)
    g = torch.Generator(device='cpu').manual_seed(int(spec.get('ziarno', 0)))
    if spec.get('start', 'zero') == 'losowy':
        d0 = torch.randn(wym, generator=g) * float(spec.get('sigma_start', 0.01))
    else:
        d0 = torch.zeros(wym)
    d = d0.to(dev).requires_grad_(True)
    st.rzutuj(d.data, baza, spec)
    opt_n = spec.get('optymalizator', 'adam')
    lr = float(spec.get('lr', 0.1))
    opt = torch.optim.Adam([d], lr=lr) if opt_n == 'adam' else torch.optim.SGD([d], lr=lr)
    lam = float(spec.get('lam', 0.0)); lam_fp = float(spec.get('lam_fp', 1.0)); cel = spec.get('cel', 'strata')
    kroki = int(spec.get('kroki', 200)); co = int(spec.get('zapis_co', 1))
    tau, prog, tau_fp = float(spec.get('tau', .5)), float(spec.get('prog_iou', .5)), float(spec.get('tau_fp', .3))
    metr = []; najl = None; status = 'complete'
    maska = spec['_maskaT']
    for krok in range(kroki + 1):
        zp, rr, o = st.wyjscie(d, baza, spec)
        cz = skladniki_straty(o, *cel_t, tau_fp=tau_fp)
        Lc = funkcja_celu(cel, cz, lam_fp)[0]
        L = Lc + lam * (d ** 2).mean()
        opt.zero_grad(set_to_none=True)
        L.backward()
        Lc, L = Lc.detach(), L.detach()
        gr = d.grad.detach()
        if not torch.isfinite(gr).all():
            raise FloatingPointError(f'gradient nieskończony w kroku {krok}')
        with torch.no_grad():
            m = st.ocena(o, ramki, cel_t, tau, prog, tau_fp)
            dz = (zp - baza['z'][None])[0]
            pf = st.podzial_fc2(dz)
            nr = st.naruszenia(zp, maska)
            rek = dict(krok=krok, cel=float(Lc), calkowita=float(L), norma_d=float(d.norm()), norma_dz=pf['norma'],
                       przechodzaca=pf['przechodzaca'], niewidoczna=pf['niewidoczna'],
                       udzial_niewidocznej=pf['udzial_niewidocznej'], naruszenie=nr['suma'], naruszenie_ulamek=nr['ulamek'],
                       ponizej_gelu=nr['ponizej_gelu'], **{k: v for k, v in m.items() if k in (
                           'strata', 'ogniskowa_poz', 'ogniskowa_neg', 'ramka', 'kara_fp', 'iou_top', 'pewnosc_top',
                           'tp', 'fp', 'pominiete', 'iou_najlepsze_diag')})
            if opt_n == 'adam' and d in opt.state and 'exp_avg' in opt.state[d]:
                rek['adam_m'] = float(opt.state[d]['exp_avg'].norm()); rek['adam_v'] = float(opt.state[d]['exp_avg_sq'].sum().sqrt())
            metr.append(rek)
            if najl is None or float(Lc) < najl['cel']:
                najl = dict(cel=float(Lc), krok=krok, d=d.detach().clone())
            if zapis is not None and (krok % co == 0 or krok == kroki):
                zapis.dodaj(krok, d=d.detach().cpu().numpy().astype(np.float32), g=gr.cpu().numpy().astype(np.float32))
        if krok == kroki:
            break
        if przerwij is not None and przerwij():
            status = 'cancelled'; break
        opt.step()
        st.rzutuj(d.data, baza, spec)
        if postep and krok % 10 == 0:
            postep(p0 + int((p1 - p0) * krok / max(kroki, 1)), f'krok {krok}/{kroki}: cel {float(Lc):.4f}, '
                                                                 f'IoU top {m["iou_top"] if m["iou_top"] is None else round(m["iou_top"], 3)}, '
                                                                 f'||δ|| {float(d.detach().norm()):.3g}')
    stan_opt = {}
    if opt_n == 'adam' and d in opt.state and 'exp_avg' in opt.state[d]:
        stan_opt = dict(adam_m=opt.state[d]['exp_avg'].cpu().numpy(), adam_v=opt.state[d]['exp_avg_sq'].cpu().numpy(),
                        adam_krok=float(opt.state[d]['step']))
    return dict(metryki=metr, najlepszy=najl, koniec=d.detach().clone(), status=status, stan_opt=stan_opt)


def _publiczna_spec(spec):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in spec.items() if not k.startswith('_')}


@torch.no_grad()
def _stan(st, d, baza, ramki, cel_t, spec):
    zp, rr, o = st.wyjscie(d, baza, spec)
    m = st.ocena(o, ramki, cel_t, spec.get('tau', .5), spec.get('prog_iou', .5), spec.get('tau_fp', .3))
    dz = (zp - baza['z'][None])[0]
    pf = st.podzial_fc2(dz)
    m.update(norma_d=float(d.norm()), norma_dz=pf['norma'], fc2={k: v for k, v in pf.items() if k != 'mapa_niewidocznej'},
             naruszenia=st.naruszenia(zp, spec['_maskaT']))
    return m, zp, o, pf


def kontrole_losowe(st, d_ref, baza, ramki, cel_t, spec, ziarna=(1, 2, 3)):
    """Losowe kierunki o tej samej normie co korekta (izotropowe na tym samym nośniku) i kontrola losowych znaków
    (moduły z korekty oracle, więc zależna od etykiety). Po rzucie na ograniczenia mierzona faktyczna norma."""
    wyn = []
    n_ref = float(d_ref.norm())
    for s in ziarna:
        g = torch.Generator(device='cpu').manual_seed(1000 + s)
        r = torch.randn(tuple(d_ref.shape), generator=g).to(st.dev)
        r = r * (n_ref / max(float(r.norm()), 1e-12))
        st.rzutuj(r, baza, spec)
        m, _, _, _ = _stan(st, r, baza, ramki, cel_t, spec)
        wyn.append(dict(rodzaj='izotropowa', ziarno=s, norma_przed_rzutem=n_ref, **m))
    for s in ziarna:
        g = torch.Generator(device='cpu').manual_seed(2000 + s)
        znak = (torch.randint(0, 2, tuple(d_ref.shape), generator=g) * 2 - 1).to(st.dev).float()
        r = d_ref.abs() * znak
        st.rzutuj(r, baza, spec)
        m, _, _, _ = _stan(st, r, baza, ramki, cel_t, spec)
        wyn.append(dict(rodzaj='losowe znaki (moduły z oracle)', ziarno=s, norma_przed_rzutem=n_ref, **m))
    return wyn


def kryterium(m, prog_iou_cel, prog_fp):
    return m['iou_top'] is not None and m['iou_top'] >= prog_iou_cel and m['fp'] <= prog_fp


def oblicz_korekte(st, rgb, ramki, spec, opis_obrazu, zaznaczenie=None, przerwij=None, postep=None):
    """Pełny pomiar korekty oracle jednego obrazu z zapisem w unikalnym folderze. Zwraca słownik wyników (także po
    przerwaniu: status 'cancelled' i częściowe wyniki; po błędzie: status 'failed' i opis w STATUS.json)."""
    spec = st.przygotuj_spec(spec, ramki, zaznaczenie)
    nazwa = f"{spec['tryb']}_{spec.get('cel', 'strata')}_{opis_obrazu.get('id', 'obraz')}"
    zapis = Zapis(nazwa)
    konf = dict(time_axis='latent_optimization', checkpoint=str(st.ckpt), checkpoint_sha256=st.sha, checkpoint_krok=st.krok,
                kod_modelu=str(Path(silnik.EKSPERYMENT_SD) / 'model_snapshot.py'),
                kod_modelu_sha256=sha_pliku(Path(silnik.EKSPERYMENT_SD) / 'model_snapshot.py'),
                kod_stanowiska_sha256=sha_pliku(__file__), punkt_wpiecia='bloki[-1].mlp.act (po GELU, przed fc2)',
                tryb=spec['tryb'], opis_trybu=TRYBY[spec['tryb']],
                residual=('r bez zmian (wejście norm2 z przebiegu)' if spec['tryb'] != 'wejscie_wspolne'
                          else 'r + e (zmienione wspólne wejście residualu)'),
                obraz=opis_obrazu, ramki_gt=np.asarray(ramki).tolist() if ramki is not None else None,
                skalowanie='obraz 720 x 480 -> 480 x 320 INTER_AREA, ramki w pikselach 480 x 320',
                oracle=True, opis_oracle='korekta z etykietą tego obrazu (oracle), nie wynik detektora bez etykiet',
                spec=_publiczna_spec(spec), cel=CELE.get(spec.get('cel', 'strata')), ewaluator=EWALUATOR,
                opis_ewaluatora=OPIS_EWALUATORA, urzadzenie=str(st.dev),
                gpu=torch.cuda.get_device_name() if st.dev.type == 'cuda' else None, torch=torch.__version__,
                fc2={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.fc2.items()})
    _json(zapis.kat / 'CONFIG.json', konf)
    zapis.status('running')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    wynik = dict(kat=str(zapis.kat), konfiguracja=konf, spec=spec)
    try:
        baza = st.przebieg(rgb)
        cel_t = cel_z_ramek(ramki if ramki is not None else np.zeros((0, 4)), st.dev)
        d0 = torch.zeros(st.wymiar(spec), device=st.dev)
        m0, _, o0, _ = _stan(st, d0, baza, ramki, cel_t, spec)
        wynik.update(blad_ogona=baza['blad_ogona'], przed=m0)
        krzywa = None
        spec_run = dict(spec)
        if spec.get('cel') == 'najmniejsza':
            krzywa, rho = _siatka_promieni(st, baza, ramki, cel_t, spec, przerwij, postep)
            wynik['krzywa_promieni'] = krzywa
            if rho is None:
                wynik['uwaga'] = 'żaden promień z siatki nie osiągnął progu; zapisano przebieg przy największym'
                rho = krzywa[-1]['rho'] if krzywa else float(spec.get('rho_max', 100))
            spec_run.update(norma='calosc', rho=rho, cel='strata')
            wynik['rho_wybrane'] = rho
        r = _jeden_przebieg(st, baza, ramki, cel_t, spec_run, przerwij, postep, zapis,
                            p0=60 if krzywa is not None else 0, p1=95)
        zapis.zrzuc()
        status = r['status'] if not (krzywa is not None and krzywa and krzywa[-1].get('przerwane')) else 'cancelled'
        mb, zb, ob, pfb = _stan(st, r['najlepszy']['d'], baza, ramki, cel_t, spec_run)
        mk, _, ok, _ = _stan(st, r['koniec'], baza, ramki, cel_t, spec_run)
        wynik.update(metryki=r['metryki'], najlepszy=dict(krok=r['najlepszy']['krok'], **mb), koniec=mk, status=status)
        if postep:
            postep(96, 'kontrole losowe...')
        wynik['kontrole'] = kontrole_losowe(st, r['najlepszy']['d'], baza, ramki, cel_t, spec_run)
        # kontrole poprawności
        zgodne_poza = None
        if spec['tryb'] in TRYBY_LATENTU:
            poza = torch.ones(600, 768, dtype=torch.bool, device=st.dev)
            poza[spec['_T'][:, None], spec['_C'][None, :]] = False
            zgodne_poza = bool(torch.equal(zb[0][poza], baza['z'][poza]))
        else:
            zgodne_poza = bool(torch.equal(zb[0][~spec['_maskaT']], baza['z'][~spec['_maskaT']]))
        wynik['sprawdzenia'] = dict(zerowa_interwencja_blad=baza['blad_ogona'], wagi_bez_zmian=st.zgodny(),
                                    poza_interwencja_bez_zmian=zgodne_poza)
        wynik['mapa_dz'] = (zb[0] - baza['z']).norm(dim=1).cpu().numpy()
        wynik['mapa_niewidocznej'] = pfb['mapa_niewidocznej']
        wynik['o_przed'] = o0[0].cpu().numpy(); wynik['o_po'] = ob[0].cpu().numpy()
        _npz(zapis.kat / 'KONIEC.npz', tokeny=spec['_T'].cpu().numpy(), kanaly=spec['_C'].cpu().numpy(),
             z_baza=baza['z'].cpu().numpy(), r_baza=baza['r'].cpu().numpy(), d_najlepszy=r['najlepszy']['d'].cpu().numpy(),
             d_koniec=r['koniec'].cpu().numpy(), o_baza=o0[0].cpu().numpy(), o_najlepszy=ob[0].cpu().numpy(),
             o_koniec=ok[0].cpu().numpy(), obraz=np.asarray(rgb), mapa_dz=wynik['mapa_dz'],
             mapa_niewidocznej=wynik['mapa_niewidocznej'], **{k: np.asarray(v) for k, v in r['stan_opt'].items()})
        _json(zapis.kat / 'METRYKI.json', r['metryki'])
        wyn_json = {k: v for k, v in wynik.items() if k not in ('metryki', 'spec', 'mapa_dz', 'mapa_niewidocznej', 'o_przed', 'o_po')}
        wyn_json['zapisane_kroki'] = 'traj/czesc_*.npz: d i g w krokach podanych w kluczu kroki (co %d)' % spec_run.get('zapis_co', 1)
        wyn_json['bajty_trajektorii'] = zapis.bajty
        _json(zapis.kat / 'WYNIK.json', wyn_json)
        zapis.status(status, bajty_trajektorii=zapis.bajty)
        wynik['bajty_trajektorii'] = zapis.bajty
        wynik['spec_run'] = spec_run
        return wynik
    except Exception as e:
        import traceback
        zapis.zrzuc()
        zapis.status('failed', blad=traceback.format_exc()[-4000:])
        raise


def _siatka_promieni(st, baza, ramki, cel_t, spec, przerwij, postep):
    """Najmniejsza znaleziona korekta: przebiegi celu 'strata' w kulach ||δ|| <= ρ dla siatki geometrycznej ρ
    (ρ_min..ρ_max, n), potem 3 kroki bisekcji między ostatnim nieudanym i pierwszym udanym. Kryterium: IoU
    najpewniejszej ramki >= prog_iou_cel i fałszywych detekcji <= prog_fp. Krzywa jakość–norma z wszystkich przebiegów."""
    rmin, rmax, n = float(spec.get('rho_min', 1.0)), float(spec.get('rho_max', 300.0)), int(spec.get('n_rho', 8))
    pi, pf = float(spec.get('prog_iou_cel', .7)), float(spec.get('prog_fp', 0))
    krzywa = []
    sp = dict(spec, cel='strata', norma='calosc', kroki=int(spec.get('kroki_promienia', max(50, int(spec.get('kroki', 200)) // 2))))

    def jeden(rho, i, n_all):
        s = dict(sp, rho=rho)
        r = _jeden_przebieg(st, baza, ramki, cel_t, s, przerwij)
        m, _, _, _ = _stan(st, r['najlepszy']['d'], baza, ramki, cel_t, s)
        rek = dict(rho=rho, osiaga=kryterium(m, pi, pf), przerwane=r['status'] == 'cancelled', **{k: m[k] for k in (
            'strata', 'iou_top', 'pewnosc_top', 'fp', 'pominiete', 'norma_d', 'norma_dz')})
        krzywa.append(rek)
        if postep:
            postep(int(55 * i / n_all), f'promień {rho:.3g}: IoU top {m["iou_top"]}, fp {m["fp"]}, osiąga {rek["osiaga"]}')
        return rek
    siatka = np.geomspace(rmin, rmax, n)
    udany = None; poprz = None
    for i, rho in enumerate(siatka):
        rek = jeden(float(rho), i, n + 3)
        if rek['przerwane']:
            return krzywa, None
        if rek['osiaga']:
            udany = float(rho); break
        poprz = float(rho)
    if udany is not None and poprz is not None:
        lo, hi = poprz, udany
        for j in range(3):
            mid = (lo * hi) ** .5
            rek = jeden(mid, n + j, n + 3)
            if rek['przerwane']:
                break
            if rek['osiaga']:
                hi = mid
            else:
                lo = mid
        udany = hi
    krzywa.sort(key=lambda r: r['rho'])
    return krzywa, udany


# ---------------------------------------------------------------------------------------------------- historia
_HIST = {}


def _hist():
    if not _HIST:
        t = np.load(HISTORIA / 'TARGETS_FULL400.npz')
        _HIST.update(src=t['source_indices'], J=t['channels0'], base=t['base'], target=t['target'], delta=t['delta_raw'])
        _HIST['det'] = dict(np.load(HISTORIA / 'DETECTIONS_FULL400.npz'))
        _HIST['wynik'] = json.loads((HISTORIA / 'RESULT.json').read_text(encoding='utf-8'))
    return _HIST


def ap_historyczne():
    """AP50 i AP50:95 zbioru 400 obrazów z DETECTIONS_FULL400 tym ewaluatorem, obok liczb zapisanych w RESULT.json."""
    H = _hist(); rows = manifest()
    gt = [obraz_gt(rows[int(i)]) for i in H['src']]
    out = {}
    for nazwa in ('baseline', 'optimized', 'random'):
        bx, sc = dekoduj(torch.from_numpy(H['det'][nazwa]))
        out[nazwa] = ap_zbioru(list(zip(bx.numpy(), sc.numpy())), gt)
    zap = {n: {k: H['wynik'][m][k] for k in ('AP50', 'AP50_95')} for n, m in
           (('baseline', 'baseline'), ('optimized', 'optimized'), ('random', 'random_control'))}
    return dict(policzone=out, zapisane=zap, uwaga='kontrola random dziedziczy moduły korekt oracle (zależna od etykiet)')


def obraz_gt(row):
    x, y, w, h = row['bbox_xywh']
    return np.array([[x, y, x + w, y + h]], np.float32)


def import_historyczny(st, i_manifest):
    """Para przed/po z maksymalizacji J128 dla obrazu manifestu: tylko końce korekty (nie trajektoria). Sprawdza
    checkpoint (hash), kanały, indeks obrazu i bazowy latent; ostrzega przy niezgodności."""
    H = _hist(); rows = manifest()
    ostrz = []
    pos = np.flatnonzero(H['src'] == i_manifest)
    if not len(pos):
        raise ValueError(f'obrazu {i_manifest} nie ma w TARGETS_FULL400')
    pos = int(pos[0])
    if H['wynik'].get('checkpoint_sha') != st.sha:
        ostrz.append(f'NIEZGODNY CHECKPOINT: wynik historyczny z sha {H["wynik"].get("checkpoint_sha", "?")[:12]}, '
                     f'wybrany {st.sha[:12]} (krok {st.krok}); para przed/po nie dotyczy tego modelu')
    J = kanaly_j128()
    if not np.array_equal(J, H['J']):
        ostrz.append('kanały w TARGETS różnią się od channels.npz')
    if sha_pliku(WEJSCIE_J128 / 'channels.npz') != H['wynik'].get('channels_sha'):
        ostrz.append('hash channels.npz różny od zapisanego w RESULT.json')
    row = rows[i_manifest]
    if row['index'] != i_manifest:
        ostrz.append('indeks w manifeście nie odpowiada pozycji')
    rgb, ramki = obraz_manifestu(row)
    baza = st.przebieg(rgb)
    Jt = torch.tensor(H['J'], device=st.dev)
    zJ = baza['z'][:, Jt].cpu().numpy()
    roznica_bazy = float(np.abs(zJ - H['base'][pos]).max())
    if roznica_bazy > 1e-3:
        ostrz.append(f'bazowy latent różni się od zapisanego o {roznica_bazy:.3g} (inny checkpoint albo obraz)')
    zp = baza['z'].clone(); zp[:, Jt] = torch.from_numpy(H['target'][pos]).to(st.dev)
    with torch.no_grad():
        o_po = st.ogon(zp[None], baza['r'][None])
    roznica_det = float(np.abs(o_po[0].cpu().numpy() - H['det']['optimized'][pos]).max())
    if roznica_det > 1e-3:
        ostrz.append(f'wyjście po korekcie różni się od DETECTIONS_FULL400 o {roznica_det:.3g}')
    cel_t = cel_z_ramek(ramki, st.dev)
    przed = st.ocena(baza['o'][None], ramki, cel_t); po = st.ocena(o_po, ramki, cel_t)
    dz = zp - baza['z']
    pf = st.podzial_fc2(dz)
    pfJ = st.fc2_podzbioru(H['J'])
    nr = st.naruszenia(zp[None], torch.ones(600, dtype=torch.bool, device=st.dev))
    dJ = H['delta'][pos]
    return dict(i=i_manifest, pos=pos, row=row, rgb=rgb, ramki=ramki, przed=przed, po=po, ostrzezenia=ostrz,
                roznica_bazy=roznica_bazy, roznica_detekcji=roznica_det, fc2=pf, fc2_J=pfJ, naruszenia=nr,
                mapa_dz=np.linalg.norm(dJ, axis=1), o_przed=baza['o'].cpu().numpy(), o_po=o_po[0].cpu().numpy(),
                min_target=float(H['target'][pos].min()), w_bazie_uczenia=row['frame_id'] in ramki_bazy_uczenia(),
                opis='Import historyczny: tylko stan przed i po (końce korekty 800 kroków Adam, lr 0,3, dolne '
                     'ograniczenie GELU). Zapisane są wybrane miary postępu, nie trajektoria aktywacji; interpolacja '
                     'między końcami nie jest pomiarem.')


# ---------------------------------------------------------------------------------------------------- tor uczenia
@torch.no_grad()
def tor_uczenia(rgb, ramki, postep=None, przerwij=None, dev=None):
    """Latent obrazu przy każdym zapisanym checkpoincie gęstej historii (oś network_training, co 10 kroków):
    z [K, 600, 768] float32, wyjścia [K, 20, 30, 5], kroki [K] i miary detekcji. Własna kopia sieci."""
    pliki = sorted(WAGI_GESTE.glob('k*.pt'))
    if not pliki:
        raise FileNotFoundError(f'brak checkpointów w {WAGI_GESTE}')
    dev = torch.device(dev or ('cuda' if torch.cuda.is_available() else 'cpu'))
    net = MalyViT().to(dev).eval().requires_grad_(False)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(dev) / 255
    Z = np.zeros((len(pliki), 600, 768), np.float32); O = np.zeros((len(pliki), 20, 30, 5), np.float32); kroki = []
    cel_t = cel_z_ramek(ramki if ramki is not None else np.zeros((0, 4)), dev)
    miary = []
    for i, p in enumerate(pliki):
        if przerwij is not None and przerwij():
            Z, O = Z[:i], O[:i]; break
        ck = torch.load(p, map_location='cpu', weights_only=True)
        net.load_state_dict(ck['net']); kroki.append(int(ck.get('krok', int(p.stem[1:]))))
        net.zapisuj_latent(True)
        o, z = net(x)
        net.zapisuj_latent(False)
        Z[i] = z[0].float().cpu().numpy(); O[i] = o[0].cpu().numpy()
        cz = skladniki_straty(o, *cel_t)
        bx, sc = dekoduj(o)
        m = ocena_obrazu(bx[0].numpy() if bx.device.type == 'cpu' else bx[0].cpu().numpy(), sc[0].cpu().numpy(),
                         ramki if ramki is not None else np.zeros((0, 4)))
        m.update({k: float(v[0]) for k, v in cz.items()}); miary.append(m)
        if postep and i % 5 == 0:
            postep(int(100 * i / len(pliki)), f'tor uczenia: checkpoint {p.stem}')
    return dict(z=Z, o=O, kroki=np.array(kroki), miary=miary, time_axis='network_training', interwal='co 10 kroków uczenia')


def porownaj_z_uczeniem(tor, k_c, dz, maska=None):
    """Ruch uczenia obrazu wobec korekty oracle dz [600, 768] policzonej przy checkpoincie k_c.
    D(k) = z(k) − z(k_c): kosinus z dz (pełny latent i podprzestrzeń nośnika korekty), współczynnik rzutu α = <D, dz>/||dz||^2,
    stosunek norm. Dla k > k_c porównanie retrospektywne (używa przyszłych stanów uczenia, tylko wyjaśnienie). Prędkość
    uczenia tuż przed k_c (z(k_c) − z(k_c − 10)) używa wyłącznie przeszłości."""
    kroki = tor['kroki']
    ic = int(np.flatnonzero(kroki == k_c)[0]) if (kroki == k_c).any() else None
    if ic is None:
        raise ValueError(f'checkpointu {k_c} nie ma w torze uczenia')
    Z = tor['z']; dzf = np.asarray(dz, np.float64).reshape(-1)
    nd = np.linalg.norm(dzf)
    if nd == 0:
        raise ValueError('korekta zerowa: porównanie nieokreślone')
    nos = (np.abs(np.asarray(dz)).reshape(600, 768) > 0).reshape(-1) if maska is None else maska.reshape(-1)
    out = dict(kroki=kroki, k_c=k_c, cos=[], cos_nosnik=[], alfa=[], stosunek=[])
    for i in range(len(kroki)):
        D = (Z[i].astype(np.float64) - Z[ic]).reshape(-1)
        nD = np.linalg.norm(D)
        out['cos'].append(float(D @ dzf / (nD * nd)) if nD > 0 else np.nan)
        Dn = D[nos]; nDn = np.linalg.norm(Dn)
        out['cos_nosnik'].append(float(Dn @ dzf[nos] / (nDn * np.linalg.norm(dzf[nos]))) if nDn > 0 else np.nan)
        out['alfa'].append(float(D @ dzf / nd ** 2)); out['stosunek'].append(float(nD / nd))
    for k in ('cos', 'cos_nosnik', 'alfa', 'stosunek'):
        out[k] = np.array(out[k])
    if ic > 0:
        v = (Z[ic].astype(np.float64) - Z[ic - 1]).reshape(-1)
        out['cos_predkosci_przed'] = float(v @ dzf / max(np.linalg.norm(v) * nd, 1e-30))
        out['interwal_predkosci'] = int(kroki[ic] - kroki[ic - 1])
    out['retrospektywne_od'] = k_c
    return out


def baza_projekcji(rodzaj, dz, tor, k_c, oracle_traj=None, ziarno=0):
    """Stała baza 3 wektorów [3, 600*768] dla obu torów:
    0 kierunek korekty + 2 losowe ortogonalne (bez przyszłości);
    1 PCA ruchu uczenia do k_c (tylko przeszłość);
    2 PCA obu torów razem (retrospektywna: używa przyszłych stanów)."""
    D = 600 * 768
    dzf = np.asarray(dz, np.float64).reshape(-1)
    kroki = tor['kroki']; ic = int(np.flatnonzero(kroki == k_c)[0])
    Z = tor['z'].reshape(len(kroki), -1).astype(np.float64)
    if rodzaj == 0:
        rng = np.random.default_rng(ziarno)
        B = np.stack([dzf, rng.standard_normal(D), rng.standard_normal(D)])
    else:
        X = Z[:ic + 1] - Z[ic] if rodzaj == 1 else np.concatenate([Z - Z[ic]] + ([oracle_traj] if oracle_traj is not None else []))
        X = X - X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        B = Vt[:3]
        if len(B) < 3:
            B = np.concatenate([B, np.random.default_rng(ziarno).standard_normal((3 - len(B), D))])
    Q, _ = np.linalg.qr(B.T)
    Q = Q[:, :3].T
    if Q[0] @ dzf < 0:
        Q[0] = -Q[0]
    return Q, ('kierunek korekty + 2 losowe (bez przyszłości)', 'PCA ruchu uczenia do k_c (tylko przeszłość)',
               'PCA obu torów (retrospektywna: używa przyszłych stanów)')[rodzaj]


def trajektoria_dz(st, kat, spec_run, baza, co=1):
    """Zmiana latentu Δz(t) [n, 600*768] float32 z zapisanej trajektorii δ (dla trybów wejścia przez MLP)."""
    kroki, d = wczytaj_trajektorie(kat, 'd')
    if d is None:
        return kroki, None
    out = np.zeros((len(kroki), 600 * 768), np.float32)
    with torch.no_grad():
        for i in range(0, len(kroki), co):
            zp, _, _ = st.wyjscie(torch.from_numpy(d[i]).to(st.dev), baza, spec_run)
            out[i] = (zp[0] - baza['z']).reshape(-1).cpu().numpy()
    return kroki, out


def wczytaj_pomiar(kat):
    """Ponowne wczytanie zapisanego pomiaru korekty (CONFIG, STATUS, WYNIK, METRYKI, KONIEC). Brakujące pliki dają
    jasny błąd; pomiar przerwany albo nieudany wraca ze swoim statusem."""
    kat = Path(kat)
    if not (kat / 'CONFIG.json').exists():
        raise FileNotFoundError(f'{kat} nie jest folderem pomiaru (brak CONFIG.json)')
    rd = lambda n, d=None: json.loads((kat / n).read_text(encoding='utf-8')) if (kat / n).exists() else d
    konf = rd('CONFIG.json'); st = rd('STATUS.json', {}); wyn = rd('WYNIK.json', {}); metr = rd('METRYKI.json', [])
    out = dict(wyn, kat=str(kat), konfiguracja=konf, metryki=metr, status=st.get('status', wyn.get('status', '?')),
               status_pelny=st)
    if (kat / 'KONIEC.npz').exists():
        with np.load(kat / 'KONIEC.npz') as f:
            k = {n: f[n] for n in f.files}
        out.update(o_przed=k['o_baza'], o_po=k['o_najlepszy'], mapa_dz=k.get('mapa_dz'), mapa_niewidocznej=k.get('mapa_niewidocznej'),
                   obraz=k.get('obraz'), koniec_npz=k)
    spec = dict(konf['spec']); spec['tryb'] = konf['tryb']
    out['spec_publiczna'] = spec
    return out


def spec_z_zapisu(st, konf, ramki, zaznaczenie=None):
    """Specyfikacja interwencji (z tensorami) odtworzona z CONFIG.json zapisanego pomiaru."""
    spec = dict(konf['spec']); spec['tryb'] = konf['tryb']
    m = np.asarray(spec.pop('maska_tokenow'), bool)
    spec['tokeny'] = 1
    return st.przygotuj_spec(spec, ramki, np.flatnonzero(m))


def rzut_trajektorii(st, kat, spec_run, baza, Q, co=1):
    """Rzut zapisanej trajektorii korekty na stałą bazę Q [3, 600*768]: współrzędne [n, 3], kroki [n] i prędkość
    ||Δz(t_i) − Δz(t_(i−1))|| / (t_i − t_(i−1)) [n] w pełnym latencie (z rzeczywistych odstępów zapisu)."""
    kroki, d = wczytaj_trajektorie(kat, 'd')
    if d is None:
        return kroki, None, None
    Qt = torch.from_numpy(np.asarray(Q, np.float32)).to(st.dev)                # [3, D]
    xyz = np.zeros((len(kroki), 3)); pr = np.full(len(kroki), np.nan)
    poprz = None
    with torch.no_grad():
        for i in range(0, len(kroki), co):
            zp, _, _ = st.wyjscie(torch.from_numpy(d[i]).to(st.dev), baza, spec_run)
            dz = (zp[0] - baza['z']).reshape(-1)
            xyz[i] = (Qt @ dz).cpu().numpy()
            if poprz is not None:
                pr[i] = float((dz - poprz[1]).norm()) / max(int(kroki[i] - poprz[0]), 1)
            poprz = (int(kroki[i]), dz)
    return kroki, xyz, pr


def probka_trajektorii(st, kat, spec_run, baza, n=50):
    """Do n stanów Δz z zapisanej trajektorii (równo rozłożone), [n, 600*768] float32: do bazy retrospektywnej."""
    kroki, d = wczytaj_trajektorie(kat, 'd')
    if d is None:
        return None
    wyb = np.unique(np.linspace(0, len(kroki) - 1, min(n, len(kroki))).astype(int))
    out = []
    with torch.no_grad():
        for i in wyb:
            zp, _, _ = st.wyjscie(torch.from_numpy(d[i]).to(st.dev), baza, spec_run)
            out.append((zp[0] - baza['z']).reshape(-1).cpu().numpy())
    return np.stack(out)
