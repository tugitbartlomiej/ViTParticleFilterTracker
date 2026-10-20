"""Detekcja narzędzia małym ViT i optymalizacja latentu pod IoU/pewność (zakładka „Detekcja ViT”).

Latent = aktywacje po GELU ukrytej warstwy MLP ostatniego bloku (600 tokenów x 768 kanałów).
Ogon sieci od latentu do mapy detekcji jest liniowy w zmianie latentu:
    wyjście_bloku' = wyjście_bloku + fc2(z' - z),  mapa = glowa(norm(wyjście_bloku'))
(jak `fast` w experiments/vit_latent128_jedno_zdjecie_90/wyznacz.py), więc gradient po latencie jest dokładny.
Strata optymalizacji jest ta sama co w tamtym eksperymencie: dodatni logit środka etykiety, tłumienie
pozostałych pozycji, ranking, log-odległości ramki i GIoU. Wagi ViT i piksele nie są zmieniane.
"""
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import silnik  # ustawia sys.path na eksperyment SD (model_snapshot.MalyViT)
from model_snapshot import MalyViT, _giou, dekoduj

ADNOTACJE = silnik.ROOT.parent / 'Dataset' / 'Cataract-LMM' / '3_Object_Tracking' / 'annotations'
KROK = 16


def iou(b, g):
    x0, y0 = np.maximum(b[..., 0], g[0]), np.maximum(b[..., 1], g[1])
    x1, y1 = np.minimum(b[..., 2], g[2]), np.minimum(b[..., 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)


def ap(detekcje, ramki_gt, prog=0.5):
    """AP jednej klasy przy progu IoU, interpolacja 101-punktowa jak w COCO.

    detekcje: lista (boxes [k,4], scores [k]) na obraz; ramki_gt: lista tablic [n,4] (może być pusta).
    """
    wszystkie, n_gt = [], 0
    for i, ((bx, sc), gt) in enumerate(zip(detekcje, ramki_gt)):
        n_gt += len(gt)
        wszystkie += [(float(s), i, b) for b, s in zip(bx, sc)]
    if n_gt == 0:
        return float('nan')
    wszystkie.sort(key=lambda x: -x[0])
    uzyte = [np.zeros(len(g), bool) for g in ramki_gt]
    tp = np.zeros(len(wszystkie))
    for j, (_, i, b) in enumerate(wszystkie):
        g = ramki_gt[i]
        if len(g):
            v = np.array([iou(b, gg) for gg in g]); v[uzyte[i]] = -1
            k = int(v.argmax())
            if v[k] >= prog:
                tp[j] = 1; uzyte[i][k] = True
    ctp = np.cumsum(tp); rec = ctp / n_gt; prec = ctp / np.arange(1, len(tp) + 1)
    for j in range(len(prec) - 2, -1, -1):
        prec[j] = max(prec[j], prec[j + 1])
    wynik = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        wynik += prec[k] if k < len(prec) else 0.0
    return wynik / 101


def metryki(boxes, scores, gt):
    """Pewność i IoU najpewniejszej ramki, AP@0,5 i mAP@0,5:0,95 na jednym obrazie."""
    m = dict(pewnosc=float(scores[0]), ramka=boxes[0].tolist())
    if gt is None:
        return dict(m, iou=None, ap50=None, map5095=None)
    g = np.asarray(gt, np.float32)
    det, gts = [(boxes, scores)], [g[None]]
    return dict(m, iou=float(iou(boxes[:1], g)[0]), ap50=ap(det, gts, .5),
                map5095=float(np.mean([ap(det, gts, p) for p in np.arange(.5, .96, .05)])))


def etykieta_dla_zdjecia(sciezka):
    """Ramka narzędzia (xyxy w pikselach natywnej klatki) z annotation.csv filmu, jeśli nazwa pliku ją wskazuje.

    Rozpoznaje nazwy typu TR_0001_S1_P03_klatka_000870.jpg oraz TR_0001_S1_P03_000870.jpg.
    Zwraca (ramka albo None, opis).
    """
    m = re.search(r'(TR_\d+_S\d+_P\d+)\D*?(\d+)\.(?:jpg|jpeg|png)$', Path(sciezka).name, re.I)
    if not m:
        return None, 'brak etykiety: nazwa pliku nie wskazuje filmu i klatki'
    film, klatka = m.group(1), int(m.group(2))
    zipp = ADNOTACJE / f'{film}.zip'
    if not zipp.exists():
        return None, f'brak etykiety: nie ma {zipp.name}'
    with zipfile.ZipFile(zipp) as z:
        nazwa = next(n for n in z.namelist() if n.endswith('annotation.csv'))
        tekst = z.read(nazwa).decode('utf-8-sig')
    import json
    csv.field_size_limit(1 << 25)
    for w in csv.DictReader(io.StringIO(tekst)):
        if int(w['frame_number']) == klatka:
            surowe = (w.get('cystotome_bbox') or '').strip()
            if surowe and surowe not in ('[]', 'null'):
                x, y, bw, bh = map(float, json.loads(surowe))
                return [x, y, x + bw, y + bh], f'etykieta: {film} kl. {klatka} (cystotom)'
            return None, f'{film} kl. {klatka}: na tej klatce nie oznaczono narzędzia'
    return None, f'brak klatki {klatka} w {film}'


class Detektor:
    """Zamrożony mały ViT z checkpointu (słownik z kluczem net, dowolny krok)."""

    def __init__(self, sciezka, urzadzenie='cuda'):
        ck = torch.load(sciezka, map_location='cpu', weights_only=True)
        self.krok = ck.get('krok')
        self.net = MalyViT()
        self.net.load_state_dict(ck.get('net', ck), strict=True)
        self.dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
        self.net.to(self.dev).eval().requires_grad_(False)
        self.sciezka = Path(sciezka)

    @torch.no_grad()
    def analizuj(self, rgb480):
        """rgb480: uint8 [320,480,3]. Zwraca słownik: mapa, latent z [600,768], wyjście bloku q, ramki, pewności."""
        blok, pam = self.net.bloki[-1], {}
        h1 = blok.mlp.act.register_forward_hook(lambda m, a, o: pam.update(z=o.detach().clone()))
        h2 = blok.register_forward_hook(lambda m, a, o: pam.update(q=o.detach().clone()))
        try:
            x = torch.from_numpy(np.ascontiguousarray(rgb480)).permute(2, 0, 1)[None].float().to(self.dev) / 255
            mapa, _ = self.net(x)
        finally:
            h1.remove(); h2.remove()
        bx, sc = dekoduj(mapa.detach().cpu())
        return dict(mapa=mapa, z=pam['z'], q=pam['q'], boxes=bx[0].numpy(), scores=sc[0].numpy())

    @torch.no_grad()
    def atencja(self, rgb480, blok=-1):
        """Wagi atencji wybranego bloku [głowy, 600 zapytań, 600 kluczy] (softmax(q k^T / sqrt(d))).

        Liczone jak w timm Attention z wejścia bloku atencji. Latent (ukryta warstwa MLP ostatniego bloku)
        leży za atencją, więc jego zmiana w auto-szukaniu nie zmienia tych wag.
        """
        b, pam = self.net.bloki[blok], {}
        h = b.attn.register_forward_hook(lambda m, a, o: pam.update(x=a[0].detach()))
        try:
            x = torch.from_numpy(np.ascontiguousarray(rgb480)).permute(2, 0, 1)[None].float().to(self.dev) / 255
            self.net(x)
        finally:
            h.remove()
            self.net._latent = None
        at, x = b.attn, pam['x']
        B, N, C = x.shape
        qkv = at.qkv(x).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q, k = getattr(at, 'q_norm', torch.nn.Identity())(q), getattr(at, 'k_norm', torch.nn.Identity())(k)
        skala = getattr(at, 'scale', (C // at.num_heads) ** -0.5)
        return ((q * skala) @ k.transpose(-2, -1)).softmax(-1)[0].float().cpu().numpy()

    def ogon(self, a, z_nowy):
        """Mapa detekcji dla podmienionego latentu z_nowy [1,600,768] (ten sam obraz, te same wagi)."""
        blok = self.net.bloki[-1]
        delta = blok.ls2(F.linear(z_nowy - a['z'], blok.mlp.fc2.weight, None))
        return self.net.glowa(self.net.norm(a['q'] + delta)).reshape(1, 20, 30, 5)

    def detekcja_z_latentu(self, a, z_nowy):
        with torch.no_grad():
            bx, sc = dekoduj(self.ogon(a, z_nowy).cpu())
        return bx[0].numpy(), sc[0].numpy()

    def optymalizuj(self, a, gt480, kanaly, lim_lo, lim_hi, kroki=2000, lr=0.05, cel_pewnosc=.99, cel_iou=.99,
                    postep=None, przerwij=lambda: False, cel_map=1.0, waga_pewnosci=1.0, waga_ramki=1.0,
                    harmonogram='staly', lr_min=0.005, rozgrzewka=0, tokeny=None):
        """Adam po latencie na wybranych kanałach, z rzutem na limity makrowektorów po każdym kroku.

        gt480: ramka xyxy w pikselach wejścia 480x320. kanaly: indeksy kanałów do zmiany (pozostałe bez zmian).
        Cele (0..1): pewność najpewniejszej ramki, jej IoU do etykiety, mAP@0,5:0,95 tego obrazu; koniec, gdy
        wszystkie są spełnione albo skończą się kroki. Cel pewności jest przeliczany na wewnętrzną wartość
        przed sigmoidą (logit(cel) + 0,5 zapasu), do której dąży punkt środka etykiety. waga_pewnosci mnoży
        człony pewności (środek, tło, ranking), waga_ramki człony ramki (odległości do krawędzi, GIoU).
        Harmonogram lr: 'staly', 'cos' (cosinusowo do lr_min), 'liniowy' (do lr_min), 'krokowy' (połowa
        4 razy w trakcie), 'plateau' (połowa, gdy strata nie spadła o 1 % przez 100 kroków, nie niżej
        niż lr_min), 'restarty' (3 etapy lr x1, x3, x6 od najlepszego wyniku, nowy Adam; jak w
        eksperymencie J128). rozgrzewka: liczba pierwszych kroków z liniowym wzrostem lr od 0.
        tokeny: indeksy patchy (0..599), w których wolno zmieniać latent; None = wszystkie 600 (jak dotąd).
        Pozostałe patche mają latent bez zmian, więc ViT musi osiągnąć cel zmianą tylko w nich.
        Zwraca (z_po [1,600,768], historia). postep(krok, slownik) co 25 kroków.
        """
        import math
        p_cel = min(max(float(cel_pewnosc), 1e-3), 1 - 1e-4)
        logit_cel = math.log(p_cel / (1 - p_cel)) + 0.5
        g = np.asarray(gt480, np.float32)
        z0 = a['z']
        J = torch.as_tensor(np.asarray(kanaly), dtype=torch.long, device=self.dev)
        T = torch.as_tensor(np.arange(600) if tokeny is None else np.unique(np.asarray(tokeny, int)),
                            dtype=torch.long, device=self.dev)
        TJ = (T[:, None], J[None, :])                                       # [nT, nJ]: wybrane patche x kanały
        lo = torch.as_tensor(np.asarray(lim_lo)[np.asarray(kanaly)], dtype=torch.float32, device=self.dev)
        hi = torch.as_tensor(np.asarray(lim_hi)[np.asarray(kanaly)], dtype=torch.float32, device=self.dev)
        cx = int(np.clip(round((g[0] + g[2]) / (2 * KROK) - .5), 0, 29))
        cy = int(np.clip(round((g[1] + g[3]) / (2 * KROK) - .5), 0, 19))
        srodek = cy * 30 + cx
        px, py = (cx + .5) * KROK, (cy + .5) * KROK
        odl = torch.tensor([[px - g[0], py - g[1], g[2] - px, g[3] - py]], device=self.dev).clamp(min=.05) / KROK
        cel_log = odl.log()
        neg = torch.ones(1, 600, device=self.dev); neg[0, srodek] = 0

        def strata(o):
            flat = o.reshape(1, 600, 5); logit = flat[:, :, 0]
            pozyt = F.softplus(logit_cel - logit[0, srodek])
            negat = (F.softplus(logit + 2) * neg).sum() / 599
            rank = F.cross_entropy(logit, torch.tensor([srodek], device=self.dev))
            ramka = F.smooth_l1_loss(flat[:, srodek, 1:], cel_log)
            giou = _giou(flat[:, srodek, 1:].clamp(-10, 10).exp(), odl).mean()
            return waga_pewnosci * (pozyt + negat + .5 * rank) + waga_ramki * (20 * ramka + 10 * (1 - giou))

        def pelny(zm):
            z = z0.clone(); z[(slice(None),) + TJ] = zm
            return z

        start = torch.max(torch.min(z0[(slice(None),) + TJ], hi), lo)
        kroki = int(kroki)
        etapy = [(1.0, kroki)] if harmonogram != 'restarty' else \
            [(1.0, kroki // 3), (3.0, kroki // 3), (6.0, kroki - 2 * (kroki // 3))]

        def lr_bazowe(k, n):
            """lr w kroku k (od 1) etapu o długości n, bez plateau."""
            if rozgrzewka and k <= rozgrzewka:
                return lr * k / rozgrzewka
            t = (k - rozgrzewka) / max(1, n - rozgrzewka)
            if harmonogram == 'cos':
                return lr_min + (lr - lr_min) * .5 * (1 + math.cos(math.pi * min(t, 1)))
            if harmonogram == 'liniowy':
                return lr + (lr_min - lr) * min(t, 1)
            if harmonogram == 'krokowy':
                return max(lr_min, lr * .5 ** min(int(t * 5), 4))
            return lr

        najl, najl_strata, historia = start.clone(), float('inf'), []
        krok_globalny, koniec = 0, False
        for mnoznik, dlugosc in etapy:
            zm = najl.clone().requires_grad_(True)
            opt = torch.optim.Adam([zm], lr=lr * mnoznik)
            plateau_mn, ostatnie = 1.0, []
            for k in range(1, dlugosc + 1):
                krok_globalny += 1
                lr_teraz = lr_bazowe(k, dlugosc) * mnoznik * plateau_mn
                if harmonogram == 'plateau':
                    lr_teraz = max(lr_teraz, lr_min) if plateau_mn < 1 else lr_teraz
                for grupa in opt.param_groups:
                    grupa['lr'] = lr_teraz
                loss = strata(self.ogon(a, pelny(zm)))
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    zm.copy_(torch.max(torch.min(zm, hi), lo))          # limity makrowektorów
                    o = self.ogon(a, pelny(zm)); s = float(strata(o))
                    if s < najl_strata:
                        najl_strata, najl = s, zm.detach().clone()
                    if krok_globalny % 25 == 0 or krok_globalny == kroki:
                        bx, sc = dekoduj(o.cpu())
                        m = metryki(bx[0].numpy(), sc[0].numpy(), g)
                        historia.append(dict(krok=krok_globalny, strata=s, lr=lr_teraz, pewnosc=m['pewnosc'],
                                             iou=m['iou'], map5095=m['map5095']))
                        if postep:
                            postep(krok_globalny, historia[-1])
                        if harmonogram == 'plateau':
                            ostatnie.append(s)
                            if len(ostatnie) > 4 and s > min(ostatnie[:-4]) * 0.99:
                                plateau_mn *= .5
                                ostatnie = [s]
                        if (m['pewnosc'] >= cel_pewnosc and m['iou'] >= cel_iou and m['map5095'] >= cel_map - 1e-9) or przerwij():
                            najl, koniec = zm.detach().clone(), True
                            break
            if koniec:
                break
        return pelny(najl).detach(), historia


def do_480(rgb):
    """Dowolny RGB uint8 HxWx3 albo float [3,H,W] w [0,1] -> uint8 [320,480,3] (INTER_AREA jak w treningu ViT)."""
    import cv2
    x = np.asarray(rgb)
    if x.ndim == 3 and x.shape[0] == 3 and x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.resize(np.ascontiguousarray(x), (480, 320), interpolation=cv2.INTER_AREA)
