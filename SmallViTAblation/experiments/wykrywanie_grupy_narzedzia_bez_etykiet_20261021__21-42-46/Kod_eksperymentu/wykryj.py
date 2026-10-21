"""Wykrywanie grupy tokenów narzędzia BEZ etykiet; etykieta (obrys) tylko do oceny.

Protokół (ustalony przed obliczeniami, 2026-10-21 21:45):
  * Klatki: MANIFEST_1500 spoza 35000 uczących, z obrysem narzędzia. Podział PO FILMACH: sha1(film) % 10 < 4 → DEV, reszta TEST.
  * DEV: wszystkie konfiguracje (cechy × metoda × parametry × spójność), checkpoint seed 3047 epoka 400 (k437500).
    Wybór: największa średnia Dice z rdzeniem narzędzia (udział pikseli ≥ 0,25) — miara główna ustalona z góry.
  * TEST: tylko wybrana konfiguracja (+ odniesienia), jednorazowo; dodatkowo jej przebieg w checkpointach.
  Cechy tokenu (600 na klatkę): wyjścia bloków 3–6 (b3…b6), stan przed MLP bloku 6 (x6), aktywacje MLP bloku 6 (z6, 768D),
  wejście głowicy po LayerNorm (n6).
  Metody (bez etykiet):
    kurt  – PCA-k wybielona, kierunek maks. kurtozy, grupa = odporne z > próg (Peña i Prieto 2001);
    odst  – odporna odległość od mediany w wybielonej PCA-k, z > próg (tokeny odstające od tła);
    cut   – TokenCut (Wang i in. 2022): graf kosinusowy A = 1 dla cos ≥ τ (inaczej 1e-5), drugi wektor własny znormalizowanego
            laplasjanu, podział wg średniej, grupa = strona z największym |y| (reguła TokenCut) albo strona mniejsza;
    + opcjonalnie największa spójna składowa na siatce 20 × 30 (sąsiedztwo 8).
  Odniesienie uczone z etykietami: mapa środków głowicy (sigmoid ≥ 0,5 × maks.), oraz wagi nieuczone (s47 k00000).
Miary: Dice i czułość wobec rdzenia (udział ≥ 0,25), precyzja (udział > 0), precyzja z sąsiedztwem 1 łaty.
"""
import sys, json, random, zipfile, hashlib, time, itertools
from pathlib import Path
import numpy as np, cv2, torch
from scipy.ndimage import label as etykietuj, binary_dilation
APP = Path(__file__).resolve().parents[3] / 'app'
EXP = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'Wyniki' / 'Dane'
sys.path.insert(0, str(APP))
import lasery_dane as LD, etykiety_narzedzia as EN

C200 = EXP / 'vit_kontynuacja_do_200epok_20261020__19-06-42/Wyniki/Dane/checkpoints'
C400 = EXP / 'vit_kontynuacja_do_400epok_20261020__23-50-57/Wyniki/Dane/checkpoints'
S47 = LD.ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
CKPT = {'s3047_ep400': C400 / 'k437500.pt', 's3047_ep300': C400 / 'k328125.pt', 's3047_ep200': C200 / 'k218750.pt',
        's3047_ep100': C200 / 'k109375.pt', 's3047_ep60': C200 / 'k065640.pt',
        's47_k65640': S47 / 'k65640.pt', 's47_k28444': S47 / 'k28444.pt', 's47_k10100': S47 / 'k10100.pt',
        's47_k01000': S47 / 'k01000.pt', 's47_k00000_nieuczony': S47 / 'k00000.pt'}
CECHY = ('b3', 'b4', 'b5', 'b6', 'x6', 'z6', 'n6')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True


# ------------------------------------------------------------------ dane
def klatki():
    uczace = {(r['film'], int(r['klatka'])) for r in LD.zbior_uczacy()}
    w = [json.loads(l) for l in (LD.ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_1500.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in w:
        r['film'] = r['frame_id'].split('/')[-2]; r['klatka'] = int(r['frame_id'].split('/')[-1])
    w = sorted((r for r in w if (r['film'], r['klatka']) not in uczace), key=lambda r: (r['film'], r['klatka']))
    out = []
    for r in w:
        e = EN.etykieta(r['film'], r['klatka'])
        if e is not None and (e['udzial'] >= 0.25).sum() >= 1:
            r['udzial'] = e['udzial']
            r['podzial'] = 'DEV' if int(hashlib.sha1(r['film'].encode()).hexdigest(), 16) % 10 < 4 else 'TEST'
            out.append(r)
    return out


def obraz(r):
    with zipfile.ZipFile(LD.ZIPY / f"{r['film']}.zip") as z:
        raw = z.read(f"{r['film']}/{r['film']}_{r['klatka']:06d}.jpg")
    assert hashlib.sha256(raw).hexdigest() == r['image_sha256']
    return cv2.resize(cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (480, 320),
                      interpolation=cv2.INTER_AREA)


class Siec:
    def __init__(self):
        M = LD.modul_modelu()
        self.net = M.MalyViT().to(dev).eval().requires_grad_(False)
        self.p = {}
        for b in (2, 3, 4, 5):
            self.net.bloki[b].register_forward_hook(lambda m, a, o, b=b: self.p.__setitem__(f'b{b + 1}', o.detach()))
        self.net.bloki[5].norm2.register_forward_hook(lambda m, a, o: self.p.__setitem__('x6', a[0].detach()))
        self.net.bloki[5].mlp.act.register_forward_hook(lambda m, a, o: self.p.__setitem__('z6', o.detach()))
        self.net.norm.register_forward_hook(lambda m, a, o: self.p.__setitem__('n6', o.detach()))

    def wczytaj(self, p):
        st = torch.load(p, map_location='cpu', weights_only=False)
        self.net.load_state_dict(st.get('net', st), strict=True)

    def __call__(self, ob):
        with torch.no_grad():
            o, _ = self.net(torch.from_numpy(ob).permute(0, 3, 1, 2).float().div(255).to(dev))
        return {k: v.float() for k, v in self.p.items()}, o[..., 0].sigmoid().reshape(len(ob), -1).float()


# ------------------------------------------------------------------ metody (wsadowo na karcie; etykiet nie widzą)
def wybiel(F, k):
    Fc = F - F.mean(1, keepdim=True)
    U, S, Vh = torch.linalg.svd(Fc, full_matrices=False)
    return U[:, :, :k] * np.sqrt(F.shape[1])


def odporne_z(s):
    med = s.median(1, keepdim=True).values
    mad = 1.4826 * (s - med).abs().median(1, keepdim=True).values + 1e-12
    return (s - med) / mad


def kierunek_kurtozy(Y, starty=20, iteracje=60, ziarno=0):
    """Y (B, n, k) wybielone → s (B, n) rzut na kierunek maks. kurtozy (wsadowo, wszystkie starty naraz)."""
    B, n, k = Y.shape
    g = torch.Generator(device='cpu').manual_seed(ziarno)
    W = torch.cat([torch.eye(k).expand(B, k, k), torch.randn(B, starty - k if starty > k else 1, k, generator=g)], 1).to(Y) if starty > k \
        else torch.eye(k).expand(B, k, k).to(Y)
    W = W / W.norm(dim=2, keepdim=True)
    for _ in range(iteracje):
        S = Y @ W.transpose(1, 2)                                         # (B, n, starty)
        W = (S ** 3).transpose(1, 2) @ Y / n - 3 * W
        W = W / W.norm(dim=2, keepdim=True).clamp_min(1e-12)
    S = Y @ W.transpose(1, 2)
    kurt = (S ** 4).mean(1) - 3
    return S.gather(2, kurt.argmax(1)[:, None, None].expand(-1, n, 1))[..., 0]


def grupa_z_z(z, prog):
    plus, minus = (z > prog).sum(1), (z < -prog).sum(1)
    z = torch.where((minus > plus)[:, None], -z, z)
    return z > prog


def tokencut(F, tau, centruj):
    if centruj:
        F = F - F.mean(1, keepdim=True)
    Fn = F / F.norm(dim=2, keepdim=True).clamp_min(1e-12)
    S = Fn @ Fn.transpose(1, 2)
    A = torch.where(S >= tau, torch.ones_like(S), torch.full_like(S, 1e-5))
    d = A.sum(2)
    Dm = d.rsqrt()
    L = torch.eye(F.shape[1], device=F.device)[None] - Dm[:, :, None] * A * Dm[:, None, :]
    _, V = torch.linalg.eigh(L)
    y = V[:, :, 1] * Dm
    strona = y > y.mean(1, keepdim=True)
    jmax = y.abs().argmax(1)
    stronaY = strona.gather(1, jmax[:, None])                            # reguła TokenCut: strona z największym |y|
    tc = torch.where(stronaY, strona, ~strona)
    mniejsza = torch.where((strona.sum(1) <= F.shape[1] // 2)[:, None], strona, ~strona)
    return tc, mniejsza


def spojna(G):
    """Największa spójna składowa (8-sąsiedztwo) na siatce 20 × 30; G (B, 600) bool numpy."""
    out = np.zeros_like(G)
    for i, g in enumerate(G):
        lab, n = etykietuj(g.reshape(20, 30), structure=np.ones((3, 3)))
        if n:
            roz = np.bincount(lab.ravel())[1:]
            out[i] = (lab == 1 + roz.argmax()).ravel()
    return out


def konfiguracje(tylko=None):
    k = []
    for c in CECHY:
        for kp, pr in itertools.product((5, 10, 20), (2.5, 3.5)):
            k.append(dict(cechy=c, metoda='kurt', k=kp, prog=pr))
            k.append(dict(cechy=c, metoda='odst', k=kp, prog=pr))
        for tau, cen in itertools.product((0.0, 0.1, 0.2, 0.3, 0.4), (False, True)):
            k.append(dict(cechy=c, metoda='cut', tau=tau, centruj=cen, strona='tokencut'))
            k.append(dict(cechy=c, metoda='cut', tau=tau, centruj=cen, strona='mniejsza'))
    k = [dict(kk, spojna=s) for kk in k for s in (False, True)]
    for kk in k:
        kk['id'] = json.dumps(kk, sort_keys=True)
    return k if tylko is None else [kk for kk in k if kk['id'] in tylko]


def grupy(cechy, mapa, konf):
    """{id konfiguracji: (B, 600) bool numpy} + odniesienie głowicy."""
    out = {}
    pam = {}
    for kk in konf:
        if kk['spojna']:
            continue
        F = cechy[kk['cechy']]
        if kk['metoda'] in ('kurt', 'odst'):
            klucz = (kk['cechy'], kk['metoda'], kk['k'])
            if klucz not in pam:
                Y = wybiel(F, kk['k'])
                if kk['metoda'] == 'kurt':
                    pam[klucz] = odporne_z(kierunek_kurtozy(Y))
                else:
                    med = Y.median(1, keepdim=True).values
                    pam[klucz] = odporne_z((Y - med).norm(dim=2))
            z = pam[klucz]
            G = grupa_z_z(z, kk['prog']) if kk['metoda'] == 'kurt' else z > kk['prog']
        else:
            klucz = (kk['cechy'], 'cut', kk['tau'], kk['centruj'])
            if klucz not in pam:
                pam[klucz] = tokencut(F, kk['tau'], kk['centruj'])
            G = pam[klucz][0] if kk['strona'] == 'tokencut' else pam[klucz][1]
        out[kk['id']] = G.cpu().numpy()
    for kk in konf:
        if kk['spojna']:
            out[kk['id']] = spojna(out[json.dumps(dict({a: b for a, b in kk.items() if a != 'id'}, spojna=False), sort_keys=True)])
    out['glowica'] = (mapa >= 0.5 * mapa.max(1, keepdim=True).values).cpu().numpy()
    return out


def miary(G, u):
    """G (B, 600) bool, u (B, 600) udział → (B, 4): Dice z rdzeniem, czułość rdzenia, precyzja, precyzja z sąsiedztwem 1 łaty."""
    rdz = u >= 0.25; narz = u > 0
    sas = np.stack([binary_dilation(n.reshape(20, 30), np.ones((3, 3))).ravel() for n in narz])
    n = G.sum(1)
    inter = (G & rdz).sum(1)
    dice = 2 * inter / np.maximum(n + rdz.sum(1), 1)
    czul = inter / np.maximum(rdz.sum(1), 1)
    prec = np.where(n > 0, (G & narz).sum(1) / np.maximum(n, 1), 0.0)
    prec1 = np.where(n > 0, (G & sas).sum(1) / np.maximum(n, 1), 0.0)
    return np.stack([dice, czul, prec, prec1, n], 1)


def ocen(siec, ckpt, ramki, obrazy, konf, paczka=32):
    siec.wczytaj(ckpt)
    wyn = {}
    for a in range(0, len(ramki), paczka):
        cechy, mapa = siec(obrazy[a:a + paczka])
        u = np.stack([r['udzial'] for r in ramki[a:a + paczka]])
        for kid, G in grupy(cechy, mapa, konf).items():
            wyn.setdefault(kid, []).append(miary(G, u))
    return {k: np.concatenate(v) for k, v in wyn.items()}


if __name__ == '__main__':
    etap = sys.argv[1]                                                     # 'dev' albo 'test'
    t0 = time.time()
    R = klatki()
    print('klatek z obrysem:', len(R), 'DEV', sum(r['podzial'] == 'DEV' for r in R), 'TEST', sum(r['podzial'] == 'TEST' for r in R),
          'filmów DEV', len({r['film'] for r in R if r['podzial'] == 'DEV'}), 'TEST', len({r['film'] for r in R if r['podzial'] == 'TEST'}), flush=True)
    siec = Siec()
    if etap == 'dev':
        D = [r for r in R if r['podzial'] == 'DEV']
        ob = np.stack([obraz(r) for r in D])
        konf = konfiguracje()
        wyn = ocen(siec, CKPT['s3047_ep400'], D, ob, konf)
        ranking = sorted(((float(v[:, 0].mean()), k) for k, v in wyn.items()), reverse=True)
        np.savez(OUT / 'dev_miary.npz', **{f'k{i}': wyn[k] for i, (_, k) in enumerate(ranking)}, ids=np.array([k for _, k in ranking]),
                 frame_id=np.array([r['frame_id'] for r in D]))
        (OUT / 'dev_ranking.json').write_text(json.dumps([dict(dice=d, **({'konfiguracja': json.loads(k)} if k != 'glowica' else {'konfiguracja': 'glowica'}),
                                                               czulosc=float(wyn[k][:, 1].mean()), precyzja=float(wyn[k][:, 2].mean()),
                                                               precyzja_sas1=float(wyn[k][:, 3].mean()), n_med=float(np.median(wyn[k][:, 4])))
                                                          for d, k in ranking], indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
        wybrana = next(k for _, k in ranking if k != 'glowica')
        (OUT / 'WYBRANA_KONFIGURACJA.json').write_text(json.dumps(dict(konfiguracja=json.loads(wybrana), dice_dev=float(wyn[wybrana][:, 0].mean()), czas=time.strftime('%Y-%m-%d %H:%M:%S')),
                                                                   indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
        for d, k in ranking[:15]:
            print(round(d, 3), k, flush=True)
    else:
        T = [r for r in R if r['podzial'] == 'TEST']
        ob = np.stack([obraz(r) for r in T])
        wyb = json.loads((OUT / 'WYBRANA_KONFIGURACJA.json').read_text(encoding='utf-8'))['konfiguracja']
        wyb['id'] = json.dumps(wyb, sort_keys=True)
        konf = konfiguracje({wyb['id']} | ({json.dumps(dict({a: b for a, b in wyb.items() if a != 'id'}, spojna=False), sort_keys=True)}
                                                                  if wyb['spojna'] else set()))
        out = {}
        for nazwa, p in CKPT.items():
            w = ocen(siec, p, T, ob, konf)
            out[nazwa] = {k: v for k, v in w.items()}
            v = w[wyb['id']]
            print(nazwa, 'Dice %.3f czułość %.3f precyzja %.3f prec±1 %.3f | głowica Dice %.3f' % (*v[:, :4].mean(0), w['glowica'][:, 0].mean()), flush=True)
        np.savez(OUT / 'test_miary.npz', **{f'{c}__{"wybrana" if k == wyb["id"] else ("glowica" if k == "glowica" else "bez_spojnej")}': v
                                            for c, w in out.items() for k, v in w.items()},
                 frame_id=np.array([r['frame_id'] for r in T]), film=np.array([r['film'] for r in T]),
                 czestosc_narz=np.array([(r['udzial'] > 0).mean() for r in T]), n_rdzen=np.array([(r['udzial'] >= 0.25).sum() for r in T]))
    print('czas', round(time.time() - t0), 's')
