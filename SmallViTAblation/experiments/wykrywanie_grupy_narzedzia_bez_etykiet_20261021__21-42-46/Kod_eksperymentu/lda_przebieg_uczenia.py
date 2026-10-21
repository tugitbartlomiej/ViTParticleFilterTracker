"""Czy LDA (etykieta z obrysu) rozdziela narzędzie od tła na POCZĄTKU uczenia? Ten sam pomiar co część 3 eksperymentu
separacja_narzedzia_replikacja_klatek_20261021__19-11-15 (wiele_klatek_obrys.py), ale na wczesnych checkpointach seedu 47
(krok 0 … 65640, 806 zapisów co 10 kroków na początku) i tych samych 300 klatkach spoza zbioru uczącego.

Klasy: narzędzie = udział ≥ próg, tło = udział 0, tokeny mieszane (0 < udział < próg) usunięte z chmury dla tej etykiety.
Widoki: 192D, PCA3 chmury (bez etykiet), LDA ze skurczem (λ = 0,1) dopasowana na tej klatce. Zero przestrzenne: ta sama maska
tokenów przesunięta w każde położenie siatki rozłączne z jakimkolwiek pikselem narzędzia; dla LDA oś dopasowana od nowa.
kNN: k = min(10, n_narz − 1). Wynik: Wyniki/Dane/wiele_klatek_obrys.jsonl.
"""
import sys, json, time, random, zipfile, hashlib
from pathlib import Path
import numpy as np, cv2, torch
APP = Path(__file__).resolve().parents[3] / 'app'
EXP = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'Wyniki' / 'Dane'
sys.path.insert(0, str(APP))
import lasery_dane as LD, rzuty_dane as RD, etykiety_narzedzia as EN

C200 = EXP / 'vit_kontynuacja_do_200epok_20261020__19-06-42/Wyniki/Dane/checkpoints'
C400 = EXP / 'vit_kontynuacja_do_400epok_20261020__23-50-57/Wyniki/Dane/checkpoints'
S47 = LD.ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
KROKI = [0, 30, 100, 300, 1000, 3000, 10100, 28444, 65640]
_dost = sorted(int(p.stem[1:]) for p in S47.glob('k*.pt'))
CKPT = {k: S47 / f'k{min(_dost, key=lambda d: abs(d - k)):05d}.pt' for k in KROKI}
PROGI = (0.25,)
MIARY = ('sil', 'hit', 'marg', 'fisher')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
dev = RD._dev()

uczace = {(r['film'], int(r['klatka'])) for r in LD.zbior_uczacy()}
wiersze = [json.loads(l) for l in (LD.ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_1500.jsonl').read_text(encoding='utf-8').splitlines()]
for r in wiersze:
    r['film'] = r['frame_id'].split('/')[-2]; r['klatka'] = int(r['frame_id'].split('/')[-1])
poza = [r for r in wiersze if (r['film'], r['klatka']) not in uczace]
random.Random(0).shuffle(poza)
wybor, etyk = [], []
for r in poza:
    e = EN.etykieta(r['film'], r['klatka'])
    if e is not None:
        wybor.append(r); etyk.append(e)
    if len(wybor) == N:
        break
print('klatek z obrysem:', len(wybor), flush=True)


def obraz(r):
    with zipfile.ZipFile(LD.ZIPY / f"{r['film']}.zip") as z:
        raw = z.read(f"{r['film']}/{r['film']}_{r['klatka']:06d}.jpg")
    assert hashlib.sha256(raw).hexdigest() == r['image_sha256']
    im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)


obrazy = np.stack([obraz(r) for r in wybor])


def pca3(X):
    Xc = X - X.mean(0, keepdim=True)
    return Xc @ torch.linalg.eigh(Xc.T @ Xc)[1][:, -3:]


def miary(P, M, k):
    return {a: v.double().cpu().numpy() for a, v in RD.miary_separacji_t(P, M, k).items()}


M = LD.modul_modelu()
net = M.MalyViT().to(dev).eval().requires_grad_(False)
pam = {}
net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
plik = (OUT / 'lda_przebieg_uczenia_s47.jsonl').open('w', encoding='utf-8', newline='\n')
t0 = time.time()
for ep, p in CKPT.items():
    st = torch.load(p, map_location='cpu', weights_only=False)
    net.load_state_dict(st.get('net', st), strict=True)
    for a in range(0, len(wybor), 32):
        x = torch.from_numpy(obrazy[a:a + 32]).permute(0, 3, 1, 2).float().div(255).to(dev)
        with torch.no_grad():
            o, _ = net(x)
            bx, sc = M.dekoduj(o.float().cpu(), k=1)
            Xb = pam['x'].float()
        for j in range(len(x)):
            n = a + j; e = etyk[n]; u = e['udzial']
            gt = list(map(float, wybor[n]['bbox_xyxy_480']))
            rec = dict(frame_id=wybor[n]['frame_id'], epoka=ep, krok=int(CKPT[ep].stem[1:]), kategorie=e['kategorie'],
                       pew=float(sc[j, 0]), iou=float(LD.iou(bx[j, 0].tolist(), gt)), n_ramka=int(LD.maska_narzedzia(gt).sum()),
                       n_obrys_dowolny=int((u > 0).sum()))
            for prog in PROGI:
                narz = u >= prog; mies = (u > 0) & (u < prog); keep = ~mies
                nt = int(narz.sum()); r = dict(n_narz=nt, n_mieszane=int(mies.sum()))
                if nt >= 2:
                    k = int(min(10, nt - 1))
                    X = Xb[j][torch.as_tensor(keep, device=dev)]
                    m1 = torch.as_tensor(narz[keep], device=dev)[None]
                    P3 = pca3(X)
                    for nazwa, P in (('192D', X[None]), ('PCA3', P3[None]), ('LDA', X[None] @ RD.baza_L_t(X, m1))):
                        r[nazwa] = {q: float(v[0]) for q, v in miary(P, m1, k).items()}
                    R = EN.przesuniecia(narz, u > 0)
                    r['n_polozen'] = len(R)
                    if len(R) >= 19:
                        Mr = torch.as_tensor(R[:, keep], device=dev)
                        z = {w: {q: [] for q in MIARY} for w in ('192D', 'PCA3', 'LDA')}
                        for b in range(0, len(Mr), 64):
                            mb = Mr[b:b + 64]
                            for w, P in (('192D', X[None].expand(len(mb), -1, -1)), ('PCA3', P3[None].expand(len(mb), -1, -1)),
                                         ('LDA', X[None] @ RD.baza_L_t(X, mb))):
                                oo = miary(P, mb, k)
                                for q in MIARY: z[w][q].append(oo[q])
                        r['p_przestrz'] = {w: {q: float((1 + (np.concatenate(z[w][q]) >= r[w][q]).sum()) / (1 + len(R))) for q in MIARY} for w in z}
                        r['mediana_przestrz'] = {w: {q: float(np.median(np.concatenate(z[w][q]))) for q in MIARY} for w in z}
                rec[f'prog_{prog}'] = r
            plik.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('krok', ep, round(time.time() - t0), 's', flush=True)
plik.close()
