"""Separacja narzędzia od tła na wielu klatkach spoza zbioru uczącego (MANIFEST_1500 bez 200 klatek uczących), w kilku
checkpointach seedu 47; zestawienie z IoU i pewnością detektora na tej samej klatce.

Dla klatki: x̃ przed MLP bloku 6 (600 × 192), maska narzędzia z ramki GT (środki łat w ramce, jak LD.maska_narzedzia),
miary w 192D i PCA3 chmury (bez etykiet), zero przestrzenne PCA3 = wszystkie położenia prostokąta o kształcie ramki GT
rozłączne z nią (p = (1 + #≥) / (1 + #)). Wynik: Wyniki/Dane/wiele_klatek.jsonl (wiersz = klatka × checkpoint).
"""
import sys, json, time, zipfile, hashlib, random
from pathlib import Path
import numpy as np, cv2, torch
APP = Path(__file__).resolve().parents[3] / 'app'
OUT = Path(__file__).resolve().parents[1] / 'Wyniki' / 'Dane'
sys.path.insert(0, str(APP)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lasery_dane as LD, rzuty_dane as RD
from bateria import ramki, pca3, MIARY

N_KLATEK = int(sys.argv[1]) if len(sys.argv) > 1 else 300
KROKI = [1000, 10100, 28444, 65640]                    # indeksy 100, 200, 400, 805 historii (806 checkpointów)
MAN = LD.ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_1500.jsonl'
WAGI = LD.ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
dev = RD._dev()

uczace = {(r['film'], int(r['klatka'])) for r in LD.zbior_uczacy()}
wiersze = [json.loads(l) for l in MAN.read_text(encoding='utf-8').splitlines()]
for r in wiersze:
    r['film'] = r['frame_id'].split('/')[-2]; r['klatka'] = int(r['frame_id'].split('/')[-1])
poza = [r for r in wiersze if (r['film'], r['klatka']) not in uczace]
random.Random(0).shuffle(poza)
wybor = poza[:N_KLATEK]
print(len(wiersze), 'w manifeście,', len(poza), 'poza uczącym, wybrano', len(wybor), flush=True)


def obraz(r):
    with zipfile.ZipFile(LD.ZIPY / f"{r['film']}.zip") as z:
        raw = z.read(f"{r['film']}/{r['film']}_{r['klatka']:06d}.jpg")
    assert hashlib.sha256(raw).hexdigest() == r['image_sha256'], r['frame_id']
    im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (LD.W * LD.KOMORKA, LD.H * LD.KOMORKA), interpolation=cv2.INTER_AREA)


obrazy = np.stack([obraz(r) for r in wybor])
gt = [list(map(float, r['bbox_xyxy_480'])) for r in wybor]
maski = [LD.maska_narzedzia(g) for g in gt]
print('obrazy wczytane', obrazy.shape, flush=True)

M = LD.modul_modelu()
net = M.MalyViT().to(dev).eval().requires_grad_(False)
pam = {}
net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
OUT.mkdir(parents=True, exist_ok=True)
plik = (OUT / 'wiele_klatek.jsonl').open('w', encoding='utf-8', newline='\n')
t0 = time.time()
for krok in KROKI:
    st = torch.load(WAGI / f'k{krok:05d}.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    for a in range(0, len(wybor), 32):
        x = torch.from_numpy(obrazy[a:a + 32]).permute(0, 3, 1, 2).float().div(255).to(dev)
        with torch.no_grad():
            o, _ = net(x)
            bx, sc = M.dekoduj(o.float().cpu(), k=1)
            Xb = pam['x'].float()
        for j in range(len(x)):
            n = a + j; m = maski[n]; nt = int(m.sum())
            rec = dict(frame_id=wybor[n]['frame_id'], krok=krok, n_narz=nt, gt=gt[n], ramka=bx[j, 0].tolist(),
                       pew=float(sc[j, 0]), iou=float(LD.iou(bx[j, 0].tolist(), gt[n])))
            if 2 <= nt <= 300:
                X = Xb[j]
                M1 = torch.as_tensor(m, device=dev)[None]
                P3 = pca3(X)
                for nazwa, P in (('192D', X), ('PCA3', P3)):
                    oo = RD.miary_separacji_t(P[None], M1)
                    rec[nazwa] = {k: float(v[0]) for k, v in oo.items()}
                R, ks = ramki(m)
                rec['ksztalt'] = ks; rec['n_polozen'] = len(R)
                if len(R) >= 19:
                    Mr = torch.as_tensor(R, device=dev); z = {k: [] for k in MIARY}
                    for b in range(0, len(Mr), 64):
                        oo = RD.miary_separacji_t(P3[None].expand(len(Mr[b:b + 64]), -1, -1), Mr[b:b + 64])
                        for k in MIARY: z[k].append(oo[k].double().cpu().numpy())
                    z = {k: np.concatenate(v) for k, v in z.items()}
                    rec['p_przestrz_PCA3'] = {k: float((1 + (z[k] >= rec['PCA3'][k]).sum()) / (1 + len(z[k]))) for k in MIARY}
                    rec['mediana_przestrz_PCA3'] = {k: float(np.median(z[k])) for k in MIARY}
            plik.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('krok', krok, round(time.time() - t0), 's', flush=True)
plik.close()
