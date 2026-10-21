"""Odkrycie bez etykiet, walidacja po fakcie. Procedura ustalona PRZED obejrzeniem wyników (2026-10-21, ok. 20:50):

  1. Klatka → x̃ przed MLP bloku 6 (600 × 192), bez żadnej etykiety.
  2. PCA do 10 wymiarów i wybielenie; kierunek maksymalnej kurtozy (pogoń za rzutem, Peña i Prieto 2001: małe skupiska dają
     dużą kurtozę), iteracja stałopunktowa w ← E[y (wᵀy)³] − 3w z 20 startami (10 osi PCA + 10 losowych, ziarno 0).
  3. Zwrot osi: ogon z większą liczbą punktów poza progiem. Grupa = tokeny z odpornym z = (s − mediana) / (1,4826·MAD) > 3,5.
     Grupa < 2 tokenów → klatka bez odkrycia. Kandydat 2 i 3: kolejne kierunki maks. kurtozy w dopełnieniu poprzednich.
  4. Dopiero teraz etykieta z obrysu (udział pikseli narzędzia w łacie): precyzja = udział tokenów grupy z jakimkolwiek
     pikselem narzędzia, czułość = udział łat z udziałem ≥ 0,25 znalezionych, Dice z maską ≥ 0,25.
  Zera: ta sama procedura na wagach nieuczonych (s47 k00000) oraz losowa grupa tej samej liczności (precyzja = częstość
  łat z narzędziem). Model: seed 3047, epoka 400 (k437500); 300 klatek z obrysem spoza zbioru uczącego (jak część 3).
Wynik: Wyniki/Dane/odkrycie_bez_etykiet.jsonl.
"""
import sys, json, random, zipfile, hashlib
from pathlib import Path
import numpy as np, cv2, torch
APP = Path(__file__).resolve().parents[3] / 'app'
EXP = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'Wyniki' / 'Dane'
sys.path.insert(0, str(APP))
import lasery_dane as LD, etykiety_narzedzia as EN

CKPT = {'epoka400_s3047': EXP / 'vit_kontynuacja_do_400epok_20261020__23-50-57/Wyniki/Dane/checkpoints/k437500.pt',
        'nieuczony_s47_k0': LD.ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi/k00000.pt'}
KOMP, Z_PROG, KANDYDACI = 10, 3.5, 3
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kierunki_kurtozy(Y, ile=KANDYDACI, ziarno=0):
    """Kolejne kierunki maksymalnej kurtozy wybielonych danych Y (n, k), każdy w dopełnieniu poprzednich."""
    rng = np.random.default_rng(ziarno)
    out = []
    for _ in range(ile):
        Q = np.eye(Y.shape[1]) - sum(np.outer(v, v) for v in out) if out else np.eye(Y.shape[1])
        best = None
        starty = list(np.eye(Y.shape[1])) + list(rng.normal(size=(10, Y.shape[1])))
        for w in starty:
            w = Q @ w; w /= np.linalg.norm(w) + 1e-12
            for _ in range(100):
                s = Y @ w
                w2 = Q @ ((Y * (s ** 3)[:, None]).mean(0) - 3 * w)
                w2 /= np.linalg.norm(w2) + 1e-12
                if abs(abs(w2 @ w) - 1) < 1e-9:
                    w = w2; break
                w = w2
            k = float(((Y @ w) ** 4).mean() - 3)
            if best is None or k > best[0]:
                best = (k, w)
        out.append(best[1])
    return out


def grupa(s):
    med = np.median(s); mad = 1.4826 * np.median(np.abs(s - med)) + 1e-12
    z = (s - med) / mad
    if (z < -Z_PROG).sum() > (z > Z_PROG).sum():
        z = -z
    return z > Z_PROG


def odkryj(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = U[:, :KOMP] * np.sqrt(len(X))                    # wybielone współrzędne PCA
    return [grupa(Y @ w) for w in kierunki_kurtozy(Y)]


def ocena(g, u):
    rdzen = u >= 0.25
    if g.sum() < 2:
        return None
    return dict(n=int(g.sum()), precyzja=float((u[g] > 0).mean()), czulosc=float(g[rdzen].mean()) if rdzen.any() else None,
                dice=float(2 * (g & rdzen).sum() / (g.sum() + rdzen.sum())) if rdzen.any() else None)


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
    if len(wybor) == 300:
        break


def obraz(r):
    with zipfile.ZipFile(LD.ZIPY / f"{r['film']}.zip") as z:
        raw = z.read(f"{r['film']}/{r['film']}_{r['klatka']:06d}.jpg")
    assert hashlib.sha256(raw).hexdigest() == r['image_sha256']
    return cv2.resize(cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (480, 320),
                      interpolation=cv2.INTER_AREA)


obrazy = np.stack([obraz(r) for r in wybor])
M = LD.modul_modelu(); net = M.MalyViT().to(dev).eval().requires_grad_(False)
pam = {}
net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
plik = (OUT / 'odkrycie_bez_etykiet.jsonl').open('w', encoding='utf-8', newline='\n')
for nazwa, p in CKPT.items():
    st = torch.load(p, map_location='cpu', weights_only=False)
    net.load_state_dict(st.get('net', st), strict=True)
    for a in range(0, len(wybor), 32):
        with torch.no_grad():
            net(torch.from_numpy(obrazy[a:a + 32]).permute(0, 3, 1, 2).float().div(255).to(dev))
        Xb = pam['x'].double().cpu().numpy()
        for j in range(len(Xb)):
            n = a + j; u = etyk[n]['udzial']
            gr = odkryj(Xb[j])                                  # etykieta NIE jest tu używana
            plik.write(json.dumps(dict(model=nazwa, frame_id=wybor[n]['frame_id'], czestosc_narz=float((u > 0).mean()),
                                       n_rdzen=int((u >= 0.25).sum()), kandydaci=[ocena(g, u) for g in gr],
                                       tokeny=[np.flatnonzero(g).tolist() for g in gr]), ensure_ascii=False) + '\n')
    print(nazwa, 'gotowe', flush=True)
plik.close()
