"""Ile ruchu tokenów widzi baza PCA32 z kroku 0? Z gęstego zapisu sond: energia kroku w pełnych 768D (normy dp z obserwatora)
kontra energia kroku w rzucie PCA32 (różnice kolejnych rekordów), dla tych samych sond i rekordów. Udział = E_pca32 / E_768.
Wynik: UDZIAL_PCA32.json (krok, udział, oba E) + wydruk średnich w fazach. Uruchomienie: python udzial_pca32.py [NS]"""
import glob, json, os, sys, time
import numpy as np
G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
t0 = time.time()
kr, e768, e32, prev = [], [], [], None
for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
    z = np.load(p)
    k = z['kroki']; P = z['pca32'][:, :NS].astype(np.float32).reshape(len(k), -1, 32); nrm = z['normy'][:, :NS, :, 1].astype(np.float32).reshape(len(k), -1)
    for i in range(len(k)):
        if prev is not None:
            kr.append(int(k[i])); e768.append(float((nrm[i] ** 2).mean())); e32.append(float(((P[i] - prev) ** 2).sum(1).mean()))
        prev = P[i]
    print(os.path.basename(p), f'[{time.time() - t0:.0f} s]', flush=True)
kr, e768, e32 = np.array(kr), np.array(e768), np.array(e32)
# dp w normach to |z(t) − z(poprzedni rekord)| w 768D, więc porównywalne z różnicą kolejnych rekordów PCA32
u = e32 / np.maximum(e768, 1e-20)
json.dump(dict(opis=__doc__, NS=NS, krok=kr.tolist(), E_768=e768.tolist(), E_pca32=e32.tolist(), udzial=u.tolist()),
          open('UDZIAL_PCA32.json', 'w', encoding='utf-8'), ensure_ascii=False)
for lo, hi in ((1, 100), (100, 300), (300, 1000), (1000, 2188), (2188, 5000), (5000, 10000), (10000, 15000), (15000, 22970)):
    m = (kr > lo) & (kr <= hi)
    if m.any():
        print(f'kroki {lo:>5}–{hi:<5}: udział PCA32 w energii kroku = {np.median(u[m]):.3f} (śr {u[m].mean():.3f}); E768 {np.median(e768[m]):.4g}, E_pca32 {np.median(e32[m]):.4g}')
