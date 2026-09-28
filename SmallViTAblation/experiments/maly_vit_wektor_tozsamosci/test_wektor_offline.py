# Offline: prognoza na wszystkich 768 neuronach (wersja 2) na latentach kampanii 1 (ramię bez), prawdziwa przyszłość.
import sys, json, numpy as np
from pathlib import Path
E = Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments')
sys.path.insert(0, str(E / 'maly_vit_od_zera'))
import analiza_wsteczna as A
import importlib.util
spec = importlib.util.spec_from_file_location('prog2', E / 'maly_vit_wektor_tozsamosci/kod/prognoza.py'); P2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P2)
import tozsamosc as TZ
man = A.jz(A.PROBE / 'manifest.json'); obecn = np.array([bool(m['semantic_presence']) for m in man]); opis = [dict(wariant=m['variant']) for m in man]
H = 10; POCZ = list(range(1000, 4001, 250)); wyn = {}
for d in A.katalogi():
    wl = [np.load(d / 'wspolny' / f'k{k:02d}' / 'LATENT.npz') for k in (1, 2, 3)]
    wsp = dict(kroki=np.concatenate([w['kroki'] for w in wl]), latent=np.concatenate([w['latent'] for w in wl]))
    Lb = A.bloki_ramienia(d, 'bez', wsp)
    rr = []
    for t in POCZ:
        b0 = t // 50 - 1
        st = TZ.statystyki(Lb[b0], obecn, opis)
        Z = (Lb[:, obecn, 0] - st['mu']) / st['sd']
        hist = Z[:b0 + 1]
        rr.append(dict(krok=t, **P2.porownaj(P2.prognoza_bloku(hist, H), hist[-1], Z[b0 + H])))
    wyn[d.name] = rr
    print(d.name, [(r['krok'], round(r['langevin']['wektor'], 2), round(r['koopman']['wektor'], 2)) for r in rr[::3]], flush=True)
print('krok  ep | wektor: K med(<1) L med(<1) | sondy: K L | znak K L')
for i, t in enumerate(POCZ):
    g = lambda m, k: np.array([w[i][m][k] for w in wyn.values()])
    print(t, round(t / 46.875), '|', f"{np.median(g('koopman','wektor')):.2f}({(g('koopman','wektor')<1).sum()}) {np.median(g('langevin','wektor')):.2f}({(g('langevin','wektor')<1).sum()})",
          '|', f"{np.median(g('koopman','sondy')):.2f} {np.median(g('langevin','sondy')):.2f}", '|', f"{np.median(g('koopman','znak')):.2f} {np.median(g('langevin','znak')):.2f}")
Path(sys.argv[1]).write_text(json.dumps(wyn, default=float), encoding='utf-8')
