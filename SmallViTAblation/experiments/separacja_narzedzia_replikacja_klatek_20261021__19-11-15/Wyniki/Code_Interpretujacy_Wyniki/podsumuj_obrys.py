"""Podsumowanie Wyniki/Dane/wiele_klatek_obrys.jsonl (etykieta z obrysu, seed 3047, epoki 60–400): odsetek klatek, w których
narzędzie wyróżnia się wobec zera przestrzennego (p ≤ 0,05), z przedziałami bootstrap po filmach; mediany miar; związek z IoU."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parents[1] / 'Dane'
W = [json.loads(l) for l in (D / 'wiele_klatek_obrys.jsonl').read_text(encoding='utf-8').splitlines()]
rng = np.random.default_rng(0)


def boot_film(y, filmy, n=2000):
    u = np.unique(filmy); idx = {f: np.flatnonzero(filmy == f) for f in u}
    b = [y[np.concatenate([idx[f] for f in rng.choice(u, len(u))])].mean() for _ in range(n)]
    return [round(float(y.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


out = {}
for prog in ('0.5', '0.25'):
    for ep in sorted({w['epoka'] for w in W}):
        r = [w for w in W if w['epoka'] == ep and 'p_przestrz' in w[f'prog_{prog}']]
        if not r:
            continue
        g = [w[f'prog_{prog}'] for w in r]
        filmy = np.array([w['frame_id'].split('/')[-2] for w in r])
        iou = np.array([w['iou'] for w in r])
        s = dict(n=len(r), filmy=len(set(filmy)), iou=round(float(iou.mean()), 3),
                 n_narz_med=float(np.median([x['n_narz'] for x in g])), n_mies_med=float(np.median([x['n_mieszane'] for x in g])))
        for widok in ('192D', 'PCA3', 'LDA'):
            for q in ('fisher', 'sil', 'hit'):
                p = np.array([x['p_przestrz'][widok][q] for x in g])
                s[f'{widok}_{q}_p05'] = boot_film((p <= 0.05).astype(float), filmy)
                s[f'{widok}_{q}_med'] = round(float(np.median([x[widok][q] for x in g])), 3)
                s[f'{widok}_{q}_med_tla'] = round(float(np.median([x['mediana_przestrz'][widok][q] for x in g])), 3)
        fis = np.array([x['LDA']['fisher'] for x in g])
        s['spearman_iou_fisherLDA'] = [round(float(v), 3) for v in spearmanr(iou, fis)]
        out[f'prog {prog}, epoka {ep}'] = s
        print(f'próg {prog} ep {ep}: n={s["n"]} IoU {s["iou"]} narz med {s["n_narz_med"]} | p≤0,05: '
              f'LDA Fisher {s["LDA_fisher_p05"]} sil {s["LDA_sil_p05"]} | PCA3 sil {s["PCA3_sil_p05"]} hit {s["PCA3_hit_p05"]} | '
              f'192D Fisher {s["192D_fisher_p05"]} sil {s["192D_sil_p05"]} | Fisher LDA med {s["LDA_fisher_med"]} vs tło {s["LDA_fisher_med_tla"]}')
(D / 'wiele_klatek_obrys_podsumowanie.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
