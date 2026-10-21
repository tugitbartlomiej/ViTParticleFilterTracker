"""Podsumowanie Wyniki/Dane/wiele_klatek.jsonl: odsetek klatek z narzędziem wyróżnionym wobec zera przestrzennego, związek z IoU."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parents[1] / 'Dane'
W = [json.loads(l) for l in (D / 'wiele_klatek.jsonl').read_text(encoding='utf-8').splitlines()]
rng = np.random.default_rng(0)
def boot(x, f=np.mean, n=2000):
    x = np.asarray(x); b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return f(x), np.percentile(b, 2.5), np.percentile(b, 97.5)
out = {}
for krok in sorted({w['krok'] for w in W}):
    r = [w for w in W if w['krok'] == krok and 'p_przestrz_PCA3' in w]
    iou = np.array([w['iou'] for w in r]); pew = np.array([w['pew'] for w in r])
    ph = np.array([w['p_przestrz_PCA3']['hit'] for w in r]); ps = np.array([w['p_przestrz_PCA3']['sil'] for w in r])
    hit = np.array([w['PCA3']['hit'] for w in r]); hit192 = np.array([w['192D']['hit'] for w in r])
    medh = np.array([w['mediana_przestrz_PCA3']['hit'] for w in r])
    films = np.array([w['frame_id'].split('/')[-2] for w in r])
    dobre = iou >= 0.5
    s = dict(n=len(r), filmy=len(set(films)), iou_srednie=float(iou.mean()), iou_ge05=float(dobre.mean()),
             frac_p05_hit=boot(ph <= 0.05), frac_p05_sil=boot(ps <= 0.05),
             frac_p05_hit_iou_ge05=float((ph[dobre] <= 0.05).mean()) if dobre.any() else None,
             frac_p05_hit_iou_lt05=float((ph[~dobre] <= 0.05).mean()) if (~dobre).any() else None,
             hit_PCA3_med=float(np.median(hit)), hit_przestrz_med=float(np.median(medh)), hit_192D_med=float(np.median(hit192)),
             spearman_iou_hit=spearmanr(iou, hit)[:2], spearman_iou_logp=spearmanr(iou, -np.log10(ph))[:2],
             spearman_pew_hit=spearmanr(pew, hit)[:2])
    # korelacja z poprawką na film: rangi w obrębie filmów
    rr = []
    for f in set(films):
        k = films == f
        if k.sum() >= 3:
            rr.append(spearmanr(iou[k], hit[k])[0])
    s['spearman_w_filmach_mediana'] = float(np.nanmedian(rr)) if rr else None
    s['n_filmow_z_3+'] = len(rr)
    out[krok] = s
    print(krok, json.dumps(s, ensure_ascii=False, default=float))
n_all = {k: sum(1 for w in W if w['krok'] == k) for k in out}
print('wszystkich klatek na krok', n_all, '; bez zera przestrzennego (ramka za mała/duża lub <19 położeń):',
      {k: n_all[k] - out[k]['n'] for k in out})
(D / 'wiele_klatek_podsumowanie.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding='utf-8', newline='\n')
