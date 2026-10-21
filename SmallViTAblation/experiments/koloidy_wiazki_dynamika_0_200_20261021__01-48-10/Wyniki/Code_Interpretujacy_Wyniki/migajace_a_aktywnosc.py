"""Is the dark-laser count on the plateau just a global activity level? Spearman of first differences with permutation p."""
from pathlib import Path
import csv,sys
import numpy as np

D=Path(__file__).resolve().parents[1]/'Dane';sys.stdout.reconfigure(encoding='utf-8');rng=np.random.default_rng(3047)
met=list(csv.DictReader(open(D/'METRYKI_CHECKPOINTY.csv',encoding='utf-8')));las=list(csv.DictReader(open(D/'LASERY_PODSUMOWANIE.csv',encoding='utf-8')))
ep=np.array([float(r['epoch']) for r in met]);late=ep>=40
x={'martwe lasery':[float(r['lasers_never_active']) for r in las],'aktywne lasery na cząstkę':[float(r['active_beams']) for r in met],
   'udział siły ciągnącej':[float(r['pull_power_share']) for r in met],'lasery narzędzia':[float(r['tool_selective']) for r in las],
   'kopnięcie / R':[float(r['kick_rel']) for r in met]}
x={k:np.diff(np.array(v)[late]) for k,v in x.items()}
def sp(a,b):
    ra,rb=np.argsort(np.argsort(a)),np.argsort(np.argsort(b));r=np.corrcoef(ra,rb)[0,1]
    return r,np.mean(np.abs([np.corrcoef(ra,rng.permutation(rb))[0,1] for _ in range(10000)])>=abs(r))
for k in list(x)[1:]:
    r,p=sp(x['martwe lasery'],x[k]);print(f'zmiany: martwe lasery vs {k:28s} r={r:+.2f} (p={p:.3f})')
