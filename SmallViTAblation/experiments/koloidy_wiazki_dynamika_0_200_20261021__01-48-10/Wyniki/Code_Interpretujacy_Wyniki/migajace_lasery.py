"""Who are the flickering (sometimes dark) lasers on the plateau (epoch >= 40), and does their flicker follow IoU?"""
from pathlib import Path
import csv,sys
import numpy as np

EXP=Path(__file__).resolve().parents[2];DANE=EXP/'Wyniki/Dane'
DET=EXP.parent/'pary_przeciwne_do_200epok_20261020__23-21-30/Wyniki/Dane/detekcja_0_200/DETEKCJA_0_200.csv'
sys.stdout.reconfigure(encoding='utf-8');rng=np.random.default_rng(3047)
L=np.load(DANE/'LASERY_26_CHECKPOINTOW.npz');ep=L['steps']*32/35000;late=ep>=40
act=L['active_frac'][late];sel=L['selectivity'][late];use=L['s_tool'][late]*0+L['mean_abs_z'][late]*L['power'][late]
dark=act==0
flick=dark.any(0)&~dark.all(0);always=~dark.any(0);never=dark.all(0)
sel_alive=np.array([np.nanmedian(np.where(dark[:,c],np.nan,sel[:,c])) if not never[c] else np.nan for c in range(768)])
print(f'plateau (16 checkpointów): zawsze świecą {always.sum()}, migają {flick.sum()}, ciemne cały czas {never.sum()}')
for name,m in [('migające',flick),('zawsze świecące',always)]:
    s=sel_alive[m];print(f'{name:16s}: mediana selektywności {np.nanmedian(s):+.2f} | lasery narzędzia (sel>0,5) {np.mean(s>.5)*100:4.1f}% | lasery tła (sel<-0,5) {np.mean(s<-.5)*100:4.1f}% | '
                         f'mediana aktywności gdy świecą {np.nanmedian(np.where(dark[:,m],np.nan,act[:,m])):.4f}')
det=list(csv.DictReader(open(DET,encoding='utf-8')))
iou={s:np.array([float(next(d for d in det if abs(float(d['epoch_equivalent'])-e)<1e-6 and d['split']==s)['mean_top1_iou']) for e in ep[late]]) for s in ('VAL','TEST')}
tool_flick=flick&(sel_alive>.5);other_flick=flick&~(sel_alive>.5)
def sp(a,b):
    ra,rb=np.argsort(np.argsort(a)),np.argsort(np.argsort(b));r=np.corrcoef(ra,rb)[0,1]
    null=[np.corrcoef(ra,rng.permutation(rb))[0,1] for _ in range(10000)];return r,np.mean(np.abs(null)>=abs(r))
for name,m in [('ciemne lasery narzędzia (migające)',tool_flick),('ciemne pozostałe migające',other_flick)]:
    n=dark[:,m].sum(1)
    for s in ('VAL','TEST'):
        r,p=sp(np.diff(iou[s]),np.diff(n));print(f'{name:36s} ({m.sum():3d} laserów) vs zmiany IoU {s}: r={r:+.2f} (p={p:.3f})')
