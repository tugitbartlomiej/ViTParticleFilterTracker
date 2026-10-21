"""Correlations between detection (VAL/TEST IoU, confidence) and colloid/laser observables over 26 checkpoints.
Three views: whole training (trend-dominated), plateau levels (epoch >= 40) and plateau changes between consecutive
checkpoints (first differences). Spearman with permutation p-values (10 000)."""
from pathlib import Path
import csv,sys
import numpy as np

EXP=Path(__file__).resolve().parents[2];DANE=EXP/'Wyniki/Dane'
DET=EXP.parent/'pary_przeciwne_do_200epok_20261020__23-21-30/Wyniki/Dane/detekcja_0_200/DETEKCJA_0_200.csv'
rng=np.random.default_rng(3047)
def rows(p):return list(csv.DictReader(open(p,encoding='utf-8')))
def f(x):return float(x) if x not in ('',None,'None','nan') else np.nan
def rank(x):return np.argsort(np.argsort(x)).astype(float)
def spearman(a,b,perm=10000):
    ok=np.isfinite(a)&np.isfinite(b);a,b=rank(a[ok]),rank(b[ok]);n=len(a)
    if n<5:return np.nan,np.nan,n
    r=np.corrcoef(a,b)[0,1];null=np.array([np.corrcoef(a,rng.permutation(b))[0,1] for _ in range(perm)])
    return r,float((np.abs(null)>=abs(r)).mean()),n

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    met=rows(DANE/'METRYKI_CHECKPOINTY.csv');mv=rows(DANE/'RUCH_MIEDZY_CHECKPOINTAMI.csv');kick=rows(DANE/'KOPNIECIE_A_ROZDZIAL.csv')
    las=rows(DANE/'LASERY_PODSUMOWANIE.csv');det=rows(DET)
    ep=np.array([f(r['epoch']) for r in met])
    get=lambda split,key:np.array([f(next(d for d in det if abs(f(d['epoch_equivalent'])-e)<1e-6 and d['split']==split)[key]) for e in ep])
    T=np.array([np.nan]+[f(r['speed_bg_aligned_per_epoch'])**2 for r in mv]);Tt=np.array([np.nan]+[f(r['speed_tool_aligned_per_epoch'])**2 for r in mv])
    knn=np.array([np.nan]+[f(r['knn_kept']) for r in mv])
    series={
        'temperatura tła (prędkość²)':T,'temperatura narzędzia':Tt,'trwałość sąsiedztwa':knn,
        'obrót laserów [°/ep]':np.array([f(r['beam_rotation_deg_per_epoch']) for r in met]),
        'moc laserów':np.array([f(r['beam_power_mean']) for r in met]),'wymiar wiązki laserów':np.array([f(r['beam_bundle_dims']) for r in met]),
        'rozdział narzędzie-tło':np.array([f(r['sep']) for r in met]),'narzędzie poza chmurą':np.array([f(r['tool_outside']) for r in met]),
        'wymiar chmury':np.array([f(r['cloud_dims']) for r in met]),'kopnięcie / R':np.array([f(r['kick_rel']) for r in met]),
        'dywergencja (narzędzie)':np.array([f(r['div_tool']) for r in met]),
        'hamowanie rozdziału [%]':np.array([(f(r['sep_after'])/f(r['sep_before'])-1)*100 for r in kick]),
        'wspólny wiatr / R':np.array([f(r['common_push']) for r in kick]),
        'martwe lasery':np.array([f(r['lasers_never_active']) for r in las]),'elita zachowana':np.array([f(r['elite_kept']) for r in las]),
        'lasery narzędzia':np.array([f(r['tool_selective']) for r in las]),'lasery tła':np.array([f(r['bg_selective']) for r in las]),
        'hamowanie (suma laserów)':np.array([f(r['brake_total']) for r in las])}
    targets={'IoU VAL':get('VAL','mean_top1_iou'),'IoU TEST':get('TEST','mean_top1_iou'),
             'pewność VAL (z narz.)':get('VAL','mean_score_positive'),'pewność VAL (puste)':get('VAL','empty_mean_top1_score')}
    late=ep>=40
    dIv,dIt=np.diff(targets['IoU VAL'][late]),np.diff(targets['IoU TEST'][late])
    r,p,n=spearman(dIv,dIt);print(f'Sufit szumu: zmiany IoU VAL vs TEST na plateau: r={r:+.2f} (p={p:.3f}, n={n})')
    out=[]
    for tname,t in targets.items():
        print(f'\n=== {tname}')
        print('miara                          | cały trening r (p)  | plateau poziomy r (p) | plateau zmiany r (p)')
        for name,s in series.items():
            a=spearman(t,s);b=spearman(t[late],s[late]);c=spearman(np.diff(t[late]),np.diff(s[late]))
            out.append(dict(target=tname,metric=name,r_all=a[0],p_all=a[1],r_plateau=b[0],p_plateau=b[1],r_changes=c[0],p_changes=c[1],n_changes=c[2]))
            flag=' *' if c[1]<.05 else ''
            print(f'{name:30s} | {a[0]:+.2f} ({a[1]:.3f})      | {b[0]:+.2f} ({b[1]:.3f})        | {c[0]:+.2f} ({c[1]:.3f}){flag}')
    with open(DANE/'KORELACJE_IOU_TEMPERATURA.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('\nUwaga: na plateau 16 checkpointów (15 zmian); testów jest wiele, więc pojedyncze p<0,05 mogą być przypadkowe.')

if __name__=='__main__':main()
