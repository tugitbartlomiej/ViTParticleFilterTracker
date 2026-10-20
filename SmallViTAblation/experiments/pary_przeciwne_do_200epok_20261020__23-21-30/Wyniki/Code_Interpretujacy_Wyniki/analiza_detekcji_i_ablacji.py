"""Film-level bootstrap for detection over epochs and for the pair ablation; prints the numbers used in the report."""
from pathlib import Path
import csv,json,sys
import numpy as np

EXP=Path(__file__).resolve().parents[2];DANE=EXP/'Wyniki/Dane'
DET=DANE/'detekcja_0_200';ZN=DANE/'znoszenie_370_391';REPS=2000;SEED=3047
def rows(p):return list(csv.DictReader(open(p,encoding='utf-8')))
def f(x):return float(x) if x not in ('',None) else np.nan
def csvwrite(path,rr):
    with open(path,'w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def boot(values,films,stat,rng):
    # Cluster bootstrap over films: resample whole films, keep their images together.
    uf=np.unique(films);idx={u:np.flatnonzero(films==u) for u in uf};out=np.empty(REPS)
    for r in range(REPS):
        take=np.concatenate([idx[u] for u in rng.choice(uf,len(uf))]);out[r]=stat(values[take])
    return np.percentile(out,[2.5,97.5])

def main():
    sys.stdout.reconfigure(encoding='utf-8');rng=np.random.default_rng(SEED)
    img=rows(DET/'DETEKCJA_OBRAZY_0_200.csv');det=rows(DET/'DETEKCJA_0_200.csv')
    out=[]
    for r in det:
        if r['split']=='DIAG50':continue
        sel=[x for x in img if x['step']==r['step'] and x['split']==r['split']]
        films=np.array([x['film'] for x in sel]);iou=np.array([f(x['iou']) for x in sel]);sc=np.array([f(x['score']) for x in sel])
        pos=np.isfinite(iou)
        for metric,vals,mask in [('mean_top1_iou',iou,pos),('mean_score_positive',sc,pos),('empty_mean_top1_score',sc,~pos),('fraction_iou_ge_05',(iou>=.5).astype(float),pos)]:
            lo,hi=boot(vals[mask],films[mask],np.mean,rng)
            out.append(dict(step=r['step'],epoch_equivalent=r['epoch_equivalent'],split=r['split'],metric=metric,value=float(vals[mask].mean()),lo=lo,hi=hi,films=len(np.unique(films[mask]))))
    csvwrite(DET/'DETEKCJA_BOOTSTRAP.csv',out)
    print('epoch | VAL IoU [95%] | VAL score(pos) | VAL empty score | TEST IoU [95%] | TEST score(pos) | TEST empty score | VAL IoU>=0.5')
    for step in sorted({int(r['step']) for r in out}):
        g=lambda s,m:next(r for r in out if int(r['step'])==step and r['split']==s and r['metric']==m)
        v,t=g('VAL','mean_top1_iou'),g('TEST','mean_top1_iou')
        print(f"{step*32/35000:6.1f} | {v['value']:.4f} [{v['lo']:.3f},{v['hi']:.3f}] | {g('VAL','mean_score_positive')['value']:.3f} | {g('VAL','empty_mean_top1_score')['value']:.3f} | "
              f"{t['value']:.4f} [{t['lo']:.3f},{t['hi']:.3f}] | {g('TEST','mean_score_positive')['value']:.3f} | {g('TEST','empty_mean_top1_score')['value']:.3f} | {g('VAL','fraction_iou_ge_05')['value']:.3f}")
    # Paired ablation effect of the pair at epoch 200, film bootstrap.
    npz=np.load(ZN/'ABLACJA_OBRAZY_0_200.npz');ab=[]
    for s in ('VAL','TEST'):
        sel=[x for x in img if int(x['step'])==218750 and x['split']==s];films=np.array([x['film'] for x in sel])
        base=npz[f'k218750_{s}_clean'];pos=np.isfinite(base[0])
        for name in ('a','b','ab'):
            cur=npz[f'k218750_{s}_{name}'];d_iou=(cur[0]-base[0])[pos];d_sc=cur[1]-base[1]
            li,hi=boot(d_iou,films[pos],np.mean,rng);ls,hs=boot(d_sc,films,np.mean,rng)
            ab.append(dict(split=s,ablated=name,mean_dIoU=float(d_iou.mean()),dIoU_lo=li,dIoU_hi=hi,mean_dscore=float(d_sc.mean()),dscore_lo=ls,dscore_hi=hs,
                images_changed_iou=int((np.abs(d_iou)>1e-6).sum()),winner_changed=int((cur[2]!=base[2]).sum()),max_abs_dIoU=float(np.abs(d_iou).max())))
    csvwrite(ZN/'ABLACJA_BOOTSTRAP_k218750.csv',ab)
    print('\nablation at epoch 200');[print(r) for r in ab]
    k=rows(ZN/'KONTROLA_k218750.csv');t=next(r for r in k if r['group']=='target')
    print('\ncontrols: group | n | median ratio | target ratio pct | median |dlogit_ab| | target pct | median VAL dIoU | TEST')
    for g in ('uniform','strength_matched','near_parallel_w2'):
        rr=[r for r in k if r['group']==g]
        for key in ('interaction_ratio_logit','diag50_mean_abs_dlogit_ab','VAL_dIoU_ab','TEST_dIoU_ab'):
            v=np.array([f(r[key]) for r in rr]);tv=f(t[key])
            print(f"{g:18s} {key:28s} n={len(v)} median={np.median(v):.4g} q05={np.quantile(v,.05):.4g} q95={np.quantile(v,.95):.4g} target={tv:.4g} pct={(v<tv).mean()*100:.0f}")
    single=rows(ZN/'POJEDYNCZE_k218750.csv');eff=np.array([f(r['mean_abs_dlogit']) for r in single])
    for n in (370,391):print('single neuron',n,'dlogit',eff[n],'rank',int((eff>eff[n]).sum())+1,'of 768')
    print('\nablation over epochs (diag50 ratio, VAL dIoU ab):')
    for r in rows(ZN/'ABLACJA_PARY_0_200.csv'):
        print(f"{f(r['epoch_equivalent']):6.1f} ratio={f(r['interaction_ratio_logit']):.3f} dlogit_ab={f(r['diag50_mean_abs_dlogit_ab']):.4f} VAL_dIoU_ab={f(r['VAL_dIoU_ab']):+.2e} TEST_dIoU_ab={f(r['TEST_dIoU_ab']):+.2e} rank_a={r['neuron_strength_rank_a']} rank_b={r['neuron_strength_rank_b']}")
    print('\nweights:')
    for r in rows(ZN/'PARA_WAGI_0_200.csv'):
        print(f"{f(r['epoch_equivalent']):6.1f} w2_angle={f(r['w2_angle_deg']):6.2f} w1cos={f(r['w1_cos_ln_folded']):+.3f} norms_w2={f(r['w2_norm_a']):.2f}/{f(r['w2_norm_b']):.2f} bhat={f(r['b_hat_a']):+.2f}/{f(r['b_hat_b']):+.2f}")
    print('\ntokens:')
    for r in rows(ZN/'PARA_TOKENY_0_200.csv'):
        if r['category'] in ('both_nonzero','opposite_sign','opposite_sign_both_active05') and int(r['step']) in (0,10000,55000,65640,109375,164063,218750):
            print(f"{f(r['epoch_equivalent']):6.1f} {r['category']:28s} n={r['tokens']:>6s} kappa_agg={f(r['aggregate_kappa']):.3f} ratio50={f(r['strength_ratio_q50']):.3f} canc50={f(r['cancellation_q50']):.3f} share50={f(r['pair_share_of_mlp_q50']):.4f} sumshare50={f(r['sum_share_of_mlp_q50']):.4f} negA={r['negative_is_a']} negB={r['negative_is_b']} gt={f(r['fraction_in_gt']):.3f} corr={r.get('corr_za_zb_all_tokens','')}")
    (DANE/'detekcja_0_200/BOOTSTRAP_MANIFEST.json').write_text(json.dumps(dict(reps=REPS,seed=SEED,unit='film (cluster bootstrap)',interval='percentile 95%'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
