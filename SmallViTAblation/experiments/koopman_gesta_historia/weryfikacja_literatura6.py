"""Post-run provenance, serialized-operator check, matched horizon table and plot."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
from audyt_literatura6 import HERE,OUT,SOURCE,training,load,sha,write

def main():
    assert json.loads((OUT/'STATUS.json').read_text())['stage']=='completed'
    names=['koopman_pamiec_full.py','audyt_literatura6.py','koopman_przyrostowy.py',
           'macierz_kazde_przejscie.py','test_literatura6.py','LITERATURA6_PROTOCOL.json']
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    deps={}
    for name in names:
        deps[str(HERE/name)]=sha(HERE/name)
        dest=snapshot/name
        if dest.exists():assert sha(dest)==deps[str(HERE/name)]
        else:shutil.copyfile(HERE/name,dest)
    deps[str(SOURCE/'MASK_WEIGHTS.npy')]=sha(SOURCE/'MASK_WEIGHTS.npy')
    deps[str(SOURCE/'CONFIG.json')]=sha(SOURCE/'CONFIG.json')
    write(OUT/'DEPENDENCIES_SHA256.json',dict(captured_after_run=True,files=deps,
        note='Supplemental provenance, not falsely represented as pre-run registration'))
    with (OUT/'ERRORS_PER_IMAGE.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    common=set.intersection(*(set(int(r['origin']) for r in rows if int(r['horizon_updates'])==h) for h in [5,20,100]))
    matched={}
    for h in [5,20,100]:
        matched[str(h)]={}
        for method in sorted(set(r['method'] for r in rows)):
            selected=[r for r in rows if r['role']=='holdout' and r['method']==method and int(r['horizon_updates'])==h and int(r['origin']) in common]
            matched[str(h)][method]=dict(rmse=float(np.sqrt(np.mean([float(r['mse']) for r in selected]))),pairs=len(selected))
    write(OUT/'MATCHED_HORIZONS.json',dict(origins=sorted(common),holdout=matched))
    # Independent reconstruction of first saved operator's one-step prediction.
    cfg=json.loads((SOURCE/'CONFIG.json').read_text());steps=[w['step'] for w in cfg['weights']]
    fit=np.array([i for i,r in enumerate(cfg['rows']) if r['role']=='fit'])
    x,v,y,pairs=training(steps,210,fit,4)
    assert pairs==json.loads((OUT/'PAIRS_0000210.json').read_text())
    with np.load(OUT/'OPERATOR_DELAY_0000210.npz') as op:
        u=(x-op['mu_x'])/op['scale_x'];vv=(v-op['mu_velocity'])/op['scale_velocity']
        index=next(i for i,r in enumerate(cfg['rows']) if r['role']=='holdout')
        z=np.array(load(210)[index],dtype=np.float64);vel=z-load(205)[index]
        q=((z-op['mu_x'])/op['scale_x'])@u.T+((vel-op['mu_velocity'])/op['scale_velocity'])@vv.T
        pred=z+vel+op['mean_acceleration']+q@op['coeff']
        mse=float(np.mean((pred-load(215)[index])**2))
    recorded=next(float(r['mse']) for r in rows if r['method']=='delay_full' and int(r['origin'])==210 and int(r['horizon_updates'])==5 and int(r['image_index'])==index)
    np.testing.assert_allclose(mse,recorded,rtol=1e-10,atol=1e-14)
    write(OUT/'VERIFICATION.json',dict(serialized_operator_reproduces_recorded_mse=True,
        mse=mse,recorded_mse=recorded,checked_origin=210,checked_image=index,
        paired_horizon_origins=sorted(common),no_state_compression=True))
    del x,v,y,u,vv,pred
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods={'delay_full':'Pełny stan + pamięć','previous_K':'Poprzednia macierz',
             'last_increment':'Ostatni przyrost','persistence':'Brak zmian',
             'past_mean_increment':'Średni przyrost fit'}
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,h in zip(axes,[5,20,100]):
        for method,label in methods.items():
            selected=[r for r in rows if r['role']=='holdout' and r['method']==method and int(r['horizon_updates'])==h]
            origins=sorted(set(int(r['origin']) for r in selected))
            values=[np.sqrt(np.mean([float(r['mse']) for r in selected if int(r['origin'])==t])) for t in origins]
            ax.plot(origins,values,marker='o',markersize=3,label=label)
        ax.set(title=f'Horyzont {h} aktualizacji',xlabel='Początek prognozy (krok treningu)',ylabel='RMSE pełnego latentu; holdout')
        ax.grid(alpha=.25);ax.set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.suptitle('Prognozy na tych samych obrazach: porównanie metod w obrębie każdego horyzontu')
    fig.savefig(OUT/'PROGNOZY.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(verified=True,matched=matched)))

if __name__=='__main__':main()
