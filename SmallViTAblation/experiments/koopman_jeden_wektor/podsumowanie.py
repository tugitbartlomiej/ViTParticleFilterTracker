"""Post-run verification and a deterministic illustrative vector, not new selection.

The example is the first held-out image with token mask coverage >= .5, at
the first forecast origin. It is not chosen for a small forecast error.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json,numpy as np
from dane import HERE,metadata,write,sha

def run():
    p=HERE/'wyniki';r=json.loads((p/'REPORT.json').read_text());token=r['selection']['token_index']
    cfg,rows,steps,inner,val,fit,hold,mask,tokens=metadata()
    checks=[]
    for f in sorted(p.glob('MATRIX_*.npz')):
        step=int(f.stem.split('_')[-1]);m=np.load(f);q=np.load(p/f'FORECAST_{step:07d}.npz')
        state=q['z'] if r['selection']['name']=='state' else np.c_[q['z'],q['v']]
        actual=(state@m['A']+m['b'])[:,:384]
        np.testing.assert_allclose(actual,q['prediction'],atol=1e-10,rtol=1e-10)
        checks.append(dict(step=step,alpha=float(m['A'][384,384]) if r['selection']['name']=='scalar_velocity' else None,
            max_abs_error=float(abs(actual-q['prediction']).max()),matrix_sha256=sha(f)))
    write(p/'VERIFICATION.json',dict(serialized_matrices_checked=len(checks),checks=checks,all_passed=True))
    valid=[int(i) for i in hold if mask[i,token]>=.5]
    if not valid:
        write(p/'EXAMPLE.json',dict(available=False,reason='no held-out foreground token'));return
    index=valid[0];origin=210
    q=np.load(p/f'FORECAST_{origin:07d}.npz');z=q['z'][index];v=q['v'][index];truth=q['truth'][index];pred=q['prediction'][index]
    delta=truth-z;dh=pred-z
    info=dict(available=True,selection_rule='first outer image in stored order with mask>=.5; first origin 210; post-run illustration, no error-based selection',
        image_index=index,group_id=rows[index]['group_id'],variant=rows[index]['variant'],token_index=token,
        mask_coverage=float(mask[index,token]),origin=origin,target=215,
        rmse=float(np.sqrt(np.mean((pred-truth)**2))),last_rmse=float(np.sqrt(np.mean((z+v-truth)**2))),
        persistence_rmse=float(np.sqrt(np.mean(delta**2))),relative_change_error=float(np.linalg.norm(dh-delta)/np.linalg.norm(delta)),
        cosine=float(dh@delta/(np.linalg.norm(dh)*np.linalg.norm(delta))))
    write(p/'EXAMPLE.json',info)
    np.savez(p/'WEKTOR_PRZYKLAD.npz',z=z,v=v,truth=truth,prediction=pred,token=token,image_index=index)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    ax.plot(delta,label='Rzeczywisty przyrost',linewidth=1.3)
    ax.plot(dh,label='Prognozowany przyrost',linewidth=1,alpha=.8)
    ax.set(xlabel='Współrzędna pełnego wektora (384 liczby)',ylabel='Zmiana wartości',
        title=f'Token {token}, obraz {index}, krok 210 → 215; przykład wybrany według maski')
    ax.grid(alpha=.2);ax.legend();fig.savefig(p/'WEKTOR_PRZYKLAD.png',dpi=150);plt.close(fig)
    print(json.dumps(info,ensure_ascii=False))

if __name__=='__main__':run()
