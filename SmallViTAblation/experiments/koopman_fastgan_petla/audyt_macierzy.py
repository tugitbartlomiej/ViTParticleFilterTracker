"""Rozstrzyga błąd rachunku kontra słaba prognoza macierzy; niezależny solver
obala poprawność, jeśli nie odtwarza zapisanych parametrów. Analiza po wyniku,
bez zmiany bramki, bez GPU i bez uczenia detektora. Warianty okien są opisowe.
"""
from pathlib import Path
import argparse
import numpy as np
from dane import read_json, write_json, sha256
from petla import history
from dynamika import fit_directions, project, fit_koopman, forecast, to_jsonable


def run(root):
    root=Path(root)
    records=read_json(root/'probe/manifest.json')
    fit=np.array([r['role']=='fit' for r in records]); hold=~fit
    steps,features=history([root/'pretrain'])
    original=read_json(root/'round_01/FORECAST.json')
    basis=fit_directions(features,np.array([r['labels'] for r in records]),fit,2,rank=8)
    c=project(features,basis)
    stop=original['model']['fit_stop']; ridge=original['model']['ridge']
    model=fit_koopman(c,steps,fit,stop,ridge)
    scale=np.asarray(model['scale']); mean=np.asarray(model['mean'])
    z=(c[:stop,fit]-mean)/scale
    x=z[:-1].reshape(-1,z.shape[-1]); y=z[1:].reshape(x.shape)
    xc=x-x.mean(0); yc=y-y.mean(0)
    penalty=np.sqrt(len(x)*ridge)*np.eye(x.shape[1])
    # Independent augmented least squares, not the SVD ridge implementation.
    coef=np.linalg.lstsq(np.vstack([xc,penalty]),np.vstack([yc,np.zeros_like(penalty)]),rcond=None)[0]
    rhs=xc.T@yc
    normal=(xc.T@xc+len(x)*ridge*np.eye(x.shape[1]))@np.array(model['A_scaled']).T-rhs
    audit=dict(posthoc=True,changes_original_gate=False,
        original_forecast_sha256=sha256(root/'round_01/FORECAST.json'),
        max_recomputed_A_difference=float(np.max(np.abs(np.array(original['model']['A'])-model['A']))),
        max_recomputed_b_difference=float(np.max(np.abs(np.array(original['model']['b'])-model['b']))),
        independent_solver_max_difference=float(np.max(np.abs(coef-np.array(model['A_scaled']).T))),
        normal_equation_relative_residual=float(np.linalg.norm(normal)/max(np.linalg.norm(rhs),1e-20)),
        convention='row vectors: next = current @ A.T + b; horizon in snapshot intervals',
        steps=steps.tolist(),matrices=[],forecast_comparisons=[])
    if max(audit['max_recomputed_A_difference'],audit['independent_solver_max_difference'])>1e-10:
        raise ValueError('Independent matrix audit failed')
    # Matrices always expressed in the SAME frozen c coordinates and same
    # comparison scaling; eigenvector ordering/sign is not compared as identity.
    for start,end in [(0,6),(5,11),(10,16),(15,21),(0,21)]:
        m=fit_koopman(c[start:end],steps[start:end],fit,end-start,ridge)
        a=np.array(m['A']); b=np.array(m['b']); eig=np.linalg.eigvals(a)
        common=a*scale[None,:]/scale[:,None]
        audit['matrices'].append(dict(start_step=int(steps[start]),end_step=int(steps[end-1]),
            A=a,b=b,eigenvalues_real=eig.real,eigenvalues_imag=eig.imag,
            spectral_radius=float(np.abs(eig).max()),
            common_scaled_distance_from_identity=float(np.linalg.norm(common-np.eye(len(a))))))
    full=fit_koopman(c,steps,fit,len(c),ridge)
    write_json(root/'round_01/OPERATOR_FULL_4_EPOCHS.json',to_jsonable(dict(
        model=full,basis=basis,forecast_step=500,forecast=forecast(full,c[-1],5),
        current=c[-1],validated_on_future=False,
        note='Fit uses all four epochs for fit-image groups. No fifth epoch exists; forecast is unverified.')))
    # Same forecast origins, holdout images and metric for all models.
    # Identity-regularized model is a diagnostic control for shrinkage bias,
    # not a post-hoc replacement of the original accepted=False decision.
    for name,window,increment in [('original_frozen',None,False),('expanding',None,False),
                                ('recent_5',5,False),('recent_10',10,False),
                                ('expanding_increment_ridge',None,True)]:
        rows=[]
        for origin in [14,15]:
            start=0 if window is None else origin+1-window
            m=model if name=='original_frozen' else fit_koopman(c[start:origin+1],steps[start:origin+1],fit,origin+1-start,ridge)
            if increment:
                mu=np.array(m['mean']); sc=np.array(m['scale'])
                zz=(c[start:origin+1,fit]-mu)/sc
                xx=zz[:-1].reshape(-1,zz.shape[-1]); dy=(zz[1:]-zz[:-1]).reshape(xx.shape)
                xx0=xx-xx.mean(0); dy0=dy-dy.mean(0)
                bb=np.linalg.lstsq(np.vstack([xx0,np.sqrt(len(xx)*ridge)*np.eye(xx.shape[1])]),
                    np.vstack([dy0,np.zeros((xx.shape[1],xx.shape[1]))]),rcond=None)[0]
                aa=np.eye(xx.shape[1])+bb.T; bs=dy.mean(0)-xx.mean(0)@bb
                m=dict(m,A=sc[:,None]*aa/sc[None,:])
                m['b']=mu+sc*bs-m['A']@mu
            anchor=c[origin,hold]; truth=c[origin+5,hold]
            predicted=forecast(m,anchor,5)
            dp=(predicted-anchor)/scale; dt=(truth-anchor)/scale
            pn=np.linalg.norm(dp,axis=1); tn=np.linalg.norm(dt,axis=1)
            rows.append(dict(origin_step=int(steps[origin]),
                error=float(np.linalg.norm(dp-dt,axis=1).mean()),constant_error=float(tn.mean()),
                predicted_length=float(pn.mean()),actual_length=float(tn.mean()),
                cosine=float(((dp*dt).sum(1)/np.maximum(pn*tn,1e-20)).mean())))
        audit['forecast_comparisons'].append(dict(name=name,mean_error=float(np.mean([r['error'] for r in rows])),folds=rows))
    out=root/'round_01/AUDYT_MACIERZY.json'
    write_json(out,to_jsonable(audit))
    print({k:v for k,v in audit.items() if k not in ('matrices','forecast_comparisons','steps')})
    print([(r['name'],r['mean_error']) for r in audit['forecast_comparisons']])
    print('Output:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    run(p.parse_args().run)
