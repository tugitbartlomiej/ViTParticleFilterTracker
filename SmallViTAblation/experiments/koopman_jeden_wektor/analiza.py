"""Test one full token chosen on earlier groups; held-out failure is a negative result.

The chosen local linear model must beat persistence AND last increment. No
retuning after final errors and no claim of full-image Koopman identification.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import json,time,csv,sys,platform
from pathlib import Path
import numpy as np
from dane import HERE,SOURCE,metadata,load_candidates,write,sha
from modele import TokenModel,CONFIGS,training,metrics

OUT=HERE/'wyniki'
def status(stage,**kw):
    write(OUT/'STATUS.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
    print(json.dumps(dict(stage=stage,**kw)),flush=True)

def aggregate(records,role='holdout',horizon=1,method='selected'):
    rr=[r for r in records if r['role']==role and r['horizon']==horizon and r['method']==method]
    if not rr:return None
    fg=[r for r in rr if r['mask_coverage']>=.5]
    return dict(pairs=len(rr),rmse=float(np.sqrt(np.mean([r['mse'] for r in rr]))),
        change_rmse=float(np.sqrt(np.mean([r['change_mse'] for r in rr]))),
        relative_change_error=float(np.sqrt(sum(r['mse'] for r in rr)/max(sum(r['change_mse'] for r in rr),1e-30))),
        mean_cosine=float(np.mean([r['cosine'] for r in rr])),
        median_increment_ratio=float(np.median([r['increment_ratio'] for r in rr])),
        foreground_pairs=len(fg),foreground_rmse=float(np.sqrt(np.mean([r['mse'] for r in fg]))) if fg else None,
        group_rmse={g:float(np.sqrt(np.mean([r['mse'] for r in rr if r['group_id']==g]))) for g in sorted({r['group_id'] for r in rr})})

def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'REPORT.json').exists():raise RuntimeError('Completed report exists; no automatic rerun')
    cfg,rows,steps,inner,val,fit,hold,mask,tokens=metadata()
    protocol=dict(date='2026-09-25',calibration_origins=[105,145,185],evaluation_origins=list(range(210,365,5)),
        secondary_rollouts=dict(origins=[210,230,250],horizon=20),window=8,configs=CONFIGS,
        candidate_tokens=tokens.tolist(),candidate_mask_coverage=mask[fit][:,tokens].mean(0).tolist(),
        inner_fit=inner.tolist(),inner_validation=val.tolist(),fit=fit.tolist(),holdout=hold.tolist(),
        success=dict(rmse_vs_last_max=.95,rmse_vs_persistence_max=.8,cosine_min=.5,winning_groups_min=3),
        source_config_sha256=sha(SOURCE/'CONFIG.json'),mask_sha256=sha(SOURCE/'MASK_WEIGHTS.npy'),
        scripts={p.name:sha(p) for p in HERE.glob('*.py')},plan_sha256=sha(HERE/'PLAN.md'),
        no_pca=True,token_dimension=384,scope='one full token, not full-image operator',
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        prior_holdout_exposure=True)
    if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text())==json.loads(json.dumps(protocol))
    write(OUT/'PROTOCOL.json',protocol)
    data,proofs=load_candidates(steps,tokens,status)
    write(OUT/'INPUT_SHA256.json',proofs)
    calibration=[]
    for t,token in enumerate(tokens):
        for name,ridge in CONFIGS:
            errors=[];last_errors=[];pers_errors=[];cs=[]
            for origin in protocol['calibration_origins']:
                j=steps.index(origin)
                x,v,y=training(data,steps,origin,inner,t,8)
                m=TokenModel().fit(x,v,y,name,ridge)
                z=data[j,val,t].astype('f8');vv=z-data[j-1,val,t].astype('f8');truth=data[j+1,val,t].astype('f8')
                pred=m.predict(z,vv)
                err,change,cos,ratio=metrics(pred,truth,z)
                errors.extend(err.tolist());last_errors.extend(np.mean((z+vv-truth)**2,axis=1).tolist())
                pers_errors.extend(change.tolist());cs.extend(cos.tolist())
            rec=dict(candidate_index=t,token_index=int(token),name=name,ridge=ridge,
                mse=float(np.mean(errors)),last_mse=float(np.mean(last_errors)),persistence_mse=float(np.mean(pers_errors)),
                score=float(sum(errors)/max(sum(last_errors),1e-30)),mean_cosine=float(np.mean(cs)))
            calibration.append(rec);status('calibration',token=int(token),model=name,ridge=ridge,score=rec['score'])
    chosen=min(calibration,key=lambda r:(r['score'],r['token_index'],r['name'],r['ridge']))
    selection=dict(**chosen,selected_using='inner validation only, max target 190',frozen_before_final_evaluation=True)
    write(OUT/'CALIBRATION.json',calibration);write(OUT/'SELECTION.json',selection)
    status('selection_frozen',selection=selection)
    t=chosen['candidate_index'];token=chosen['token_index'];records=[];fit_records=[]
    for origin in protocol['evaluation_origins']:
        j=steps.index(origin);x,v,y=training(data,steps,origin,fit,t,8)
        m=TokenModel().fit(x,v,y,chosen['name'],chosen['ridge'])
        fit_pred=m.predict(x,v)
        fit_records.append(dict(origin=origin,pairs=len(x),fit_rmse=float(np.sqrt(np.mean((fit_pred-y)**2))),
            relative_fit_change_error=float(np.linalg.norm(fit_pred-y)/max(np.linalg.norm(y-x),1e-30))))
        A,b=m.explicit();np.savez(OUT/f'MATRIX_{origin:07d}.npz',A=A,b=b,token_index=token,
            state='z' if chosen['name']=='state' else '[z,v]',convention='row state @ A + b',max_train_target=origin)
        z=data[j,:,t].astype('f8');vel=z-data[j-1,:,t].astype('f8')
        horizons=[1]+([20] if origin in [210,230,250] else [])
        for h in horizons:
            truth=data[j+h,:,t].astype('f8')
            candidates=dict(selected=m.predict(z,vel,h),last_increment=z+h*vel,persistence=z)
            for method,pred in candidates.items():
                if not np.isfinite(pred).all():raise RuntimeError(f'Nonfinite prediction {origin} {h} {method}')
                err,change,cos,ratio=metrics(pred,truth,z)
                for i in range(len(rows)):
                    records.append(dict(origin=origin,target=steps[j+h],horizon=h,method=method,
                        image_index=i,group_id=rows[i]['group_id'],role=rows[i]['role'],
                        token_index=token,mask_coverage=float(mask[i,token]),mse=float(err[i]),
                        change_mse=float(change[i]),cosine=float(cos[i]),increment_ratio=float(ratio[i])))
            if h==1:
                np.savez(OUT/f'FORECAST_{origin:07d}.npz',prediction=candidates['selected'],truth=truth,z=z,v=vel)
        status('evaluation',origin=origin)
    with (OUT/'ERRORS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(OUT/'FIT_ERRORS.json',fit_records)
    totals={str(h):{method:aggregate(records,horizon=h,method=method) for method in ['selected','last_increment','persistence']} for h in [1,20]}
    best,last,pers=[totals['1'][k] for k in ['selected','last_increment','persistence']]
    wins=sum(best['group_rmse'][g]<last['group_rmse'][g] for g in best['group_rmse'])
    passed=dict(vs_last=best['rmse']<=.95*last['rmse'],vs_persistence=best['rmse']<=.8*pers['rmse'],
        direction=best['mean_cosine']>=.5,groups=wins>=3)
    report=dict(state='completed',selection=selection,totals=totals,winning_groups=wins,
        acceptance_checks=passed,accepted=all(passed.values()),fit_errors=fit_records,
        limitations=['diagnostic, previously exposed dataset','4 held-out source groups, not independent per-frame samples',
            'spatial token may contain background/context','one-step local operators are refitted using past fit data',
            'no proof of semantic identity or exact Koopman closure','not full-image operator'])
    write(OUT/'REPORT.json',report)
    lines=['# Wynik: jeden pełny wektor tokenu','',f"Token {token}, siatka 24×24: wiersz {token//24}, kolumna {token%24} (indeksy od zera).",
        f"Model: {chosen['name']}; ridge {chosen['ridge']}. Wybrany na wcześniejszej walidacji, bez błędów końcowych.",
        'Wektor ma wszystkie 384 liczby. To test lokalny wybranego tokenu, nie operator całego obrazu.','',
        '| Horyzont | Metoda | RMSE | RMSE / wielkość zmiany | Cosinus przyrostu |',
        '|---:|---|---:|---:|---:|']
    for h,dd in totals.items():
        for name,q in dd.items():lines.append(f"| {int(h)*5} aktualizacji | {name} | {q['rmse']:.8f} | {q['relative_change_error']:.4f} | {q['mean_cosine']:.4f} |")
    lines+=['',f"Spełnienie ustalonych kryteriów: {report['accepted']}; {passed}; przewaga w {wins}/4 grup.",
        'Protokół: PROTOCOL.json; dobór: CALIBRATION.json, SELECTION.json; błędy: ERRORS.csv; jawne macierze: MATRIX_*.npz.',
        'Wynik jest diagnostyczny. Nie dowodzi pełnej tożsamości narzędzia ani poprawnej prognozy całej sieci.',
        'Długi horyzont używa trzech początków, krótki 31; metod nie dobierano na długim horyzoncie.']
    (HERE/'WYNIKI_TEST.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(records,token,report)
    status('completed',accepted=report['accepted'],token=token,model=chosen['name'])
    print(json.dumps(report,ensure_ascii=False),flush=True)

def plot(records,token,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names={'selected':'Wybrany Koopman','last_increment':'Ostatni przyrost','persistence':'Brak zmian'}
    for method,label in names.items():
        rr=[r for r in records if r['role']=='holdout' and r['horizon']==1 and r['method']==method]
        origins=sorted({r['origin'] for r in rr})
        err=[np.sqrt(np.mean([r['mse'] for r in rr if r['origin']==o])) for o in origins]
        axes[0].plot(origins,err,label=label)
        if method!='persistence':axes[1].plot(origins,[np.mean([r['cosine'] for r in rr if r['origin']==o]) for o in origins],label=label)
    axes[0].set(ylabel='RMSE pełnego wektora 384 liczb',xlabel='Początek prognozy (aktualizacja)',title='Prognoza jednego checkpointu')
    axes[1].set(ylabel='Cosinus prognozowanej i rzeczywistej zmiany',xlabel='Początek prognozy (aktualizacja)',title='Kierunek zmiany')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.suptitle(f'Token {token}: wybór na wcześniejszych grupach, ocena na innych obrazach')
    fig.savefig(OUT/'PROGNOZA_TOKENU.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    try:run()
    except Exception as e:
        if OUT.exists():status('failed',error=repr(e))
        raise
