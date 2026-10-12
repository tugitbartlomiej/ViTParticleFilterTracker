"""Prospectively specified diagnostic comparison on already available full tokens.

No detector/GAN training, no PCA, no transition between different images.
Every forecast uses only states at or before its origin. Calibration groups
are inside the original fit set. Outer image holdout never selects lambda.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import csv,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
from koopman_pamiec_full import FullDelayKoopman,history_indices,self_test
from koopman_przyrostowy import IncrementKoopman
from macierz_kazde_przejscie import metric_rows

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'wyniki/macierze_wiele_obrazow'
OUT=HERE/'wyniki/audyt_literatura6'

def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def status(stage,**kw):
    write(OUT/'STATUS.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def load(step):
    return np.load(SOURCE/f'TOKENS_{step:07d}.npy',mmap_mode='r').reshape(72,-1)

def training(steps,origin,indices,window):
    triples=history_indices(steps,origin,window)
    x=[];v=[];y=[];pairs=[]
    for ia,ib,ic in triples:
        previous=np.array(load(steps[ia])[indices],dtype=np.float64)
        current=np.array(load(steps[ib])[indices],dtype=np.float64)
        following=np.array(load(steps[ic])[indices],dtype=np.float64)
        x.append(current);v.append(current-previous);y.append(following)
        pairs.extend(dict(image_index=int(i),source_step=steps[ib],target_step=steps[ic],history_step=steps[ia]) for i in indices)
    assert all(p['target_step']<=origin for p in pairs)
    return np.concatenate(x),np.concatenate(v),np.concatenate(y),pairs

def groups_split(rows):
    groups=sorted(set(r['group_id'] for r in rows if r['role']=='fit'))
    assert len(groups)==8
    inner=np.array([i for i,r in enumerate(rows) if r['role']=='fit' and r['group_id'] in groups[:6]])
    valid=np.array([i for i,r in enumerate(rows) if r['role']=='fit' and r['group_id'] in groups[6:]])
    fit=np.array([i for i,r in enumerate(rows) if r['role']=='fit'])
    hold=np.array([i for i,r in enumerate(rows) if r['role']=='holdout'])
    assert len(inner)==36 and len(valid)==12 and len(fit)==48 and len(hold)==24
    assert set(fit).isdisjoint(hold)
    assert set(rows[i]['group_id'] for i in fit).isdisjoint(rows[i]['group_id'] for i in hold)
    return inner,valid,fit,hold

def select_ridge(scores):
    # Deterministic ties prefer more regularization, without consulting final data.
    means={float(k):float(np.mean(v)) for k,v in scores.items()}
    return min(means,key=lambda k:(means[k],-k)),means

def run():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((HERE/'LITERATURA6_PROTOCOL.json').read_text())
    cfg=json.loads((SOURCE/'CONFIG.json').read_text())
    assert json.loads((SOURCE/'STATUS.json').read_text())['stage']=='completed'
    steps=[w['step'] for w in cfg['weights']];rows=cfg['rows']
    inner,valid,fit,hold=groups_split(rows)
    frozen=dict(protocol=protocol,source_config_sha256=sha(SOURCE/'CONFIG.json'),
                script_sha256=sha(Path(__file__)),model_sha256=sha(HERE/'koopman_pamiec_full.py'),
                inner_fit=inner.tolist(),inner_validation=valid.tolist(),fit=fit.tolist(),holdout=hold.tolist())
    if (OUT/'CONFIG.json').exists():
        assert json.loads((OUT/'CONFIG.json').read_text())==frozen,'Frozen audit config changed'
    else:write(OUT/'CONFIG.json',frozen)
    write(OUT/'TESTS.json',self_test())
    proofs=[]
    for step in steps:
        status('verifying_tokens',step=step)
        path=SOURCE/f'TOKENS_{step:07d}.npy'
        proof=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==proof['sha256']
        proofs.append(dict(step=step,sha256=proof['sha256']))
    write(OUT/'INPUT_SHA256.json',proofs)
    scores={str(r):[] for r in protocol['relative_ridges']};calibration=[]
    for origin in protocol['calibration_origins']:
        x,v,y,pairs=training(steps,origin,inner,protocol['window_intervals'])
        z=np.array(load(origin)[valid],dtype=np.float64)
        vel=z-load(origin-5)[valid];truth=load(origin+5)[valid]
        denom=max(float(np.mean((truth-z)**2)),1e-30)
        for ridge in protocol['relative_ridges']:
            status('calibrating',origin=origin,relative_ridge=ridge)
            m=FullDelayKoopman().fit(x,v,y,ridge)
            score=float(np.mean((m.predict(z,vel)-truth)**2)/denom)
            scores[str(ridge)].append(score)
            calibration.append(dict(origin=origin,ridge=ridge,normalized_mse=score,
                                    diagnostics=m.diagnostics,pairs=len(pairs)))
            del m
        del x,v,y
    selected,means=select_ridge(scores)
    write(OUT/'CALIBRATION.json',dict(selected_ridge=selected,means=means,records=calibration,
        final_holdout_used=False,max_calibration_target=max(protocol['calibration_origins'])+5))
    records=[];diagnostics=[];mask=np.load(SOURCE/'MASK_WEIGHTS.npy')
    for origin in protocol['evaluation_origins']:
        status('fitting_forecast',origin=origin,relative_ridge=selected)
        x,v,y,pairs=training(steps,origin,fit,protocol['window_intervals'])
        m=FullDelayKoopman().fit(x,v,y,selected)
        train_pred=m.predict(x,v)
        diag=dict(origin=origin,**m.diagnostics,
            train_rmse=float(np.sqrt(np.mean((train_pred-y)**2))),
            train_pairs=len(pairs),max_train_target=max(p['target_step'] for p in pairs))
        diagnostics.append(diag)
        m.save(OUT/f'OPERATOR_DELAY_{origin:07d}.npz')
        write(OUT/f'PAIRS_{origin:07d}.json',pairs)
        drift=(y-x).mean(0);del train_pred,x,v,y
        z=np.array(load(origin),dtype=np.float64);vel=z-load(origin-5)
        previous=IncrementKoopman().fit(np.array(load(origin-5)[fit],dtype=np.float64),z[fit],1e-3)
        for h in protocol['horizons_snapshots']:
            target=origin+5*h
            if target>steps[-1]:continue
            status('evaluating_forecast',origin=origin,target=target)
            truth=np.array(load(target),dtype=np.float64)
            candidates={
                'delay_full':lambda:m.predict(z,vel,h),
                'previous_K':lambda:previous.predict(z,h),
                'persistence':lambda:z,
                'last_increment':lambda:z+h*vel,
                'past_mean_increment':lambda:z+h*drift}
            for method,fn in candidates.items():
                pred=fn()
                if not np.isfinite(pred).all():raise RuntimeError(f'Nonfinite forecast {method} {origin} {h}')
                for i,ms in enumerate(metric_rows(pred,truth,z,mask)):
                    records.append(dict(origin=origin,target=target,horizon_updates=5*h,
                        method=method,image_index=i,group_id=rows[i]['group_id'],role=rows[i]['role'],**ms))
                del pred
            del truth
        del m,previous,z,vel
        write(OUT/'DIAGNOSTICS.json',diagnostics)
        write(OUT/'PARTIAL_ERRORS.json',records)
    with (OUT/'ERRORS_PER_IMAGE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    totals={}
    for h in sorted(set(r['horizon_updates'] for r in records)):
        totals[str(h)]={}
        for method in sorted(set(r['method'] for r in records)):
            totals[str(h)][method]={}
            for role in ['fit','holdout']:
                rs=[r for r in records if r['horizon_updates']==h and r['method']==method and r['role']==role]
                totals[str(h)][method][role]=dict(rmse=float(np.sqrt(np.mean([r['mse'] for r in rs]))),
                    foreground_rmse=float(np.sqrt(np.mean([r['foreground_mse'] for r in rs]))),pairs=len(rs),
                    group_rmse={g:float(np.sqrt(np.mean([r['mse'] for r in rs if r['group_id']==g]))) for g in sorted(set(r['group_id'] for r in rs))})
    report=dict(state='completed',selected_ridge=selected,totals=totals,full_state=True,pca=False,
        learned_bottleneck=False,deep_koopman=False,
        limitations=['diagnostic: previous runs exposed outer holdout',
            'only 4 held-out source groups, correlated transformed images',
            'past increment is a proxy for memory, not full optimizer state',
            'calibration selects one-step error, longer horizons are separate checks',
            'no theorem that raw detector activations close under affine delay dynamics',
            'no change to historical acceptance gates and no semantic identity proof'])
    write(OUT/'REPORT.json',report)
    lines=['# Audyt po sześciu pracach: pełny latent z pamięcią','',
        f'Wybrana regularyzacja: {selected}; wyłącznie wcześniejsza walidacja wewnątrz grup fit.',
        'Stan to pełne [z, z−z_poprzedni]. Nie ma PCA, obcięcia rzędu ani przejść między różnymi obrazami.',
        'Każda prognoza korzysta z czterech ostatnich odcinków i 48 obrazów fit: 192 par. Grupy holdout nie wybierają parametrów.',
        'To dodatkowa analiza diagnostyczna; ten holdout był już oglądany we wcześniejszych eksperymentach.','',
        '| Horyzont (aktualizacje) | Metoda | RMSE fit | RMSE holdout | RMSE narzędzie holdout |','|---:|---|---:|---:|---:|']
    for h,methods in totals.items():
        for name,a in methods.items():lines.append(f"| {h} | {name} | {a['fit']['rmse']:.7f} | {a['holdout']['rmse']:.7f} | {a['holdout']['foreground_rmse']:.7f} |")
    lines+=['','To afiniczny model z opóźnieniem, nie deep Koopman. Dobry wynik nie dowodzi poprawnej tożsamości narzędzia.',
            'Operator jest zapisany dokładnie przez czynniki dualne i listę par; składowe stanu nie są redukowane.',
            'Haseli–Cortés: warunek pełnego rzędu kolumn nie zachodzi. Nie wyliczamy pozornego certyfikatu z różnicy projektorów ani z cyklu forward/backward.']
    (OUT/'RAPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    status('completed');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tests-only',action='store_true');args=p.parse_args()
    if args.tests_only:
        assert select_ridge({'0.001':[1,2],'0.1':[1,2]})[0]==0.1
        print(self_test())
    else:
        try:run()
        except Exception as e:
            if OUT.exists():status('failed',error=repr(e))
            raise
