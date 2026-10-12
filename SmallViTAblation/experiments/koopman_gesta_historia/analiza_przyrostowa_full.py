"""Diagnostic comparison: identity prior and increasing history, frozen protocol."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import hashlib,json,time
from pathlib import Path
import numpy as np
from koopman_przyrostowy import IncrementKoopman,self_test

HERE=Path(__file__).resolve().parent
OUT=HERE/'wyniki'/'koopman_przyrostowy_full'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def metrics(pred,actual,start):
    diff=pred-start;truth=actual-start
    norm=np.linalg.norm(diff,axis=1);tn=np.linalg.norm(truth,axis=1)
    valid=(norm*tn)>0
    cos=np.divide((diff*truth).sum(1),norm*tn,out=np.zeros(len(norm)),where=valid)
    errors=np.mean((pred-actual)**2,axis=1)
    return dict(rmse=float(np.sqrt(errors.mean())),mse=float(errors.mean()),
        mean_cosine=float(cos[valid].mean()) if valid.any() else None,
        mean_norm_ratio=float(np.mean(norm/tn)),per_origin_mse=errors.tolist())

def main():
    OUT.mkdir(exist_ok=True)
    save('TESTS.json',self_test())
    # Predeclared diagnostic protocol; previous holdout has already been seen.
    save('CONFIG.json',dict(dimension=221184,shape=[576,384],pca=False,
         origins_updates=list(range(500,701,5)),horizons_updates=[5,100],
         fit_ends_updates=[500,600,700],relative_ridge=1e-3,
         expanded_fit='use latest fit end <= forecast origin',
         final_fit_end_update=800,final_fit_future_verified=False,
         note='Diagnostic follow-up on previously inspected holdout; no hyperparameter search. One image only.'))
    root=HERE/'wyniki'/'run_1810662'
    manifest={r['path']:r for r in json.loads((root/'REMOTE_MANIFEST.json').read_text())}
    history=[];verified=[]
    for step in range(0,801,5):
        rel=f'pretrain/CZASTKI_{step:07d}.npz';p=root/rel
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha==manifest[rel]['sha256']
        with np.load(p) as d:v=d['tokens'].copy()
        assert v.shape==(576,384) and np.isfinite(v).all()
        history.append(v.reshape(-1));verified.append(dict(path=rel,sha256=sha))
    history=np.asarray(history,dtype=np.float64)
    save('VERIFIED_INPUTS.json',verified)
    origins=np.arange(100,141);start=history[origins]
    results={};predictions={1:{},20:{}}
    for end in [100,120,140,160]:
        save('STATUS.json',dict(state='fitting',fit_end_update=end*5,pid=os.getpid(),time=time.time()))
        model=IncrementKoopman().fit(history[:end],history[1:end+1])
        if end in (100,160):model.save(OUT/f'OPERATOR_{end*5:04d}.npz')
        if end==160:
            np.savez_compressed(OUT/'FUTURE_UNVERIFIED.npz',step805=model.predict(history[-1],1).reshape(576,384),step900=model.predict(history[-1],20).reshape(576,384))
            break
        choose=(origins>=end)&(origins<end+20)
        for h in [1,20]:
            if end==100:
                predictions[h]['residual_fixed']=model.predict(start,h)
                predictions[h]['residual_expanding']=np.empty_like(start)
            predictions[h]['residual_expanding'][choose]=model.predict(start[choose],h)
    # Original direct-state operator, not refitted or retuned.
    with np.load(HERE/'wyniki'/'analiza_czesciowa_full'/'OPERATOR_DUAL.npz') as old:
        mx,my,xc,yc,inv=[old[k] for k in ['mu_x','mu_y','X_centered','Y_centered','gram_inverse']]
        transition=yc@xc.T@inv;offset=(my-mx)@xc.T@inv
        for h in [1,20]:
            weights=(start-mx)@xc.T@inv
            for _ in range(h-1):weights=offset+weights@transition
            predictions[h]['old_direct']=my+weights@yc
    for h in [1,20]:
        actual=history[origins+h]
        predictions[h]['persistence']=start
        predictions[h]['last_increment']=start+h*(start-history[origins-1])
        predictions[h]['trend_100']=start+h/20*(start-history[origins-20])
        results[str(h*5)]={name:metrics(pred,actual,start) for name,pred in predictions[h].items()}
        np.savez_compressed(OUT/f'PREDICTIONS_H{h*5}.npz',origins=origins*5,actual=actual.reshape(-1,576,384),
            residual_fixed=predictions[h]['residual_fixed'].reshape(-1,576,384),residual_expanding=predictions[h]['residual_expanding'].reshape(-1,576,384))
    report=dict(state='completed',probe_images=1,verified_snapshots=161,results=results,
        full_fit_pairs=160,heldout_forecast_fit_pairs=[100,120,140],
        limitations=['single probe; not held-out image generalization','diagnostic follow-up after seeing original results','prior identity preserves unseen input perturbations; not guarantee of correct physics','no proof of identity completeness or detector improvement','full 800-update fit forecasts unverified'],
        identity_prior=True,pca=False)
    save('REPORT.json',report)
    lines=['# Pełny operator przyrostowy — wynik diagnostyczny','',
        '161 zweryfikowanych stanów jednego obrazu, wszystkie 221184 współrzędne. Bez PCA i obcięcia SVD. To analiza po obejrzeniu wcześniejszych wyników, nie nowe niezależne potwierdzenie.','',
        'Macierz stała: 100 par do kroku 500. Wariant aktualizowany: 100, 120, 140 par do kroków 500, 600, 700; żadna prognoza nie używa danych późniejszych niż jej początek. Te same 41 początków 500–700 dla wszystkich metod. Osobno zapisano operator ze wszystkich 160 par do kroku 800; jego przyszłe prognozy są niesprawdzone.','']
    for horizon,rows in results.items():
        lines += [f'## Horyzont {horizon} aktualizacji','','| Metoda | RMSE | Cosinus przyrostu | Stosunek długości |','|---|---:|---:|---:|']
        for name,r in rows.items():lines.append(f"| {name} | {r['rmse']:.7f} | {r['mean_cosine']} | {r['mean_norm_ratio']:.4f} |")
    lines+=['','Testy numeryczne potwierdziły równoważność regresji dualnej i bezpośredniej, wielokrokowej iteracji, zachowanie niewidzianych perturbacji wejścia, stałego dryfu i identyczności. Nie badano nowych sond, tożsamości ani jakości detekcji; nie uruchamiano FastGAN.']
    (OUT/'RAPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save('STATUS.json',dict(state='completed',pid=os.getpid(),time=time.time()))
    print(json.dumps({h:{k:{a:b for a,b in v.items() if a!='per_origin_mse'} for k,v in rows.items()} for h,rows in results.items()}))

if __name__=='__main__':main()
