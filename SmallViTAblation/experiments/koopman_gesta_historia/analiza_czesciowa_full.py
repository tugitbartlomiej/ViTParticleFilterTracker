"""Partial chronological full-tensor forecast; one recorded probe, no PCA."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'wyniki'/'analiza_czesciowa_full'

def fit(x,y):
    mx,my=x.mean(0),y.mean(0)
    xc,yc=x-mx,y-my
    gram=xc@xc.T
    ridge=1e-3*np.trace(gram)/len(x)
    inv=np.linalg.solve(gram+ridge*np.eye(len(x)),np.eye(len(x)))
    return mx,my,xc,yc,inv,ridge

def forecast(z,model,h):
    mx,my,xc,yc,inv,_=model
    a=(z-mx)@xc.T@inv
    transition=yc@xc.T@inv
    offset=(my-mx)@xc.T@inv
    for _ in range(h-1): a=offset+a@transition
    return my+a@yc

def main():
    OUT.mkdir(exist_ok=True)
    # Numerical identity check: dual recurrence equals explicit primal ridge.
    rng=np.random.default_rng(17)
    x=rng.normal(size=(12,7)); y=x@rng.normal(size=(7,7))*.1+.2
    m=fit(x,y); z=x[:2].copy(); explicit=z.copy()
    a=m[2].T@m[4]@m[3]
    for _ in range(5): explicit=(explicit-m[0])@a+m[1]
    np.testing.assert_allclose(forecast(z,m,5),explicit,atol=1e-10)
    run=HERE/'wyniki'/'run_1810662'
    manifest={r['path']:r for r in json.loads((run/'REMOTE_MANIFEST.json').read_text())}
    paths=[run/'pretrain'/f'CZASTKI_{s:07d}.npz' for s in range(0,801,5)]
    verified=[]; arrays=[]
    for p in paths:
        rel=p.relative_to(run).as_posix()
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha==manifest[rel]['sha256'],rel
        with np.load(p) as data: v=data['tokens'].copy()
        assert v.shape==(576,384) and np.isfinite(v).all()
        arrays.append(v.reshape(-1));verified.append(dict(path=rel,sha256=sha))
    history=np.array(arrays,dtype=np.float64)
    # Fixed BEFORE evaluating: fit transitions steps 0..500, 20-step horizon.
    # 41 rolling origins 500..700; no image holdout is available here.
    cfg=dict(fit_end_update=500,horizon_updates=100,origins=list(range(500,701,5)),ridge_relative=1e-3,trend_window_updates=100,state_shape=[576,384],pca=False,probe_images=1)
    (OUT/'CONFIG.json').write_text(json.dumps(cfg,indent=2))
    model=fit(history[:100],history[1:101])
    start=history[100:141]; actual=history[120:161]
    pred=forecast(start,model,20)
    trend=start+(start-history[80:121])
    metrics={}
    true_delta=actual-start
    for name,value in [('koopman',pred),('persistence',start),('linear_trend',trend)]:
        error=value-actual;delta=value-start
        norm=np.linalg.norm(delta,axis=1);tn=np.linalg.norm(true_delta,axis=1)
        cosine=np.divide((delta*true_delta).sum(1),norm*tn,out=np.zeros(len(norm)),where=norm*tn>0)
        metrics[name]=dict(mse=float(np.mean(error**2)),rmse=float(np.sqrt(np.mean(error**2))),per_origin_mse=np.mean(error**2,axis=1).tolist(),mean_increment_cosine=None if name=='persistence' else float(cosine.mean()),mean_increment_norm_ratio=float(np.mean(norm/tn)))
    mx,my,xc,yc,inv,ridge=model
    spectrum=np.linalg.eigvals(yc@xc.T@inv)
    np.savez_compressed(OUT/'OPERATOR_DUAL.npz',mu_x=mx,mu_y=my,X_centered=xc,Y_centered=yc,gram_inverse=inv,ridge=ridge,eigenvalues_nonzero=spectrum)
    np.savez_compressed(OUT/'PREDICTIONS.npz',origins=np.arange(500,701,5),predicted=pred.reshape(-1,576,384),actual=actual.reshape(-1,576,384),initial=start.reshape(-1,576,384))
    report=dict(status='partial_single_probe',config=cfg,sha256_verified_files=len(verified),numerical_dual_test='passed',metrics=metrics,better_than_both_controls=metrics['koopman']['mse']<min(metrics['persistence']['mse'],metrics['linear_trend']['mse']),spectral_radius=float(abs(spectrum).max()),limitations=['one original probe only; no held-out image groups','no identity attribution or RGB256 test','rolling origins overlap; not independent replicates','forecast improvement is not detection improvement','LR and run duration changed together; no separate test set'],dense_matrix_gib=221184**2*8/1024**3)
    (OUT/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (OUT/'VERIFIED_INPUTS.json').write_text(json.dumps(verified,indent=2))
    lines=['# Częściowa analiza pełnego latentu','', 'Wszystkie 161 pełnych zrzutów jednego obrazu potwierdzono przez SHA256. Bez PCA i kompresji współrzędnych. Operator afiniczny jest zapisany dokładnie w postaci dualnej; nie jest operatorem 384x384 stosowanym niezależnie do cząstek.','', 'Dopasowanie: aktualizacje 0–500. Ocena: 41 początków od 500 do 700, prognoza o 100 aktualizacji. Trend używa poprzednich 100 aktualizacji. Parametry ustalono przed oceną.','', '| Model | RMSE | Cosinus przyrostu | Stosunek długości przyrostu |','|---|---:|---:|---:|']
    for name,r in metrics.items():lines.append(f"| {name} | {r['rmse']:.6g} | {r['mean_increment_cosine']} | {r['mean_increment_norm_ratio']:.4g} |")
    lines+=['','Koopman lepszy od obu kontroli: '+str(report['better_than_both_controls']), '', 'Ograniczenia: to jedna trajektoria jednego obrazu, bez osobnych grup obrazów. Nakładające się prognozy nie są niezależnymi próbami. Nie wykonano identyfikacji tożsamości, testu 256 teł ani generowania FastGAN. Wynik nie rozstrzyga domknięcia tożsamości ani poprawy detekcji. Nie zastępuje głównego eksperymentu.','', 'Macierz gęsta wymagałaby około 365 GiB float64. Postać dualna nie obcina rzędu ani kanałów, ale 100 par treningowych nie identyfikuje dowolnej dynamiki w 221184 wymiarach. Regularyzacja wybiera jedno przybliżenie spośród niedookreślonych rozwiązań.']
    (OUT/'RAPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','config')}))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_origin_mse'} for k,v in metrics.items()}))

if __name__=='__main__':main()
