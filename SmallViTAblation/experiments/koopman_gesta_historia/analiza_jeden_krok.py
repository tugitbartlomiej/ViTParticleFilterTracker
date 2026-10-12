"""One saved step (5 optimizer updates), frozen previously fitted operator."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import hashlib,json
from pathlib import Path
import numpy as np

def main():
    here=Path(__file__).resolve().parent
    root=here/'wyniki'/'run_1810662'
    out=here/'wyniki'/'analiza_czesciowa_full'
    manifest={r['path']:r for r in json.loads((root/'REMOTE_MANIFEST.json').read_text())}
    history=[]
    for step in range(400,801,5):
        rel=f'pretrain/CZASTKI_{step:07d}.npz'
        p=root/rel
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[rel]['sha256']
        with np.load(p) as d: history.append(d['tokens'].reshape(-1))
    history=np.asarray(history,dtype=np.float64)
    with np.load(out/'OPERATOR_DUAL.npz') as m:
        start=history[20:80]; actual=history[21:81]
        pred=m['mu_y']+((start-m['mu_x'])@m['X_centered'].T@m['gram_inverse'])@m['Y_centered']
    controls={'koopman':pred,'persistence':start,
              'linear_trend_100_updates':start+(start-history[:60])/20,
              'last_increment_5_updates':2*start-history[19:79]}
    reports={}
    for label,n in [('same_origins_500_700',41),('all_origins_500_795',60)]:
        metrics={};delta=actual[:n]-start[:n];tn=np.linalg.norm(delta,axis=1)
        for name,value in controls.items():
            inc=value[:n]-start[:n];norm=np.linalg.norm(inc,axis=1)
            cos=np.divide((inc*delta).sum(1),norm*tn,out=np.zeros(n),where=norm*tn>0)
            mse=np.mean((value[:n]-actual[:n])**2,axis=1)
            metrics[name]={'rmse':float(np.sqrt(mse.mean())),'mse':float(mse.mean()),
                'mean_increment_cosine':None if name=='persistence' else float(cos.mean()),
                'mean_increment_norm_ratio':float(np.mean(norm/tn)), 'per_origin_mse':mse.tolist()}
        reports[label]=metrics
    report=dict(horizon_saved_steps=1,horizon_optimizer_updates=5,fit_end_optimizer_update=500,
        refitted=False,pca=False,full_state_shape=[576,384],probe_images=1,results=reports,
        note='Diagnostic follow-up after 100-update forecast; same frozen operator, no tuning. No held-out image groups.')
    (out/'ONE_STEP.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Prognoza o jeden zapis do przodu','',
        'Jeden zapis = pięć aktualizacji optymalizatora. Ta sama zamrożona macierz dopasowana do aktualizacji 0–500; bez ponownego dopasowania, PCA i strojenia. Pełny tensor jednego obrazu 576×384. SHA256 użytych zrzutów sprawdzono.','']
    for label,metrics in reports.items():
        lines += [f'## {label}','','| Metoda | RMSE | Cosinus przyrostu | Stosunek długości |','|---|---:|---:|---:|']
        for name,r in metrics.items():lines.append(f"| {name} | {r['rmse']:.7f} | {r['mean_increment_cosine']} | {r['mean_increment_norm_ratio']:.4f} |")
    lines+=['','Wynik dotyczy jednej trajektorii; nie jest testem tożsamości ani skuteczności detekcji. Krótszy horyzont naturalnie zmniejsza błąd bezwzględny, dlatego porównujemy z kontrolami przy tym samym horyzoncie.']
    (out/'JEDEN_KROK.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:{a:{b:c for b,c in v.items() if b!='per_origin_mse'} for a,v in m.items()} for k,m in reports.items()}))

if __name__=='__main__':main()
