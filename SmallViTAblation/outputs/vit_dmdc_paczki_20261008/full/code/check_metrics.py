"""Compare Gram-matrix full-space errors with explicitly reconstructed forecasts."""
import json
import numpy as np
from run import metrics_baseline, metrics_projection

rng=np.random.default_rng(71)
X=rng.normal(size=(32,21));S=rng.normal(size=(32,21))
C=np.linalg.qr(rng.normal(size=(21,6)))[0]
p=dict(xx=X@X.T,ss=S@S.T,xs=X@S.T,xc=X@C,cc=C.T@C)
ts=np.arange(2,20)
checks=[]
for h in [1,5,10]:
    truth=X[ts+h]-X[ts]
    dc=rng.normal(size=(len(ts),6));explicit=dc@C.T
    result=metrics_projection(p,dc,ts,h,21)
    ref=((truth-explicit)**2).sum()
    checks.append(abs(result['sse']-ref)/ref)
    for kind,beta,b in [('persistence',0,1),('velocity',0,1),('force_direct',0,1),('force_inertia',.87,.98)]:
        if kind=='persistence':explicit=np.zeros_like(truth)
        elif kind=='velocity':explicit=h*(X[ts]-X[ts-1])
        else:
            v=X[ts]-X[ts-1];explicit=np.zeros_like(truth)
            for k in range(h):v=beta*v+b*S[ts+k];explicit+=v
        result=metrics_baseline(p,ts,h,21,kind,beta,b)
        ref=((truth-explicit)**2).sum()
        checks.append(abs(result['sse']-ref)/max(ref,1e-30))
assert max(checks)<1e-12,checks
mask=np.arange(21)%3==0
p_mask=dict(xx=X[:,mask]@X[:,mask].T,ss=S[:,mask]@S[:,mask].T,
            xs=X[:,mask]@S[:,mask].T,xc=X[:,mask]@C[mask],cc=C[mask].T@C[mask])
for h in [1,5,10]:
    dc=rng.normal(size=(len(ts),6));truth=X[ts+h][:,mask]-X[ts][:,mask]
    ref=((truth-dc@C[mask].T)**2).sum()
    result=metrics_projection(p_mask,dc,ts,h,int(mask.sum()))
    checks.append(abs(result['sse']-ref)/ref)
assert max(checks)<1e-12,checks
print(json.dumps({'checks':len(checks),'max_relative_sse_error':max(checks),'status':'passed'}))
