"""Independent, read-only numerical audit of reflector equations and stored visualization.

Runs three CPU inference passes on existing s47 weights; does not train or render.
All new evidence is written separately under outputs/audyt_reflektory_20261012.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.special import erf

sys.dont_write_bytecode = True
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'outputs/porownanie_szklo_20261010/film_krysztal'
OUT = ROOT / 'outputs/audyt_reflektory_20261012'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod/model.py'
WEIGHTS = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
spec = importlib.util.spec_from_file_location('audited_model', MODEL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
net = module.MalyViT().eval()
captured = {}
net.bloki[5].norm2.register_forward_pre_hook(lambda m, a: captured.__setitem__('x', a[0].detach().clone()))
net.bloki[5].mlp.register_forward_hook(lambda m, a, y: captured.__setitem__('f', y.detach().clone()))
im = cv2.cvtColor(cv2.imread(str(SRC / 'zdjecie_train0.jpg')), cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)
image = torch.from_numpy(im.copy()).permute(2, 0, 1)[None].float() / 255

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def norm(a):
    return float(np.linalg.norm(a))

def gelu(a):
    return .5 * a * (1 + erf(a / np.sqrt(2)))

def dgelu(a):
    return .5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a*a/2) / np.sqrt(2*np.pi)

def quant(a):
    return dict(zip(['min', 'median', 'max'], map(float, [np.min(a), np.median(a), np.max(a)])))

def matrices(w, x, eps):
    u = x - x.mean()
    d = len(x)
    sd = np.sqrt(np.mean(u*u) + eps)
    a = w['W1'] @ (w['gamma'] * u/sd + w['beta']) + w['b1']
    jln = w['gamma'][:, None] * ((np.eye(d)-np.ones((d,d))/d)/sd - np.outer(u,u)/(d*sd**3))
    j = (w['W2'] * dgelu(a)[None,:]) @ w['W1'] @ jln
    return a, j

snapshots = {}
for p in [SRC/'ZASADY_LASER_KOLOID.md', MODEL, SRC/'film_lasery_12_reflektory_stale.py', SRC/'film_lasery_17_pulapka_ognisko.py', SRC/'film_lasery_19_pulapka_dwa_widoki_sync.py']:
    snapshots[str(p.relative_to(ROOT))] = dict(sha256=sha(p), size=p.stat().st_size)
(OUT/'ZASADY_LASER_KOLOID_snapshot.md').write_bytes((SRC/'ZASADY_LASER_KOLOID.md').read_bytes())

# Same fixed three-dimensional subspace used by the right panel of movie 19.
with np.load(SRC/'pulapka_18_cache.npz') as f:
    stored_j = f['JJ'].astype(np.float64)
je = stored_j[-1]
se, ae = (je+je.T)/2, (je-je.T)/2
_, ev = np.linalg.eigh(se)
e1,e2 = ev[:,0],ev[:,-1]
u3,_,_ = np.linalg.svd((np.eye(192)-np.outer(e1,e1)-np.outer(e2,e2)) @ ae @ np.stack([e1,e2],1), full_matrices=False)
q = np.stack([e1,e2,u3[:,0]],1)
with np.load(SRC/'szescian.npz') as f:
    recorded_steps = f['kroki'].astype(int)

result = {'created_utc':datetime.now(timezone.utc).isoformat(), 'sources':snapshots,
          'scope':'TRAIN 0 JPEG used by original movies; block 6; checkpoints 0,10940,65640; 8 token Jacobians per checkpoint; no training',
          'tokens':[0,100,200,300,316,400,500,599], 'checkpoints':{}}
for step in [0,10940,65640]:
    path = WEIGHTS / f'k{step:05d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    net.load_state_dict(checkpoint['net'])
    with torch.no_grad():
        net(image)
    x = captured['x'][0].double().numpy()
    f_model = captured['f'][0].double().numpy()
    b = net.bloki[5]
    arr = lambda t:t.detach().double().numpy()
    w = dict(W1=arr(b.mlp.fc1.weight),b1=arr(b.mlp.fc1.bias),W2=arr(b.mlp.fc2.weight),b2=arr(b.mlp.fc2.bias),gamma=arr(b.norm2.weight),beta=arr(b.norm2.bias))
    eps = b.norm2.eps
    xc = x-x.mean(1,keepdims=True)
    u = w['gamma'] * xc / np.sqrt(np.mean(xc*xc,1,keepdims=True)+eps) + w['beta']
    a = u @ w['W1'].T + w['b1']
    z = gelu(a)
    exact = z @ w['W2'].T + w['b2']
    clipped = np.maximum(z,0) @ w['W2'].T + w['b2']
    neg = np.minimum(z,0) @ w['W2'].T
    center = lambda f:f-f.mean(0)
    rec = {'weight_sha256':sha(path),'epsilon':eps,
           'mlp_formula_relative_error':norm(exact-f_model)/norm(f_model),
           'negative_activation_fraction':float(np.mean(z<0)),
           'negative_derivative_fraction':float(np.mean(dgelu(a)<0)),
           'clipping_error_relative_full_F':norm(clipped-exact)/norm(exact),
           'clipping_error_relative_centered_F':norm(center(clipped)-center(exact))/norm(center(exact)),
           'bias_norm_relative_full_F':np.sqrt(len(x))*norm(w['b2'])/norm(exact),
           'F_over_x_norm_ratio':quant(np.linalg.norm(exact,axis=1)/np.linalg.norm(x,axis=1)),
           'tokens':{}}
    for token in result['tokens']:
        _,j = matrices(w,x[token],eps)
        sym=(j+j.T)/2
        sing=np.linalg.svd(np.eye(192)+j,compute_uv=False)
        svj=np.linalg.svd(j,compute_uv=False)
        sign,logdet=np.linalg.slogdet(np.eye(192)+j)
        t={'J_norm2':float(svj[0]),'trace_J_per_dim':float(np.trace(j)/192),
           'map_log_abs_det_per_dim':float(logdet/192),'map_det_sign':float(sign),
           'map_singular_min':float(sing[-1]),'map_singular_max':float(sing[0]),
           'map_fraction_singular_gt_1':float(np.mean(sing>1+1e-8)),
           'S_fraction_negative':float(np.mean(np.linalg.eigvalsh(sym)<-1e-10)),
           'neutral_direction_residual':norm(j@np.ones(192))/max(norm(j),1e-30),
           'F_norm':norm(exact[token]),'x_norm':norm(x[token])}
        if token==316:
            jj=q.T@j@q
            leak=j@q-q@jj
            t.update(projection_leakage_relative_JQ=norm(leak)/norm(j@q),
                     projected_J_norm2=float(np.linalg.norm(jj,2)),
                     cached_J_relative_error=norm(stored_j[int(np.flatnonzero(recorded_steps==step)[0])]-j)/norm(j))
            if step==65640:
                wt={k:torch.from_numpy(v) for k,v in w.items()}
                def ff(t):
                    u=torch.nn.functional.layer_norm(t,(192,),wt['gamma'],wt['beta'],eps)
                    return wt['W2']@torch.nn.functional.gelu(wt['W1']@u+wt['b1'])+wt['b2']
                xx=torch.from_numpy(x[token]).requires_grad_()
                auto=torch.autograd.functional.jacobian(ff,xx,vectorize=True).numpy()
                t['analytic_J_vs_autograd_relative_error']=norm(j-auto)/norm(auto)
        rec['tokens'][str(token)]=t
    result['checkpoints'][str(step)]=rec
    print(step, json.dumps({k:v for k,v in rec.items() if k not in ['tokens','weight_sha256']},ensure_ascii=False),flush=True)

# Explicit finite-dimensional counterexample to the necessity of an opposite partner.
directions=np.array([[1.,0.],[-.1,1.],[-.1,-1.]])
result['counterexample_opposite_pair']={'directions':directions.tolist(),'activations':[0.,10.,10.],
    'sum':(np.array([0.,10.,10.])@directions).tolist(),
    'partner_cosines':((directions[1:]@directions[0])/np.linalg.norm(directions[1:],axis=1)).tolist()}
target=OUT/'numerical_spotchecks.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved',target,flush=True)
