"""Sprawdza mechanikę zapisanych ruchów. Duży błąd replay obali zgodność osi czasu.

Odczyt 16 istniejących NPZ panelu; bez zapisów na Edenie, jobów i fit.
Wynik JSON stdout. Wnioskowanie statystyczne ograniczone do panelu.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json, pathlib, hashlib, time
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=pathlib.Path('/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008')
def q(a):
    a=np.asarray(a)
    return {'n':int(a.size),'q':{str(k):float(np.quantile(a,k)) for k in [0,.1,.25,.5,.75,.9,1]}} if a.size else {'n':0}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cosine(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.sum(a*b,axis=-1)/np.maximum(den,1e-30)
started=time.time()
panel=json.loads((ROOT/'provenance/CONTROL_PANEL.json').read_text())
cp=ROOT/'results/packages/model_mean/MODEL_VIT_K1000.pt'
ckpt=torch.load(cp,map_location='cpu',weights_only=False)
state=ckpt['net'] if 'net' in ckpt else ckpt.get('model',ckpt.get('state_dict',ckpt))
keys=[k for k in state if k.endswith('mlp.fc2.weight')]
key=sorted(keys)[-1]
W=state[key].detach().numpy().astype(float)
_,s,V=np.linalg.svd(W,full_matrices=False)
rank=int((s>max(W.shape)*np.finfo(float).eps*s[0]).sum()); V=V[:rank]
dm=np.load(ROOT/'results/dmdc/DMDC_MODEL.npz',allow_pickle=False)
mean=np.load(ROOT/'results/regimes/MEAN_DELTA_ALL.npz')['mean_delta'].astype(float)
records=[]; tensors=[]; gradients=[]; moments1=[]; moments2=[]; phases=[]
endpoint_energy={'success':[0.,0.], 'budget_exhausted':[0.,0.]}
for image_text,tokens in panel.items():
    index=int(image_text); p=ROOT/f'results/production/latents/{index:06d}.npz'
    with np.load(p,allow_pickle=False) as a:
        meta=json.loads(a['metadata_json_utf8'].tobytes())
        z=a['control_z'].astype(float); g=a['control_gradient_before_step'].astype(float)
        m=a['control_adam_m1_before_step'].astype(float); v=a['control_adam_m2_before_step'].astype(float)
        dt=np.diff(z,axis=0); t=np.arange(1,501)[:,None,None]
        lr=np.select([t<=100,t<=250,t<=400],[.05,.15,.05],default=.01)
        mh=(.9*m+.1*g)/(1-.9**t); vh=(.999*v+.001*g*g)/(1-.999**t)
        predicted=np.maximum(z[:-1]-lr*mh/(np.sqrt(vh)+1e-8),-0.16997121)
        phase=a['telemetry_phase_min_norm'][:-1].astype(bool)
        # phase[t] denotes availability of first success at source state t.
        rec={'image':index,'npz_sha256':sha(p),'status':meta['status'],'first_success_step':meta['first_success_step'],'after_step':meta['after_step'],
          'target_token':meta['target']['token_index'],'target_in_panel':bool(meta['target']['token_index'] in tokens),
          'schema':{k:list(a[k].shape) for k in ['before','after','delta','trajectory','control_z','control_gradient_before_step','control_adam_m1_before_step','telemetry_step']},
          'trajectory_steps':a['trajectory_steps'].tolist(),'adam_replay_max_abs_error':float(np.max(np.abs(predicted-z[1:]))),'adam_replay_rmse':float(np.sqrt(np.mean((predicted-z[1:])**2))),
          'path_length':q(np.linalg.norm(dt,axis=-1).sum(axis=0)), 'net_movement':q(np.linalg.norm(z[-1]-z[0],axis=-1)),
          'step_direction_cosine':{},'endpoint':{}}
        norms=np.linalg.norm(dt,axis=-1)
        turns=cosine(dt[1:],dt[:-1]);valid=(norms[1:]>1e-4)&(norms[:-1]>1e-4)
        for name,mask in [('fit',~phase[1:]&~phase[:-1]),('min_norm',phase[1:]&phase[:-1]),('switch',phase[1:]!=phase[:-1])]:
            sel=valid&mask[:,None]; rec['step_direction_cosine'][name]={'cosine':q(turns[sel]),'negative':int((turns[sel]<0).sum())}
        for name,delta in [('after',a['delta'].astype(float)),('k100',a['trajectory'][0].astype(float)-a['before']),('k500',a['trajectory'][1].astype(float)-a['before'])]:
            total=float(np.square(delta).sum()); row=float(np.square(delta@V.T).sum())
            target=meta['target']['token_index']; mask=np.arange(600)!=target
            rec['endpoint'][name]={'energy':total,'null_energy_fraction':1-row/max(total,1e-30),'target_norm':float(np.linalg.norm(delta[target])),'other_norms':q(np.linalg.norm(delta[mask],axis=1)),
                'cosine_with_mean_for_active_tokens':q(cosine(delta,mean)[np.linalg.norm(delta,axis=1)>1e-4])}
            if name=='after':
                endpoint_energy[meta['status']][0]+=total; endpoint_energy[meta['status']][1]+=total-row
        tensors.append(z.transpose(1,0,2)); gradients.append(g.transpose(1,0,2)); moments1.append(m.transpose(1,0,2)); moments2.append(v.transpose(1,0,2)); phases.append(np.broadcast_to(phase,(8,500)).copy()); records.append(rec)
z=np.concatenate(tensors).astype(np.float32);g=np.concatenate(gradients).astype(np.float32);m=np.concatenate(moments1).astype(np.float32);v=np.concatenate(moments2).astype(np.float32)
phase=np.concatenate(phases);image_ids=np.repeat([int(i) for i in panel],8);test=np.isin(image_ids,[0,500,999,1499])
x=(z[:,:-1]-dm['state_mean'])@dm['state_components'].T; y=(z[:,1:]-dm['state_mean'])@dm['state_components'].T
u=np.concatenate([g,m,v],axis=-1); control=(u-dm['control_mean'])@dm['control_components'].T
ones=np.ones((*x.shape[:2],1),np.float32)
pred={'persistence':x,'dmd':np.concatenate([x,ones],axis=-1)@dm['dmd_coefficients'],'dmdc':np.concatenate([x,control,ones],axis=-1)@dm['dmdc_coefficients']}
metrics={}
for name,mask in [('heldout_all',np.broadcast_to(test[:,None],phase.shape)),('heldout_fit',test[:,None]&~phase),('heldout_min_norm',test[:,None]&phase)]:
    metrics[name]={'transitions':int(mask.sum()),'rmse_reduced_state':{k:float(np.sqrt(np.mean((val[mask]-y[mask]).astype(float)**2))) for k,val in pred.items()}}
# Measure scale imbalance in *training* controls without refitting PCA.
train=~test
control_scale={name:{'rms':float(np.sqrt(np.mean(a[train].astype(float)**2))),'sum_centered_squares':float(np.sum((a[train].astype(float)-a[train].mean(axis=(0,1),dtype=float))**2))} for name,a in [('gradient',g),('m1',m),('m2',v)]}
report={'status':'complete','scope':'16 existing control-panel files only; no fit or image inference','elapsed_seconds':time.time()-started,
 'checkpoint_sha256':sha(cp),'fc2_key':key,'fc2_shape':list(W.shape),'fc2_rank':rank,'fc2_nullity':W.shape[1]-rank,
 'sample_after_energy':{k:{'total':v[0],'null_fraction':v[1]/v[0]} for k,v in endpoint_energy.items()},
 'control_train_scales':control_scale,'existing_dmd_model_metrics_by_phase':metrics,'records':records}
print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
