"""Historical checkpoint summaries. Δweights are NOT gradients; sparse AdamW labels stay sparse."""
from pathlib import Path
import numpy as np
import torch
from przygotuj_eden import HERE,SOURCE,read,js,sha,stamp
def rms(values):
 n=sum(v.numel() for v in values)
 return float(torch.sqrt(sum(v.double().square().sum() for v in values)/n)) if n else 0.
def prepare():
 from model_snapshot import MalyViT
 net=MalyViT().cpu();params=list(net.named_parameters());buffers=list(net.named_buffers())
 groups=sorted({'.'.join(n.split('.')[:2]) for n,_ in params+buffers})
 pg={g:[n for n,_ in params if '.'.join(n.split('.')[:2])==g] for g in groups}
 bg={g:[n for n,v in buffers if v.is_floating_point() and '.'.join(n.split('.')[:2])==g] for g in groups}
 order=[[n for n,p in params if p.ndim>=2],[n for n,p in params if p.ndim<2]]
 names=[kind+':'+g for kind in ('rms_delta_parameter','rms_delta_bn_buffer','rms_actual_adamw_m_hold_last','rms_actual_adamw_v_hold_last') for g in groups]+['optimizer_snapshot_age_updates','optimizer_moments_available','delta_checkpoint_available']
 Q=len(names);aux=np.zeros((101,Q),np.float32);sparse=np.zeros_like(aux);valid=np.ones_like(aux,dtype=bool);label_valid=np.ones_like(valid)
 G=len(groups);last_m=np.zeros(G);last_v=np.zeros(G);last_cp=None;previous=None;source=[];maps=[]
 for cp in range(101):
  path=SOURCE/'wagi'/('k%04d.pt'%(cp*10));ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['krok']==cp*10;state=ck['net'];net.load_state_dict(state,strict=True)
  if previous is not None:
   aux[cp,:G]=[rms([state[n]-previous[n] for n in pg[g]]) for g in groups]
   aux[cp,G:2*G]=[rms([state[n]-previous[n] for n in bg[g]]) for g in groups]
  else:valid[cp,:2*G]=False;label_valid[cp,:2*G]=False
  observed=False
  if cp%10==0:
   fullpath=SOURCE/'pelny'/('k%04d.pt'%(cp*10));full=torch.load(fullpath,map_location='cpu',weights_only=True);assert full['krok']==cp*10
   assert all(torch.equal(state[n],full['net'][n]) for n in state)
   opt=full['opt'];assert len(opt['param_groups'])==2;states={}
   for paramgroup,names_order in zip(opt['param_groups'],order):
    assert len(paramgroup['params'])==len(names_order)
    for ident,name in zip(paramgroup['params'],names_order):
     if ident not in opt['state']:continue
     ss=opt['state'][ident];assert ss['exp_avg'].shape==state[name].shape and ss['exp_avg_sq'].shape==state[name].shape;states[name]=ss
   if states:
    assert len(states)==len(params)
    last_m=np.array([rms([states[n]['exp_avg'] for n in pg[g]]) for g in groups]);last_v=np.array([rms([states[n]['exp_avg_sq'] for n in pg[g]]) for g in groups]);last_cp=cp;observed=True
   maps.append(dict(cp=cp,step=cp*10,path=str(fullpath),sha256=sha(fullpath),state_count=len(states),param_order_verified=True))
  aux[cp,2*G:3*G]=last_m;aux[cp,3*G:4*G]=last_v
  aux[cp,-3:]=[(cp-last_cp)*10 if last_cp is not None else cp*10,float(last_cp is not None),float(cp>0)]
  valid[cp,2*G:4*G]=last_cp is not None
  label_valid[cp,2*G:4*G]=observed
  sparse[cp]=aux[cp]
  source.append(dict(cp=cp,step=cp*10,path=str(path),sha256=sha(path),moments_actual_observed=observed,last_actual_moments_cp=last_cp))
  previous=state
  print('telemetry',cp,'actual_moments',observed,flush=True)
 sparse[~label_valid]=0
 np.savez_compressed(HERE/'AUX_HISTORY.npz',input_values=aux,input_valid=valid,target_values=sparse,target_valid=label_valid,moment_age_steps=aux[:,-3].copy())
 js(HERE/'AUX_HISTORY.json',dict(status='complete',timestamp=stamp(),shape=list(aux.shape),names=names,groups=groups,data_sha256=sha(HERE/'AUX_HISTORY.npz'),sources=source,full_snapshots=maps,actual_optimizer_cadence_updates=100,delta_weight_cadence_updates=10,batch_ids_available=False,augmentations_available=False,gradients_available=False,moment_semantics='input: last actually observed snapshot with age; target moment loss only on actually observed snapshots',missing_initial_optimizer_state='cp0 empty state is missing, not measured zero moments',parameter_order=order))
if __name__=='__main__':prepare()
