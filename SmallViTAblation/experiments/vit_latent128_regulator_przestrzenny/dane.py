"""Czy poprawa przenosi się między filmami? Dane i augmentacje nie mieszają splitów."""
import numpy as np
import torch
from torch.nn import functional as F
from dane_bazowe import *

SEED=20261005

def photo(x,index):
    rng=np.random.default_rng(SEED+index)
    a,b=rng.uniform(.8,1.2,2)
    mean=x.mean((1,2,3),keepdim=True)
    return ((x-mean)*float(a)+mean*float(b)).clamp(0,1)

def load_data(augment=True):
    net,rows,sets,J,T,gt,H,Q,base,mu,sd,prov=load()
    sets['train_original']=sets['train'].copy()
    for i,r in enumerate(rows):r.update(source_index=i,augmentation='original')
    if augment:
        cache={};block=net.bloki[-1]
        handles=[block.mlp.act.register_forward_pre_hook(lambda m,a:cache.update(pre=a[0].detach())),
                 block.register_forward_hook(lambda m,a,o:cache.update(state=o.detach()))]
        hs=[H];qs=[Q];outs=[base];extra=[];gs=[]
        originals=sets['train_original']
        for variant in ('flip','photo'):
            for start in range(0,len(originals),8):
                ids=originals[start:start+8]
                x=torch.stack([image_tensor(rows[i]) for i in ids])
                x=x.flip(3) if variant=='flip' else torch.cat([photo(x[k:k+1],i) for k,i in enumerate(ids)])
                with torch.no_grad():
                    o,_=net(x);hs.append(cache['pre'][:,:,J].clone());qs.append(cache['state'].clone());outs.append(o.cpu())
                for i in ids:
                    rec=dict(rows[i],augmentation=variant)
                    g=gt[i].copy()
                    if variant=='flip':g[:,[0,2]]=480-g[:,[2,0]]
                    extra.append(rec);gs.append(g)
        for handle in handles:handle.remove()
        offset=len(rows);rows=rows+extra;gt=gt+gs
        H=torch.cat(hs);Q=torch.cat(qs);base=torch.cat(outs)
        extraT=cele(gs,20,30)
        T=tuple(torch.cat([a,b.cuda()]) for a,b in zip(T,extraT))
        sets['train']=originals+list(range(offset,len(rows)))
        assert all(rows[i]['split']=='train' for i in range(offset,len(rows)))
        assert set(rows[i]['film'] for i in sets['train']).isdisjoint(rows[i]['film'] for i in sets['val']+sets['test'])
        print('AUGMENTED',len(sets['train']),flush=True)
    Z=F.gelu(H);del H
    original=sets['train_original']
    stats=dict(z_mu=Z[original].mean((0,1)),z_sd=Z[original].std((0,1)).clamp(min=.05),
               q_mu=Q[original].mean((0,1)),q_sd=Q[original].std((0,1)).clamp(min=.05))
    centers=T[0].flatten(1).argmax(1)
    batch=torch.arange(len(rows),device='cuda')
    center_log=T[1].reshape(len(rows),600,4)[batch,centers].clamp(min=.05).log()
    prov.update(augmented_train=len(sets['train']),seed=SEED,
                historical_benchmark_reused=True,test_labels_in_gradient=False,
                augmentation='full frozen forwards: horizontal flip and deterministic brightness/contrast')
    return dict(net=net,rows=rows,sets=sets,J=J,T=T,gt=gt,Z=Z,Q=Q,base=base,stats=stats,
                centers=centers,center_log=center_log,provenance=prov)

def objective(o,targets,centers,center_log,strong=True):
    heat,ltrb,poz=targets
    logit=o[...,0];p=logit.sigmoid();pos=heat.eq(1).float()
    focal=(-(F.logsigmoid(logit)*(1-p).square()*pos+F.logsigmoid(-logit)*p.square()*(1-heat).pow(4)*(1-pos))).sum((1,2))/pos.sum((1,2)).clamp(min=1)
    pred=o[...,1:].clamp(-10,10).exp()
    g=_giou(pred.reshape(-1,4),ltrb.reshape(-1,4)).reshape(p.shape)
    box=((1-g)*poz).sum((1,2))/poz.sum((1,2)).clamp(min=1e-12)
    loss=focal+2*box
    if strong:
        flat=o.reshape(len(o),600,5);bi=torch.arange(len(o),device=o.device)
        rank=F.cross_entropy(flat[:,:,0],centers,reduction='none')
        logbox=F.smooth_l1_loss(flat[bi,centers,1:],center_log,reduction='none').mean(1)
        loss=loss+.5*rank+2*logbox
    return loss

# Wildcard imports intentionally exclude the private GIoU helper.
from model_snapshot import _giou
