"""Porownuje te same natywne cele poza dopasowaniem; zachowuje wyniki ponizej kontroli."""
import numpy as np

def measure(pred,truth,start,w,epsilon,majority):
    error=pred-truth;real=truth-start;forecast=pred-start;D=real.shape[-1]
    mse_origin=np.einsum('ond,n->o',error**2,w)/D;persist_origin=np.einsum('ond,n->o',real**2,w)/D
    rn=np.linalg.norm(real,axis=2);pn=np.linalg.norm(forecast,axis=2);valid=(rn>1e-12)&(pn>1e-12)
    cos=np.zeros_like(rn);cos[valid]=(forecast*real).sum(2)[valid]/(rn*pn)[valid]
    cosnum=(cos*w).sum(1);cosden=(valid*w).sum(1)
    same=(np.sign(real)==np.sign(forecast));signnum=np.einsum('ond,n->o',same,w)/D
    valid_sign=np.abs(real)>epsilon;near_num=np.einsum('ond,n->o',same&valid_sign,w)/D;near_den=np.einsum('ond,n->o',valid_sign,w)/D
    majority_same=np.sign(real)==majority;majority_full=np.einsum('ond,n->o',majority_same,w)/D
    majority_near=np.einsum('ond,n->o',majority_same&valid_sign,w)/D
    mse=float(mse_origin.mean());pmse=float(persist_origin.mean())
    summary=dict(MSE=mse,persistence_MSE=pmse,skill=1-mse/pmse if pmse>1e-30 else None,cosine=float(cosnum.sum()/cosden.sum()) if cosden.sum()>0 else None,
        cosine_valid=int(valid.sum()),cosine_total=valid.size,sign_accuracy=float(signnum.mean()),sign_filtered=float(near_num.sum()/near_den.sum()) if near_den.sum()>0 else None,
        sign_filtered_valid=int(valid_sign.sum()),sign_total=valid_sign.size,majority_sign_accuracy=float(majority_full.mean()),majority_filtered=float(majority_near.sum()/near_den.sum()) if near_den.sum()>0 else None,
        per_token_MSE=(error**2).mean((0,2)),per_token_persistence_MSE=(real**2).mean((0,2)),per_coordinate_MSE=np.einsum('ond,n->d',error**2,w)/len(pred),per_coordinate_sign=np.einsum('ond,n->d',same,w)/len(pred))
    details=dict(mse=mse_origin,persistence=persist_origin,cosnum=cosnum,cosden=cosden,sign=signnum,near_num=near_num,near_den=near_den)
    return summary,details

def block_bootstrap(details,origin_stride,H,n=300):
    if H>=100:return dict(reported=False,reason='about one independent block at H100')
    N=len(details['mse']);block=max(50,H)//origin_stride;block=max(block,1);rng=np.random.default_rng(917+origin_stride+H);values=[]
    for _ in range(n):
        starts=rng.integers(0,N,int(np.ceil(N/block)));ids=np.concatenate([(s+np.arange(block))%N for s in starts])[:N]
        m=details['mse'][ids].mean();p=details['persistence'][ids].mean();cd=details['cosden'][ids].sum();nd=details['near_den'][ids].sum()
        values.append([1-m/p,details['sign'][ids].mean(),details['near_num'][ids].sum()/nd if nd else np.nan,details['cosnum'][ids].sum()/cd if cd else np.nan])
    a=np.array(values);ci={}
    for j,k in enumerate(('skill','sign_accuracy','sign_filtered','cosine')):ci[k]=np.quantile(a[np.isfinite(a[:,j]),j],[.025,.975]) if np.isfinite(a[:,j]).any() else None
    return dict(reported=True,descriptive_only=True,origin_stride=origin_stride,block_origins=block,block_updates=block*origin_stride,effective_block_count=N/block,repeats=n,CI95=ci)

