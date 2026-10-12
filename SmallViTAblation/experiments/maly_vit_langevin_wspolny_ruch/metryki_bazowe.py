"""Joint drift/memory fitting must predict from a prefix, in physical coordinates."""
import numpy as np

def fit(q,w,L,ridge,dt,end):
    anchor=q[500].copy();x=q-anchor[None];targets=np.arange(500+dt,end+1,dt);current=targets-dt
    level=x[current];y=x[targets]-level;T=len(targets);D=q.shape[-1]
    weights=np.tile(w/T,T);levelmean=np.einsum('tnd,n->d',level,w)/T;ym=np.einsum('tnd,n->d',y,w)/T
    scale_x=np.sqrt(np.einsum('tnd,n->d',(level-levelmean)**2,w)/T);scale_y=np.sqrt(np.einsum('tnd,n->d',(y-ym)**2,w)/T)
    scale_x=np.maximum(scale_x,1e-10);scale_y=np.maximum(scale_y,1e-10)
    parts=[level/scale_x]
    for lag in range(L):parts.append((x[current-lag*dt]-x[current-(lag+1)*dt])/scale_y)
    X=np.concatenate(parts,axis=-1).reshape(-1,D*(1+L));Y=(y/scale_y).reshape(-1,D)
    mux=weights@X;muy=weights@Y;XX=X-mux;YY=Y-muy;G=XX.T@(XX*weights[:,None]);R=XX.T@(YY*weights[:,None]);gamma=ridge*np.trace(G)/len(G)
    slope=np.linalg.solve(G+gamma*np.eye(len(G)),R);intercept=muy-mux@slope
    b=intercept*scale_y;A=((slope[:D]/scale_x[:,None])*scale_y[None,:]).T
    K=np.stack([((slope[D*(j+1):D*(j+2)]/scale_y[:,None])*scale_y[None,:]).T for j in range(L)]) if L else np.empty((0,D,D))
    physical=b+level.reshape(-1,D)@A.T
    for j in range(L):physical+=(x[current-j*dt]-x[current-(j+1)*dt]).reshape(-1,D)@K[j].T
    standardized=(X@slope+intercept)*scale_y
    max_error=float(np.max(np.abs(physical-standardized)));assert max_error<1e-10
    eig=np.linalg.eigvalsh(G);rank=int((eig>max(eig[-1],1e-30)*1e-10).sum());condition=float((eig[-1]+gamma)/(max(eig[0],0)+gamma))
    companion=np.zeros((D*(L+1),D*(L+1)));companion[:D,:D]=np.eye(D)+A
    if L:
        companion[D:2*D,:D]=A
        for lag in range(L):companion[:D,D*(lag+1):D*(lag+2)]=K[lag];companion[D:2*D,D*(lag+1):D*(lag+2)]=K[lag]
        for lag in range(1,L):companion[D*(lag+1):D*(lag+2),D*lag:D*(lag+1)]=np.eye(D)
    radius=float(np.max(np.abs(np.linalg.eigvals(companion))))
    majority=np.sign(np.einsum('tnd,n->d',np.sign(y),w));majority[majority==0]=1
    return dict(b=b,A=A,K=K,L=L,ridge=ridge,dt=dt,anchor=anchor,state_scale=scale_x,delta_scale=scale_y,
        feature_mean=mux,target_mean=muy,standardized_slope=slope,standardized_intercept=intercept,
        epsilon=.1*scale_y,majority_sign=majority,fit_targets=targets,gram_rank=rank,feature_count=len(G),regularized_condition=condition,ridge_gamma=gamma,
        native_vs_scaled_max_error=max_error,companion_spectral_radius=radius)

def rollout(q,origins,max_h,model,kind='gle'):
    origins=np.array(origins,int);dt=model['dt'];assert max_h%dt==0
    current=q[origins].copy();initial=current.copy();anchor=model['anchor'];x=current-anchor[None];out={0:initial.copy()};L=model['L']
    if kind=='velocity':velocity=q[origins]-q[origins-dt]
    if kind=='gle' and L:history=np.stack([q[origins-lag*dt]-q[origins-(lag+1)*dt] for lag in range(L)],axis=1)
    for h in range(dt,max_h+1,dt):
        if kind=='persistence':current=initial.copy()
        elif kind=='velocity':current=initial+(h/dt)*velocity
        elif kind=='gle':
            delta=model['b'][None,None]+x@model['A'].T
            for lag in range(L):delta+=history[:,lag]@model['K'][lag].T
            x=x+delta;current=x+anchor[None]
            if L:
                if L>1:history[:,1:]=history[:,:-1].copy()
                history[:,0]=delta
        else:raise ValueError(kind)
        assert np.isfinite(current).all(),'Nonfinite autonomous forecast'
        out[h]=current.copy()
    return out

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
