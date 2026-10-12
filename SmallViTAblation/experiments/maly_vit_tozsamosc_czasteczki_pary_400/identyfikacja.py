"""Czy podprzestrzeń rozdziela pary innych filmów? Słaby test/sham obala swoistość."""
import json,time,os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score,roc_curve
from threadpoolctl import threadpool_limits
from dane import HERE,load,weights,js,sha

threadpool_limits(4)

def basis(cache,rows,ids):
    X=[];w=[];wi=weights(rows,ids)
    for j,i in enumerate(ids):
        a,b=cache['offsets'][i:i+2];s=cache['states'][a:b];c=cache['coverage'][a:b].astype(np.float64)
        X.append((s[:,0]-s[:,1]).astype(np.float64));w.extend(wi[j]*c/c.sum())
    X=np.concatenate(X);w=np.array(w);C=X.T@(X*w[:,None]);values,vectors=np.linalg.eigh(C)
    order=np.argsort(values)[::-1];values=np.maximum(values[order],0);vectors=vectors[:,order]
    mean=(X*w[:,None]).sum(0);sign=np.sign(mean@vectors);sign[sign==0]=1;vectors*=sign
    assert np.isfinite(values).all() and np.allclose(vectors.T@vectors,np.eye(768),atol=1e-9)
    return vectors,values,mean

def dataset(pooled,ids,U):
    return np.concatenate([pooled[ids,0]@U,pooled[ids,1]@U]),np.r_[np.ones(len(ids)),np.zeros(len(ids))]

def metrics(y,s,threshold=.5,w=None):
    return dict(auroc=float(roc_auc_score(y,s,sample_weight=w)),balanced_accuracy=float(balanced_accuracy_score(y,s>=threshold,sample_weight=w)),accuracy=float(accuracy_score(y,s>=threshold,sample_weight=w)))

def fit_probe(pooled,rows,ids,U,C):
    X,y=dataset(pooled,ids,U);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1
    ww=np.tile(weights(rows,ids),2)*len(ids)
    clf=LogisticRegression(C=C,solver='lbfgs',max_iter=500).fit((X-mu)/sd,y,sample_weight=ww)
    return dict(mu=mu,sd=sd,coef=clf.coef_[0],intercept=clf.intercept_[0])

def score(pooled,ids,U,probe):
    X,y=dataset(pooled,ids,U);v=((X-probe['mu'])/probe['sd'])@probe['coef']+probe['intercept']
    return y,1/(1+np.exp(-np.clip(v,-60,60)))

def threshold(y,s,w):
    fpr,tpr,t=roc_curve(y,s,sample_weight=w);valid=np.isfinite(t);best=np.argmax((tpr-fpr)[valid]);return float(t[valid][best])

def boot_presence(rows,ids,s,threshold,n=500):
    rng=np.random.default_rng(908);films=sorted({rows[i]['video_id'] for i in ids});out=[]
    for _ in range(n):
        draw=rng.choice(films,len(films),replace=True);pos=[];ww=[]
        for f in draw:
            p=[j for j,i in enumerate(ids) if rows[i]['video_id']==f];pos.extend(p);ww.extend([1/len(p)/len(films)]*len(p))
        pos=np.array(pos);sel=np.r_[pos,pos+len(ids)];y=np.r_[np.ones(len(pos)),np.zeros(len(pos))]
        out.append(list(metrics(y,s[sel],threshold,np.tile(ww,2)).values()))
    return {k:list(np.quantile(np.array(out)[:,j],[.025,.975])) for j,k in enumerate(('auroc','balanced_accuracy','accuracy'))}

def main():
    start=time.time();rows,t=load();cache=dict(np.load(HERE/'TOKENY_PARY.npz'));pooled=cache['pooled'].astype(np.float64)
    ids={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','val','test')}
    U,eig,mean=basis(cache,rows,ids['train']);tr,va,te=ids.values()
    validation=[];chosen=None
    for rank in (1,2,4,8,16):
        for C in (.01,.1,1,10):
            probe=fit_probe(pooled,rows,tr,U[:,:rank],C);y,s=score(pooled,va,U[:,:rank],probe)
            m=metrics(y,s,w=np.tile(weights(rows,va),2));rec=dict(rank=rank,C=C,**m);validation.append(rec)
            if chosen is None or m['auroc']>chosen['validation']['auroc']+1e-12:
                chosen=dict(rank=rank,C=C,probe=probe,validation=m)
    rank=chosen['rank'];chosenU=U[:,:rank];probe=chosen['probe'];y,s=score(pooled,va,chosenU,probe)
    tau=threshold(y,s,np.tile(weights(rows,va),2));chosen['threshold']=tau
    reports={};scores={}
    for part in ('train','val','test'):
        y,s=score(pooled,ids[part],chosenU,probe);scores[part]=s
        reports[part]=dict(film_weighted=metrics(y,s,tau,np.tile(weights(rows,ids[part]),2)),image_weighted=metrics(y,s,tau),n_images=len(ids[part]),n_films=len({rows[i]['video_id'] for i in ids[part]}))
    reports['test']['film_bootstrap95']=boot_presence(rows,te,scores['test'],tau)
    # LDA: separate comparator, training-only covariance and direction; rank1.
    w=weights(rows,tr);pos=pooled[tr,0];neg=pooled[tr,1];mp=w@pos;mn=w@neg
    cp=pos-mp;cn=neg-mn;cov=(cp.T@(cp*w[:,None])+cn.T@(cn*w[:,None]))/2
    lda_candidates=[];lda_best=None
    scale=max(np.trace(cov)/768,1e-12)
    for ridge in (.01,.1,1):
        direction=np.linalg.solve(cov+ridge*scale*np.eye(768),mp-mn);direction/=np.linalg.norm(direction)
        center=(mp+mn)/2
        val=np.r_[(pooled[va,0]-center)@direction,(pooled[va,1]-center)@direction]
        yy=np.r_[np.ones(len(va)),np.zeros(len(va))];auc=roc_auc_score(yy,val,sample_weight=np.tile(weights(rows,va),2))
        lda_candidates.append(dict(ridge=ridge,val_auroc=auc))
        if lda_best is None or auc>lda_best['val_auroc']+1e-12:lda_best=dict(ridge=ridge,val_auroc=auc,direction=direction,center=center)
    yy=np.r_[np.ones(len(va)),np.zeros(len(va))];sv=np.r_[(pooled[va,0]-lda_best['center'])@lda_best['direction'],(pooled[va,1]-lda_best['center'])@lda_best['direction']]
    ldatau=threshold(yy,sv,np.tile(weights(rows,va),2));st=np.r_[(pooled[te,0]-lda_best['center'])@lda_best['direction'],(pooled[te,1]-lda_best['center'])@lda_best['direction']]
    lda_best.update(threshold=ldatau,test=metrics(np.r_[np.ones(len(te)),np.zeros(len(te))],st,ldatau,np.tile(weights(rows,te),2)))
    # Axis scores are paired, image-/film-weighted, with no pooling substituted for token storage.
    delta=pooled[:,0]-pooled[:,1];mu=w@delta[tr];sd=np.sqrt(np.maximum(w@((delta[tr]-mu)**2),0));effect=mu/np.maximum(sd,1e-12)
    axis_order=np.argsort(np.abs(effect))[::-1];filmmeans=np.array([delta[[i for i in tr if rows[i]['video_id']==f]].mean(0) for f in sorted({rows[i]['video_id'] for i in tr})])
    rng=np.random.default_rng(909);boots=np.array([filmmeans[rng.integers(0,len(filmmeans),len(filmmeans))].mean(0) for _ in range(500)])
    ci=np.quantile(boots,[.025,.975],axis=0)
    axes=[]
    for position,j in enumerate(axis_order):
        axes.append(dict(rank=position+1,channel0=int(j),channel1=int(j+1),train_mean_delta=mu[j],train_std_delta=sd[j],paired_effect=effect[j],
            train_film_sign_agreement=float((np.sign(filmmeans[:,j])==np.sign(mu[j])).mean()),mean_delta_film_bootstrap95=ci[:,j],
            test_mean_delta=float(weights(rows,te)@delta[te,j]),test_image_sign_agreement=float((np.sign(delta[te,j])==np.sign(mu[j])).mean())))
    js(HERE/'MAKROWEKTORY.json',axes)
    mixed=[]
    for k in range(rank):
        order=np.argsort(np.abs(chosenU[:,k]))[::-1]
        mixed.append(dict(direction=k+1,top_channels=[dict(channel0=int(j),channel1=int(j+1),weight=float(chosenU[j,k])) for j in order[:20]],train_paired_mean_projection=float(mu@chosenU[:,k])))
    # Per-token ranking within each original image, not universal patch IDs.
    token_rank=[];maps=np.zeros((400,600),np.float32);bg_projection=[]
    for i,r in enumerate(rows):
        a,b=cache['offsets'][i:i+2];D=cache['states'][a:b,0]-cache['states'][a:b,1];P=D@chosenU
        norms=np.linalg.norm(D,axis=1);pnorms=np.linalg.norm(P,axis=1);maps[i,cache['token_ids'][a:b]]=pnorms
        order=np.argsort(pnorms)[::-1]
        token_rank.append(dict(source_index=i,frame_id=r['frame_id'],split=r['split'],tokens=[dict(rank=j+1,token0=int(cache['token_ids'][a+k]),token1=int(cache['token_ids'][a+k]+1),row=int(cache['token_ids'][a+k]//30),column=int(cache['token_ids'][a+k]%30),coverage=float(cache['coverage'][a+k]),delta_l2=float(norms[k]),identity_projection_l2=float(pnorms[k])) for j,k in enumerate(order)]))
        bg=cache['background_states'][i*8:(i+1)*8];bD=bg[:,0]-bg[:,1]
        bg_projection.append(dict(source_index=i,tool_projection_mean=float(np.average(pnorms,weights=cache['coverage'][a:b])),far8_projection_mean=float(np.linalg.norm(bD@chosenU,axis=1).mean()),tool_delta_mean=float(np.average(norms,weights=cache['coverage'][a:b])),far8_delta_mean=float(np.linalg.norm(bD,axis=1).mean())))
    js(HERE/'CZASTECZKI_RANKING.json',token_rank);js(HERE/'PROJEKCJA_TLO.json',bg_projection)
    np.savez_compressed(HERE/'PODPRZESTRZEN.npz',U=chosenU.astype(np.float32),U16=U[:,:16].astype(np.float32),eigenvalues=eig,mean_delta=mean,
        probe_mu=probe['mu'],probe_sd=probe['sd'],probe_coef=probe['coef'],probe_intercept=probe['intercept'],threshold=tau,rank=rank,C=chosen['C'],
        lda_direction=lda_best['direction'],lda_center=lda_best['center'],lda_threshold=ldatau,identity_projection_maps=maps)
    # Angle transfer: refit basis/probe outside angle; settings inherited from main validation.
    angle=[];transfer=[]
    for binid in range(12):
        jj=np.array([i for i in te if int(rows[i]['angle_axis_deg']//15)==binid])
        if len(jj):
            yy,ss=score(pooled,jj,chosenU,probe);angle.append(dict(bin=binid,degrees=[15*binid,15*(binid+1)],n=len(jj),**metrics(yy,ss,tau)))
        if len(jj)>=3:
            fit=np.array([i for i in tr if int(rows[i]['angle_axis_deg']//15)!=binid]);vb=np.array([i for i in va if int(rows[i]['angle_axis_deg']//15)!=binid])
            ub,_,_=basis(cache,rows,fit);ub=ub[:,:rank];pb=fit_probe(pooled,rows,fit,ub,chosen['C'])
            vy,vs=score(pooled,vb,ub,pb);tt=threshold(vy,vs,np.tile(weights(rows,vb),2));yy,ss=score(pooled,jj,ub,pb)
            transfer.append(dict(heldout_angle_bin=binid,n_test=len(jj),train_n=len(fit),basis_refitted=True,settings_exposed_via_original_val=True,**metrics(yy,ss,tt)))
    js(HERE/'KATY.json',dict(test_bins=angle,leave_angle_bin_out_diagnostic=transfer))
    report=dict(chosen_rank=rank,chosen_C=chosen['C'],chosen_threshold=tau,selection='train only U/probe; film-weighted val AUROC; ascending rank and regularization ties',validation_grid=validation,
        presence=reports,lda_candidates=lda_candidates,lda=dict(ridge=lda_best['ridge'],val_auroc=lda_best['val_auroc'],test=lda_best['test'],threshold=ldatau),
        r50=int(np.searchsorted(np.cumsum(eig)/eig.sum(),.5)+1),r90=int(np.searchsorted(np.cumsum(eig)/eig.sum(),.9)+1),energy_selected_rank=float(eig[:rank].sum()/eig.sum()),mixed_directions=mixed,
        zero_std_axes=int((sd<1e-12).sum()),all_finite=True,seconds=time.time()-start,subspace_sha256=sha(HERE/'PODPRZESTRZEN.npz'))
    js(HERE/'IDENTYFIKACJA.json',report);print(json.dumps(dict(rank=rank,presence=reports,lda=report['lda'],r50=report['r50'],r90=report['r90'],seconds=report['seconds']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
