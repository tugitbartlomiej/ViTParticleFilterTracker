"""U500/top5 osi nie mogą korzystać z tokenów testu ani checkpointu1000."""
import time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,torch,psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from threadpoolctl import threadpool_limits
from dane import HERE,SOURCE,PAIRS,PREVIOUS,read,js,sha,image,tensor,init_net

def main():
    start=time.time();threadpool_limits(4);assert shutil.disk_usage(HERE).free>1.15e9
    rows=read(PAIRS/'manifest.json');pairs=read(PAIRS/'PARY.json');split=read(PREVIOUS/'PODZIAL.json')['records'];m=dict(np.load(PAIRS/'TOKEN_MASKI.npz'))
    assert sha(PAIRS/'manifest.json')==read(PREVIOUS/'POMIAR.json')['manifest_sha256']
    train=np.array([r['source_index'] for r in split if r['split']=='train']);assert len(train)==240
    trainset=set(train);films=sorted({rows[i]['video_id'] for i in train});counts={f:sum(rows[i]['video_id']==f for i in train) for f in films}
    weights={i:1/len(films)/counts[rows[i]['video_id']] for i in train}
    net=init_net(500);buffers={n:b.clone() for n,b in net.named_buffers()};cov=np.zeros((768,768));mean=np.zeros(768);pool=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            assert r['frame_id']==pairs[i]['frame_id']==split[i]['frame_id']
            assert sha(PAIRS/r['image_path'])==r['image_sha256'] and sha(PAIRS/pairs[i]['synthetic'])==pairs[i]['synthetic_sha256']
            _,z=net(tensor([image(PAIRS/r['image_path']),image(PAIRS/pairs[i]['synthetic'])]));z=z.cpu().numpy().astype(np.float64);ids=np.flatnonzero(m['tool'][i]);c=m['area_coverage'][i,ids].astype(np.float64);c/=c.sum()
            assert z.shape==(2,600,768) and np.isfinite(z).all() and len(ids)>0
            pool.append((z[:,ids]*c[None,:,None]).sum(1))
            if i in trainset:
                D=z[0,ids]-z[1,ids];w=c*weights[i];cov+=D.T@(D*w[:,None]);mean+=(D*w[:,None]).sum(0)
            if (i+1)%50==0:print('calibrate k500',i+1,'/400',round(time.time()-start,1),'s',flush=True)
    assert all(torch.equal(b,buffers[n]) for n,b in net.named_buffers())
    eig,U=np.linalg.eigh(cov);order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0);U=U[:,order[:8]];orient=np.sign(mean@U);orient[orient==0]=1;U*=orient
    pooled=np.array(pool);delta=pooled[:,0]-pooled[:,1];wi=np.array([weights[i] for i in train]);mu=wi@delta[train];sd=np.sqrt(wi@((delta[train]-mu)**2));effect=mu/np.maximum(sd,1e-12);axes=np.argsort(np.abs(effect))[::-1][:5]
    # Fixed probe C1, no selection on future trajectory or identity evaluation.
    X=np.concatenate([pooled[train,0]@U,pooled[train,1]@U]);y=np.r_[np.ones(len(train)),np.zeros(len(train))];pm=X.mean(0);ps=X.std(0);ps[ps<1e-12]=1
    probe=LogisticRegression(C=1,max_iter=500).fit((X-pm)/ps,y,sample_weight=np.tile(wi,2)*len(train));evaluation={}
    for part in ('train','val','test'):
        ids=np.array([r['source_index'] for r in split if r['split']==part]);xx=np.concatenate([pooled[ids,0]@U,pooled[ids,1]@U]);yy=np.r_[np.ones(len(ids)),np.zeros(len(ids))];score=probe.predict_proba((xx-pm)/ps)[:,1]
        evaluation[part]=dict(n=len(ids),AUROC=float(roc_auc_score(yy,score)),balanced_accuracy_threshold0_5=float(balanced_accuracy_score(yy,score>=.5)))
    np.savez_compressed(HERE/'IDENTYFIKACJA_K500.npz',U500=U,eigenvalues=eig,mean_delta=mean,axes500_0=axes,axes500_1=axes+1,pooled400=pooled.astype(np.float32),probe_mu=pm,probe_sd=ps,probe_coef=probe.coef_,probe_intercept=probe.intercept_)
    record=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),anchor_updates=500,rank_predeclared=8,train_pairs=240,train_films=69,all_pairs_measured=400,
        basis_fit_excludes_val_test=True,axes500=[dict(channel0=int(j),channel1=int(j+1),train_mean_delta=mu[j],train_std_delta=sd[j],effect=effect[j]) for j in axes],identity_evaluation=evaluation,
        checkpoint500_sha256=sha(SOURCE/'wagi/k0500.pt'),model_sha256=sha(HERE/'model_snapshot.py'),manifest_sha256=sha(PAIRS/'manifest.json'),split_sha256=sha(PREVIOUS/'PODZIAL.json'),U500_file_sha256=sha(HERE/'IDENTYFIKACJA_K500.npz'),buffers_unchanged=True,all_finite=True,seconds=time.time()-start,
        batches_observed=False,limitations='same LaMa/knownROI train images; direction calibration causal from anchor500; no future trajectory in fit')
    js(HERE/'IDENTYFIKACJA_K500.json',record);print(record,flush=True)

if __name__=='__main__':main()
