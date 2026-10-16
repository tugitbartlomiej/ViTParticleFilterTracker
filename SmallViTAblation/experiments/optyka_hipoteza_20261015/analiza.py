"""Chronological out-of-sample tests; no fitting on test or third image."""
from pathlib import Path
import json, time, hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/optyka_hipoteza_20261015'

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def interval(e0,e1,seed=1510,block=20,nrep=1000):
    # e0/e1 are one mean squared error per checkpoint, not independent tokens.
    rng=np.random.default_rng(seed);n=len(e0)
    starts=rng.integers(0,n,size=(nrep,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(nrep,-1)[:,:n]
    gains=100*(1-e1[idx].mean(1)/np.maximum(e0[idx].mean(1),1e-30))
    return [float(v) for v in np.quantile(gains,[.025,.975])]

def phase_probe(y,steps,cut1,cut2,levels=False):
    """AR of increments; explicitly tests the false phase caused by differencing noise."""
    d=y if levels else np.diff(y,axis=0)
    cur=np.arange(1,len(d)-1)
    a=d[cur]; b=d[cur-1]; target=d[cur+1]
    # Target increment ends at state cur+2.
    train=cur+(1 if levels else 2)<cut1;test=cur+(0 if levels else 1)>=cut2
    predictions={n:np.zeros_like(target) for n in ['zero','ar1','ar2','oscillator']}
    if levels: predictions['zero']=a.copy()  # zero change = persistence of level
    fit=[]
    for role in range(3):
        sl=slice(8*role,8*(role+1))
        for channel in range(5):
            x1=a[train,:2,sl,channel].reshape(-1);x2=b[train,:2,sl,channel].reshape(-1)
            yt=target[train,:2,sl,channel].reshape(-1)
            beta1=np.linalg.lstsq(np.c_[np.ones_like(x1),x1],yt,rcond=None)[0]
            beta2=np.linalg.lstsq(np.c_[np.ones_like(x1),x1,x2],yt,rcond=None)[0]
            roots=np.roots([1,-beta2[1],-beta2[2]])
            complex_pair=bool(np.max(np.abs(roots.imag))>1e-8)
            stable=bool(np.max(np.abs(roots))<1)
            period=float(2*np.pi/abs(np.angle(roots[0]))*100) if complex_pair else None
            # Fit stable damped oscillator directly by training SSE.
            best=(float('inf'),None)
            for radius in [.2,.4,.6,.8,.9,.95,.98,.995]:
                for period_samples in [3,4,5,6,8,10,15,20,30,50,80,120,200]:
                    aa=2*radius*np.cos(2*np.pi/period_samples);bb=-radius**2
                    intercept=float(np.mean(yt-aa*x1-bb*x2))
                    loss=float(np.mean((yt-intercept-aa*x1-bb*x2)**2))
                    if loss<best[0]:best=(loss,(intercept,aa,bb,radius,period_samples))
            cc,aa,bb,rad,per=best[1]
            aa_all=a[:,:,sl,channel];bb_all=b[:,:,sl,channel]
            predictions['ar1'][:,:,sl,channel]=beta1[0]+beta1[1]*aa_all
            predictions['ar2'][:,:,sl,channel]=beta2[0]+beta2[1]*aa_all+beta2[2]*bb_all
            predictions['oscillator'][:,:,sl,channel]=cc+aa*aa_all+bb*bb_all
            fit.append(dict(role=role,channel=channel,beta_ar2=beta2.tolist(),complex=complex_pair,
                            stable=stable,period_steps=period,radius=float(np.max(np.abs(roots))),
                            oscillator_radius=rad,oscillator_period_steps=per*100))
    evaluation=[]
    for image in range(3):
        for group,ch,scale in [('logit',slice(0,1),np.array([1.])),('box',slice(1,5),np.array([480,320,480,320]))]:
            obs=target[test,image,0,ch]
            errors={n:np.mean(((p[test,image,0,ch]-obs)*scale)**2,axis=-1) for n,p in predictions.items()}
            for n,e in errors.items():
                evaluation.append(dict(image=image,group=group,model=n,rmse=float(np.sqrt(e.mean())),
                    gain_vs_ar1_pct=float(100*(1-e.mean()/errors['ar1'].mean())),
                    ci_vs_ar1_pct=interval(errors['ar1'],e),
                    gain_vs_ar2_pct=float(100*(1-e.mean()/errors['ar2'].mean())),
                    ci_vs_ar2_pct=interval(errors['ar2'],e)))
    # Independent white position noise, differenced exactly like the observations.
    # This is a diagnostic null, not a calibrated significance test for the ViT.
    rng=np.random.default_rng(2510);null=[]
    for rep in range(200):
        q=rng.normal(size=(cut1,16));inc=q if levels else np.diff(q,axis=0)
        be=np.linalg.lstsq(np.c_[np.ones((len(inc)-2)*16),inc[1:-1].ravel(),inc[:-2].ravel()],inc[2:].ravel(),rcond=None)[0]
        rr=np.roots([1,-be[1],-be[2]])
        cc=bool(np.max(np.abs(rr.imag))>1e-8)
        null.append(dict(complex=cc,stable=bool(np.max(np.abs(rr))<1),radius=float(np.max(np.abs(rr))),
                         period_steps=float(2*np.pi/abs(np.angle(rr[0]))*100) if cc else None))
    return dict(fits=fit,evaluation=evaluation,white_position_noise_null=null,levels=levels,
                note='Raw readout levels' if levels else 'AR models describe increments, not optical phases; differencing independent position noise can create complex AR poles.')

def main():
    start=time.time();d=np.load(OUT/'pomiar.npz');steps=d['steps'];F=d['features'].astype(float)
    assert all(np.isfinite(d[k]).all() for k in ['features','x_post','u','y'])
    fidx={n:i for i,n in enumerate(d['feature_names'])}
    descriptive=[]
    windows=[('initial',steps==0),('early', (steps>=1000)&(steps<10000)),
             ('middle',(steps>=20000)&(steps<40000)),('late',steps>=55000)]
    for label,mask in windows:
        for im in range(3):
            for role in range(3):
                row=dict(window=label,image=im,role=role,n_checkpoints=int(mask.sum()))
                for key in ['ratio_power','ratio_length','effective_channels','ratio_logit','ratio_box']:
                    vals=F[mask,im,role*8:(role+1)*8,fidx[key]].ravel()
                    row[key]=dict(median=float(np.median(vals)),mean=float(np.mean(vals)),
                                  q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75)),
                                  fraction_below_one=float(np.mean(vals<1)))
                descriptive.append(row)
    dump('opis_superpozycji.json',descriptive)
    # Uniform interval; exclude extra epoch-end snapshots rather than resample.
    sel=(steps>=1100)&(steps%100==0);ss=steps[sel]
    assert np.all(np.diff(ss)==100)
    x=d['x_post'][sel].astype(float);u=d['u'][sel].astype(float);y=d['y'][sel].astype(float);f=F[sel]
    T=len(ss);cut1=int(T*.5);cut2=int(T*.65);cur=np.arange(1,T-1)
    dx=x[1:]-x[:-1];dy=y[1:]-y[:-1]
    rel=np.linalg.norm(dx,axis=-1)/np.maximum(np.linalg.norm(x[:-1],axis=-1),1e-12)
    target=np.concatenate([rel[1:,...,None],dy[1:]],axis=-1)
    shape=target.shape[:-1];timeid=np.broadcast_to(cur[:,None,None],shape)
    imageid=np.broadcast_to(np.arange(3)[None,:,None],shape)
    tokenid=np.broadcast_to(np.arange(24)[None,None,:],shape)
    train=(timeid+1<cut1)&(imageid<2)
    val=(timeid>=cut1)&(timeid+1<cut2)&(imageid<2)
    test=(timeid>=cut2)
    # Every preprocessing fit uses exactly the same training mask.
    normidx=[0,1,2,3,4,14]
    norm=np.log1p(f[1:-1,...,normidx])
    timing=np.broadcast_to(np.stack([ss[1:-1]/10000,np.log1p(ss[1:-1])],-1)[:,None,None,:],shape+(2,))
    xy=np.stack([(d['ids']%30)/29,(d['ids']//30)/19],-1)
    xy=np.broadcast_to(xy[None],shape+(2,))
    rolehot=np.eye(3)[np.arange(24)//8];rolehot=np.broadcast_to(rolehot[None,None],shape+(3,))
    past=dx[:-1];U=u[1:-1]
    cos=np.sum(past*U,axis=-1)/(np.linalg.norm(past,axis=-1)*np.linalg.norm(U,axis=-1)+1e-20)
    base=np.concatenate([timing,norm,y[1:-1],dy[:-1],rel[:-1,...,None],cos[...,None],xy,rolehot],-1)
    pcameta={}
    for name,arr in [('state',x[1:-1]),('push',u[1:-1])]:
        pca=PCA(n_components=8,svd_solver='randomized',random_state=1510)
        pca.fit(arr[train]);pcs=pca.transform(arr.reshape(-1,192)).reshape(shape+(8,))
        base=np.concatenate([base,pcs],-1);pcameta[name]=float(pca.explained_variance_ratio_.sum())
    optical=np.log(np.maximum(f[1:-1,...,5:14],1e-12))
    shuffled=optical.copy();rng=np.random.default_rng(1510)
    for t in range(len(cur)):
        for im in range(3):
            for role in range(3):
                j=np.arange(role*8,(role+1)*8)
                shuffled[t,im,j]=optical[t,im,rng.permutation(j)]
    sets={'base':base,'optical':np.concatenate([base,optical],-1),
          'shuffled_optical':np.concatenate([base,shuffled],-1)}
    ys=StandardScaler().fit(target[train]);yt=ys.transform(target.reshape(-1,6)).reshape(target.shape)
    preds={};selection=[]
    naive=np.zeros_like(target);naive[...,0]=rel[:-1]
    preds['naive']=naive
    for kind,features in sets.items():
        scaler=StandardScaler().fit(features[train]);A=scaler.transform(features.reshape(-1,features.shape[-1]))
        tr=train.ravel();va=val.ravel();YY=yt.reshape(-1,6)
        best=(np.inf,None,None)
        for alpha in [.1,1,10,100,1000]:
            model=Ridge(alpha=alpha).fit(A[tr],YY[tr])
            loss=float(np.mean((model.predict(A[va])-YY[va])**2))
            if loss<best[0]:best=(loss,alpha,model)
        preds['ridge_'+kind]=ys.inverse_transform(best[2].predict(A)).reshape(target.shape)
        selection.append(dict(model='ridge_'+kind,alpha=best[1],validation_mse_standardized=best[0]))
        model=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,
             l2_regularization=10,learning_rate=.06,early_stopping=False,random_state=1510))
        model.fit(A[tr],YY[tr]);pred=model.predict(A)
        preds['hgb_'+kind]=ys.inverse_transform(pred).reshape(target.shape)
        selection.append(dict(model='hgb_'+kind,validation_mse_standardized=float(np.mean((pred[va]-YY[va])**2))))
        print('fit',kind,'elapsed',round(time.time()-start,1),flush=True)
    evaluation=[]
    specifications=[('latent_all',slice(0,1),np.array([1.]),slice(None)),
                    ('logit_center',slice(1,2),np.array([1.]),slice(0,1)),
                    ('box_center',slice(2,6),np.array([480,320,480,320]),slice(0,1))]
    tidx=cur>=cut2
    for im in range(3):
        for label,ch,scale,tokens in specifications:
            errors={name:np.mean(((p[tidx,im,tokens,ch]-target[tidx,im,tokens,ch])*scale)**2,axis=(-1,-2)) for name,p in preds.items()}
            for name,e in errors.items():
                ref=name.split('_')[0]+'_base' if name!='naive' else 'naive'
                eb=errors[ref]
                evaluation.append(dict(image=im,group=label,model=name,n_times=len(e),
                    rmse=float(np.sqrt(e.mean())),gain_vs_base_pct=float(100*(1-e.mean()/eb.mean())),
                    ci_vs_base_pct=interval(eb,e),gain_vs_naive_pct=float(100*(1-e.mean()/errors['naive'].mean()))))
    phase=phase_probe(y,ss,cut1,cut2)
    dump('faza.json',phase)
    dump('faza_poziomy.json',phase_probe(y,ss,cut1,cut2,levels=True))
    result=dict(regular_steps=len(ss),start=int(ss[0]),end=int(ss[-1]),
        split=dict(train_target_before=int(ss[cut1]),validation_inputs_from=int(ss[cut1]),
                   test_inputs_from=int(ss[cut2]),train_rows=int(train.sum()),validation_rows=int(val.sum()),
                   test_rows=int(test.sum()),third_image_never_fit=True),
        base_feature_count=base.shape[-1],added_feature_count=optical.shape[-1],pca=pcameta,
        selection=selection,evaluation=evaluation,bootstrap=dict(type='circular moving block of checkpoint means',block=20,reps=1000),
        limitations=['One training run and three images; image identities are not independent training seeds.',
          'Forecasting existing detector outputs does not improve detection accuracy.',
          'Phase probe fits increments; differencing observation noise can create spurious complex poles.',
          'Optical ratios of P_net/P_diag are deterministic transforms already available to the baseline.',
          'Reported bootstrap intervals are exploratory and conditional, not corrected for multiple comparisons.'])
    dump('prognozy.json',result)
    np.savez_compressed(OUT/'prognozy.npz',steps=ss[1:-1],target=target,train=train,val=val,test=test,**preds)
    # Figures show all three images, no cherry-picked result.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    colors=['#0072B2','#D55E00','#009E73']
    for im in range(3):
        for role in range(3):
            median=np.median(F[:,im,role*8:(role+1)*8,5],axis=1)
            axes[0,im].plot(steps,median,color=colors[role],label=['środek','brzeg','tło'][role],lw=1)
        axes[0,im].axhline(1,color='black',ls='--',lw=.8)
        axes[0,im].set(yscale='log',title=f'Obraz {im+1}: wypadkowa superpozycja',ylabel='P_net / P_diag',xlabel='Krok uczenia')
        axes[0,im].legend(fontsize=8)
        rows=[r for r in evaluation if r['image']==im and r['model']=='hgb_optical']
        gains=[r['gain_vs_base_pct'] for r in rows];cis=np.array([r['ci_vs_base_pct'] for r in rows])
        axes[1,im].bar(np.arange(3),gains,color=colors)
        # Percentile CI need not contain point estimate; draw independently.
        for j in range(3):axes[1,im].plot([j,j],cis[j],color='black',lw=1.5)
        axes[1,im].axhline(0,color='black',lw=.8)
        axes[1,im].set(xticks=np.arange(3),xticklabels=['ruch latentu','logit środka','ramka środka'],
            ylabel='Zmniejszenie MSE względem baseline [%]',title='Test przyszłości'+(' i nowego obrazu' if im==2 else ''))
    fig.suptitle('Analogia światła: opis superpozycji i niezależny test prognozy\nPasmo: bootstrap bloków czasu; wartości ujemne oznaczają pogorszenie',fontsize=14)
    fig.savefig(OUT/'superpozycja_i_prognoza.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    periods=[r['period_steps'] for r in phase['fits'] if r['complex']]
    nullperiods=[r['period_steps'] for r in phase['white_position_noise_null'] if r['complex']]
    axes[0].hist(nullperiods,bins=15,alpha=.5,label='Różnice niezależnego szumu położeń',density=True)
    if periods:axes[0].hist(periods,bins=10,alpha=.6,label='Różnice odczytów ViT',density=True)
    axes[0].set(xlabel='Okres z biegunów AR(2) [kroki]',ylabel='Gęstość',title='Zespolone bieguny mogą powstać po różnicowaniu szumu')
    axes[0].legend(fontsize=8)
    for i,group in enumerate(['logit','box']):
        rr=[r for r in phase['evaluation'] if r['image']==2 and r['group']==group and r['model'] in ['ar2','oscillator']]
        for j,r in enumerate(rr):axes[1].bar(i*3+j,r['gain_vs_ar1_pct'],color=colors[j])
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(xticks=[0,1,3,4],xticklabels=['logit AR2','logit osc.','ramka AR2','ramka osc.'],
        ylabel='Zmniejszenie MSE względem AR1 [%]',title='Trzeci obraz, późniejsze checkpointy')
    fig.savefig(OUT/'faza_i_kontrola.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(seconds=time.time()-start,split=result['split'],evaluation_image3=[r for r in evaluation if r['image']==2]),ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
