"""Exact MSD and increment correlation from full-768D Gram matrices.
Bootstrap resamples 7 films (one has two images), conditional on one training run.
These are descriptive image/film uncertainty bands, not uncertainty across training seeds.
"""
from pathlib import Path
import json, math, argparse, csv
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation')
OUT=ROOT/'outputs/audyt_szkla_768d_20261010'
FILMS=[[0],[1],[2],[3,4],[5],[6],[7]]
def native(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x))
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=native),encoding='utf-8')
def load(mode):
    fs=[np.load(OUT/f'{mode}_image{i}.npz') for i in range(8)]
    s=fs[0]['steps']; assert all(np.array_equal(s,f['steps']) for f in fs)
    g=np.stack([f['g'] for f in fs]); c=np.stack([f['centroid'] for f in fs]).astype(np.float64)
    c-=c.mean(1,keepdims=True)
    gc=np.einsum('itd,isd->its',c,c,optimize=True)
    avg=c.mean(0); mixed=np.einsum('itd,sd->its',c,avg,optimize=True)
    glob=g-mixed-mixed.transpose(0,2,1)+avg@avg.T
    return s,dict(raw=g,global_relative=glob,image_relative=g-gc,cage=np.stack([f['cage_g'] for f in fs]))
def dist(g,i,j):
    return g[:,i,i]+g[:,j,j]-2*g[:,i,j]
def film_ci(per):
    per=np.asarray(per); rng=np.random.default_rng(20261010)
    picks=rng.integers(0,7,(2000,7)); vals=[]
    for row in picks:
        ids=[i for f in row for i in FILMS[f]];vals.append(per[ids].mean(0))
    return np.quantile(vals,[.025,.975],axis=0)
def slope(t,v,low,hi):
    mask=(t>=low)&(t<=hi)&(v>0)
    return float(np.polyfit(np.log(t[mask]),np.log(v[mask]),1)[0]) if mask.sum()>=3 else None
def timeavg(g,steps,start,end,dt,lags):
    ids=np.flatnonzero((steps>=start)&(steps<=end)&(steps%dt==0))
    st=steps[ids];assert np.all(np.diff(st)==dt)
    curves=[]; counts=[]; used=[]
    for lag in lags:
        if lag%dt or lag>=st[-1]-st[0]:continue
        n=lag//dt;curves.append(dist(g,ids[:-n],ids[n:]).mean(1));counts.append(len(ids)-n);used.append(lag)
    return np.array(used),np.stack(curves,1),np.array(counts)
def vacf(g,steps,start,end,dt,maxlag):
    ids=np.flatnonzero((steps>=start)&(steps<=end)&(steps%dt==0));s=steps[ids]
    assert np.all(np.diff(s)==dt)
    a,b=ids[:-1],ids[1:];energy=dist(g,a,b).mean(1)
    per=[np.ones(8)]
    for lag in range(1,min(maxlag,len(a)-1)+1):
        u,v=a[:-lag],b[:-lag];w,z=a[lag:],b[lag:]
        numerator=(g[:,v,z]-g[:,v,w]-g[:,u,z]+g[:,u,w]).mean(1)
        per.append(numerator/energy)
    per=np.stack(per,1);return dict(lag=np.arange(per.shape[1])*dt,per_image=per,mean=per.mean(0),ci=film_ci(per),step_energy=energy)
def curves(g,s,mode):
    if mode=='dense':
        ages=[1000,3000,6000,10000,15000,19000]; lags=[50,100,150,200,300,500,750,1000,1500,2000,3000,5000,7500,10000,15000,20000];dt=50
    else:
        ages=[1000,3000,6000,10000,15000,19000,24000,30000,40000,50000,60000];lags=[100,200,300,500,700,1000,1500,2000,3000,4000,5000,8000,10000,15000,20000,30000];dt=100
    idx={int(k):i for i,k in enumerate(s)};out=[]
    for age in ages:
        origins=np.arange(age,age+501,dt);vals=[];ls=[]
        for lag in lags:
            if origins[-1]+lag>s[-1]:continue
            if mode=='long' and any(a<22974<a+lag for a in origins):continue
            a=[idx[int(v)] for v in origins];b=[idx[int(v+lag)] for v in origins]
            vals.append(dist(g,a,b).mean(1));ls.append(lag)
        v=np.stack(vals,1)
        out.append(dict(age=age,origin_width=500,origins=len(origins),lag=ls,per_image=v,mean=v.mean(0),ci=film_ci(v),normalized=v/v[:,0,None]))
    return out
def crossings(aging):
    # Same thresholds for all ages, sensitivity from the first age's MSD at lag 1000.
    r=aging[0];ref=r['mean'][list(r['lag']).index(1000)];out=[]
    for factor in [.5,1,2]:
        th=ref*factor;times=[]
        for a in aging:
            t=np.array(a['lag']);v=np.asarray(a['mean']);hit=np.flatnonzero(v>=th)
            tau=None;left=False
            if len(hit):
                k=hit[0]
                if k==0: tau=float(t[0]);left=True
                else: tau=float(np.exp(np.interp(np.log(th),np.log(v[k-1:k+1]),np.log(t[k-1:k+1]))))
            times.append(dict(age=a['age'],time=tau,left_censored=left,right_censored=tau is None,max_lag=int(t[-1])))
        out.append(dict(threshold=float(th),factor=factor,times=times))
    return out
def shape_times(aging):
    # A common normalization horizon, NOT an assumed asymptotic plateau.
    out=[]
    for a in aging:
        t=np.asarray(a['lag']);v=np.asarray(a['mean']);j=list(t).index(2000)
        for frac in [.5,.8]:
            th=frac*v[j];hit=np.flatnonzero((v>=th)&(t<=2000));tau=None;left=False
            if len(hit):
                k=hit[0];left=k==0
                tau=float(t[0]) if left else float(np.exp(np.interp(np.log(th),np.log(v[k-1:k+1]),np.log(t[k-1:k+1]))))
            out.append(dict(age=a['age'],fraction=frac,horizon=2000,time=tau,left_censored=left))
    return out
def model_fits(t,y):
    # Descriptive fit on log MSD; holdout alternating lags. No AIC on correlated lag averages.
    t=np.asarray(t,dtype=float); y=np.asarray(y);out={}
    funcs={
        'OU':(lambda p:p[0]*(1-np.exp(-t/p[1])),[y.max(),50],[1e-10,.01],[y.max()*100,1e7]),
        'OU_plus_diffusion':(lambda p:p[0]*(1-np.exp(-t/p[1]))+p[2]*t,[y.max()*.7,30,y.max()/t.max()*.3],[1e-10,.01,0],[y.max()*100,1e7,y.max()]),
        'two_OU':(lambda p:p[0]*(1-np.exp(-t/p[1]))+p[2]*(1-np.exp(-t/p[3])),[y.max()*.6,20,y.max()*.4,1000],[1e-10,.01,1e-10,.01],[y.max()*100,1e7,y.max()*100,1e7]),
        'colored_OU_plus_diffusion':(lambda p:p[0]*(1-((p[1]+p[2])*np.exp(-t/(p[1]+p[2]))-p[1]*np.exp(-t/p[1]))/p[2])+p[3]*t,
                                    [y.max()*.7,1,20,y.max()/t.max()*.3],[1e-10,.001,.001,0],[y.max()*100,1e6,1e7,y.max()])}
    for name,(fn,p,lo,hi) in funcs.items():
        mask=np.arange(len(t))%2==0
        fit=least_squares(lambda z:(np.log(np.maximum(fn(z),1e-30))-np.log(y))[mask],p,bounds=(lo,hi),max_nfev=3000)
        pred=fn(fit.x);error=np.log(pred)-np.log(y)
        out[name]=dict(params=fit.x,prediction=pred,train_log_rmse=float(np.sqrt(np.mean(error[mask]**2))),heldout_log_rmse=float(np.sqrt(np.mean(error[~mask]**2))))
    return out
def plateau_scan(t,y):
    # All resolvable full decades; no post-hoc hand selection of a narrow flat segment.
    scans=[]
    for start in t:
        if start*10>t[-1]:continue
        m=(t>=start)&(t<=10*start)
        if m.sum()<4:continue
        al=slope(t,y,start,start*10)
        scans.append(dict(start=int(start),end=int(start*10),alpha=al,msd_max_min=float(y[m].max()/y[m].min()),
                          passes_02=bool(abs(al)<=.2 and y[m].max()/y[m].min()<=10**.2),
                          passes_035=bool(abs(al)<=.35 and y[m].max()/y[m].min()<=10**.35)))
    return scans

def run(mode):
    s,gs=load(mode);results={};print('analyzing',mode,flush=True)
    for name,g in gs.items():
        # A max lag <= half each window avoids near-empty averages.
        if mode=='dense': windows=[('warmup',0,300,1,150),('early',300,1094,1,350),('late1',1000,6000,50,2000),('late2',10000,15000,50,2000),('late3',17500,22500,50,2000),('late_all',1000,22950,50,10000)]
        else:windows=[('early',300,1000,10,300),('late1',1000,6000,100,2000),('late2',10000,15000,100,2000),('late3',17500,22500,100,2000),('postresume',25000,40000,100,6000),('end',50000,65000,100,6000),('postresume_all',24000,65600,100,20000)]
        wout={}
        for wn,lo,hi,dt,mx in windows:
            lags=np.unique(np.r_[np.arange(1,min(20,mx//dt)+1)*dt,np.round(np.geomspace(dt,mx,50)/dt).astype(int)*dt])
            t,v,cnt=timeavg(g,s,lo,hi,dt,lags);mean=v.mean(0)
            band_slopes=[]
            for a,b in [(1,5),(5,20),(10,100),(35,80),(50,500),(100,1000),(500,2000),(500,5000)]:
                if a<t.min() or b>t.max():continue
                ps=np.array([slope(t,z,a,b) for z in v]);band_slopes.append(dict(low=a,high=b,mean=slope(t,mean,a,b),per_image=ps,ci=film_ci(ps)))
            wout[wn]=dict(start=lo,end=hi,dt=dt,lag=t,per_image=v,mean=mean,ci=film_ci(v),counts=cnt,
                         local_alpha=np.gradient(np.log(mean),np.log(t)),slopes=band_slopes,plateau_scan=plateau_scan(t,mean),vacf=vacf(g,s,lo,hi,dt,40),fits=model_fits(t,mean))
        # Early matched-age dense curves, identical origin widths and lag support.
        short=[]
        if mode=='dense':
            idx={int(k):i for i,k in enumerate(s)}
            lags=np.unique(np.r_[np.arange(1,21),np.round(np.geomspace(20,300,35)).astype(int)])
            for age in [300,500,700]:
                starts=np.arange(age,age+95);v=np.stack([dist(g,[idx[k] for k in starts],[idx[k+lag] for k in starts]).mean(1) for lag in lags],1)
                short.append(dict(age=age,lag=lags,per_image=v,mean=v.mean(0),normalized=v/v[:,0,None]))
        ages=curves(g,s,mode)
        trend=[]
        for tau in ([50,100,500,1000,2000] if mode=='dense' else [100,500,1000,2000,5000]):
            for label,amin,amax in [('all',0,70000),('late',6000,22000),('postresume',24000,70000)]:
                selected=[a for a in ages if amin<=a['age']<=amax and tau in a['lag']]
                if len(selected)<3:continue
                ag=np.array([a['age'] for a in selected]);v=np.stack([a['per_image'][:,a['lag'].index(tau)] for a in selected],1)
                betas=np.array([np.polyfit(np.log(ag),np.log(z),1)[0] for z in v])
                trend.append(dict(tau=tau,period=label,ages=ag,amplitude_exponent=float(np.polyfit(np.log(ag),np.log(v.mean(0)),1)[0]),per_image=betas,ci=film_ci(betas),ratio_last_first=float(v.mean(0)[-1]/v.mean(0)[0])))
        results[name]=dict(windows=wout,aging=ages,short_aging=short,thresholds=crossings(ages),shape_times=shape_times(ages),amplitude_trends=trend)
    save(f'{mode}_metrics.json',results)
    plot(mode,results)
    print('finished',mode,flush=True)

def plot(mode,r):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(16,9))
    title={'raw':'raw 768D','global_relative':'global drift removed','image_relative':'image drift removed','cage':'6-neighbor relative (neighbors at step 300)'}
    for name,col in [('raw','#0072B2'),('image_relative','#009E73'),('cage','#D55E00')]:
        w=r[name]['windows']['early'];ax[0,0].loglog(w['lag'],w['mean'],label=title[name],color=col)
        v=w['vacf'];ax[0,1].plot(v['lag'],v['mean'],label=title[name],color=col)
        ax[0,2].semilogx(w['lag'],w['local_alpha'],label=title[name],color=col)
    for a in r['cage']['aging']:
        ax[1,0].loglog(a['lag'],a['mean'],label=f"age {a['age']}")
        ax[1,1].semilogx(a['lag'],a['mean']/a['mean'][0],label=f"age {a['age']}")
    for name,col in [('raw','#0072B2'),('image_relative','#009E73'),('cage','#D55E00')]:
        ages=r[name]['aging'];x=[a['age'] for a in ages];y=[a['mean'][a['lag'].index(1000)] for a in ages]
        ax[1,2].loglog(x,y,'o-',label=title[name],color=col)
    ax[0,0].set(title='MSD after warmup: early training',xlabel='Lag [optimizer steps]',ylabel='Mean squared displacement [activation units²]')
    ax[0,1].set(title='Increment autocorrelation',xlabel='Lag [optimizer steps]',ylabel='VACF');ax[0,1].axhline(0,color='gray',lw=1)
    ax[0,2].set(title='Local log-slope of MSD',xlabel='Lag [optimizer steps]',ylabel='alpha');ax[0,2].axhline(.2,color='gray',ls='--');ax[0,2].axhline(1,color='gray',ls=':')
    ax[1,0].set(title='Two-time MSD, fixed-width origin windows',xlabel='Lag [optimizer steps]',ylabel='Cage-relative MSD')
    ax[1,1].set(title='Shape after dividing by shortest-lag MSD',xlabel='Lag [optimizer steps]',ylabel='MSD / shortest-lag MSD')
    ax[1,2].set(title='Motion at fixed lag = 1000 steps',xlabel='Age [optimizer steps]',ylabel='MSD')
    for a in ax.flat: a.grid(alpha=.2);a.legend(fontsize=7)
    fig.suptitle(f'Full 768D audit | {mode} run | 8 images, 4800 tokens | warmup ends at step 300',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/f'{mode}_diagnostics.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dense','long']);a=p.parse_args();run(a.mode)
