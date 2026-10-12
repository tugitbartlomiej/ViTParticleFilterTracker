"""Collective POD/DMDc on recorded SmallViT token trajectories, with real batch forces.

No network training is performed. GPU: block Gram matrices and mode projection.
PyDMD and regularized regression run on CPU. All model selection is chronological.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse, csv, hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
torch = None
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]

def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if isinstance(o,np.ndarray) else float(o))+'\n', encoding='utf8')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def blocks(Z,S,lo,hi,block):
    for f in range(Z.shape[1]):
        for j in range(0,600,block):
            z=np.array(Z[lo:hi+1,f,j:j+block],dtype=np.float32).reshape(hi-lo+1,-1)
            ss=np.array(S[lo:hi+1,f,j:j+block],dtype=np.float32)
            s=ss.reshape(len(ss),-1)
            # Final state has no outgoing force; pad zero only beyond final recorded input.
            if len(s)<len(z): s=np.concatenate([s,np.zeros((len(z)-len(s),s.shape[1]),np.float32)])
            yield f,j,z,s

def prepare(a,out):
    t0=time.perf_counter(); dev='cuda'
    Z=np.load(a.source/'Z.npy',mmap_mode='r'); S=np.load(a.source/'S_nowa.npy',mmap_mode='r')
    meta=json.loads((a.source/'metadata.json').read_text(encoding='utf8'))
    assert Z.shape==(1001,3,600,768) and S.shape==(1000,3,600,768)
    assert meta['status']=='complete' and meta['seed']==47 and meta['klatki']==[2,3,6]
    assert a.start>=meta['hyper']['rozgrzewka_lr']
    masks=np.load(ROOT/'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/maski.npz')
    tool=masks['tool'][meta['klatki']]; back=masks['background'][meta['klatki']]
    assert np.array_equal(tool,np.asarray(meta['narzedzie']))
    lo,hi=a.start-1,a.end; T=hi-lo+1; nt=a.train_end-lo+1
    source_shape=list(Z.shape);source_dtype=str(Z.dtype)
    # Contiguous float16 read avoids thousands of strided disk seeks. Keep plenty
    # of RAM for the desktop and other processes; otherwise retain mmap fallback.
    import psutil
    cache_bytes=(T+min(T,len(S)-lo))*3*600*768*2
    if psutil.virtual_memory().available>cache_bytes+5*2**30:
        Z=np.array(Z[lo:hi+1],copy=True);S=np.array(S[lo:hi+1],copy=True)
        read_lo,read_hi=0,T-1
        print(f'Contiguous float16 source cache {cache_bytes/2**30:.2f}GiB ready',flush=True)
    else:read_lo,read_hi=lo,hi
    gr={g:{k:np.zeros((T,T),np.float64) for k in ['xx','ss','xs']} for g in ['all','tool','background']}
    quant_bound2=0.; sample_d2=0.; count=0
    check_pairs=[(1,2),(nt-2,nt-1),(T-11,T-1),(T-2,T-1)]
    check_norms={g:np.zeros(len(check_pairs)) for g in gr}
    for f,j,z,s in blocks(Z,S,read_lo,read_hi,a.block):
        assert np.isfinite(z).all() and np.isfinite(s).all()
        zt=torch.from_numpy(z).to(dev); st=torch.from_numpy(s).to(dev)
        zt-=zt[1:nt].mean(0,keepdim=True)
        for g,m in [('all',np.ones(z.shape[1]//768,bool)),('tool',tool[f,j:j+a.block]),('background',back[f,j:j+a.block])]:
            if not m.any(): continue
            ix=np.repeat(m,768); x=zt[:,ix]; u=st[:,ix]
            for ci,(ta,tb) in enumerate(check_pairs):
                check_norms[g][ci]+=np.square(z[tb,ix].astype(np.float64)-z[ta,ix],dtype=np.float64).sum()
            gr[g]['xx']+=(x@x.T).double().cpu().numpy()
            gr[g]['ss']+=(u@u.T).double().cpu().numpy()
            gr[g]['xs']+=(x@u.T).double().cpu().numpy()
        # Rigorous upper envelope from half-ULP rounding of each stored state.
        z16=z.astype(np.float16)
        spacing=np.abs(np.spacing(np.abs(z16))).astype(np.float32)
        bnd=(spacing[1:]+spacing[:-1])*.5
        quant_bound2+=float(np.square(bnd,dtype=np.float64).sum())
        sample_d2+=float(np.square(np.diff(z,axis=0),dtype=np.float64).sum())
        count+=1
        if count%10==0: print(f'Gram blocks {count}, seconds {time.perf_counter()-t0:.1f}',flush=True)
        del zt,st
    train=np.arange(1,nt)
    def eig(G):
        ev,V=np.linalg.eigh((G+G.T)*.5); ix=np.argsort(ev)[::-1]
        return np.maximum(ev[ix],0),V[:,ix]
    ev,V=eig(gr['all']['xx'][np.ix_(train,train)])
    r=min(a.max_rank,len(train)-2); C=V[:,:r]/np.sqrt(np.maximum(ev[:r],1e-12))
    coeff=gr['all']['xx'][:,train]@C
    # Force POD is learned only from train transitions, centered by train input mean.
    it=np.arange(1,nt-1)
    H=np.eye(len(it))-np.ones((len(it),len(it)))/len(it)
    ef,Vf=eig(H@gr['all']['ss'][np.ix_(it,it)]@H)
    Cf=H@Vf[:,:r]/np.sqrt(np.maximum(ef[:r],1e-12))
    force=gr['all']['ss'][:,it]@Cf
    force-=force[it].mean(0)
    proj={}
    for g,gg in gr.items():
        proj[g]={'xx':gg['xx'],'ss':gg['ss'],'xs':gg['xs'],
                 'xc':gg['xx'][:,train]@C,'cc':C.T@gg['xx'][np.ix_(train,train)]@C,
                 'sc':gg['xs'][train,:].T@C}
    data=dict(coeff=coeff,force=force,ev=ev,ef=ef,C=C,train=train,proj=proj,lo=lo,hi=hi,nt=nt,meta=meta,
              groups_n={'all':1800,'tool':int(tool.sum()),'background':int(back.sum())},
              quantization_rms_upper_over_motion_rms=np.sqrt(quant_bound2/max(sample_d2,1e-30)))
    import pickle
    with open(out/'prepared.pkl','wb') as f: pickle.dump(data,f,protocol=5)
    numerical_errors={}
    for g in gr:
        check_gram=np.array([gr[g]['xx'][ta,ta]+gr[g]['xx'][tb,tb]-2*gr[g]['xx'][ta,tb] for ta,tb in check_pairs])
        numerical_errors[g]=float(np.max(np.abs(check_gram-check_norms[g])/np.maximum(check_norms[g],1e-30)))
    numerical_error=max(numerical_errors.values())
    assert numerical_error<.002, f'Gram numerical error {numerical_error}'
    write(out/'PREPARE.json',{'seconds':time.perf_counter()-t0,'shape':source_shape,'source_dtype':source_dtype,
          'range':[lo,hi],'cuda_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
          'cuda_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
          'quantization_rms_upper_over_motion_rms':data['quantization_rms_upper_over_motion_rms'],
          'direct_float64_vs_Gram_relative_error_max':numerical_error,
          'direct_float64_vs_Gram_by_group':numerical_errors,
          'POD_train_explained_variance':{str(k):float(ev[:k].sum()/ev.sum()) for k in [4,8,16,32,64] if k<=r},
          'force_POD_train_explained_variance':{str(k):float(ef[:k].sum()/ef.sum()) for k in [4,8,16,32,64] if k<=r}})
    # Save full-dimensional collective modes as 3 x 600 x 768 arrays.
    modes=np.lib.format.open_memmap(out/'POD_MODES.npy',mode='w+',dtype=np.float32,shape=(r,3,600,768))
    Ct=torch.from_numpy(C.T.astype(np.float32)).cuda();frameproj=np.zeros((T,3,r),np.float64)
    for f,j,z,s in blocks(Z,S,read_lo,read_hi,a.block):
        xt=torch.from_numpy(z).cuda(); xt-=xt[1:nt].mean(0,keepdim=True)
        mode=Ct@xt[train]
        modes[:,f,j:j+a.block]=mode.cpu().numpy().reshape(r,-1,768)
        frameproj[:,f]+=(xt@mode.T).double().cpu().numpy()
    modes.flush(); del modes,Ct
    np.save(out/'FRAME_MODE_CONTRIBUTIONS.npy',frameproj)
    corr=[]
    for k in range(min(8,r)):
        corr.append({'mode':k+1,'train_corr':np.corrcoef(frameproj[1:nt,:,k].T),
                     'test_corr':np.corrcoef(frameproj[a.val_end-lo:,:,k].T)})
    write(out/'COHERENCE.json',corr)
    return data

def fit_ridge(y,u,ti,r,q,lag,ulags,lam,kind):
    y=y[:,:r]; sd=y[ti].std(0); sd=np.maximum(sd,1e-8); yy=y/sd
    uu=u[:,:q]; us=uu[ti].std(0); us=np.maximum(us,1e-8); uu=uu/us
    def design(t):
        parts=[yy[t]]
        if lag: parts+=[yy[t]-yy[t-1]]
        if q:
            parts+=[uu[t]]
            if ulags: parts+=[uu[t-1]]
        return np.concatenate(parts+[np.ones((len(t),1))],axis=1)
    X=design(ti); target=yy[ti+1]-yy[ti]
    P=np.eye(X.shape[1]); P[-1,-1]=0
    W=np.linalg.solve(X.T@X/len(ti)+lam*P+1e-12*np.eye(X.shape[1]),X.T@target/len(ti))
    return dict(kind=kind,r=r,q=q,lag=lag,ulags=ulags,lam=lam,W=W,sd=sd,us=us)

def predict(m,y,u,ts,h):
    r=m['r']; cur=y[ts,:r]/m['sd']; prev=y[ts-1,:r]/m['sd']; start=cur.copy()
    for k in range(h):
        if m['kind']=='PyDMDc':
            inp=np.concatenate([u[ts+k,:m['q']]/m['us'],np.ones((len(ts),1))],axis=1)
            nxt=cur@m['A'].T+inp@m['B'].T
        elif m['kind']=='PyDMD': nxt=cur@m['A'].T
        else:
            p=[cur]
            if m['lag']: p+=[cur-prev]
            if m['q']:
                p+=[u[ts+k,:m['q']]/m['us']]
                if m['ulags']:p+=[u[ts+k-1,:m['q']]/m['us']]
            nxt=cur+np.concatenate(p+[np.ones((len(ts),1))],axis=1)@m['W']
        prev,cur=cur,nxt
    return (cur-start)*m['sd']

def metrics_projection(p,d,ts,h,nfeatures):
    xx=p['xx']; r=d.shape[1]; gt=p['xc'][ts+h,:r]-p['xc'][ts,:r]
    true=np.maximum(xx[ts+h,ts+h]+xx[ts,ts]-2*xx[ts+h,ts],0)
    pred=np.einsum('ni,ij,nj->n',d,p['cc'][:r,:r],d); dot=(d*gt).sum(1)
    err=np.maximum(true+pred-2*dot,0)
    return summary(err,true,pred,dot,nfeatures)

def summary(err,true,pred,dot,nfeatures):
    cos=dot/np.sqrt(np.maximum(true*pred,1e-30))
    valid=pred>1e-20
    return {'rmse':float(np.sqrt(err.sum()/(len(err)*nfeatures))),
            'skill':float(1-err.sum()/max(true.sum(),1e-30)),
            'sse':float(err.sum()),'persistence_sse':float(true.sum()),
            'cos_motion_mean':float(cos[valid].mean()) if valid.any() else None,'n_origins':len(err)}

def metrics_baseline(p,ts,h,nfeatures,kind,beta=0.,b=1.):
    xx,ss,xs=p['xx'],p['ss'],p['xs']
    true=np.maximum(xx[ts+h,ts+h]+xx[ts,ts]-2*xx[ts+h,ts],0)
    if kind=='persistence': return summary(true,true,np.zeros_like(true),np.zeros_like(true),nfeatures)
    if kind=='velocity':
        vv=xx[ts,ts]+xx[ts-1,ts-1]-2*xx[ts,ts-1]
        dot=h*(xx[ts+h,ts]-xx[ts+h,ts-1]-xx[ts,ts]+xx[ts,ts-1])
        pred=h*h*vv
    else:
        weights=np.array([b*sum(beta**j for j in range(h-i)) for i in range(h)])
        av=sum(beta**j for j in range(1,h+1))
        si=ts[:,None]+np.arange(h)[None]
        pred=np.einsum('i,nij,j->n',weights,ss[si[:,:,None],si[:,None,:]],weights)
        dot=(xs[(ts+h)[:,None],si]-xs[ts[:,None],si])@weights
        if av:
            vv=xx[ts,ts]+xx[ts-1,ts-1]-2*xx[ts,ts-1]
            vs=(xs[ts[:,None],si]-xs[(ts-1)[:,None],si])@weights
            pred+=av*av*vv+2*av*vs
            dot+=av*(xx[ts+h,ts]-xx[ts+h,ts-1]-xx[ts,ts]+xx[ts,ts-1])
    return summary(np.maximum(true+pred-2*dot,0),true,np.maximum(pred,0),dot,nfeatures)

def run_models(a,out,d):
    # Use unchanged installed PyDMD modules without importing every optional family
    # from its eager package __init__ (very slow filesystem imports on this host).
    import importlib.util, importlib, types
    if 'pydmd' not in sys.modules:
        spec=importlib.util.find_spec('pydmd')
        pkg=types.ModuleType('pydmd');pkg.__path__=list(spec.submodule_search_locations)
        pkg.__spec__=spec;sys.modules['pydmd']=pkg
    DMD=importlib.import_module('pydmd.dmd').DMD
    DMDc=importlib.import_module('pydmd.dmdc').DMDc
    y,u=d['coeff'],d['force']; lo=d['lo']; ti=np.arange(a.start-lo,a.train_end-lo)
    val=(a.train_end-lo,a.val_end-lo); test=(a.val_end-lo,a.end-lo)
    def origins(span,h):return np.arange(span[0],span[1]-h+1)
    grid=[]; best={}; ranklist=[k for k in [4,8,16,32,64] if k<=a.max_rank]
    for kind in ['DMD_ridge','DMD_delay','DMDc_ridge','DMDc_delay','DMDc_delay_inputlag']:
        for r in ranklist:
            qs=[0] if kind.startswith('DMD_') else [q for q in [4,16,32] if q<=a.max_rank]
            for q in qs:
                for lam in [.001,.03,1.]:
                    m=fit_ridge(y,u,ti,r,q,'delay' in kind,'inputlag' in kind,lam,kind)
                    scores=[]
                    for h in [1,5,10]:
                        ts=origins(val,h); dm=predict(m,y,u,ts,h)
                        scores.append(metrics_projection(d['proj']['all'],dm,ts,h,1800*768)['skill'])
                    score=float(np.mean(scores)); row={k:m[k] for k in ['kind','r','q','lag','ulags','lam']};row.update(val_skill_h1_h5_h10=scores,val_mean_skill=score);grid.append(row)
                    if kind not in best or score>best[kind]['score']:best[kind]={'score':score,'model':m,'selection':row}
        print('Selected',kind,best[kind]['selection'],flush=True)
    # Actual library reference models, full state/omega ranks in reduced coordinates.
    for kind in ['PyDMD','PyDMDc']:
        for r in ranklist[:4]:
            q=min(16,a.max_rank); sd=np.maximum(y[ti,:r].std(0),1e-8);us=np.maximum(u[ti,:q].std(0),1e-8)
            yy=(y[ti[0]:ti[-1]+2,:r]/sd).T
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if kind=='PyDMDc':
                        lib=DMDc(svd_rank=-1,svd_rank_omega=-1)
                        lib.fit(yy,np.concatenate([u[ti,:q]/us,np.ones((len(ti),1))],1).T)
                        A=lib.basis@lib.operator.as_numpy_array@lib.basis.T; B=lib.B
                    else:
                        lib=DMD(svd_rank=-1,exact=True);lib.fit(yy)
                        A=(lib.modes@np.diag(lib.eigs)@np.linalg.pinv(lib.modes)).real;B=None
                m=dict(kind=kind,r=r,q=q if kind=='PyDMDc' else 0,sd=sd,us=us,A=A.real,B=B,lag=False,ulags=False)
                scores=[metrics_projection(d['proj']['all'],predict(m,y,u,origins(val,h),h),origins(val,h),h,1800*768)['skill'] for h in [1,5,10]]
                row=dict(kind=kind,r=r,q=m['q'],val_skill_h1_h5_h10=scores,val_mean_skill=float(np.mean(scores)))
                grid.append(row)
                if kind not in best or row['val_mean_skill']>best[kind]['score']:best[kind]={'score':row['val_mean_skill'],'model':m,'selection':row}
            except Exception as e: grid.append(dict(kind=kind,r=r,error=str(e)))
    # Two-scalar baseline fitted on train full-dimensional dot products, no PCA.
    p=d['proj']['all'];xx,ss,xs=p['xx'],p['ss'],p['xs'];t=ti
    vv=np.sum(xx[t,t]+xx[t-1,t-1]-2*xx[t,t-1]);ff=np.sum(ss[t,t]);vf=np.sum(xs[t,t]-xs[t-1,t])
    vy=np.sum(xx[t,t+1]-xx[t-1,t+1]-xx[t,t]+xx[t-1,t]);fy=np.sum(xs[t+1,t]-xs[t,t])
    beta,b=np.linalg.solve(np.array([[vv,vf],[vf,ff]]),np.array([vy,fy]))
    write(out/'MODEL_SELECTION_BEFORE_TEST.json',{'selected':{k:v['selection'] for k,v in best.items()},'baseline_beta_b':[beta,b], 'criterion':'mean full-dimensional skill on validation horizons1,5,10; each family selected separately', 'grid':grid})
    import pickle
    with open(out/'MODELS.pkl','wb') as f:pickle.dump(best,f)
    rows=[]
    for split,span in [('validation',val),('test',test)]:
        for h in [1,5,10]:
            ts=origins(span,h)
            for group,p in d['proj'].items():
                n=d['groups_n'][group]*768
                for kind in ['persistence','velocity','force_direct','force_inertia']:
                    met=metrics_baseline(p,ts,h,n,kind,beta if kind=='force_inertia' else 0,b if kind=='force_inertia' else 1)
                    rows.append(dict(split=split,group=group,horizon=h,model=kind,**met))
                for kind,v in best.items():
                    met=metrics_projection(p,predict(v['model'],y,u,ts,h),ts,h,n)
                    rows.append(dict(split=split,group=group,horizon=h,model=kind,**met))
                # Diagnostic ceiling only: uses actual future displacement, never fit as an input.
                dm=y[ts+h]-y[ts]
                met=metrics_projection(p,dm,ts,h,n)
                rows.append(dict(split=split,group=group,horizon=h,model='POD_oracle_ceiling_NOT_FORECAST',**met))
    write(out/'METRICS.json',rows)
    with open(out/'METRICS.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'BASELINE.json',{'beta':beta,'b':b})
    np.savez(out/'REDUCED_TRAJECTORIES.npz',coeff=y,force=u,state_spectrum=d['ev'],force_spectrum=d['ef'])
    return best,rows

def main():
    global torch
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--start',type=int,default=300);ap.add_argument('--train-end',type=int,default=700);ap.add_argument('--val-end',type=int,default=800);ap.add_argument('--end',type=int,default=1000)
    ap.add_argument('--max-rank',type=int,default=64);ap.add_argument('--block',type=int,default=30);ap.add_argument('--reuse',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();torch.cuda.set_per_process_memory_fraction(.28)
    write(a.out/'RUNNING.json',{'pid':os.getpid(),'args':vars(a)|{'source':str(a.source),'out':str(a.out)},'python':sys.executable})
    if a.reuse:
        import pickle
        with open(a.out/'prepared.pkl','rb') as f:d=pickle.load(f)
        assert d['lo']==a.start-1 and d['hi']==a.end and d['nt']==a.train_end-d['lo']+1
        assert d['coeff'].shape[1]==a.max_rank
    else:d=prepare(a,a.out)
    best,rows=run_models(a,a.out,d)
    write(a.out/'RUNTIME.json',{'seconds':time.perf_counter()-t0,'pid':os.getpid(),'status':'complete','torch':torch.__version__,
      'pydmd':version('pydmd'),'numpy':np.__version__,'device':torch.cuda.get_device_name(),
      'cuda_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'cuda_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
      'code_sha256':sha(__file__),'argv':sys.argv})
    print('COMPLETE',a.out,flush=True)

if __name__=='__main__':main()
