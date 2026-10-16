"""CPU temporal analysis, real PyDMD/deeptime, bounded raw working set (one image)."""
import argparse, csv, hashlib, importlib.metadata, inspect, json, os, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from deeptime.decomposition import DMD, VAMP
from pydmd import DMDc
from temporal_core import Scaler, film_split, window_starts, adam_input, validate_arrays, fit_pydmd_pairs, fit_deeptime_pairs, error_metrics, causal_motion_features

CONTROL_KEYS=['control_gradient_before_step','control_adam_m1_before_step','control_adam_m2_before_step']
CKPT='3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e'
MANIFEST='33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def serial(o):
    if isinstance(o,np.ndarray):return o.tolist()
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,Path):return str(o)
    raise TypeError(type(o).__name__)

def save(path,o):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(o,default=serial,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def metadata(path):
    with np.load(path,allow_pickle=False) as f:
        return json.loads(f['metadata_json_utf8'].tobytes().decode('utf-8'))

def sources(args):
    if args.panel_manifest:
        m=json.loads(args.panel_manifest.read_text())
        if m.get('status')!='complete' or m.get('step_stride')!=1 or m.get('dimensions')!=768:raise ValueError('Panel manifest not complete/full768/unit step')
        if m.get('source_checkpoint_sha256')!=CKPT or m.get('source_manifest_sha256')!=MANIFEST:raise ValueError('Wrong oracle provenance')
        validation=args.panel_manifest.parent/m['validation_path']
        if not validation.is_file():raise ValueError('Missing replay validation')
        v=json.loads(validation.read_text())
        if v.get('status')!='pass' or v.get('weights_buffers_unchanged') is not True:raise ValueError('Replay validation not passed')
        rows=[dict(e,path=str((args.panel_manifest.parent/e['path']).resolve())) for e in m['files']]
        if len(rows)!=m.get('count_images',len(rows)):raise ValueError('Panel count mismatch')
    else:
        panel=json.loads((args.old_root/'provenance/CONTROL_PANEL.json').read_text())
        rows=[{'image_index':int(k),'path':str(args.old_root/'results/production/latents'/f'{int(k):06d}.npz')} for k in panel]
    if len(set(r['image_index'] for r in rows))!=len(rows):raise ValueError('Duplicate image')
    for r in rows:
        actual=sha(r['path'])
        if 'sha256' in r and r['sha256']!=actual:raise ValueError('Panel checksum mismatch')
        r['sha256']=actual; m=metadata(r['path']);r['metadata']=m
        r['film']=m['frame_id'].rsplit('/',1)[0]
        if m['checkpoint_sha256']!=CKPT or m['manifest_sha256']!=MANIFEST:raise ValueError('NPZ provenance mismatch')
        if args.panel_manifest:
            if m['analysis_split']!=r['analysis_split'] or not m.get('video_disjoint_split'):raise ValueError('Split provenance mismatch')
    if not args.panel_manifest:
        for r,s in zip(rows,film_split([r['film'] for r in rows])):r['analysis_split']=s
    for r in rows:
        if r['analysis_split']=='val':r['analysis_split']='validation'
        if r['analysis_split'] not in ['train','validation','test']:raise ValueError('Unknown split')
    byfilm={}
    for r in rows:byfilm.setdefault(r['film'],set()).add(r['analysis_split'])
    if any(len(v)!=1 for v in byfilm.values()):raise ValueError('Film leakage')
    if set(r['analysis_split'] for r in rows)!={'train','validation','test'}:raise ValueError('Missing split')
    return rows

def load(r):
    with np.load(r['path'],allow_pickle=False) as f:
        keys=['control_z','control_tokens',*CONTROL_KEYS,'telemetry_step','telemetry_learning_rate','telemetry_phase_min_norm','telemetry_top_token']
        a={k:f[k] for k in keys}
        for k in ['control_token_role_mask','phase_before_update']:
            if k in f:a[k]=f[k]
    validate_arrays(a)
    if 'phase_before_update' in a and not np.array_equal(a['phase_before_update'],a['telemetry_phase_min_norm'][:-1]):raise ValueError('Phase alignment')
    return a

def training_basis(rows,rank,samples_per_image=1800):
    samples=[];ids=[]
    for r in rows:
        if r['analysis_split']!='train':continue
        a=load(r);z=a['control_z'];d=(z-z[:1]).reshape(-1,768)
        select=np.linspace(0,len(d)-1,min(samples_per_image,len(d)),dtype=int)
        samples.append(d[select]);ids.append(r['image_index'])
    sample=np.concatenate(samples).astype(np.float64)
    pca=PCA(n_components=rank,svd_solver='randomized',random_state=20261009).fit(sample)
    return pca.components_,{'fit_images':ids,'fit_rows':len(sample),'variance_ratio':pca.explained_variance_ratio_.tolist(),'variance_sum':float(pca.explained_variance_ratio_.sum()),'input':'full768 displacement z_t-z0; no channel normalization; common train basis'}

def project(a,C):
    z=a['control_z'].astype(np.float64);T,P,_=z.shape;T-=1
    d=z-z[:1];p=d@C.T
    u=np.concatenate([a[k].astype(np.float64)@C.T for k in CONTROL_KEYS],axis=-1)
    lr=a['telemetry_learning_rate'][1:];steps=np.arange(1,T+1)
    extras=np.broadcast_to(np.stack([lr,steps/500],axis=-1)[:,None,:],(T,P,2))
    u=np.concatenate([u,extras],axis=-1)
    known=adam_input(z[:-1],*(a[k].astype(np.float64) for k in CONTROL_KEYS),steps[:,None],lr[:,None])
    maxerr=float(np.abs(known-np.diff(z,axis=0)).max())
    return p,u,known@C.T,maxerr

def pool(reduced,split,phase,what):
    out=[]
    for r in reduced:
        if r['row']['analysis_split']!=split:continue
        mask=r['phase']==phase
        if what in ['fx','fy']:mask=mask&r['vamp_valid']
        value=r[what]
        out.append(value[mask].reshape(-1,value.shape[-1]))
    return np.concatenate(out)

def aggregate(rows):
    result=[]
    # Include trajectory-weighted and sample-weighted records; no pseudo-independent CI.
    for split in ['validation','test']:
      for phase in ['fit','min_norm']:
       for h in [1,5,20]:
        for method in sorted(set(r['method'] for r in rows)):
         sub=[r for r in rows if r['split']==split and r['phase']==phase and r['horizon']==h and r['method']==method]
         if not sub:continue
         n=sum(r['n'] for r in sub)
         result.append({'split':split,'phase':phase,'horizon':h,'method':method,'n':n,'trajectories':len(sub),
          'rmse':float(np.sqrt(sum(r['rmse']**2*r['n'] for r in sub)/n)),
          'true_net_norm_mean':sum(r['true_net_norm_mean']*r['n'] for r in sub)/n,
          'pred_net_norm_mean':sum(r['pred_net_norm_mean']*r['n'] for r in sub)/n})
    return result

def main():
    parser=argparse.ArgumentParser();src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--old-root',type=Path);src.add_argument('--panel-manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--rank',type=int,default=16)
    parser.add_argument('--max-windows',type=int,default=80)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);started=time.time()
    rows=sources(args);save(args.output/'INPUTS.json',rows)
    print('Validated',len(rows),'images; split',[(s,sum(r['analysis_split']==s for r in rows)) for s in ['train','validation','test']],flush=True)
    C,basis_info=training_basis(rows,args.rank);save(args.output/'BASIS.json',basis_info)
    reduced=[];baseline={p:{'sum':np.zeros(768),'n':0,'xy':0.,'xx':0.} for p in [0,1]}
    for r in rows:
        a=load(r);p,u,k,maxerr=project(a,C);phase=a['telemetry_phase_min_norm'][:-1].astype(int)
        reduced.append({'row':r,'p':p,'u':u,'k':k,'phase':phase,'adam_alignment_max_abs':maxerr})
        if r['analysis_split']=='train':
            z=a['control_z'].astype(np.float64);d=z-z[:1]
            for ph in [0,1]:
                x=d[:-1][phase==ph];y=d[1:][phase==ph];b=baseline[ph]
                b['sum']+=(y-x).sum(axis=(0,1));b['n']+=x.shape[0]*x.shape[1]
                b['xy']+=float(np.sum(x*y));b['xx']+=float(np.sum(x*x))
    s=Scaler.fit(np.concatenate([r['p'].reshape(-1,args.rank) for r in reduced if r['row']['analysis_split']=='train']))
    us=Scaler.fit(np.concatenate([r['u'].reshape(-1,r['u'].shape[-1]) for r in reduced if r['row']['analysis_split']=='train']))
    for r in reduced:
        scaled=s.transform(r['p']);r['x']=scaled[:-1];r['y']=scaled[1:]
        r['u']=np.concatenate([us.transform(r['u']),np.ones((*r['u'].shape[:-1],1))],axis=-1)
        r['k']/=s.scale
        f=causal_motion_features(scaled);r['fx']=f[:-1];r['fy']=f[1:]
        r['vamp_valid']=np.r_[False,r['phase'][1:]==r['phase'][:-1]]
    np.savez_compressed(args.output/'BASIS_SCALERS.npz',components=C,state_mean=s.mean,state_scale=s.scale,control_mean=us.mean,control_scale=us.scale)
    models={};regimes={};diagnostics={};baselines={}
    for ph in [0,1]:
        x,y,u,k,fx,fy=[pool(reduced,'train',ph,w) for w in ['x','y','u','k','fx','fy']]
        if len(x)<max(100,args.rank*4):raise ValueError('Insufficient train phase pairs')
        models[ph]={'pydmd_learnedB':fit_pydmd_pairs(x,y,u),'pydmd_knownB':fit_pydmd_pairs(x,y,k,np.eye(args.rank)), 'deeptime_DMD':fit_deeptime_pairs(x,y)}
        fs=Scaler.fit(fx);vamp=VAMP(dim=min(8,args.rank),epsilon=1e-8).fit((fs.transform(fx),fs.transform(fy))).fetch_model()
        embedding=vamp.transform(fs.transform(fx));es=Scaler.fit(embedding)
        km=KMeans(n_clusters=8,random_state=20261009,n_init=10).fit(es.transform(embedding))
        regimes[ph]=(fs,vamp,es,km)
        diagnostics[str(ph)]={name:{key:value for key,value in m.items() if key not in ['A','B','b','eigenvalues']}|{'spectral_radius':float(max(abs(np.linalg.eigvals(m['A']))))} for name,m in models[ph].items()}
        diagnostics[str(ph)]['vamp']={'backend':'deeptime.decomposition.VAMP.fit','singular_values':vamp.singular_values,'kmeans_count':8,'features':'train-scaled causal previous increment only; phase boundary excluded; no position/velocity algebraic overlap','pairs':len(fx)}
        b=baseline[ph];baselines[ph]={'increment':b['sum']/b['n'],'shrink':b['xy']/max(b['xx'],1e-30),'n':b['n']}
        print('Fitted phase',ph,'pairs',len(x),flush=True)
    save(args.output/'MODEL_DIAGNOSTICS.json',diagnostics);save(args.output/'BASELINES.json',baselines)
    # Store actual fitted operators; VAMP matrices are sufficient to reproduce transforms.
    matrices={}
    for ph in [0,1]:
        for name,m in models[ph].items():
            for key in ['A','B','b']:
                if key in m:matrices[f'phase{ph}_{name}_{key}']=m[key]
        fs,vm,es,km=regimes[ph]
        for key,value in {'feature_mean':fs.mean,'feature_scale':fs.scale,'vamp_mean':vm.mean_0,'vamp_coefficients':vm.singular_vectors_left,'embedding_mean':es.mean,'embedding_scale':es.scale,'centroids':km.cluster_centers_}.items():matrices[f'phase{ph}_{key}']=value
    np.savez_compressed(args.output/'OPERATORS.npz',**matrices)
    metrics=[];occupancy=[];motion={};excluded=[];transitions=[];turning=[];event_windows=[];control_ood=[]
    for reduced_row in reduced:
        r=reduced_row['row'];a=load(r);z=a['control_z'].astype(np.float64);T,P,_=z.shape;T-=1
        target=int(r['metadata']['target']['token_index']);status=r['metadata']['status'];phase=reduced_row['phase']
        roles=a.get('control_token_role_mask',np.array([1 if int(t)==target else 0 for t in a['control_tokens']]))
        for ph in [0,1]:
            mask=phase==ph
            if mask.any():
                control_ood.append({'image':r['image_index'],'split':r['analysis_split'],'phase':ph,'max_abs_scaled_control':float(np.abs(reduced_row['u'][mask,:,:-1]).max()),'p99_abs_scaled_control':float(np.quantile(np.abs(reduced_row['u'][mask,:,:-1]),.99))})
        dz=np.diff(z,axis=0);norm=np.linalg.norm(dz,axis=2)
        for j,token in enumerate(a['control_tokens']):
            for ph in [0,1]:
                valid=(phase[1:]==ph)&(phase[:-1]==ph)&(norm[1:,j]>1e-4)&(norm[:-1,j]>1e-4)
                cosine=np.sum(dz[1:,j]*dz[:-1,j],axis=1)/np.maximum(norm[1:,j]*norm[:-1,j],1e-30)
                turning.append({'image':r['image_index'],'split':r['analysis_split'],'phase':ph,'status':status,'token':int(token),'role_mask':int(roles[j]),'n_active_pairs':int(valid.sum()),'cos_mean':float(cosine[valid].mean()) if valid.any() else None,'reversal_fraction':float((cosine[valid]<0).mean()) if valid.any() else None})
        first=r['metadata'].get('first_success_step')
        if first is not None:
            for offset in range(-20,21):
                t=int(first)+offset
                if 0<=t<T:event_windows.append({'image':r['image_index'],'split':r['analysis_split'],'event':'first_success','relative_step':offset,'phase':int(phase[t]),'mean_step_norm':float(norm[t].mean()),'target_present':bool(np.any(roles&1))})
        for ph in [0,1]:
            select=np.flatnonzero((phase==ph)&reduced_row['vamp_valid'])
            if not len(select):continue
            fs,vm,es,km=regimes[ph]
            f=reduced_row['fx'][select];labels=km.predict(es.transform(vm.transform(fs.transform(f.reshape(-1,f.shape[-1]))))).reshape(len(select),P)
            for j,token in enumerate(a['control_tokens']):
                counts=np.zeros((8,8),dtype=int)
                adjacent=np.flatnonzero(np.diff(select)==1)
                np.add.at(counts,(labels[adjacent,j],labels[adjacent+1,j]),1)
                transitions.append({'image':r['image_index'],'split':r['analysis_split'],'phase':ph,'status':status,'token':int(token),'role_mask':int(roles[j]),'counts':counts})
                for regime in range(8):
                    idx=select[labels[:,j]==regime]
                    if not len(idx):continue
                    dz=z[idx+1,j]-z[idx,j];norm=np.linalg.norm(dz,axis=1);active=norm>1e-4
                    key=(r['analysis_split'],ph,status,'target' if roles[j]&1 else 'other',regime)
                    acc=motion.setdefault(key,{'sum':np.zeros(768),'unit_sum':np.zeros(768),'n':0,'active_n':0,'normsum':0.,'norm2sum':0.})
                    acc['sum']+=dz.sum(0);acc['unit_sum']+=(dz[active]/norm[active,None]).sum(0);acc['n']+=len(idx);acc['active_n']+=int(active.sum());acc['normsum']+=float(norm.sum());acc['norm2sum']+=float(np.sum(norm**2))
                    occupancy.append({'image':r['image_index'],'film':r['film'],'split':r['analysis_split'],'phase':ph,'status':status,'token':int(token),'role_mask':int(roles[j]),'regime':regime,'count':len(idx)})
        if r['analysis_split']=='train':continue
        for h in [1,5,20]:
          valid=window_starts(phase,h)
          excluded.append({'image':r['image_index'],'horizon':h,'phase_crossing_excluded':T-h+1-len(valid)})
          for ph in [0,1]:
            starts=valid[phase[valid]==ph]
            if not len(starts):continue
            starts=starts[np.linspace(0,len(starts)-1,min(args.max_windows,len(starts)),dtype=int)]
            for j,token in enumerate(a['control_tokens']):
                start=z[starts,j];true=z[starts+h,j];initial=reduced_row['x'][starts,j]
                predictions={'persistence':start,'constant_increment':start+h*baselines[ph]['increment'],
                 'phase_shrink':z[0,j]+baselines[ph]['shrink']**h*(start-z[0,j])}
                for name,model in models[ph].items():
                    state=initial.copy()
                    for offset in range(h):
                        state=state@model['A'].T
                        if name=='deeptime_DMD':state+=model['b']
                        else:state+=reduced_row['k' if name=='pydmd_knownB' else 'u'][starts+offset,j]@model['B'].T
                    predictions[name]=start+((state-initial)*s.scale)@C
                for name,pred in predictions.items():
                    if not np.isfinite(pred).all():raise ValueError(f'Nonfinite rollout {name} image{r["image_index"]} h{h}; do not conceal divergence')
                    metrics.append({'image':r['image_index'],'film':r['film'],'split':r['analysis_split'],'phase':'fit' if ph==0 else 'min_norm','status':status,'token':int(token),'role_mask':int(roles[j]),'role':'target' if roles[j]&1 else 'other','horizon':h,'method':name,**error_metrics(pred,true,start)})
        print('Evaluated',r['image_index'],flush=True)
    summary=aggregate(metrics);save(args.output/'SUMMARY_METRICS.json',summary)
    for name,data in [('METRICS_PER_TRAJECTORY.csv',metrics),('REGIME_OCCUPANCY.csv',occupancy)]:
        with open(args.output/name,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    directions={};motion_report=[]
    for number,(key,a) in enumerate(motion.items()):
        split,ph,status,role,regime=key
        directions[str(number)]=a['sum']/a['n']
        motion_report.append({'direction_key':str(number),'split':split,'phase':ph,'status':status,'role':role,'regime':regime,'n':a['n'],'active_n':a['active_n'],'mean_step_norm':a['normsum']/a['n'],'rms_step_norm':np.sqrt(a['norm2sum']/a['n']),'mean_vector_norm':float(np.linalg.norm(a['sum']/a['n'])),'unit_direction_coherence':float(np.linalg.norm(a['unit_sum'])/max(a['active_n'],1))})
    save(args.output/'REGIME_MOTION.json',motion_report);np.savez_compressed(args.output/'REGIME_DIRECTIONS_768D.npz',**directions)
    save(args.output/'EXCLUDED_WINDOWS.json',excluded)
    save(args.output/'REGIME_TRANSITIONS.json',transitions);save(args.output/'DIRECTION_CHANGES.json',turning)
    save(args.output/'SUCCESS_EVENT_ALIGNED.json',event_windows);save(args.output/'CONTROL_SCALE_DIAGNOSTICS.json',control_ood)
    breakdown={}
    for field in ['status','role','role_mask','film']:
        breakdown[field]={str(value):aggregate([r for r in metrics if r[field]==value]) for value in sorted(set(r[field] for r in metrics))}
    save(args.output/'METRICS_BREAKDOWN.json',breakdown)
    libraries={name:{'version':importlib.metadata.version(name)} for name in ['pydmd','deeptime','numpy','scipy','scikit-learn']}
    for name,cls in [('pydmd',DMDc),('deeptime',VAMP)]:
        path=inspect.getfile(cls);libraries[name].update(source=path,sha256=sha(path))
    save(args.output/'PROVENANCE.json',{'status':'complete','elapsed_seconds':time.time()-started,'job_id':os.getenv('SLURM_JOB_ID'),'libraries':libraries,'source_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'arguments':vars(args),'train_only_fit_images':basis_info['fit_images'],'video_disjoint':True,'input_panel_images':len(rows),'input_trajectories':sum(r['p'].shape[1] for r in reduced),'transitions':sum(np.prod(r['x'].shape[:2]).item() for r in reduced),'max_adam_alignment_abs':max(r['adam_alignment_max_abs'] for r in reduced),'analytical_rank':args.rank,'metric_dimensions':768,'max_windows_per_trajectory_phase_horizon':args.max_windows,'scope':'Selected full768 control trajectories, not all900000. Displacement PCA fitted only train. Residual outside basis held at rollout start. Phases and future controls are oracle information. No detection evaluation/corrector export.','knownB_interpretation':'B=I after projecting analytically clamped Adam update from pre-step g,m,v,lr and true current state; teacher-forced integration upper baseline, not identified causality or deployable predictor.'})
    lines=['# Wynik wykonania bibliotek temporalnych','',f'Obrazy: {len(rows)}. Rank analityczny: {args.rank}. Metryki: pełne 768D.','', '| Split | Faza | h | Metoda | RMSE 768D | n |','|---|---|---:|---|---:|---:|']
    for r in summary:lines.append(f'|{r["split"]}|{r["phase"]}|{r["horizon"]}|{r["method"]}|{r["rmse"]:.8g}|{r["n"]}|')
    lines+=['','To prognoza ruchu latentów, nie dowód poprawy detekcji. Kontrole DMDc i identyfikacja faz są teacher-forced. KnownB to analityczny Adam po projekcji. Nowy panel rozdziela filmy według kontraktu producenta; stary panel ma własny deterministyczny split filmów. Żaden test/validation nie dopasowuje PCA, skal, operatorów, VAMP, kmeans ani baseline. Parametry wybrano przed odczytem wyników testu.','', 'Źródła API: [PyDMD DMDc](https://pydmd.github.io/PyDMD/dmdc.html), [deeptime DMD](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.DMD.html), [deeptime VAMP](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.VAMP.html).']
    (args.output/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE',args.output,flush=True)

if __name__=='__main__':main()
