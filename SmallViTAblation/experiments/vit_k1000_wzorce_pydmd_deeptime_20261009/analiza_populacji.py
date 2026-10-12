"""Population assignment/summary, streaming one image. PCA3 is never an input."""
import argparse, hashlib, importlib.metadata, inspect, json, os, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from deeptime.decomposition import VAMP
from temporal_core import Scaler, fit_deeptime_pairs
from analiza_temporalna import sha, save, CKPT, MANIFEST

def population_features(a,analytical_key):
    if not np.array_equal(a['steps'],np.arange(0,501,5)):raise ValueError('Expected physical lag5')
    if analytical_key:
        z=a[analytical_key]
        if z.shape!=(101,600,32):raise ValueError('Expected analytical PCA32, never visualization PCA3')
        features=np.diff(z.astype(np.float64),axis=0)
    else:
        # Completed-interval outcomes used only for descriptive regime assignment.
        # These are not controls for predicting the same interval.
        features=np.stack([np.log1p(a['interval_increment_norm']),a['interval_previous_increment_cosine'],np.log1p(a['movement_norm'][1:])],axis=-1)
    fraction=a['interval_phase_min_norm_fraction']
    phase=np.where(fraction==0,0,np.where(fraction==1,1,-1))
    pairs=(phase[:-1]>=0)&(phase[:-1]==phase[1:])
    if not analytical_key:pairs[0]=False # first adjacent-angle placeholder
    if not np.isfinite(features).all():raise ValueError('Nonfinite population features')
    return features,phase,pairs

def assign_splits(rows,panel):
    known={}
    for r in panel['entries']:
        f=r['frame_id'].rsplit('/',1)[0];s=r['analysis_split']
        if f in known and known[f]!=s:raise ValueError('Producer film leakage')
        known[f]=s
    for r in rows:
        f=r['frame_id'].rsplit('/',1)[0];r['film']=f
        # Retain every producer panel film's split; hash unlisted films independently.
        u=int(hashlib.sha256(('population-20261009|'+f).encode()).hexdigest()[:8],16)%10
        r['split']=known.get(f,'train' if u<6 else 'validation' if u<8 else 'test')
        if r['split']=='val':r['split']='validation'
    return rows

def read(row,key,verify=False):
    if verify and sha(row['resolved_path'])!=row['sha256']:raise ValueError('Population file hash mismatch')
    with np.load(row['resolved_path'],allow_pickle=False) as f:
        keys=['steps','movement_norm','interval_increment_norm','interval_previous_increment_dot','interval_previous_increment_cosine','interval_previous_increment_valid','interval_phase_min_norm_fraction','token_role_mask','metadata_json_utf8']
        if key:keys.extend([key,'analysis_state_residual_norm','analysis_increment_residual_norm'])
        a={k:f[k] for k in keys}
    meta=json.loads(a.pop('metadata_json_utf8').tobytes().decode())
    if meta['checkpoint_sha256']!=CKPT or meta['manifest_sha256']!=MANIFEST:raise ValueError('Population provenance mismatch')
    if key and meta.get('analysis_pca32_basis_sha256')!=row.get('basis_sha256'):raise ValueError('Population analytical basis mismatch')
    return a,meta

def forecast_metrics(a,features,phase,models,split,image_index):
    """Forecast next 5/20 actual updates from an already observed velocity.

    At h5 the omitted-component error norm is known exactly. At h20 only
    rigorous triangle bounds are available; never call PCA32 error full768.
    """
    rows=[]
    for ph in [0,1]:
      fit=models[ph]
      for intervals in [1,4]:
        starts=np.array([i for i in range(99-intervals+1) if phase[i]==ph and np.all(phase[i+1:i+1+intervals]==ph)],dtype=int)
        if not len(starts):continue
        starts=starts[np.linspace(0,len(starts)-1,min(12,len(starts)),dtype=int)]
        initial=features[starts];true=sum(features[starts+j] for j in range(1,intervals+1))
        pred=np.zeros_like(initial);state=fit['scaler'].transform(initial)
        for step in range(intervals):
            state=state@fit['dmd']['A'].T+fit['dmd']['b'];pred+=fit['scaler'].inverse(state)
        predictions={'deeptime_velocity_DMD':pred,'persistence':np.zeros_like(initial),'last_increment':intervals*initial,'mean_increment':np.broadcast_to(intervals*fit['mean_increment'],initial.shape),'phase_velocity_shrink':sum(fit['shrink']**j for j in range(1,intervals+1))*initial}
        if intervals==1:lo=hi=a['analysis_increment_residual_norm'][starts+1].astype(float)
        else:
            residues=np.stack([a['analysis_increment_residual_norm'][starts+j] for j in range(1,intervals+1)],axis=0).astype(float)
            r0=a['analysis_state_residual_norm'][starts+1].astype(float);r1=a['analysis_state_residual_norm'][starts+1+intervals].astype(float)
            lo=np.maximum(np.abs(r1-r0),np.maximum(2*residues.max(0)-residues.sum(0),0))
            hi=np.minimum(r0+r1,residues.sum(0));hi=np.maximum(hi,lo) # rounding guard
        for name,value in predictions.items():
            if not np.isfinite(value).all():raise ValueError('Divergent population rollout, do not silently omit')
            err2=np.sum((value-true)**2,axis=-1)
            rows.append({'image':image_index,'split':split,'phase':ph,'horizon_actual_steps':5*intervals,'method':name,'n':int(err2.size),'rmse32':float(np.sqrt(err2.mean()/32)),'rmse768_lower':float(np.sqrt(np.mean(err2+lo**2)/768)),'rmse768_upper':float(np.sqrt(np.mean(err2+hi**2)/768)),'full768_exact_within_float_rounding':intervals==1,'mean_true_net_norm32':float(np.linalg.norm(true,axis=-1).mean()),'max_abs_scaled_initial_velocity':float(np.abs(fit['scaler'].transform(initial)).max())})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--panel-contract',type=Path,required=True)
    p.add_argument('--analytical-key',default='');p.add_argument('--analytical-provenance',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fit-image-count',type=int,default=160)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);(args.output/'assignments').mkdir(exist_ok=True);started=time.time()
    validation=json.loads((args.index.parent/'REPLAY_VALIDATION.json').read_text())
    if validation['status']!='pass' or validation['count_particles']!=900000:raise ValueError('Replay is not validated complete population')
    rows=[json.loads(s) for s in args.index.read_text().splitlines() if s.strip()]
    if len(rows)!=1500 or sorted(r['image_index'] for r in rows)!=list(range(1500)):raise ValueError('Not1500uniqueimages')
    descriptor_path=args.index.parent/'GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json'
    descriptor=json.loads(descriptor_path.read_text())
    if descriptor.get('status')!='complete' or descriptor.get('particle_count')!=900000 or descriptor.get('step_stride')!=5:raise ValueError('Incomplete population descriptor manifest')
    def identities(entries):return sorted((r['image_index'],r['path'],r['sha256']) for r in entries)
    if identities(descriptor['files'])!=identities(rows):raise ValueError('Descriptor/index file mismatch')
    rows=assign_splits(rows,json.loads(args.panel_contract.read_text()))
    for r in rows:r['resolved_path']=str((args.index.parent/r['path']).resolve())
    if args.analytical_key:
        if args.analytical_provenance is None:raise ValueError('Analytical provenance required')
        provenance=json.loads(args.analytical_provenance.read_text())
        if provenance.get('status')!='complete' or provenance.get('validation_and_test_used_for_fit') is not False:raise ValueError('Unverified train-only PCA provenance')
        basis=(args.analytical_provenance.parent/provenance['basis_path']).resolve()
        if sha(basis)!=provenance['container_sha256']:raise ValueError('Analytical basis container hash')
        if provenance.get('per_image_key')!=args.analytical_key:raise ValueError('Analytical array key mismatch')
        if identities(provenance['files'])!=identities(rows):raise ValueError('PCA/index file mismatch')
        if descriptor['analysis_pca32']['basis_sha256']!=provenance['basis_sha256']:raise ValueError('Global/PCA manifest basis mismatch')
        with np.load(basis,allow_pickle=False) as basis_arrays:
            components=basis_arrays['pca_components']
            if components.shape!=(768,32) or not np.allclose(components.T@components,np.eye(32),atol=1e-7,rtol=0):raise ValueError('Nonorthonormal PCA32')
        fit_ids=provenance.get('fit_image_indices')
        if not fit_ids:raise ValueError('PCA basis fit_image_indices required in consumer provenance')
        byid={r['image_index']:r for r in rows}
        if any(byid[i]['split']!='train' for i in fit_ids):raise ValueError('PCA32 basis film leakage')
        for r in rows:r['basis_sha256']=provenance['basis_sha256']
    train=[r for r in rows if r['split']=='train']
    # Fixed stratification independent of target motions/outcomes, endpoints included.
    sample=[train[i] for i in np.linspace(0,len(train)-1,min(args.fit_image_count,len(train)),dtype=int)]
    fitx={0:[],1:[]};fity={0:[],1:[]}
    for r in sample:
        a,_=read(r,args.analytical_key,verify=True);f,phase,pairs=population_features(a,args.analytical_key)
        tokens=np.unique(np.r_[np.linspace(0,599,24,dtype=int),np.flatnonzero(a['token_role_mask'])])
        for ph in [0,1]:
            ix=np.flatnonzero(pairs&(phase[:-1]==ph));ix=ix[np.linspace(0,len(ix)-1,min(24,len(ix)),dtype=int)] if len(ix) else ix
            fitx[ph].append(f[ix][:,tokens].reshape(-1,f.shape[-1]));fity[ph].append(f[ix+1][:,tokens].reshape(-1,f.shape[-1]))
    models={};dynamics={};info={};matrices={}
    for ph in [0,1]:
        x=np.concatenate(fitx[ph]);y=np.concatenate(fity[ph])
        if len(x)<100:raise ValueError('Insufficient population train pairs')
        scaler=Scaler.fit(x);vm=VAMP(dim=min(8,x.shape[1]),epsilon=1e-8).fit((scaler.transform(x),scaler.transform(y))).fetch_model()
        embedding=vm.transform(scaler.transform(x));es=Scaler.fit(embedding);km=KMeans(n_clusters=8,random_state=20261009,n_init=10).fit(es.transform(embedding))
        models[ph]=(scaler,vm,es,km)
        info[str(ph)]={'pairs':len(x),'feature_dimension':x.shape[1],'singular_values':vm.singular_values,'backend':'deeptime.decomposition.VAMP.fit','physical_lag':5}
        if args.analytical_key:
            dmd=fit_deeptime_pairs(scaler.transform(x),scaler.transform(y))
            singular=np.linalg.svd(np.column_stack([scaler.transform(x),np.ones(len(x))]),compute_uv=False)
            dynamics[ph]={'scaler':scaler,'dmd':dmd,'mean_increment':y.mean(0),'shrink':float(np.sum(x*y)/max(np.sum(x*x),1e-30))}
            info[str(ph)]['dynamics']={'backend':dmd['backend'],'spectral_radius':float(max(abs(np.linalg.eigvals(dmd['A'])))),'design_singular_values':singular,'condition':float(singular[0]/max(singular[-1],1e-30)),'input':'past completed 5-step velocity, no future controls','phase_velocity_shrink':dynamics[ph]['shrink']}
            matrices[f'phase{ph}_DMD_A']=dmd['A'];matrices[f'phase{ph}_DMD_b']=dmd['b'];matrices[f'phase{ph}_mean_increment']=y.mean(0)
        for k,v in {'scale_mean':scaler.mean,'scale':scaler.scale,'vamp_mean':vm.mean_0,'vamp_coefficients':vm.singular_vectors_left,'embedding_mean':es.mean,'embedding_scale':es.scale,'centroids':km.cluster_centers_}.items():matrices[f'phase{ph}_{k}']=v
    save(args.output/'FIT.json',{'fit_images':[r['image_index'] for r in sample],'models':info,'split_rule':'preserve producer panel films; other films sha256(population-20261009|film) modulo10: 0..5train,6..7validation,8..9test','PCA3_used':False})
    np.savez_compressed(args.output/'OPERATORS.npz',**matrices);save(args.output/'INPUTS_SPLITS.json',rows)
    stats={};perimage=[];transition_counts={};assigned=0;mixed=0;forecasts=[]
    for n,r in enumerate(rows):
        a,meta=read(r,args.analytical_key,verify=True);features,phase,pairs=population_features(a,args.analytical_key)
        if args.analytical_key and r['split']!='train':forecasts.extend(forecast_metrics(a,features,phase,dynamics,r['split'],r['image_index']))
        labels=np.full((100,600),255,dtype=np.uint8)
        for ph in [0,1]:
            idx=np.flatnonzero(phase==ph)
            if not args.analytical_key:idx=idx[idx>0]
            if not len(idx):continue
            scaler,vm,es,km=models[ph];x=features[idx]
            labels[idx]=km.predict(es.transform(vm.transform(scaler.transform(x.reshape(-1,x.shape[-1]))))).reshape(len(idx),600).astype(np.uint8)
        assigned+=int((labels!=255).sum());mixed+=int((phase<0).sum())*600
        np.savez_compressed(args.output/'assignments'/f'{r["image_index"]:06d}.npz',regime=labels,interval_end_steps=np.arange(5,501,5,dtype=np.int32))
        norm=a['interval_increment_norm'].astype(float);cos=a['interval_previous_increment_cosine'];validangle=a['interval_previous_increment_valid'].astype(bool)
        active=norm>1e-4;previous=np.r_[np.zeros((1,600),bool),active[:-1]]
        samephase=np.r_[False,(phase[1:]==phase[:-1])&(phase[1:]>=0)]
        validangle=validangle[:,None]&active&previous&samephase[:,None]
        energy32=np.sum(features**2,axis=-1) if args.analytical_key else None
        for ph in [0,1]:
          for role in ['target','other']:
            tokenmask=(a['token_role_mask']&1)!=0
            if role=='other':tokenmask=~tokenmask
            for regime in range(8):
                mask=(phase[:,None]==ph)&tokenmask[None,:]&(labels==regime)
                count=int(mask.sum())
                if not count:continue
                key=(r['split'],ph,meta['status'],role,regime)
                rec=stats.setdefault(key,{'n':0,'norm_sum':0.,'energy768':0.,'energy32':0.,'angle_n':0,'cos_sum':0.,'reversals':0,'feature_sum':np.zeros(features.shape[-1])})
                rec['n']+=count;rec['norm_sum']+=float(norm[mask].sum());rec['energy768']+=float(np.sum(norm[mask]**2));rec['feature_sum']+=features[mask].sum(0)
                if energy32 is not None:rec['energy32']+=float(energy32[mask].sum())
                va=mask&validangle;rec['angle_n']+=int(va.sum());rec['cos_sum']+=float(cos[va].sum());rec['reversals']+=int((cos[va]<0).sum())
                perimage.append({'image':r['image_index'],'film':r['film'],'split':r['split'],'phase':ph,'status':meta['status'],'role':role,'regime':regime,'n':count,'norm_mean768':float(norm[mask].mean())})
            pairmask=pairs[:,None]&tokenmask[None,:]&(labels[:-1]!=255)&(labels[1:]!=255)&(phase[:-1,None]==ph)
            key=(r['split'],ph,role);cm=transition_counts.setdefault(key,np.zeros((8,8),np.int64));np.add.at(cm,(labels[:-1][pairmask],labels[1:][pairmask]),1)
        if n%100==0:print('Population images',n+1,flush=True)
    output=[]
    for (split,ph,status,role,regime),v in stats.items():
        output.append({'split':split,'phase':ph,'status':status,'role':role,'regime':regime,'n':v['n'],'mean_increment_norm768':v['norm_sum']/v['n'],'rms_increment_norm768':float(np.sqrt(v['energy768']/v['n'])),'active_angle_n':v['angle_n'],'mean_adjacent_cos768':v['cos_sum']/max(v['angle_n'],1),'reversal_fraction768':v['reversals']/max(v['angle_n'],1),'energy_retained32':v['energy32']/max(v['energy768'],1e-30) if args.analytical_key else None,'mean_feature_vector':v['feature_sum']/v['n']})
    save(args.output/'REGIME_SUMMARY.json',output);save(args.output/'OCCUPANCY_PER_IMAGE.json',perimage)
    save(args.output/'TRANSITIONS.json',[{'split':k[0],'phase':k[1],'role':k[2],'counts':v} for k,v in transition_counts.items()])
    save(args.output/'DYNAMICS_PER_IMAGE.json',forecasts)
    summary=[]
    for split in ['validation','test']:
      for phase in [0,1]:
       for h in [5,20]:
        for method in sorted(set(r['method'] for r in forecasts)):
            selected=[r for r in forecasts if r['split']==split and r['phase']==phase and r['horizon_actual_steps']==h and r['method']==method]
            if not selected:continue
            count=sum(r['n'] for r in selected)
            summary.append({'split':split,'phase':phase,'horizon':h,'method':method,'n':count,**{metric:float(np.sqrt(sum(r[metric]**2*r['n'] for r in selected)/count)) for metric in ['rmse32','rmse768_lower','rmse768_upper']},'full768_exact_within_float_rounding':h==5})
    save(args.output/'DYNAMICS_SUMMARY.json',summary)
    libraries={name:importlib.metadata.version(name) for name in ['pydmd','deeptime','numpy','scipy','scikit-learn']}
    save(args.output/'PROVENANCE.json',{'status':'complete','job_id':os.getenv('SLURM_JOB_ID'),'elapsed_seconds':time.time()-started,'arguments':vars(args),'images':1500,'particles':900000,'intervals_per_particle':100,'assigned_particle_intervals':assigned,'mixed_phase_particle_intervals':mixed,'input_index_sha256':sha(args.index),'input_descriptor_manifest_sha256':sha(descriptor_path),'analytical_manifest_sha256':sha(args.analytical_provenance) if args.analytical_provenance else None,'source_sha256':sha(__file__),'libraries':libraries,'deeptime_vamp_source_sha256':sha(inspect.getfile(VAMP)),'scope':'Population descriptive VAMP regimes on analytical PCA32 increments' if args.analytical_key else 'Population descriptive VAMP regimes of scalar full768 geometry, no common directional operator','no_global_DMDc':True,'prediction_causality':'features describe already completed intervals; not future controls','full768_trajectory_recovered':False})
    lines=['# Populacja900k — wykonana analiza','',f'Przetworzono1500 obrazów i900000 cząsteczek. Etykiety reżimów: {assigned} interwałów; wykluczone mieszane fazy: {mixed}.', '', 'Fit tylko próbka train; streaming inference wszystkich danych. Nie użyto PCA3. Regimy opisują zakończony ruch, nie siłę dostępną przed tym ruchem. DMD jest autonomiczne; kontrolowane DMDc pozostaje panelem.', '', '| Split | Faza | h | Metoda | RMSE32 | Dolna RMSE768 | Górna RMSE768 |', '|---|---|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f'|{r["split"]}|{r["phase"]}|{r["horizon"]}|{r["method"]}|{r["rmse32"]:.7g}|{r["rmse768_lower"]:.7g}|{r["rmse768_upper"]:.7g}|')
    lines+=['','Dla h5 obie granice są równe i dają pełny błąd768 z dokładnością zapisu. Dla h20 to granice, nie dokładny błąd768. h1 nie jest testowalne z zapisów co5. Normy/kąty768 i energia przyrostów zachowana przez PCA32 są w REGIME_SUMMARY.json. Etykiety255 są wykluczeniami. Nie wykonano oceny mAP ani eksportu correctora.']
    (args.output/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE population',flush=True)

if __name__=='__main__':main()
