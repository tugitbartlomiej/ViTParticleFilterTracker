"""Whole-population PyDMD DMDc with causal scalar amplitudes, physical lag5.

No future delta, mean-gradient, last-gradient or target labels enter U.
Full768 omitted displacement is preserved at h5 and bounded at h20.
"""
import argparse,csv,hashlib,importlib.metadata,inspect,json,os,time
from pathlib import Path
import numpy as np
from pydmd import DMDc
from temporal_core import Scaler,fit_pydmd_pairs,fit_deeptime_pairs

CONTROL_FIELDS=['interval_gradient_l2_start','interval_adam_m1_l2_start','interval_adam_m2_l2_start']
CHECKPOINT='3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e'
SOURCE_MANIFEST='33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    def serial(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,default=serial,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def causal_controls(a):
    steps=np.asarray(a['steps']);start=steps[:-1]
    if not np.array_equal(steps,np.arange(len(steps))*5):raise ValueError('Expected consecutive physical lag5')
    shape=a[CONTROL_FIELDS[0]].shape
    if len(shape)!=2 or shape[0]!=len(start):raise ValueError('Control shape')
    for key in CONTROL_FIELDS:
        if a[key].shape!=shape or not np.isfinite(a[key]).all() or (a[key]<0).any():raise ValueError('Invalid start norm '+key)
    lr=a['telemetry_learning_rate'][start+1]
    clock=start.astype(float)/500
    extras=np.broadcast_to(np.stack([lr,clock],axis=-1)[:,None,:],(*shape,2))
    u=np.concatenate([np.stack([a[k] for k in CONTROL_FIELDS],axis=-1),extras],axis=-1).astype(float)
    if not np.isfinite(u).all():raise ValueError('Nonfinite causal input')
    return u

def phase_windows(fraction,h):
    fraction=np.asarray(fraction)
    return np.array([0 if np.all(fraction[t:t+h]==0) else 1 if np.all(fraction[t:t+h]==1) else -1 for t in range(len(fraction)-h+1)],dtype=int)

def residual_bounds(a,h):
    r=a['analysis_increment_residual_norm'].astype(float)
    if h==1:return r,r
    n=len(r)-h+1;blocks=np.stack([r[j:j+n] for j in range(h)])
    first=a['analysis_state_residual_norm'][:n].astype(float);last=a['analysis_state_residual_norm'][h:h+n].astype(float)
    lo=np.maximum(np.abs(last-first),np.maximum(2*blocks.max(0)-blocks.sum(0),0))
    hi=np.minimum(first+last,blocks.sum(0))
    # Float32 norms can very slightly violate a triangle equality.
    return lo,np.maximum(lo,hi)

def rollout(A,B,initial,input_sequence,h,frozen_start):
    state=initial.copy()
    for step in range(h):state=state@A.T+input_sequence[0 if frozen_start else step]@B.T
    return state

def select_fit_rows(rows,count=160):
    train=sorted((r for r in rows if r['split']=='train'),key=lambda r:r['image_index'])
    if not train:raise ValueError('No train films')
    return [train[i] for i in np.linspace(0,len(train)-1,min(count,len(train)),dtype=int)]

def fit_shared_models(x,y,u):
    s=Scaler.fit(np.concatenate([x,y]));us=Scaler.fit(u)
    xs,ys=s.transform(x),s.transform(y);controls=np.column_stack([us.transform(u),np.ones(len(u))])
    dmdc=fit_pydmd_pairs(xs,ys,controls)
    dmdc['physical_lag']=5 # core's generic one-step adapter; actual pair spacing here is five updates.
    dmd=fit_deeptime_pairs(xs,ys)
    return {'state_scaler':s,'control_scaler':us,'dmdc':dmdc,'dmd':dmd,'control_train_min':u.min(0),'control_train_max':u.max(0)}

def identities(rows):return sorted((r['image_index'],r['path'],r['sha256']) for r in rows)

def sources(root):
    merged=root/'results/merged'
    validation=json.loads((merged/'REPLAY_VALIDATION.json').read_text())
    if validation.get('status')!='pass' or validation.get('count_particles')!=900000 or validation.get('weights_buffers_unchanged') is not True:raise ValueError('Replay not complete validated population')
    rows=[json.loads(line) for line in (merged/'REPLAY_INDEX.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=1500 or sorted(r['image_index'] for r in rows)!=list(range(1500)):raise ValueError('Need all1500images')
    descriptor=json.loads((merged/'GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json').read_text());pca=json.loads((merged/'ANALYTICAL_PCA32_MANIFEST.json').read_text())
    if descriptor.get('status')!='complete' or descriptor.get('step_stride')!=5 or pca.get('status')!='complete':raise ValueError('Incomplete descriptor/PCA manifest')
    if identities(rows)!=identities(pca['files']) or identities(rows)!=identities(descriptor['files']):raise ValueError('Manifest file lists differ')
    if pca.get('validation_and_test_used_for_fit') is not False or pca.get('per_image_key')!='analysis_positions_pca32':raise ValueError('Not train-only analytical PCA32')
    basis=(merged/pca['basis_path']).resolve()
    if sha(basis)!=pca['container_sha256'] or descriptor['analysis_pca32']['basis_sha256']!=pca['basis_sha256']:raise ValueError('PCA basis hash mismatch')
    with np.load(basis,allow_pickle=False) as f:
        c=f['pca_components']
        if c.shape!=(768,32) or not np.allclose(c.T@c,np.eye(32),atol=1e-7,rtol=0):raise ValueError('PCA not orthonormal')
    panel=json.loads((root/'provenance/PANEL_64.json').read_text());known={}
    for r in panel['entries']:
        film=r['frame_id'].rsplit('/',1)[0];split=r['analysis_split']
        if film in known and known[film]!=split:raise ValueError('Producer film split leakage')
        known[film]=split
    for r in rows:
        film=r['frame_id'].rsplit('/',1)[0];u=int(hashlib.sha256(('population-20261009|'+film).encode()).hexdigest()[:8],16)%10
        r['film']=film;r['split']=known.get(film,'train' if u<6 else 'validation' if u<8 else 'test')
        if r['split']=='val':r['split']='validation'
        r['resolved_path']=str((merged/r['path']).resolve());r['basis_sha256']=pca['basis_sha256']
    byid={r['image_index']:r for r in rows}
    if not pca.get('fit_image_indices') or any(byid[i]['split']!='train' for i in pca['fit_image_indices']):raise ValueError('PCA basis uses heldout film')
    return rows,pca

def load(r,verify=True):
    if verify and sha(r['resolved_path'])!=r['sha256']:raise ValueError('NPZ file checksum mismatch')
    keys=['analysis_positions_pca32','analysis_state_residual_norm','analysis_increment_residual_norm','interval_increment_norm','interval_phase_min_norm_fraction','token_role_mask','steps','telemetry_learning_rate','metadata_json_utf8',*CONTROL_FIELDS]
    with np.load(r['resolved_path'],allow_pickle=False) as f:a={key:f[key] for key in keys}
    meta=json.loads(a.pop('metadata_json_utf8').tobytes().decode())
    if meta['checkpoint_sha256']!=CHECKPOINT or meta['manifest_sha256']!=SOURCE_MANIFEST or meta['analysis_pca32_basis_sha256']!=r['basis_sha256']:raise ValueError('NPZ source/basis provenance')
    if a['analysis_positions_pca32'].shape!=(101,600,32):raise ValueError('Expected full population PCA32 per image')
    for key in ['analysis_increment_residual_norm','interval_increment_norm',*CONTROL_FIELDS]:
        if a[key].shape!=(100,600) or not np.isfinite(a[key]).all():raise ValueError('Invalid array '+key)
    if a['analysis_state_residual_norm'].shape!=(101,600) or a['token_role_mask'].shape!=(600,):raise ValueError('State residual/role shapes')
    if not np.isfinite(a['analysis_positions_pca32']).all() or not np.isin(a['token_role_mask'],np.arange(8)).all():raise ValueError('Nonfinite state/invalid role')
    if a['interval_phase_min_norm_fraction'].shape!=(100,) or a['telemetry_learning_rate'].shape!=(501,):raise ValueError('Phase/LR shape')
    return a,meta

def train(rows):
    selected=select_fit_rows(rows);xs=[];ys=[];us=[];sampling=[]
    for r in selected:
        a,meta=load(r);z=a['analysis_positions_pca32'].astype(float);u=causal_controls(a);phase=phase_windows(a['interval_phase_min_norm_fraction'],1)
        tokens=np.unique(np.r_[np.linspace(0,599,24,dtype=int),np.flatnonzero(a['token_role_mask'])])
        intervals=[]
        for ph in [0,1]:
            ids=np.flatnonzero(phase==ph)
            if len(ids):intervals.extend(ids[np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int)].tolist())
        intervals=np.asarray(sorted(set(intervals)),dtype=int)
        xs.append(z[intervals][:,tokens].reshape(-1,32));ys.append(z[intervals+1][:,tokens].reshape(-1,32));us.append(u[intervals][:,tokens].reshape(-1,5))
        sampling.append({'image_index':r['image_index'],'film':r['film'],'split':r['split'],'interval_indices':intervals,'token_indices':tokens})
    fitted=fit_shared_models(np.concatenate(xs),np.concatenate(ys),np.concatenate(us))
    return fitted,sampling

def normalized_stats(s):
    n=s['n'];out={'n':int(n),'rmse32':float(np.sqrt(s['sse32']/n/32)),'rmse768_lower':float(np.sqrt(s['sse768_lower']/n/768)),'rmse768_upper':float(np.sqrt(s['sse768_upper']/n/768)),
        'true_net_norm32_mean':s['true_norm_sum']/n,'pred_net_norm32_mean':s['pred_norm_sum']/n,'direction_cos32_active':s['cos_sum']/s['active_n'] if s['active_n'] else None,'direction_active_n':int(s['active_n']),
        'retained_energy_h5':s['true_energy32']/s['true_energy768'] if s['true_energy768'] else None}
    return out

def evaluate_one(a,r,meta,fit,totals,writer):
    z=a['analysis_positions_pca32'].astype(float);u=causal_controls(a);s=fit['state_scaler'];us=fit['control_scaler']
    scaled=s.transform(z);controls=np.concatenate([us.transform(u),np.ones((100,600,1))],axis=-1)
    ood={'image':r['image_index'],'film':r['film'],'split':r['split'],'status':meta['status'],'max_abs_scaled_by_control':np.abs(controls[:,:,:5]).max(axis=(0,1)),
         'fraction_outside_train_range_by_control':((u<fit['control_train_min'])|(u>fit['control_train_max'])).mean(axis=(0,1))}
    for h in [1,4]:
        n=101-h;initial=scaled[:n];true=z[h:];start=z[:n];phases=phase_windows(a['interval_phase_min_norm_fraction'],h)
        sequence=np.stack([controls[offset:offset+n] for offset in range(h)])
        A,B=fit['dmdc']['A'],fit['dmdc']['B']
        predictions={'pydmd_DMDc_startU' if h==1 else 'pydmd_DMDc_frozen_startU':s.inverse(rollout(A,B,initial,sequence,h,True))}
        if h==4:predictions['pydmd_DMDc_futureU_oracle']=s.inverse(rollout(A,B,initial,sequence,h,False))
        state=initial.copy()
        for _ in range(h):state=state@fit['dmd']['A'].T+fit['dmd']['b']
        predictions['deeptime_DMD']=s.inverse(state);predictions['persistence']=start
        lo,hi=residual_bounds(a,h);true_delta=true-start;tn=np.linalg.norm(true_delta,axis=-1)
        group=((phases[:,None]+1)*8+a['token_role_mask'][None,:]).ravel();count=np.bincount(group,minlength=24)
        for method,pred in predictions.items():
            if not np.isfinite(pred).all():raise ValueError('Nonfinite rollout '+method)
            err2=np.sum((pred-true)**2,axis=-1);delta=pred-start;pn=np.linalg.norm(delta,axis=-1);active=(tn>1e-4)&(pn>1e-4)
            cosine=np.sum(true_delta*delta,axis=-1)/np.maximum(tn*pn,1e-30)
            fields={'sse32':err2,'sse768_lower':err2+lo**2,'sse768_upper':err2+hi**2,'true_norm_sum':tn,'pred_norm_sum':pn,'cos_sum':np.where(active,cosine,0),'active_n':active.astype(float),'true_energy32':tn**2,'true_energy768':a['interval_increment_norm'].astype(float)**2 if h==1 else np.zeros_like(tn)}
            sums={key:np.bincount(group,weights=value.ravel(),minlength=24) for key,value in fields.items()}
            for code in np.flatnonzero(count):
                key=(r['split'],meta['status'],int(code)//8-1,int(code)%8,h*5,method)
                target=totals.setdefault(key,{'n':0,**{k:0. for k in fields}});target['n']+=int(count[code])
                for k in fields:target[k]+=float(sums[k][code])
            overall={'n':n*600,**{k:float(v.sum()) for k,v in fields.items()}}
            writer.writerow({'image':r['image_index'],'film':r['film'],'split':r['split'],'status':meta['status'],'horizon':h*5,'method':method,**normalized_stats(overall)})
    return ood

def main():
    p=argparse.ArgumentParser();p.add_argument('--producer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);started=time.time();rows,pca=sources(args.producer_root)
    save(args.output/'INPUTS_SPLITS.json',rows);fit,sampling=train(rows);save(args.output/'TRAIN_SAMPLING.json',sampling)
    m=fit['dmdc'];diagnostics={k:v for k,v in m.items() if k not in ['A','B','eigenvalues']}
    diagnostics.update({'spectral_radius':float(max(abs(np.linalg.eigvals(m['A'])))),'B_frobenius_norm':float(np.linalg.norm(m['B'])),'DMD_spectral_radius':float(max(abs(np.linalg.eigvals(fit['dmd']['A'])))),
     'one_shared_A_B_across_all_images_tokens_and_phases':True,'U_columns':['gradient_l2_start','adam_m1_l2_start','adam_m2_l2_start','lr_at_t_plus_1','t_div_500','bias'],
     'identification_scope':'predictive coarse regression on observed scalar amplitudes; not directional forces, exogenous interventions or deploy without GT',
     'train_images':len(sampling),'prespecified_rank_omega_relative_threshold':1e-8,'fit_does_not_use_validation_or_test':True})
    save(args.output/'MODEL_DIAGNOSTICS.json',diagnostics)
    np.savez_compressed(args.output/'MODEL.npz',A=m['A'],B=m['B'],DMD_A=fit['dmd']['A'],DMD_b=fit['dmd']['b'],state_mean=fit['state_scaler'].mean,state_scale=fit['state_scaler'].scale,control_mean=fit['control_scaler'].mean,control_scale=fit['control_scaler'].scale,control_train_min=fit['control_train_min'],control_train_max=fit['control_train_max'])
    print('FIT COMPLETE',len(sampling),'images',m['pairs'],'train pairs',flush=True)
    totals={};ood=[]
    columns=['image','film','split','status','horizon','method','n','rmse32','rmse768_lower','rmse768_upper','true_net_norm32_mean','pred_net_norm32_mean','direction_cos32_active','direction_active_n','retained_energy_h5']
    with open(args.output/'METRICS_PER_IMAGE.csv','w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader()
        for number,r in enumerate(rows):
            a,meta=load(r);ood.append(evaluate_one(a,r,meta,fit,totals,writer))
            if number%100==0:print('EVALUATED',number+1,'images',flush=True)
    detail=[];pooled={}
    for (split,status,ph,mask,h,method),sums in totals.items():
        detail.append({'split':split,'status':status,'phase':ph,'role_mask':mask,'horizon':h,'method':method,**normalized_stats(sums)})
        key=(split,ph,h,method);dest=pooled.setdefault(key,{k:0. for k in sums})
        for k,v in sums.items():dest[k]+=v
    summary=[{'split':k[0],'phase':k[1],'horizon':k[2],'method':k[3],**normalized_stats(v)} for k,v in pooled.items()]
    save(args.output/'METRICS_STRATIFIED.json',detail);save(args.output/'SUMMARY_METRICS.json',summary);save(args.output/'CONTROL_OOD.json',ood)
    h5_count=sum(s['n'] for k,s in totals.items() if k[4]==5 and k[5]=='persistence')
    if h5_count!=90000000:raise ValueError('Not all90million physical5-step transitions evaluated')
    libraries={name:importlib.metadata.version(name) for name in ['pydmd','deeptime','numpy','scipy']}
    save(args.output/'PROVENANCE.json',{'status':'complete','job_id':os.getenv('SLURM_JOB_ID'),'elapsed_seconds':time.time()-started,'arguments':vars(args),'images':1500,'particles':900000,'h5_transitions_evaluated':h5_count,'h20_windows_per_particle':97,'libraries':libraries,'pydmd_DMDc_source_sha256':sha(inspect.getfile(DMDc)),'code_hashes':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},'PCA_basis_sha256':pca['basis_sha256'],'one_shared_A_B':True,'start_U_is_measured_oracle_amplitude_not_full_vector_control':True,'input_features_do_not_use_future_mean_last_gradient_or_delta':True,'h20_protocols':['frozen_all_startU_components','futureU_oracle_teacher_forced'],'h5_full768_error':'projected prediction preserving start residual, plus known residual increment norm; float32 precision','h20_full768_error':'triangle bounds from omitted state/increment norms, not exact','split_rule':'identical to frozen population consumer; panel film split preserved, unlisted films deterministic hash','prediction_scope':'oracle optimization trajectory forecast, not learned GT-free corrector or mAP claim'})
    lines=['# Populacyjny PyDMD DMDc ze skalarnymi wejściami','', 'Wykonano PyDMD.DMDc.fit na próbce train, następnie prognozy dla wszystkich1500×600 cząsteczek. Jedna wspólna para A/B. U zawiera normy g/m1/m2 z początku interwału, LR pierwszej aktualizacji, czas i bias. Nie zawiera średniego/końcowego gradientu ani przyszłego przemieszczenia.', '', 'Normy są mierzonymi amplitudami, nie pełnymi kierunkowymi gradientami. Nawet U start pochodzi z oracle zależnego od GT; przyczynowa dostępność w zapisanym czasie nie oznacza dostępności przy inference bez GT. KnownB nie jest tu zakładane. H20 frozenU zamraża również LR/czas; futureU jest jawnie oracle.', '', '|Split|Faza|h|Metoda|RMSE768 dolna|RMSE768 górna|n|','|---|---|---:|---|---:|---:|---:|']
    for r in summary:
        if r['split']=='test':lines.append(f'|{r["split"]}|{r["phase"]}|{r["horizon"]}|{r["method"]}|{r["rmse768_lower"]:.8g}|{r["rmse768_upper"]:.8g}|{r["n"]}|')
    lines+=['','Faza−1 oznacza okno przecinające zmianę celu;0fit;1min_norm. Wszystkie okna są ocenione, mieszane osobno. H5 ma znany błąd768 z dokładnością zapisu; h20 ma wyłącznie granice. Maski ról w METRICS_STRATIFIED: bit1target,bit2natural-top1,bit4konkurent;0pozostałe. Dobór rank/skali nie został zmieniony na podstawie testu. OOD, ranga i stabilność są oddzielnymi diagnostykami. Nie oceniono mAP ani nie wyeksportowano correctora.']
    (args.output/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('COMPLETE supplementary population DMDc',flush=True)

if __name__=='__main__':main()
