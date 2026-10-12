"""Post hoc description only. Reads collector-verified outputs; never fits models/jobs."""
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from collections import defaultdict
from pathlib import Path
import numpy as np

FAMILIES=['panel64','population900k','population900k_dmdc_supplement']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify(root,name):
    marker=read(root/f'TRANSFER_{name}.json');folder=root/name
    for item in marker['files']:
        path=(folder/item['path']).resolve();path.relative_to(folder.resolve())
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise RuntimeError('Local transfer digest failed: '+str(path))
    p=read(folder/'PROVENANCE.json')
    if p!=marker['provenance'] or p['status']!='complete':raise RuntimeError('Provenance differs from verified transfer')
    return {'job_id':p['job_id'],'file_count':len(marker['files']),'bytes':sum(r['bytes'] for r in marker['files']),'transfer_manifest_sha256':sha(root/f'TRANSFER_{name}.json'),'all_files_rehashed':True,'provenance':p}

def bootstrap_by_film(rows,metric,baseline='persistence'):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:np.zeros(2))))
    for r in rows:
        if r['split']!='test':continue
        phase=r.get('phase','all');h=int(r.get('horizon',r.get('horizon_actual_steps',0)))
        film=r.get('film',str(r.get('image')));method=r['method'];n=int(r['n']);rmse=float(r[metric])
        groups[(str(phase),h)][method][film]+=np.array([rmse*rmse*n,n])
    result=[];rng=np.random.default_rng(20261009)
    for (phase,h),methods in groups.items():
        if baseline not in methods:continue
        for method,films in methods.items():
            if method==baseline:continue
            shared=sorted(set(films)&set(methods[baseline]))
            if len(shared)<2:continue
            m=np.array([films[f] for f in shared]);b=np.array([methods[baseline][f] for f in shared])
            if not np.array_equal(m[:,1],b[:,1]):raise ValueError('Different evaluation window counts')
            indices=rng.integers(0,len(shared),size=(2000,len(shared)))
            ratios=np.sqrt(m[indices,0].sum(1)/np.maximum(b[indices,0].sum(1),1e-300))
            dimension_factor=32/768 if metric=='rmse32' else 1.
            mse_difference=dimension_factor*(m[indices,0].sum(1)-b[indices,0].sum(1))/m[indices,1].sum(1)
            perfilm=np.sqrt(m[:,0]/np.maximum(b[:,0],1e-300))
            result.append({'phase':phase,'horizon':h,'method':method,'metric':metric,'films':len(shared),'baseline':baseline,'rmse_ratio_to_baseline':float(np.sqrt(m[:,0].sum()/b[:,0].sum())),
              **({'rmse_ratio_to_persistence':float(np.sqrt(m[:,0].sum()/b[:,0].sum()))} if baseline=='persistence' else {}),
              'bootstrap95_film_cluster':np.quantile(ratios,[.025,.975]).tolist(),'fraction_films_better':float((perfilm<1).mean()),'median_film_ratio':float(np.median(perfilm)),
              'mse768_difference_model_minus_baseline':float(dimension_factor*(m[:,0].sum()-b[:,0].sum())/m[:,1].sum()),
              'mse768_difference_bootstrap95':np.quantile(mse_difference,[.025,.975]).tolist(),
              'difference_assumption':'For rmse32: every compared forecast freezes the same residual at the same start; omitted true residual cancels in paired full768 MSE difference. Individual h20 full768 RMSE remains bounded only.',
              'largest_film_sse_share':float(m[:,0].max()/max(m[:,0].sum(),1e-300)),'largest_error_film':shared[int(m[:,0].argmax())]})
    return result

def panel(root):
    folder=root/'panel64';motion=read(folder/'REGIME_MOTION.json');vectors=np.load(folder/'REGIME_DIRECTIONS_768D.npz')
    counts=defaultdict(int)
    for r in read(folder/'INPUTS.json'):counts[(r['analysis_split'],r['metadata']['status'])]+=1
    alignment=[]
    index={(r['split'],r['phase'],r['status'],r['role'],r['regime']):r for r in motion}
    for r in motion:
        if r['split']!='test':continue
        other=index.get(('train',r['phase'],r['status'],r['role'],r['regime']))
        if not other:continue
        x=vectors[r['direction_key']];y=vectors[other['direction_key']];den=np.linalg.norm(x)*np.linalg.norm(y)
        alignment.append({'phase':r['phase'],'status':r['status'],'role':r['role'],'regime':r['regime'],'test_n':r['n'],'train_n':other['n'],
           'train_test_mean_direction_cos768':float(x@y/den) if den>1e-12 else None,
           'test_mean_vector_over_mean_norm':r['mean_vector_norm']/max(r['mean_step_norm'],1e-30),
           'test_unit_direction_coherence':r['unit_direction_coherence'],'test_mean_step_norm':r['mean_step_norm'],
           'train_mean_step_norm':other['mean_step_norm']})
    rolecos=[]
    for r in motion:
        if r['split']!='test' or r['role']!='target':continue
        other=index.get(('test',r['phase'],r['status'],'other',r['regime']))
        if not other:continue
        x=vectors[r['direction_key']];y=vectors[other['direction_key']];den=np.linalg.norm(x)*np.linalg.norm(y)
        rolecos.append({'phase':r['phase'],'status':r['status'],'regime':r['regime'],'cos_target_other_mean768':float(x@y/den) if den>1e-12 else None,'target_n':r['n'],'other_n':other['n'],'target_mean_step_norm':r['mean_step_norm'],'other_mean_step_norm':other['mean_step_norm']})
    turning=defaultdict(lambda:np.zeros(3))
    for r in read(folder/'DIRECTION_CHANGES.json'):
        key=(r['split'],r['phase'],r['status'],'target' if r['role_mask']&1 else 'other');n=r['n_active_pairs']
        turning[key]+=np.array([n,(r['cos_mean'] or 0)*n,(r['reversal_fraction'] or 0)*n])
    occupancy=defaultdict(lambda:np.zeros(8,dtype=np.int64));trans=defaultdict(lambda:np.zeros((8,8),dtype=np.int64))
    with (folder/'REGIME_OCCUPANCY.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):occupancy[(r['split'],r['phase'],r['status'],'target' if int(r['role_mask'])&1 else 'other')][int(r['regime'])]+=int(r['count'])
    for r in read(folder/'REGIME_TRANSITIONS.json'):trans[(r['split'],str(r['phase']),r['status'],'target' if r['role_mask']&1 else 'other')]+=np.asarray(r['counts'],dtype=np.int64)
    with (folder/'METRICS_PER_TRAJECTORY.csv').open(encoding='utf-8') as f:metrics=list(csv.DictReader(f))
    predictions=defaultdict(lambda:np.zeros(6))
    for r in metrics:
        if r['split']!='test':continue
        k=(r['phase'],r['status'],r['role'],int(r['horizon']),r['method']);n=int(r['n']);a=int(r['direction_active_n'])
        predictions[k]+=np.array([n,float(r['rmse'])**2*n,float(r['true_net_norm_mean'])*n,float(r['pred_net_norm_mean'])*n,float(r['direction_cos_mean_active'] or 0)*a,a])
    event=defaultdict(lambda:[])
    for r in read(folder/'SUCCESS_EVENT_ALIGNED.json'):
        if r['split']=='test':event[(r['relative_step'],r['phase'])].append(r['mean_step_norm'])
    return {'split_status_counts':[{'split':k[0],'status':k[1],'images':v} for k,v in counts.items()],
     'motion':motion,'mean_direction_train_test':alignment,'target_other_direction':rolecos,
     'turning':[dict(split=k[0],phase=k[1],status=k[2],role=k[3],n=int(v[0]),mean_cos=v[1]/v[0] if v[0] else None,reversal_fraction=v[2]/v[0] if v[0] else None) for k,v in turning.items()],
     'regime_occupancy':[dict(split=k[0],phase=k[1],status=k[2],role=k[3],counts=v.tolist(),fractions=(v/max(v.sum(),1)).tolist(),p_same_next=float(np.trace(trans[k])/max(trans[k].sum(),1))) for k,v in occupancy.items()],
     'bootstrap_by_test_film':bootstrap_by_film(metrics,'rmse'),
     'prediction_test_role_status':[dict(phase=k[0],status=k[1],role=k[2],horizon=k[3],method=k[4],n=int(v[0]),rmse=float(np.sqrt(v[1]/v[0])),true_norm=v[2]/v[0],pred_norm=v[3]/v[0],cos_pred_true=v[4]/v[5] if v[5] else None) for k,v in predictions.items()],
     'first_success_aligned_test':[dict(relative_step=k[0],phase=k[1],images=len(v),mean_norm=float(np.mean(v)),median_norm=float(np.median(v))) for k,v in event.items()]}

def population(root):
    folder=root/'population900k';rows=read(folder/'REGIME_SUMMARY.json');aggregate=defaultdict(lambda:np.zeros(7));directions=defaultdict(list)
    index={(r['split'],r['phase'],r['status'],r['role'],r['regime']):r for r in rows};alignment=[]
    for r in rows:
        k=(r['split'],r['phase'],r['status'],r['role']);n=r['n'];energy=r['rms_increment_norm768']**2*n;an=r['active_angle_n']
        aggregate[k]+=np.array([n,r['mean_increment_norm768']*n,energy,(r['energy_retained32'] or 0)*energy,r['mean_adjacent_cos768']*an,an,r['reversal_fraction768']*an])
        directions[k].append({'regime':r['regime'],'n':n,'mean_norm768':r['mean_increment_norm768'],'energy_retained32':r['energy_retained32'],
         'energy768':energy,
         'projected_mean_direction_ratio_lower_bound':float(np.linalg.norm(r['mean_feature_vector'])/max(r['mean_increment_norm768'],1e-30))})
        if r['split']=='test':
            other=index.get(('train',r['phase'],r['status'],r['role'],r['regime']))
            if other:
                x=np.asarray(r['mean_feature_vector']);y=np.asarray(other['mean_feature_vector']);den=np.linalg.norm(x)*np.linalg.norm(y)
                alignment.append(dict(phase=r['phase'],status=r['status'],role=r['role'],regime=r['regime'],test_n=n,train_n=other['n'],train_test_cos32=float(x@y/den) if den>1e-12 else None))
    for k,groups in directions.items():
        for r in groups:
            r['fraction']=r['n']/aggregate[k][0];r['energy_fraction']=r['energy768']/max(aggregate[k][2],1e-30)
    means=defaultdict(lambda:np.zeros(32))
    for r in rows:means[(r['split'],r['phase'],r['status'],r['role'])]+=np.asarray(r['mean_feature_vector'])*r['n']
    for k in means:means[k]/=aggregate[k][0]
    def cs(x,y):
        den=np.linalg.norm(x)*np.linalg.norm(y)
        return float(x@y/den) if den>1e-12 else None
    comparisons=[]
    for split in ['train','validation','test']:
        for ph,status in [(0,'success'),(0,'budget_exhausted'),(1,'success')]:
            comparisons.append(dict(split=split,comparison='target_vs_other',phase=ph,status=status,cos32=cs(means[(split,ph,status,'target')],means[(split,ph,status,'other')])))
        for role in ['target','other']:
            comparisons.append(dict(split=split,comparison='fit_vs_min_norm_success',role=role,cos32=cs(means[(split,0,'success',role)],means[(split,1,'success',role)])))
    inputs=read(folder/'INPUTS_SPLITS.json');filmbyimage={r['image_index']:r['film'] for r in inputs};metrics=read(folder/'DYNAMICS_PER_IMAGE.json')
    split_by_film=defaultdict(set);split_by_image={r['image_index']:r['split'] for r in inputs}
    for r in inputs:split_by_film[r['film']].add(r['split'])
    if any(len(v)!=1 for v in split_by_film.values()):raise ValueError('Population film leakage')
    fit=read(folder/'FIT.json');producer=read(root/'PRODUCER_METADATA_VERIFIED.json')['ANALYTICAL_PCA32_MANIFEST.json']
    if any(split_by_image[i]!='train' for i in fit['fit_images']+producer['fit_image_indices']):raise ValueError('Operator/PCA fitted to held-out image')
    prov=read(folder/'PROVENANCE.json')
    if producer['sha256']!=prov['analytical_manifest_sha256']:raise ValueError('Different PCA provenance')
    counts=np.zeros(256,dtype=np.int64)
    for i in range(1500):
        with np.load(folder/'assignments'/f'{i:06d}.npz',allow_pickle=False) as a:
            labels=a['regime']
            if labels.shape!=(100,600) or not np.array_equal(a['interval_end_steps'],np.arange(5,501,5)):raise ValueError('Assignment timeline')
            counts+=np.bincount(labels.ravel(),minlength=256)
    if counts[8:255].sum() or int(counts[:8].sum())!=prov['assigned_particle_intervals'] or int(counts[255])!=prov['mixed_phase_particle_intervals']:raise ValueError('Population assignment coverage')
    for r in metrics:r['film']=filmbyimage[r['image']]
    return {'integrity':dict(video_disjoint=True,operator_and_PCA_fit_only_train=True,PCA_provenance_hash_match=True,assignment_arrays_verified=1500,assigned_intervals=int(counts[:8].sum()),excluded_mixed_phase=int(counts[255]),total_intervals=int(counts.sum())),
      'aggregate_geometry':[dict(split=k[0],phase=k[1],status=k[2],role=k[3],n=int(v[0]),mean_step_norm768=v[1]/v[0],energy768=v[2],energy_retained32=v[3]/v[2] if v[2] else None,mean_adjacent_cos768=v[4]/v[5] if v[5] else None,active_angles=int(v[5]),reversal_fraction768=v[6]/v[5] if v[5] else None) for k,v in aggregate.items()],
      'regime_direction_lower_bound':[dict(split=k[0],phase=k[1],status=k[2],role=k[3],regimes=v) for k,v in directions.items()],
      'mean_direction_train_test32':alignment,
      'pooled_mean_direction_comparisons32':comparisons,
      'transition_self_fraction':[dict(split=r['split'],phase=r['phase'],role=r['role'],n=int(np.sum(r['counts'])),p_same_next=float(np.trace(r['counts'])/max(np.sum(r['counts']),1))) for r in read(folder/'TRANSITIONS.json')],
      'bootstrap_test_film_h5_full768':bootstrap_by_film([r for r in metrics if r['horizon_actual_steps']==5],'rmse768_lower'),
      'bootstrap_vs_last_increment_h5':bootstrap_by_film([r for r in metrics if r['horizon_actual_steps']==5],'rmse768_lower','last_increment'),
      'bootstrap_vs_phase_shrink_projection':bootstrap_by_film(metrics,'rmse32','phase_velocity_shrink'),
      'bootstrap_test_film_projection32':bootstrap_by_film(metrics,'rmse32')}

def supplement(root):
    folder=root/'population900k_dmdc_supplement';strata=read(folder/'METRICS_STRATIFIED.json');byrole=defaultdict(lambda:np.zeros(5));exclusive=defaultdict(lambda:np.zeros(2))
    for r in strata:
        k=(r['split'],r['phase'],r['status'],'target' if r['role_mask']&1 else 'other',r['horizon'],r['method']);n=r['n']
        byrole[k]+=np.array([n,r['rmse32']**2*n,r['rmse768_lower']**2*n,r['rmse768_upper']**2*n,r['pred_net_norm32_mean']*n])
        mask=r['role_mask'];role='target' if mask&1 else 'natural_top1' if mask&2 else 'competitor' if mask&4 else 'other'
        exclusive[(r['split'],r['phase'],role,r['horizon'],r['method'])]+=np.array([n,r['rmse768_lower']**2*n])
    with (folder/'METRICS_PER_IMAGE.csv').open(encoding='utf-8') as f:metrics=list(csv.DictReader(f))
    sampling=read(folder/'TRAIN_SAMPLING.json');population_inputs=read(root/'population900k/INPUTS_SPLITS.json');byid={r['image_index']:r for r in population_inputs}
    if any(r['split']!='train' or byid[r['image_index']]['split']!='train' or r['film']!=byid[r['image_index']]['film'] for r in sampling):raise ValueError('Supplement sampling leakage')
    counts=defaultdict(int)
    for r in metrics:counts[(r['method'],int(r['horizon']))]+=int(r['n'])
    if any(n!=900000*(100 if h==5 else 97) for (m,h),n in counts.items()):raise ValueError('Incomplete supplement predictions')
    ood=read(folder/'CONTROL_OOD.json');ood_summary=[]
    for split in ['train','validation','test']:
        selected=[r for r in ood if r['split']==split]
        ood_summary.append(dict(split=split,images=len(selected),max_abs_scaled_by_control=np.array([r['max_abs_scaled_by_control'] for r in selected]).max(0).tolist(),fraction_outside_fit_train_range=np.array([r['fraction_outside_train_range_by_control'] for r in selected]).mean(0).tolist()))
    return {'integrity':dict(train_sampling_verified=True,train_images=len(sampling),all_900000_tokens_forecasted=True,windows_per_method=[dict(method=m,horizon=h,n=n) for (m,h),n in counts.items()]),
      'control_ood_summary':ood_summary,
      'exclusive_role_priority':'target bit1, then natural_top1 bit2, then competitor bit4, then other; status pooled; h5 full768 only, h20 remains lower bound',
      'exclusive_roles':[dict(split=k[0],phase=k[1],role=k[2],horizon=k[3],method=k[4],n=int(v[0]),rmse768_lower=np.sqrt(v[1]/v[0])) for k,v in exclusive.items()],
      'by_role':[dict(split=k[0],phase=k[1],status=k[2],role=k[3],horizon=k[4],method=k[5],n=int(v[0]),rmse32=np.sqrt(v[1]/v[0]),rmse768_lower=np.sqrt(v[2]/v[0]),rmse768_upper=np.sqrt(v[3]/v[0]),predicted_norm32=v[4]/v[0]) for k,v in byrole.items()],
      'bootstrap_test_film_h5_full768':bootstrap_by_film([r for r in metrics if int(r['horizon'])==5],'rmse768_lower'),
      'bootstrap_vs_state_DMD_h5':bootstrap_by_film([r for r in metrics if int(r['horizon'])==5],'rmse768_lower','deeptime_DMD'),
      'bootstrap_future_vs_frozen_h20':bootstrap_by_film([r for r in metrics if int(r['horizon'])==20],'rmse32','pydmd_DMDc_frozen_startU'),
      'bootstrap_test_film_projection32':bootstrap_by_film(metrics,'rmse32')}

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--available',action='store_true');args=p.parse_args()
    status=read(args.root/'TRANSFER_WYNIKOW.json');completed=set(status.get('completed_all_hashes_verified',[]))
    if not args.available and completed!=set(FAMILIES):raise RuntimeError('Collector has not verified all three outputs')
    result={'description':'Post hoc evidence summary, no model fitting/selection; film cluster bootstrap descriptive, not proof of generalization to new distributions','created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'bootstrap_replicates':2000,'seed':20261009,'verified':{}}
    for name in FAMILIES:
        if name not in completed:continue
        result['verified'][name]=verify(args.root,name)
        result[name]={'panel64':panel,'population900k':population,'population900k_dmdc_supplement':supplement}[name](args.root)
    (args.root/'SYNTEZA_DIAGNOSTYKA.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('Analyzed verified outputs:',sorted(result['verified']))

if __name__=='__main__':main()
