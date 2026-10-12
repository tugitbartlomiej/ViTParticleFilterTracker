"""Rozlicza zapisane wyniki; sprzeczność z artefaktami obala historyczny opis.

Bez fit, model inference i odczytu obrazów. Wyniki opisowe, nie test przyczynowy.
"""
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FAMILY = 'vit_k1000_analiza_rezimow_astra_20261009'
OUT = ROOT / 'outputs' / FAMILY
SRC = ROOT / 'outputs/vit_k1000_1500_pelne_latenty_90_20261008'
RES = SRC / 'eden_results'
VIS = ROOT / 'outputs/vit_k1000_taniec_czasteczek_20261009/data'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def q(x):
    a = np.asarray(x)
    return {'n': int(a.size), 'q': {str(k): float(np.quantile(a, k)) for k in [0,.1,.25,.5,.75,.9,1]}} if a.size else {'n': 0}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = read(VIS/'metadata.json')['images']
    panel = read(SRC/'provenance/CONTROL_PANEL.json')
    mov = np.load(RES/'movement/MOVEMENT_ARRAYS.npz', allow_pickle=False)
    proj = np.load(VIS/'projection.npz', allow_pickle=False)
    first = np.array([-1 if m['first_success_step'] is None else m['first_success_step'] for m in meta])
    targets = np.array([m['target_token'] if 'target_token' in m else m['target']['token_index'] for m in meta])
    success = first >= 0
    after = mov['after_step']
    a = mov['particle_l2_k0_k100'].astype(float)
    b = mov['particle_l2_k100_k500'].astype(float)
    c = mov['cosine_early_late_by_image_token']
    d = mov['particle_l2_k0_after'].astype(float)
    target_mask = np.arange(600)[None] == targets[:,None]
    mean = np.load(RES/'regimes/MEAN_DELTA_ALL.npz')['mean_delta'].astype(float)
    mnorm = np.linalg.norm(mean,axis=1)
    counts = np.bincount(targets,minlength=600)
    report = {
      'scope':'descriptive statistics on existing artifacts; no model fit',
      'first_success':q(first[success]),
      'after_steps_success':q(after[success]), 'after_steps_failure':q(after[~success]),
      'after_not_500':int((after!=500).sum()),
      'after_is_first_success':int(((after==first)&success).sum()),
      'success_counts_at_steps':{str(t):int(((proj['confidence'][:,t]>=.9)&(proj['iou'][:,t]>=.9)).sum()) for t in [0,25,50,100,250,400,500]},
      'first_success_by_100':int(((first>=0)&(first<=100)).sum()),
      'ever_success_but_not_at500':int((success & ~((proj['confidence'][:,500]>=.9)&(proj['iou'][:,500]>=.9))).sum()),
      'target_patch_counts':{'used_patches':int((counts>0).sum()),'maximum_images_one_patch':int(counts.max()),'max_fraction':float(counts.max()/1500),'counts':counts.tolist()},
      'panel_target_coverage':[{'image':int(i),'target':int(targets[int(i)]),'target_in_panel':bool(targets[int(i)] in toks),'first_success':int(first[int(i)]),'film':meta[int(i)]['frame_id'].split('/')[2]} for i,toks in panel.items()],
      'mean_delta':{'norm_per_patch':q(mnorm),'coherence_per_patch_norm_mean_over_mean_norm':q(mnorm/d.mean(0)), 'energy_fraction_of_constant_projection':float(1500*np.square(mean).sum()/np.square(d).sum())},
      'endpoint_norms':{'target':q(d[target_mask]), 'non_target':q(d[~target_mask]),'target_energy_fraction':float(np.square(d[target_mask]).sum()/np.square(d).sum())},
      'reversal_by_minimum_segment_norm':{}, 'reversal_by_phase_group':{},
      'storage_bytes':{'full_f32':1500*501*600*768*4, 'full_f16':1500*501*600*768*2, 'all_pca32_f32':1500*501*600*32*4,'panel128_full_f32':128*501*768*4,'full16images_f32':16*501*600*768*4,'full64images_f32':64*501*600*768*4},
    }
    for eps in [0,1e-6,1e-4,1e-2,1]:
        sel=(a>eps)&(b>eps)
        report['reversal_by_minimum_segment_norm'][str(eps)]={'count':int(sel.sum()),'opposite':int((sel&(c<0)).sum()),'cosine':q(c[sel])}
    for name,mask in [('success_by100',success&(first<=100)),('success_after100',first>100),('never_success',~success)]:
        report['reversal_by_phase_group'][name]={'images':int(mask.sum()),'opposite_fraction':float((c[mask]<0).mean()),'end500norm':q(mov['particle_l2_k0_k500'][mask]),'earlynorm':q(a[mask])}
    rd=read(RES/'regimes/REGIME_DIAGNOSTICS.json')
    report['regime_assignment']={}
    for name,v in [('all',rd['all'])]+list(rd['folds'].items()):
        cm=np.array(v['assignment_confusion']); total=cm.sum()
        report['regime_assignment'][name]={'accuracy':float(np.trace(cm)/total),'majority':float(cm.sum(1).max()/total),'balanced_recall':float(np.mean(np.diag(cm)/cm.sum(1))),'training_images':v['training_images']}
    report['dmdc_holdout_films']={str(i):meta[i]['frame_id'].split('/')[2] for i in [0,500,999,1499]}
    upper=np.array([m['target']['geometry_iou_upper_bound'] for m in meta])
    report['geometry_upper_bounds']={'quantiles':q(upper),'below_90':int((upper<.9).sum()),'below_90_and_failure':int(((upper<.9)&~success).sum())}
    report['endpoint_phase_split']={}
    for label,mask in [('success',success),('budget_exhausted',~success)]:
        norms=d[mask]
        report['endpoint_phase_split'][label]={'images':int(mask.sum()),'total_energy_fraction':float(np.square(norms).sum()/np.square(d).sum()),'norms':q(norms),'non_target_norms':q(d[mask][~target_mask[mask]]),'target_norms':q(d[mask][target_mask[mask]]),'particles_norm_below_1e4':int((norms<1e-4).sum())}
    report['endpoint_near_zero_counts']={str(eps):int((d<eps).sum()) for eps in [1e-6,1e-5,1e-4,.01,1]}
    rng=np.random.default_rng(20261008)
    sample=rng.choice(1500*600,size=50000,replace=False)
    report['B_direction_PCA_sample_near_zero_fraction']=float((d.reshape(-1)[sample]<1e-4).mean())
    with np.load(RES/'regimes/REGIME_MODEL_ALL.npz') as model:
        counts_rt=model['regime_token_counts']
        report['regime_token_counts']={'quantiles':q(counts_rt),'zeros':int((counts_rt==0).sum()),'below20':int((counts_rt<20).sum()),'cells':int(counts_rt.size)}
    # Bound successful contribution to mean using triangle inequality.
    success_sum_norm=d[success].sum(0)
    report['success_contribution_to_A_norm_upper_bound_per_patch']=q(success_sum_norm/1500/mnorm)
    (OUT/'DIAGNOSTYKA_LOKALNA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('saved',OUT/'DIAGNOSTYKA_LOKALNA.json')

if __name__=='__main__':
    main()
