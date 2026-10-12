"""Project real FP32 768D displacements onto the row space and null space of final MLP fc2.
This is a conditional fixed-readout decomposition, not an attribution of prediction changes.
Both endpoint readouts and frozen-BN activations are examined. No training is performed.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import torch

ROOT=Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation')
LAT=ROOT/'outputs/audyt_szkla_768d_20261010/precision_controls'
WEIGHTS=ROOT/'experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/pelny/wagi'
OUT=ROOT/'outputs/poszlaki_dynamiki_20261011'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
pairs=[(300,310),(990,1000),(1094,2188),(5470,10940),(27350,32820),(54700,60170)]
rows=[];sources={}
for start,end in pairs:
    za=np.load(LAT/f'k{start:05d}.npz');zb=np.load(LAT/f'k{end:05d}.npz')
    for anchor in (start,end):
        path=WEIGHTS/f'k{anchor:05d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)['net']
        names=[k for k in state if k.endswith('bloki.5.mlp.fc2.weight')]
        assert len(names)==1,names
        W=state[names[0]].double()
        assert tuple(W.shape)==(192,768),W.shape
        U,s,Vh=torch.linalg.svd(W,full_matrices=False)
        rank=int((s>s.max()*1e-10).sum())
        basis=Vh[:rank].T
        sources[str(path)]=dict(key=names[0],rank=rank,singular_min=float(s.min()),singular_max=float(s.max()))
        for variant in ('fp32','frozen_bn'):
            d=torch.from_numpy((zb[variant]-za[variant]).astype('float64'))
            for centering in ('raw','per_image'):
                D=d if centering=='raw' else d-d.mean(1,keepdim=True)
                coords=D@basis
                total=D.square().sum((1,2));visible=coords.square().sum((1,2));null=total-visible
                projected_null=D-coords@basis.T
                error=float(torch.linalg.vector_norm(projected_null@W.T)/(torch.linalg.vector_norm(D@W.T)+1e-30))
                rows.append(dict(start=start,end=end,lag=end-start,anchor=anchor,activation=variant,centering=centering,
                                 rank=rank,isotropic_null_fraction=(768-rank)/768,
                                 null_fraction=float(null.sum()/total.sum()),
                                 per_image_null_fraction=(null/total).tolist(),
                                 displacement_MSD=float(total.sum()/(D.shape[0]*D.shape[1])),
                                 conditional_fc2_output_MSD=float((D@W.T).square().sum(-1).mean()),
                                 null_projection_relative_output_error=error))
    print('done',start,end,flush=True)
result={'description':'Fixed-fc2 null projection of real 768D activation displacement, original 8 probes; endpoint sensitivity and frozen-BN controls.',
        'limitations':['Measured pre-fc2 activation has 768 channels; residual width is 192.',
                      'fc2 null directions are exactly invisible at fixed downstream weights and fixed residual branch, but actual training changes both.',
                      'A 75% null share is the dimensional isotropic baseline, not evidence by itself.',
                      'Frozen BN uses the pre-existing step-300 reference statistics; it is an intervention, not a physiological reference.',
                      'These are selected endpoint pairs, not a dense-time estimate or independent seed replication.'],
        'sources':sources,'rows':rows}
(OUT/'readout_nullspace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows:
    if r['anchor']==r['start'] and r['centering']=='per_image':
        print(json.dumps(r,ensure_ascii=False),flush=True)
