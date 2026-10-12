"""Reconstruct one full 384x384 matrix from exact factors and audited trajectory."""
import os
os.environ['OMP_NUM_THREADS']='4'
import argparse,json
import numpy as np
import torch
from dane import HERE,SOURCE,OUT,sha,write
from operator_lokalny import Operators

def run(image,token,stage):
    if not 0<=image<72 or not 0<=token<576:raise ValueError('Image/token index out of bounds')
    report=OUT/'REPORT.json'
    if not report.exists() or json.loads(report.read_text()).get('state')!='completed':raise RuntimeError('Analysis is not complete')
    if not (OUT/f'IMAGE_{image:03d}_DONE.json').exists():raise RuntimeError('Image output incomplete')
    inputs=json.loads((OUT/'INPUT_SHA256.json').read_text())
    if [x['step'] for x in inputs]!=list(range(5,370,5)):raise RuntimeError('Incomplete input manifest')
    for item in inputs:
        if sha(SOURCE/f"TOKENS_{item['step']:07d}.npy")!=item['sha256']:raise RuntimeError('Changed input tensor')
    manifest=json.loads((OUT/'OUTPUT_SHA256.json').read_text())
    names=['C_FINAL.npy','MEAN_X_FINAL.npy','MEAN_DELTA_FINAL.npy'] if stage=='final' else ['RELATIVE_RIDGE.npy']
    for name in names:
        if sha(OUT/name)!=manifest[name]:raise RuntimeError(f'Changed operator factor {name}')
    states=np.stack([np.load(SOURCE/f'TOKENS_{step:07d}.npy',mmap_mode='r')[image,token] for step in range(5,370,5)]).astype('float64')
    if stage=='final':
        coeff=np.array(np.load(OUT/'C_FINAL.npy',mmap_mode='r')[image,token])
        mx=np.load(OUT/'MEAN_X_FINAL.npy',mmap_mode='r')[image,token]
        md=np.load(OUT/'MEAN_DELTA_FINAL.npy',mmap_mode='r')[image,token]
        B=(states[:-1]-mx).T@coeff;K=np.eye(384)+B;b=md-mx@B
        cutoff=365;forecasts_verified=False
    else:
        relative=float(np.load(OUT/'RELATIVE_RIDGE.npy',mmap_mode='r')[image,token])
        m=Operators().fit(torch.tensor(states[None,:48]),torch.tensor(states[None,1:49]),relative)
        Kt,bt=m.explicit();K=Kt[0].numpy();b=bt[0].numpy();cutoff=245;forecasts_verified=True
    name=f'MATRIX_{stage}_{image:03d}_{token:03d}.npz';path=OUT/name
    np.savez(path,K=K,b=b,image_index=image,token_index=token,cutoff_step=cutoff,
        convention='row-vector z_next=z@K+b',forecast_evaluated=forecasts_verified)
    write(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),stage=stage,
        source='PROTOCOL.json and INPUT_SHA256.json',full_history_prediction_unverified=stage=='final'))
    print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',type=int,required=True);p.add_argument('--token',type=int,required=True)
    p.add_argument('--stage',choices=['final','evaluation'],default='final');a=p.parse_args()
    with torch.no_grad():run(a.image,a.token,a.stage)
