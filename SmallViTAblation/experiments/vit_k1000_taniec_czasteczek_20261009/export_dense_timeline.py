"""Assemble genuine image shards into verified streaming GPU frames on Eden.

Requires an atomically published producer manifest. Does not compute/interpolate
latent states; avoids saving the full 768-dimensional population to disk.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from dense_timeline import BASIS_SHA,COUNT
IMAGE_COUNT=1500
TOKENS=600

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(p,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--position-key',default='projected_positions');parser.add_argument('--norm-key',default='movement_norm');parser.add_argument('--steps-key',default='projection_steps');args=parser.parse_args()
    src=json.loads(args.source_manifest.read_text(encoding='utf-8'));steps=src['steps'];files=src['files']
    if src['status']!='complete' or src['point_count']!=COUNT or src['pca_basis_sha256']!=BASIS_SHA or src['interpolation'] is not False:raise ValueError('Producent nie opublikował zgodnego, kompletnego zapisu')
    stride=steps[1]-steps[0]
    if steps!=list(range(0,501,stride)) or stride>10 or len(files)!=IMAGE_COUNT or sorted(int(r['image_index']) for r in files)!=list(range(IMAGE_COUNT)):raise ValueError('Niepełna populacja lub timeline')
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Finalny eksport już istnieje; nie nadpisuj zweryfikowanego zapisu')
    paths={step:root/f'step_{step:03d}.bin.part' for step in steps}
    arrays={step:np.memmap(p,dtype='<f4',mode='w+',shape=(COUNT,4)) for step,p in paths.items()}
    confidence=np.empty((IMAGE_COUNT,501),np.float32);iou=np.empty_like(confidence)
    started=time.perf_counter()
    for ordinal,record in enumerate(files,1):
        i=int(record['image_index']);p=Path(record['path']);p=p if p.is_absolute() else args.source_manifest.parent/p
        if sha(p)!=record['sha256']:raise ValueError(f'Niezgodny hash obrazu {i}')
        with np.load(p,allow_pickle=False) as f:
            positions=f[args.position_key];norms=f[args.norm_key]
            if positions.shape!=(len(steps),TOKENS,3) or norms.shape!=(len(steps),TOKENS) or not np.array_equal(f[args.steps_key],steps):raise ValueError(f'Niezgodne położenia/kroki obrazu {i}')
            if positions.dtype!=np.float32 or norms.dtype!=np.float32 or not np.isfinite(positions).all() or not np.isfinite(norms).all():raise ValueError('Nieprawidłowe dane położenia')
            if not np.array_equal(f['telemetry_step'],np.arange(501)):raise ValueError('Niepełna telemetria')
            confidence[i]=f['telemetry_confidence'];iou[i]=f['telemetry_iou']
            for k,step in enumerate(steps):
                arrays[step][i*TOKENS:(i+1)*TOKENS,:3]=positions[k];arrays[step][i*TOKENS:(i+1)*TOKENS,3]=norms[k]
        if ordinal%100==0:print(json.dumps({'assembled_images':ordinal,'seconds':time.perf_counter()-started}),flush=True)
    for arr in arrays.values():arr.flush()
    initial=arrays[0][:,:3];bounds={};stats={};norm_scale=0.;heat_scale=0.
    for movement in (False,True):
        lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
        for arr in arrays.values():
            xyz=arr[:,:3]-initial if movement else arr[:,:3]
            lo=np.minimum(lo,xyz.min(axis=0));hi=np.maximum(hi,xyz.max(axis=0))
        pad=np.maximum((hi-lo)*.04,.03);bounds['movement' if movement else 'natural']=np.column_stack([lo-pad,hi+pad]).tolist()
    for step,arr in arrays.items():
        norm=arr[:,3];heat=norm.reshape(IMAGE_COUNT,TOKENS).mean(axis=0,dtype=np.float64).reshape(20,30)
        stats[str(step)]={'norm_median':float(np.median(norm)),'norm_max':float(norm.max()),'confidence':float(confidence[:,step].mean(dtype=np.float64)),'iou':float(iou[:,step].mean(dtype=np.float64)),'both90':int(((confidence[:,step]>=.9)&(iou[:,step]>=.9)).sum()),'heatmap':heat.tolist()}
        norm_scale=max(norm_scale,float(norm.max()));heat_scale=max(heat_scale,float(heat.max()))
    # Drop mmap objects before renaming, including Windows-compatible local use.
    del initial,xyz,arr,norm;arrays.clear()
    outputs={}
    for step,p in paths.items():
        final=p.with_suffix('');os.replace(p,final);outputs[str(step)]={'name':final.name,'bytes':final.stat().st_size,'sha256':sha(final)}
    manifest={'status':'complete','schema_version':1,'point_count':COUNT,'steps':steps,'stride':stride,'interpolation':False,'pca_basis_sha256':BASIS_SHA,'files':outputs,'bounds':bounds,'norm_scale':max(norm_scale,1e-6),'heat_scale':max(heat_scale,1e-6),'stats':stats,'curves':{'confidence':confidence.mean(axis=0,dtype=np.float64).tolist(),'iou':iou.mean(axis=0,dtype=np.float64).tolist(),'label':'Średnia rzeczywistego replay 1500 obrazów'},'source_manifest_sha256':sha(args.source_manifest),'source_manifest':str(args.source_manifest),'producer_provenance':{k:v for k,v in src.items() if k not in {'files','stats'}},'seconds':time.perf_counter()-started}
    atomic_json(root/'manifest.json',manifest);print('COMPLETE',root/'manifest.json',flush=True)

if __name__=='__main__':main()
