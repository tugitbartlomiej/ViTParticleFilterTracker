from pathlib import Path
import json,numpy as np
from analyze import OUT,save,film_ci
def run():
    result={}
    for mode in ['dense','long']:
        fs=[OUT/f'{mode}_image{i}_direct.json' for i in range(8)]
        if not all(f.exists() for f in fs):continue
        data=[json.loads(f.read_text()) for f in fs];res={'dynamic_cage':[],'turning':[]}
        for j,a in enumerate(data[0]['dynamic_cage']):
            v=np.array([d['dynamic_cage'][j]['mean_cage_msd'] for d in data]);q=np.array([d['dynamic_cage'][j]['shell_relative_overlap'] for d in data]);ds=np.array([d['dynamic_cage'][j]['mean_shell_distance_sq'] for d in data])
            res['dynamic_cage'].append(dict(age=a['age'],lag=a['lag'],mean=v.mean(0),per_image=v,ci=film_ci(v),shell_distance_sq=ds,relative_rms=np.sqrt(v/ds[:,None]).mean(0),overlap_mean=q.mean(0),overlap_threshold_factors=[.1,.2,.5]))
        for j,t in enumerate(data[0]['turning']):
            r=dict(start=t['start'],end=t['end'],increment=t['increment'])
            for key in ['adjacent_correlation','turning_fraction','mean_token_cosine']:
                v=np.array([d['turning'][j][key] for d in data]);r[key]=dict(mean=v.mean(),per_image=v,ci=film_ci(v))
            res['turning'].append(r)
        result[mode]=res
    # Large-lag original-checkpoint controls, same images, FP32 inference, fixed BN and unit vectors.
    cp={int(p.stem[1:]):p for p in (OUT/'precision_controls').glob('k*.npz')}
    comparisons=[]
    for a,b in [(300,310),(310,320),(900,910),(910,920),(990,1000),(300,1000),(10000,20000),(30000,40000),(50000,60000),(60000,65640)]:
        if a not in cp or b not in cp:continue
        z1=np.load(cp[a]);z2=np.load(cp[b]);r=dict(start=a,end=b,lag=b-a)
        for name in ['fp32','bf16','frozen_bn']:
            x=z1[name].astype(np.float64);y=z2[name].astype(np.float64)
            r[name]=((y-x)**2).sum(-1).mean(1)
            x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
            y/=np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-12)
            r[name+'_unit_vectors']=((y-x)**2).sum(-1).mean(1)
        comparisons.append(r)
    result['precision_bn_comparisons']=comparisons
    save('DIRECT_CONTROLS.json',result)
    print('summarized',list(result),flush=True)
if __name__=='__main__':run()
