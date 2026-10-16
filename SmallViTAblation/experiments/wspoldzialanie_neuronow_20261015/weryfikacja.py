"""Independent checks on saved artifacts, not a mirror of the selection algorithm."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015'

def overlap(a,b):
    lo=np.maximum(a[...,:2],b[...,:2]);hi=np.minimum(a[...,2:],b[...,2:])
    intersection=np.clip(hi-lo,0,None).prod(-1)
    area1=np.prod(a[...,2:]-a[...,:2],axis=-1);area2=np.prod(b[...,2:]-b[...,:2],axis=-1)
    return intersection/(area1+area2-intersection)

def main():
    d=pd.read_csv(OUT/'interwencje_lokalne.csv.gz');g=pd.read_csv(OUT/'interwencje_globalne.csv.gz')
    group=pd.read_csv(OUT/'interwencje_grupowe.csv.gz');base=pd.read_csv(OUT/'ramki_bazowe.csv')
    m=json.loads((OUT/'MANIFEST.json').read_text());gt=np.load(ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz')['gt']
    assert len(d)==34*3*8*13*4*3==m['n_local']
    assert len(g)==34*3*13*4*3==m['n_global']
    assert len(group)==34*3*7*3*3*2
    assert len(base)==34*3*8
    assert not d.duplicated(['step','image','token_slot','method','pair_slot','alpha']).any()
    assert not group.duplicated(['step','image','method','size','scope','alpha']).any()
    assert set(d.step)==set(m['steps'])
    da=d[[f'da{k}' for k in range(4)]].values;db=d[[f'db{k}' for k in range(4)]].values;dj=d[[f'dj{k}' for k in range(4)]].values
    denom=np.linalg.norm(da,axis=1)+np.linalg.norm(db,axis=1)
    valid=denom>1e-3
    assert np.array_equal(valid,d.valid_ratio.values)
    q=np.linalg.norm(dj[valid],axis=1)/denom[valid]
    errq=float(np.max(np.abs(q-d.loc[valid,'q_actual'].values)));assert errq<1e-10
    inter=np.linalg.norm(dj-da-db,axis=1)/2
    erri=float(np.max(np.abs(inter-d.interaction_px.values)));assert erri<1e-10
    assert np.allclose(np.linalg.norm(dj,axis=1)/2,d.joint_px,atol=1e-10)
    joined=d.merge(base[['step','image','token_slot']+[f'b{k}' for k in range(4)]],on=['step','image','token_slot'],validate='many_to_one',sort=False)
    assert len(joined)==len(d)
    b=joined[[f'b{k}' for k in range(4)]].values;truth=gt[joined.image.values]
    j=joined[[f'dj{k}' for k in range(4)]].values
    delta=overlap(b+j,truth)-overlap(b,truth)
    errlocal=float(np.max(np.abs(delta-joined.delta_iou.values)));assert errlocal<1e-10
    old=g[[f'base{k}' for k in range(4)]].values;new=g[[f'new{k}' for k in range(4)]].values
    errglob=float(np.max(abs(overlap(new,gt[g.image.values])-overlap(old,gt[g.image.values])-g.delta_iou.values)))
    assert errglob<2e-6
    assert np.allclose(np.linalg.norm(new-old,axis=1)/2,g.joint_px,atol=2e-5,rtol=2e-5)
    assert np.array_equal((g.new_token!=g.base_token).astype(int),g.switched)
    # The explicit replay must agree with the previously recorded outlier.
    replay=json.loads((OUT/'przelaczenie_weryfikacja.json').read_text())
    row=group[(group.step==55000)&(group.image==2)&(group.method=='reinforce')&(group['size']==32)&(group.scope=='global')&(group.alpha==1)].iloc[0]
    assert [x['winner'] for x in replay['variants']]==[369,369,369,370]
    assert all(x['winner']==x['double_winner'] for x in replay['variants'])
    assert np.allclose(replay['variants'][-1]['box'],row[[f'new{k}' for k in range(4)]].values.astype(float),atol=1e-4)
    assert abs(replay['variants'][-1]['iou']-row.joint_iou)<1e-5
    # All recorded inputs and output artifacts remain exactly the same.
    fm=json.loads((OUT/'MANIFEST_KONCOWY.json').read_text())
    for name,info in fm.items():
        file=ROOT/name;assert file.stat().st_size==info['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==info['sha256'],name
    for step,digest in m['checkpoint_sha256'].items():
        assert hashlib.sha256((ROOT/f'assets/MalyViT_35000_od_zera_s47/wagi/k{int(step):05d}.pt').read_bytes()).hexdigest()==digest
    assert hashlib.sha256(Path(__file__).with_name('pomiar.py').read_bytes()).hexdigest()==m['script_sha256']
    assert hashlib.sha256(Path(__file__).with_name('PROTOKOL.md').read_bytes()).hexdigest()==m['protocol_sha256']
    summary=dict(status='PASS',local_rows=len(d),global_pair_rows=len(g),group_rows=len(group),
                 q_recompute_max_error=errq,interaction_recompute_max_error=erri,local_iou_max_error=errlocal,
                 global_iou_max_error=errglob,checked_manifest_files=len(fm),checked_checkpoints=len(m['checkpoint_sha256'])+1,
                 decoder_switch_replayed=True,invalid_ratios=int((~valid).sum()))
    (OUT/'WERYFIKACJA.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
