"""Pomiar pełnych tokenów par; różne ROI, NaN lub zmiana BN obala zgodność."""
import json,time,shutil,platform
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,torch,psutil,timm
from dane import HERE,DATA,CHECKPOINT,load,image,tensor,init_net,sha,js

def main():
    start=time.time();rows,t=load()
    hardware=dict(disk_free=shutil.disk_usage(HERE).free,ram_free=psutil.virtual_memory().available,gpu_free=torch.cuda.mem_get_info()[0],gpu_name=torch.cuda.get_device_name(0))
    assert hardware['disk_free']>1.25e9 and hardware['ram_free']>3e9 and hardware['gpu_free']>2e9
    net=init_net();bn={n:b.clone() for n,b in net.named_buffers()}
    state=[];bgstate=[];token_ids=[];bgids=[];offset=[0];outmaps=[];full_l2=[];bgstats=[]
    pooled=[];pooledbg=[];coverage=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            a=image(DATA/r['image_path']);b=image(r['synthetic_absolute'])
            o,z=net(tensor([a,b]));z=z.cpu().numpy();assert z.shape==(2,600,768) and np.isfinite(z).all()
            oi=o.cpu().numpy();assert np.isfinite(oi).all()
            ids=np.flatnonzero(t['tool'][i]);assert len(ids)>0
            cov=t['area_coverage'][i,ids].astype(np.float32);assert (cov>0).all()
            controls=np.flatnonzero(t['background'][i]);assert len(controls)>=8
            chosen=controls[np.rint(np.linspace(0,len(controls)-1,8)).astype(int)]
            state.append(z[:,ids].transpose(1,0,2));bgstate.append(z[:,chosen].transpose(1,0,2))
            token_ids.extend(ids);bgids.append(chosen);offset.append(offset[-1]+len(ids));coverage.extend(cov)
            pooled.append((z[:,ids]*cov[None,:,None]).sum(1)/cov.sum())
            pooledbg.append(z[:,controls].mean(1))
            delta=z[0]-z[1];energy=np.linalg.norm(delta,axis=1);full_l2.append(energy)
            bgstats.append(dict(source_index=i,tool_l2_mean=float(np.average(energy[ids],weights=cov)),far_background_l2_mean=float(energy[controls].mean()),far_background_n=len(controls)))
            outmaps.append(oi)
            if (i+1)%25==0:print('paired forward',i+1,'/400',round(time.time()-start,1),'s',flush=True)
        _,repeat=net(tensor([a,b]));assert np.max(np.abs(repeat.cpu().numpy()-z))<=1e-6
    assert all(torch.equal(b,bn[n]) for n,b in net.named_buffers())
    target=HERE/'TOKENY_PARY.npz'
    np.savez_compressed(target,states=np.concatenate(state).astype(np.float32),token_ids=np.array(token_ids,np.int16),offsets=np.array(offset,np.int32),coverage=np.array(coverage,np.float32),
        background_states=np.concatenate(bgstate).astype(np.float32),background_ids=np.array(bgids,np.int16),pooled=np.array(pooled,np.float32),pooled_background=np.array(pooledbg,np.float32),
        full_delta_l2=np.array(full_l2,np.float32),outputs=np.array(outmaps,np.float32),frame_ids=np.array([r['frame_id'] for r in rows]))
    js(HERE/'TLO_DIAGNOSTYKA.json',bgstats)
    metadata=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),count=400,images=800,layer='last block mlp.act GELU',measurement='eval fp32 no_autocast noTF32',
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),checkpoint_updates=1000,
        model_source_sha256=sha(HERE/'model_snapshot.py'),manifest_sha256=sha(DATA/'manifest.json'),pair_manifest_sha256=sha(DATA/'PARY.json'),token_masks_sha256=sha(DATA/'TOKEN_MASKI.npz'),
        saved_token_states_shape=np.concatenate(state).shape,full768_float32=True,background_tokens_per_image=8,npz_bytes=target.stat().st_size,npz_sha256=sha(target),
        split=json.loads((HERE/'PODZIAL.json').read_text(encoding='utf-8'))['image_counts'],hardware=hardware,seconds=time.time()-start,buffers_unchanged=True,repeat_max_abs_error=float(np.max(np.abs(repeat.cpu().numpy()-z))),all_finite=True,empty_tool_masks=0,
        versions=dict(python=platform.python_version(),torch=torch.__version__,timm=timm.__version__,numpy=np.__version__),gpu_peak_bytes=torch.cuda.max_memory_allocated(),training=False)
    js(HERE/'POMIAR.json',metadata)
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<250e6
    assert shutil.disk_usage(HERE).free>1e9
    print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
