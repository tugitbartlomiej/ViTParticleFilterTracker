"""Stałe tokeny muszą zachować obraz, coverage i oddzielić U500 od przyszłego U1000."""
import time,shutil
import numpy as np
from dane import HERE,SOURCE,PREVIOUS,read,js,sha

def main():
    start=time.time();probe=2;meta=read(SOURCE/'dane/metadata.json');images=read(SOURCE/'dane/images.json');m=dict(np.load(SOURCE/'dane/maski.npz'))
    assert images[probe]['frame_id']=='cataract_lmm/tracking/TR_0057_S1_P03/00001431' and images[probe]['role']=='fit'
    src=SOURCE/'dane/LATENT_GESTY.npy';receipt=read(SOURCE/'pobranie_latentu_2026-10-01/MANIFEST_LOKALNY_2026-10-01.json');assert receipt['sha256_match'] and receipt['remote_sha256']==meta['latent_sha256']
    a=np.load(src,mmap_mode='r');assert a.shape==(1001,12,600,768) and a.dtype==np.float32
    from dane import init_net,tensor
    import torch
    net=init_net(500);source_image=np.load(SOURCE/'dane/obrazy.npy',mmap_mode='r')[probe]
    with torch.no_grad():_,forward=net(tensor([np.array(source_image)]))
    actual=forward[0].cpu().numpy().astype(np.float64);reference=np.array(a[500,probe],dtype=np.float64)
    relative=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference));cosine=float(np.sum(actual*reference)/np.linalg.norm(actual)/np.linalg.norm(reference));maxabs=float(np.max(np.abs(actual-reference)))
    js(HERE/'ZGODNOSC_K500.json',dict(relative_l2_error=relative,max_abs_error=maxabs,cosine=cosine,tolerance_relative=1e-4,passed=relative<1e-4,shape=list(actual.shape),source_probe_index0=probe))
    assert relative<1e-4,'Local checkpoint does not reproduce the measured token channels'
    tool=np.flatnonzero(m['tool'][probe]);background=np.flatnonzero(m['background'][probe]);control=background[np.rint(np.linspace(0,len(background)-1,8)).astype(int)];ids=np.r_[tool,control]
    z=np.empty((1001,len(ids),768),np.float32)
    for t in range(1001):z[t]=a[t,probe,ids]
    assert np.isfinite(z).all();cal=dict(np.load(HERE/'IDENTYFIKACJA_K500.npz'));U=cal['U500'];q=z.astype(np.float64)@U;axes=cal['axes500_0'];cov=m['coverage'][probe,tool].astype(np.float64);w=cov/cov.sum()
    np.savez_compressed(HERE/'TRAJEKTORIE_PRZYCZYNOWE_K500.npz',q500=q,axes500_values=z[:,:,axes],axes500_0=axes,token_ids=ids,tool_count=len(tool),tool_weights=w,coverage600=m['coverage'][probe],mask_input=m['mask_input'][probe],steps=np.arange(1001),U500=U,probe_index=probe)
    old=dict(np.load(PREVIOUS/'PODPRZESTRZEN.npz'));oldaxes=np.array([499,471,734,511,646])
    np.savez_compressed(HERE/'TRAJEKTORIE_RETROSPEKTYWNE_K1000.npz',q1000=z.astype(np.float64)@old['U'],old_axes0=oldaxes,old_axes1=oldaxes+1,old_axes_values=z[:,:,oldaxes],U1000=old['U'],token_ids=ids,description='Retrospective only: basis/axis selection from final checkpoint1000')
    import cv2
    cv2.imwrite(str(HERE/'SONDA_STALA.png'),cv2.cvtColor(source_image,cv2.COLOR_RGB2BGR))
    record=dict(probe_index0=probe,frame_id=images[probe]['frame_id'],source_role=images[probe]['role'],source_split=images[probe]['split'],tool_count=len(tool),tool_ids0=tool,tool_ids1=tool+1,background_control_ids0=control,weights=w,steps=[0,1000],
        latent_source_shape=list(a.shape),latent_source_sha256_from_verified_receipt=meta['latent_sha256'],latent_source_bytes=src.stat().st_size,source_masks_sha256=sha(SOURCE/'dane/maski.npz'),U500_file_sha256=sha(HERE/'IDENTYFIKACJA_K500.npz'),
        observed_inputs=False,source_BN_changes_included=True,raw_states_kept_in_RAM_only=True,seconds=time.time()-start,trajectory_sha256=sha(HERE/'TRAJEKTORIE_PRZYCZYNOWE_K500.npz'))
    js(HERE/'DANE.json',record);assert shutil.disk_usage(HERE).free>1e9
    print('Projected fixed probe',probe,'tool tokens',len(tool),'controls',len(control),'seconds',round(record['seconds'],2),flush=True)

if __name__=='__main__':main()
