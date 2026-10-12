import sys,time,shutil
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;OLD=HERE.parent;sys.path.insert(0,str(OLD))
from ekstrakcja import network,extract,compare,stamp
from dobor import read,sha,js,SOURCE
from threadpoolctl import threadpool_limits
threadpool_limits(4);start=time.time();m=read(HERE/'MANIFEST.json');replace=m['replaced_indices'];assert m['fit_base1500_count']==30
X=np.load(HERE/'OBRAZY.npy')[replace];assert X.shape==(5,320,480,3)
path=HERE/'LATENT_NOWE5_768.npy';assert not path.exists();assert shutil.disk_usage(HERE).free>5e9
output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(101,5,600,768));net=network();rows=[]
for slot,step in enumerate(range(0,1001,10)):
 cp=SOURCE/'wagi'/f'k{step:04d}.pt';ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['krok']==step;net.load_state_dict(ck['net'],strict=True);net.eval();z=extract(net,X,batch=4);output[slot]=z;output.flush()
 row=dict(step=step,finite=True,buffers_unchanged=True,checkpoint_sha256=sha(cp))
 if step in (0,500,1000):
  repeat=extract(net,X,batch=1);row['repeat_batch1_vs_batch4']=compare(repeat,z);assert row['repeat_batch1_vs_batch4']['within_maxabs1e_5']
 rows.append(row);js(HERE/'EKSTRAKCJA_POSTEP.json',dict(status='running',completed_steps=len(rows),records=rows));print(step,round(time.time()-start,1),flush=True)
assert all(np.isfinite(output[k]).all() for k in range(101));del output
old=read(OLD/'DANE_768.json');oldsha=sha(OLD/'LATENT_768.npy');assert oldsha==old['latent_sha256']
js(HERE/'DANE.json',dict(status='complete',completed=stamp(),logical_shape=[101,34,600,768],new_shape=[101,5,600,768],dtype='float32',new_latent_path=str(path),new_latent_sha256=sha(path),old_latent_path=str(OLD/'LATENT_768.npy'),old_latent_sha256=oldsha,old_dataset_unchanged=True,cache_sources=m['cache_sources'],fit_base1500_count=30,fit_indices=m['fit_indices'],holdout_indices=m['holdout_indices'],holdout_membership=m['holdout_membership'],manifest_sha256=sha(HERE/'MANIFEST.json'),images_sha256=sha(HERE/'OBRAZY.npy'),masks_sha256=sha(HERE/'MASKI.npz'),plan_sha256=sha(HERE/'PLAN.md'),code_sha256=sha(HERE/'ekstrakcja_poprawka.py'),channels_sha256=sha(HERE/'KANALY_J128.npz'),all_finite=True,checkpoint_steps=list(range(0,1001,10)),records=rows,eval=True,tf32=False,autocast=False,batch=4,measurement_includes='weights and BatchNorm running statistics',source_layer='bloki[-1].mlp.act',seconds=time.time()-start,new_training=False))
js(HERE/'EKSTRAKCJA_POSTEP.json',dict(status='complete',completed=stamp(),completed_steps=101,seconds=time.time()-start,records=rows));print('COMPLETE',time.time()-start,flush=True)
