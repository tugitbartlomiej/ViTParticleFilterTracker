"""Read-only full-dimensional audit. Sufficient Gram matrices for exact mean MSD/VACF.

GPU products are float32 with TF32 disabled, summed across feature blocks in float64.
Temporal centering before products prevents cancellation by large static activations.
No PCA or interpolation. Raw inputs never modified. Checks compare direct float64 distances.
"""
from pathlib import Path
import os, sys, json, zipfile, struct, time, hashlib, gc, argparse
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ROOT = Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation')
SRC = ROOT/'experiments/maly_vit_od_zera_35000_21epok_20261009'
OUT = ROOT/'outputs/audyt_szkla_768d_20261010'
OUT.mkdir(parents=True, exist_ok=True)
T0=time.time()
def log(*a): print(round(time.time()-T0,1),*a,flush=True)
def savejson(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def npz_mmap(p, name):
    with zipfile.ZipFile(p) as z:
        inf=z.getinfo(name+'.npy'); assert inf.compress_type==0
        with open(p,'rb') as f:
            f.seek(inf.header_offset); header=f.read(30)
            nl,el=struct.unpack('<HH',header[26:30]); f.seek(nl+el,1)
            ver=np.lib.format.read_magic(f)
            shape,fort,dtype=np.lib.format._read_array_header(f,ver)
            offset=f.tell()
    return np.memmap(p,mode='r',dtype=dtype,offset=offset,shape=shape,order='F' if fort else 'C')

def gram(x):
    # x CPU float16/float32, T x token x dim; all 768 coordinates used.
    flat=x.reshape(len(x),-1); g=np.zeros((len(x),len(x)),np.float64)
    for b in range(0,flat.shape[1],24576):
        v=torch.from_numpy(np.array(flat[:,b:b+24576],dtype=np.float32,copy=True)).cuda()
        v-=v.mean(0,keepdim=True)
        g+=(v@v.T).cpu().numpy().astype(np.float64)/x.shape[1]
        del v
    return g

def image_compute(x, steps, image_index):
    assert x.shape==(len(steps),600,768)
    assert np.isfinite(x).all()
    if ARGS.mode=='long':
        from direct_controls import measure
        measure(x,steps,'long',image_index,OUT)
    center=x.mean(axis=1,dtype=np.float32)
    base=np.asarray(x[np.where(steps==300)[0][0]],np.float64)
    ds=((base[:,None,:]-base[None,:,:])**2).sum(-1)
    np.fill_diagonal(ds,np.inf)
    nn=np.argsort(ds,axis=1)[:,:6]
    g=gram(x)
    # Cage is fixed at step 300, never refitted to manufacture a plateau.
    cr=np.empty_like(x,dtype=np.float32)
    for b in range(0,len(x),16):
        v=np.asarray(x[b:b+16],np.float32)
        cr[b:b+16]=v-v[:,nn,:].mean(2)
    cg=gram(cr); del cr
    checks=[]
    for a,b in [(300,301),(300,310),(300,1000),(1000,int(steps[-1])),(int(steps[-2]),int(steps[-1]))]:
        if a not in steps or b not in steps: continue
        i,j=np.searchsorted(steps,[a,b])
        direct=np.mean(np.sum((np.asarray(x[j],np.float64)-np.asarray(x[i],np.float64))**2,axis=-1))
        via=g[i,i]+g[j,j]-2*g[i,j]
        checks.append(dict(a=a,b=b,direct=float(direct),gram=float(via),relative_error=float(abs(via-direct)/max(direct,1e-20))))
    norms=np.array([(np.asarray(v,np.float64)**2).sum()/600 for v in x])
    # Quantization bound: f16 half-ULP errors, independent of compute error of bf16 forward.
    qs=[]
    for k in [300,1000,int(steps[-1])]:
        i=np.searchsorted(steps,k); v=np.asarray(x[i],np.float16)
        ulp=np.spacing(np.abs(v)).astype(np.float64)
        qs.append(dict(step=int(steps[i]),mean_half_ulp_squared=float(np.mean(np.sum((ulp/2)**2,-1)))))
    # Neighbor retention purely kinematic, in the original 768D metric.
    retention={}
    for k in [300,1000,5000,10000,20000,int(steps[-1])]:
        if k not in steps: continue
        y=np.asarray(x[np.searchsorted(steps,k)],np.float64)
        dd=(y*y).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*y@y.T
        np.fill_diagonal(dd,np.inf); nb=np.argsort(dd,axis=1)[:,:6]
        retention[str(k)]=float(np.mean([len(set(a)&set(b))/6 for a,b in zip(nn,nb)]))
    np.savez_compressed(OUT/f'{ARGS.mode}_image{image_index}.npz',steps=steps,g=g,cage_g=cg,centroid=center,
                        norms=norms,neighbors=nn)
    info=dict(image=image_index,checks=checks,f16_rounding_bound=qs,neighbor_retention=retention)
    savejson(OUT/f'{ARGS.mode}_image{image_index}_qa.json',info)
    log('image',image_index,'done','max_direct_error',max(z['relative_error'] for z in checks))

def dense():
    fs=sorted((SRC/'wyniki/gesty/gesty').glob('pelne768_*.npz'))
    steps=np.concatenate([np.load(p)['kroki'] for p in fs]); assert np.all(np.diff(steps)>0)
    expected=np.r_[np.arange(1095),np.arange(1100,22951,50)]
    assert np.array_equal(steps,expected)
    sources=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in fs]
    savejson(OUT/'dense_sources.json',dict(sources=sources,shape=[len(steps),8,600,768],warmup_end=300))
    remaining=[im for im in range(8) if not (OUT/f'dense_image{im}.npz').exists()]
    for begin in range(0,len(remaining),2):
        ims=remaining[begin:begin+2];xs=np.empty((len(steps),len(ims),600,768),np.float16);off=0
        for p in fs:
            with np.load(p) as f: a=f['latent']
            for j,im in enumerate(ims):xs[off:off+len(a),j]=a[:,im]
            off+=len(a);del a;log('read contiguous source',p.name,ims)
        for j,im in enumerate(ims):
            log('loaded dense image',im,xs[:,j].shape);image_compute(xs[:,j],steps,im);gc.collect()
        del xs;gc.collect()

def long_run():
    import cv2, importlib.util
    spec=importlib.util.spec_from_file_location('audit_model',SRC/'kod/model.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    sondy=json.loads((SRC/'wyniki/gesty/gesty/SONDY.json').read_text())['pelne']
    allrows=[json.loads(s) for s in (SRC/'wejscia/MANIFEST_35000.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[allrows[i] for i in sondy]; imgs=[]
    for r in rows:
        with zipfile.ZipFile(r['archive']) as z: raw=z.read(r['member'])
        assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
        im=cv2.cvtColor(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        imgs.append(cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA))
    X=torch.from_numpy(np.stack(imgs)).cuda().permute(0,3,1,2).float()/255
    fs=sorted((SRC/'wyniki/pelny/wagi').glob('k*.pt'),key=lambda p:int(p.stem[1:]))
    steps=np.array([int(p.stem[1:]) for p in fs]);assert steps[-1]==65640 and 300 in steps
    net=M.MalyViT().cuda().eval(); net.zapisuj_latent(True)
    cache=OUT/'long_latents_working.npy'
    zs=np.lib.format.open_memmap(cache,mode='w+',shape=(8,len(fs),600,768),dtype=np.float16)
    # BN and inference precision controls at all epoch endpoints plus selected early states.
    controls=[]; control_steps=[]; bn_ref=None; source=[]
    (OUT/'precision_controls').mkdir(exist_ok=True)
    with torch.no_grad():
        for j,(k,p) in enumerate(zip(steps,fs)):
            cp=torch.load(p,map_location='cpu',weights_only=False);state=cp['net']
            net.load_state_dict(state,strict=True)
            if k==300: bn_ref={n:v.clone() for n,v in state.items() if 'running_' in n or 'num_batches_tracked' in n}
            with torch.autocast('cuda',dtype=torch.bfloat16): _,z=net(X)
            zs[:,j]=z.float().cpu().numpy().astype(np.float16)
            source.append(dict(step=int(k),path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
            if k in [300,310,320,900,910,920,990,1000,5000,10000,20000,30000,40000,50000,60000,65640] or (k>0 and k%1094==0):
                _,zf=net(X); zf=zf.float().cpu().numpy()
                err=np.mean(np.sum((zf-zs[:,j].astype(np.float32))**2,axis=-1),axis=-1)
                if bn_ref is not None:
                    for n,v in bn_ref.items(): state[n]=v
                    net.load_state_dict(state,strict=True)
                    _,zb=net(X);zb=zb.float().cpu().numpy()
                    np.savez_compressed(OUT/'precision_controls'/f'k{k:05d}.npz',fp32=zf,frozen_bn=zb,bf16=zs[:,j])
                    control_steps.append(int(k))
                controls.append(dict(step=int(k),bf16_vs_fp32_msd=err.tolist()))
            if j%50==0: log('inference',j,len(fs),'step',k)
    zs.flush();del net,X;torch.cuda.empty_cache();gc.collect()
    savejson(OUT/'long_sources.json',dict(sources=source,probe_rows=sondy,shape=[len(fs),8,600,768],precision='bf16 forward -> f16 storage'))
    savejson(OUT/'long_precision_qa.json',controls)
    savejson(OUT/'long_bn_controls.json',dict(steps=control_steps,frozen_bn_reference=300))
    del zs;gc.collect()
    for im in range(8):
        if (OUT/f'long_image{im}.npz').exists(): continue
        mmap=np.load(cache,mmap_mode='r');x=np.array(mmap[im]);del mmap
        image_compute(x,steps,im);del x;gc.collect()
    assert cache.resolve().parent==OUT.resolve()
    cache.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['dense','long']);ARGS=parser.parse_args()
    log('start',ARGS.mode,torch.cuda.get_device_name())
    dense() if ARGS.mode=='dense' else long_run()
    log('complete')
