"""Resume the trusted full state of epoch200 (step218750); TRAIN only, sparse full checkpoints to epoch400."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,json,math,os,random,signal,time,zipfile
import numpy as np
import cv2
import torch
import model as M

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)
def tree_cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,np.ndarray):return v.copy()
    if isinstance(v,dict):return {k:tree_cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [tree_cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(tree_cpu(x) for x in v)
    return copy.deepcopy(v)
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def torch_save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);os.replace(temp,path)

class Training:
    def __init__(self,cfg,config_hash,images,targets):
        self.cfg=cfg;self.config_hash=config_hash;self.images=images;self.targets=targets
        self.net=M.MalyViT().cuda()
        dec=[v for v in self.net.parameters() if v.ndim>=2]
        nd=[v for v in self.net.parameters() if v.ndim<2]
        self.opt=torch.optim.AdamW([dict(params=dec,weight_decay=.05),dict(params=nd,weight_decay=0)],lr=.0005,betas=(.9,.999))
        self.rng=np.random.default_rng(cfg['seed']);self.queue=[];self.step=0;self.source_identity=None
    def restore(self,st,expected_config_sha256):
        # The parent's epoch200 checkpoint carries the parent config hash; this run's own checkpoints carry ours.
        required=['net','opt','queue','rng','torch_rng','cuda_rng','python_rng','numpy_rng','krok','seed','continuation_config_sha256']
        assert all(k in st for k in required),'Not a full continuation checkpoint'
        assert st['seed']==self.cfg['seed']
        assert st['training_manifest_sha256']==self.cfg['training_manifest_sha256']
        assert st['continuation_config_sha256']==expected_config_sha256,'Continuation configuration mismatch'
        self.source_identity=st['source_training_identity']
        self.net.load_state_dict(st['net'],strict=True)
        # Isolate CPU AdamW step tensors; load_state_dict may otherwise alias the source.
        self.opt.load_state_dict(copy.deepcopy(st['opt']))
        assert equal(self.net.state_dict(),st['net']),'Model/BN restore mismatch'
        assert equal(self.opt.state_dict(),st['opt']),'Optimizer restore mismatch'
        self.step=int(st['krok']);self.queue=list(st['queue']);self.rng.bit_generator.state=st['rng']
        assert self.opt.state and all(float(v['step'])==self.step for v in self.opt.state.values()),'Optimizer step differs from global step'
        self.net.train()
        torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
        random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng'])
    def pack(self):
        return dict(net=self.net.state_dict(),opt=self.opt.state_dict(),krok=self.step,seed=self.cfg['seed'],
            queue=self.queue,rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            training_manifest_sha256=self.cfg['training_manifest_sha256'],source_training_identity=self.source_identity,
            continuation_config_sha256=self.config_hash,experiment_id=self.cfg['experiment_id'],parent_experiment_id=self.cfg['parent_experiment_id'])
    def once(self):
        if len(self.queue)<32:self.queue+=self.rng.permutation(len(self.images)).tolist()
        ix=np.array(self.queue[:32]);self.queue=self.queue[32:];flip=self.rng.random(32)<.5
        x=torch.from_numpy(self.images[ix]).cuda().permute(0,3,1,2).float()/255
        fm=torch.from_numpy(flip).cuda();x=torch.where(fm[:,None,None,None],x.flip(3),x)
        ac=torch.from_numpy(self.rng.uniform(.8,1.2,(32,2)).astype(np.float32)).cuda();mean=x.mean((1,2,3),keepdim=True)
        x=((x-mean)*ac[:,0,None,None,None]+mean*ac[:,1,None,None,None]).clamp(0,1)
        ts=[]
        for k in range(3):
            t0=self.targets[0][k][ix];t1=self.targets[1][k][ix]
            mask=torch.from_numpy(flip).reshape((32,)+(1,)*(t0.ndim-1));ts.append(torch.where(mask,t1,t0).cuda())
        for g in self.opt.param_groups:g['lr']=.0005*min(1,(self.step+1)/300)
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):raw,_=self.net(x)
        loss,_=M.strata(raw,*ts)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.);self.opt.step();self.step+=1
        return ix.tolist()

def load_data(cfg,out):
    path=Path(cfg['training_manifest']);assert sha(path)==cfg['training_manifest_sha256']
    split_path=path.parent/'split_manifest.json'
    split=json.loads(split_path.read_text())['groups']
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(rows)==cfg['training_images']==35000
    assert len({r['id'] for r in rows})==35000
    assert all(r['split'].lower()=='train' and split[r['film']]=='train' for r in rows)
    X=np.empty((len(rows),320,480,3),dtype=np.uint8);boxes=[];archive=None;z=None
    archive_paths=set()
    for i,r in enumerate(rows):
        name=Path(r['archive'].replace('\\','/')).name
        if archive!=name:
            if z:z.close()
            ap=Path(cfg['archives'])/name;z=zipfile.ZipFile(ap);archive=name;archive_paths.add(str(ap))
        raw=z.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256'],'Training image changed'
        im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        assert im is not None
        h,w=im.shape[:2];assert (w,h)==(r['width'],r['height'])
        X[i]=cv2.cvtColor(cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        b=np.array([g['bbox_xywh'] for g in r['gt']],dtype=np.float32).reshape(-1,4)
        b[:,2:]+=b[:,:2];b*=np.array([480/w,320/h,480/w,320/h]);boxes.append(b)
        if i%5000==0:print('DATA_LOADED',i,'/',len(rows),flush=True)
    if z:z.close()
    targets=[M.cele(boxes,20,30)]
    flipped=[np.stack([480-b[:,2],b[:,1],480-b[:,0],b[:,3]],1) if len(b) else b for b in boxes]
    targets.append(M.cele(flipped,20,30))
    atomic_json(out/'DATA_VERIFIED.json',dict(status='PASS',images=35000,all_image_sha256_checked=True,
        manifest_sha256=sha(path),split_manifest_sha256=sha(split_path),archives=sorted(archive_paths),split='TRAIN only'))
    print('DATA_READY',len(rows),flush=True)
    return X,targets

def preflight(trainer,source,out,cfg):
    start=cfg['start_step'];original_source=tree_cpu(source)
    trainer.restore(source,cfg['parent_continuation_config_sha256']);trainer.once()
    path=out/f'preflight_roundtrip_k{start+1:06d}.pt';torch_save(path,trainer.pack())
    expected_batch=trainer.once();expected=tree_cpu(trainer.pack())
    # Reload the actual disk serialization, then execute the same next update.
    st=torch.load(path,map_location='cpu',weights_only=False);trainer.restore(st,trainer.config_hash)
    actual_batch=trainer.once();actual=tree_cpu(trainer.pack())
    checked=['net','opt','queue','rng','torch_rng','cuda_rng','python_rng','numpy_rng','krok']
    assert expected_batch==actual_batch,'Sampler next-batch mismatch'
    for key in checked:assert equal(expected[key],actual[key]),'Roundtrip diverged: '+key
    assert equal(source,original_source),'In-memory source checkpoint was mutated'
    trainer.restore(source,cfg['parent_continuation_config_sha256']);assert trainer.step==start
    result=dict(status='PASS',source_step=start,roundtrip_checkpoint_step=start+1,comparison_step=start+2,
        next_batch_equal=True,model_and_BN_exact=True,optimizer_exact=True,sampler_and_RNG_exact=True,
        production_restored_to_source_step=True,source_state_immutable=True,optimizer_step_verified=True,
        roundtrip_checkpoint_sha256=sha(path))
    atomic_json(out/'PREFLIGHT.json',result);print('PREFLIGHT_PASS restored_to_step',trainer.step,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'));config_hash=sha(args.config)
    exp=Path(__file__).resolve().parents[1];out=exp/'Wyniki/Dane';ckpts=out/'checkpoints'
    ckpts.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'Allocated Slurm GPU required'
    lock=(out/'TRAINING.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze=json.loads(Path(__file__).with_name('FREEZE.json').read_text())
    for name,h in freeze['files'].items():assert sha(Path(__file__).parent/name)==h,name
    if (out/'COMPLETE.json').exists():raise RuntimeError('Already completed; preserved')
    if (out/'LATEST.json').exists() and not args.resume:raise RuntimeError('Existing run; explicit --resume required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    assert sha(Path(M.__file__))==cfg['model_sha256']
    assert sha(cfg['source_restart'])==cfg['source_restart_sha256']
    # Trusted own parent checkpoint with pinned SHA256; full RNG objects require weights_only=False.
    source=torch.load(cfg['source_restart'],map_location='cpu',weights_only=False)
    assert source['krok']==cfg['start_step']==218750 and source['seed']==cfg['seed']==3047
    assert source['experiment_id']==cfg['parent_experiment_id'],'Source is not the parent experiment checkpoint'
    assert source['continuation_config_sha256']==cfg['parent_continuation_config_sha256']
    atomic_json(out/'SOURCE_VERIFIED.json',dict(status='PASS',source_step=cfg['start_step'],source_epoch=cfg['start_step']*32/35000,
        source_restart=cfg['source_restart'],source_restart_sha256=cfg['source_restart_sha256'],source_experiment_id=source['experiment_id'],
        full_optimizer_sampler_rng_present=all(k in source for k in ['opt','queue','rng','torch_rng','cuda_rng','python_rng','numpy_rng']),
        job_id=os.environ['SLURM_JOB_ID'],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    images,targets=load_data(cfg,out);tr=Training(cfg,config_hash,images,targets)
    if args.resume:
        latest=json.loads((out/'LATEST.json').read_text());path=ckpts/latest['file']
        assert sha(path)==latest['sha256'];tr.restore(torch.load(path,map_location='cpu',weights_only=False),config_hash)
    else:
        preflight(tr,source,out,cfg)
    del source
    stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,lambda *a:stop.__setitem__(0,True))
    def checkpoint(reason):
        path=ckpts/f'k{tr.step:06d}.pt'
        if path.exists():
            existing=torch.load(path,map_location='cpu',weights_only=False)
            assert equal(existing['net'],tr.net.state_dict()),'Refusing to replace a differing checkpoint'
        else:torch_save(path,tr.pack())
        record=dict(file=path.name,step=tr.step,epoch=tr.step*32/35000,sha256=sha(path),reason=reason)
        atomic_json(out/'LATEST.json',record);print('CHECKPOINT',json.dumps(record),flush=True)
    if not args.resume:checkpoint('start')
    schedule=set(cfg['checkpoint_steps']);start=time.monotonic();initial=tr.step
    atomic_json(out/'STATUS.json',dict(status='RUNNING',step=tr.step,epoch=tr.step*32/35000,target_step=cfg['target_step']))
    print('PRODUCTION_START',tr.step,'TARGET',cfg['target_step'],flush=True)
    while tr.step<cfg['target_step'] and not stop[0]:
        tr.once()
        if tr.step in schedule:checkpoint('scheduled')
        if tr.step==initial+1 or tr.step%1000==0:
            atomic_json(out/'STATUS.json',dict(status='RUNNING',step=tr.step,epoch=tr.step*32/35000,
                target_step=cfg['target_step'],seconds=time.monotonic()-start))
            print('PROGRESS',tr.step,'epoch',round(tr.step*32/35000,4),flush=True)
    if stop[0]:
        checkpoint('signal');atomic_json(out/'STATUS.json',dict(status='INTERRUPTED',step=tr.step,epoch=tr.step*32/35000))
        raise SystemExit(143)
    if not (out/'LATEST.json').exists() or json.loads((out/'LATEST.json').read_text())['step']!=tr.step:checkpoint('final')
    final=dict(status='COMPLETE',step=tr.step,epoch=tr.step*32/35000,target_epochs=cfg['target_epochs'],
        additional_steps=tr.step-cfg['start_step'],seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(out/'COMPLETE.json',final);atomic_json(out/'STATUS.json',final);print('COMPLETE',json.dumps(final),flush=True)

if __name__=='__main__':main()
