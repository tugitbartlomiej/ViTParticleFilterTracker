"""Rozstrzyga czy pełne32 z oryginalnym BN i siłami mieści się na współdzielonym GPU; OOM obala dopasowanie.
Nie zmienia wag ani optymalizatora i nie używa mikrobatche.
"""
import json,os,subprocess,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parents[1]/'outputs'/HERE.name
def save(v):
    p=OUT/'GPU_MEMORY_SMOKE.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf8');t.replace(p)
def free():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
save(dict(status='loading_cpu',pid=os.getpid()))
try:
    import numpy as np,torch,cv2
    from types import SimpleNamespace
    import model as M,rownanie_original as RW
    torch.set_num_threads(4);cv2.setNumThreads(1)
    initial_free=free()
    if initial_free<2500:
        save(dict(status='skipped_insufficient_free',free_mib=initial_free,minimum_to_attempt=2500));raise SystemExit(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    st=torch.load(OUT/'MODEL_AUTHENTIC_K1000.pt',map_location='cpu',weights_only=True)
    net=M.MalyViT().cuda();net.load_state_dict(st['net']);net.zapisuj_latent(True)
    opt=torch.optim.AdamW([dict(params=[p for p in net.parameters() if p.ndim>=2],weight_decay=.05),dict(params=[p for p in net.parameters() if p.ndim<2],weight_decay=0)],lr=5e-4,betas=(.9,.999));opt.load_state_dict(st['opt'])
    u=SimpleNamespace(net=net,opt=opt,dev='cuda',_lr=lambda:5e-4)
    x=torch.from_numpy(np.load(OUT/'IMAGES_selected32.npy')).permute(0,3,1,2).float().cuda()/255
    # Account for both complete input pools resident in the main worker.
    control=torch.from_numpy(np.load(OUT/'IMAGES_random32.npy')).permute(0,3,1,2).float().cuda()/255
    gt=json.loads((OUT/'GROUNDTRUTH_selected32.json').read_text())
    c=tuple(t.cuda() for t in M.cele([np.array(r['boxes_xyxy_480x320'],dtype=np.float32).reshape(-1,4) for r in gt],20,30))
    im=cv2.imread(str(OUT/'input/SONDA_ORIGINAL.jpg'));rgb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    probe=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255
    buf={n:b.clone() for n,b in net.named_buffers()}
    def read(d=None):
        net.eval();p=dict(net.named_parameters())
        if d is not None:p={n:v+d[n] for n,v in p.items()}
        with torch.no_grad():torch.func.functional_call(net,(p,buf),(probe,));return net._latent.detach().cpu().numpy().copy()
    z0=read();torch.cuda.reset_peak_memory_stats();t0=time.time()
    save(dict(status='full32_gradient_and_forces',pid=os.getpid(),initial_free_mib=initial_free))
    g=RW.gradient_tensorow(u,x,*c);new,mom=RW.czesci_kroku(u,g)
    sn=read(new)-z0;sm=read(mom)-z0
    np.savez_compressed(OUT/'SMOKE_FORCE_selected32.npz',nowa=sn,ped=sm,z0=z0)
    # True training-mode forward/backward with .grad, same32, unmodified BN.
    del g,new,mom;net.train();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):o,_=net(x)
    loss,_=M.strata(o,*c);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
    torch.cuda.synchronize()
    allocated=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
    remaining=free();safe=int(reserved+1024+255)//256*256
    save(dict(status='passed',pid=os.getpid(),batch=32,microbatch=False,bn='original train mode',original_optimizer=True,force_definition='original RW.gradient_tensorow/czesci_kroku',initial_free_mib=initial_free,free_after_peak_mib=remaining,peak_allocated_mib=allocated,peak_reserved_mib=reserved,recommended_threshold_mib=safe,reserve_mib=1024,seconds=time.time()-t0,training_step_executed=False,gradient_and_virtual_force_computed=True))
except Exception as e:
    save(dict(status='failed',error=str(e),traceback=traceback.format_exc(),no_other_process_stopped=True))
    raise
