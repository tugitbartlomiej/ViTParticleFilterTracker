"""Czy jedna wspólna korekta poprawia detekcję na filmach pilota? Brak przewagi AP nad baseline lub kontrolami obala poprawę."""
import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import time,traceback,platform,hashlib
import numpy as np,torch,cv2
from dane import *
from core import intervene,tail
from model_snapshot import MalyViT,cele,strata,dekoduj,_giou
START=time.time()
CONFIG=dict(seeds=[101,202,303],learning_rates=[.01,.05],steps=200,batch=8,validate_every=50,regularization=.001,delta_norm_limit=5.,scales=[-2.,-1.,-.5,-.25,0.,.25,.5,1.,2.],random_seeds=[11,22,33],bootstrap_resamples=1000,bootstrap_seed=991,checkpoint_step=1000,intervention='postGELU768 final MLP: native delta[128] broadcast all 600 tokens; residual unchanged',selection='checkpoint and LR: validation detection loss; signed scale and final candidate: validation AP50:95, AP50, then loss; exact tie prefers zero scale',test_used_for_selection=False)
def progress(stage,**kw):
    state=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kw);js(OUT/'PROGRESS.json',state);print(json.dumps(state,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
def state_hash(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def losses(o,t):
    heat,ltrb,poz=t;p=o[...,0].sigmoid().clamp(1e-4,1-1e-4);pos=heat.eq(1).float()
    focal=(-(p.log()*(1-p)**2*pos+(1-p).log()*p**2*(1-heat)**4*(1-pos))).sum((1,2))/pos.sum((1,2)).clamp(min=1)
    pr=o[...,1:].exp();g=_giou(pr.reshape(-1,4),ltrb.reshape(-1,4)).reshape(p.shape)
    box=((1-g)*poz).sum((1,2))/poz.sum((1,2)).clamp(min=1e-12)
    return focal+2*box
def image(path):
    im=cv2.imread(str(path));assert im is not None
    im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2,0,1).float()/255
def overlaps(boxes,gt):
    inter=np.maximum(0,np.minimum(boxes[...,2:],gt[:,None,2:])-np.maximum(boxes[...,:2],gt[:,None,:2])).prod(-1)
    return inter/(np.maximum(0,boxes[...,2:]-boxes[...,:2]).prod(-1)+np.maximum(0,gt[:,2:]-gt[:,:2]).prod(-1)[:,None]-inter+1e-9)
def aps(scores,ov):
    # Exactly one annotated target bbox per image, top-10 local peaks, no GT-based selection.
    order=np.argsort(-scores.reshape(-1),kind='stable');hit=ov[:,:,None]>=np.arange(.5,.951,.05)
    tp=hit & (np.cumsum(hit,axis=1)==1)
    tp=tp.reshape(-1,10)[order];ct=np.cumsum(tp,axis=0);rec=ct/len(scores);prec=ct/np.arange(1,len(ct)+1)[:,None]
    prec=np.maximum.accumulate(prec[::-1],axis=0)[::-1];out=[]
    for k in range(10):
        ix=np.searchsorted(rec[:,k],np.linspace(0,1,101));out.append(float(np.mean(np.r_[prec[:,k],0][ix])))
    return np.array(out)
def decoded(o,gt):
    boxes,scores=dekoduj(o);boxes=boxes.numpy();scores=scores.numpy();return boxes,scores,overlaps(boxes,gt)
def stats(o,gt,t):
    b,s,ov=decoded(o,gt);a=aps(s,ov)
    return dict(AP50=float(a[0]),AP50_95=float(a.mean()),AP_by_IoU=a,bbox_IoU_top_confidence=float(ov[:,0].mean()),confidence_top=float(s[:,0].mean()),detection_loss=float(losses(o,t).mean()))
def rank(m,scale):return (m['AP50_95'],m['AP50'],-m['detection_loss'],-abs(scale))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'RUN.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:run()
    finally:lock.unlink()
def run():
    progress('preflight');js(OUT/'CONFIG.json',CONFIG)
    rows,ids,j=load();J=torch.tensor(j,device='cuda');gt=np.array([[r['bbox_xywh'][0],r['bbox_xywh'][1],r['bbox_xywh'][0]+r['bbox_xywh'][2],r['bbox_xywh'][1]+r['bbox_xywh'][3]] for r in rows],np.float32)
    T=cele([g[None] for g in gt],20,30)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    ck=torch.load(CKPT,map_location='cpu',weights_only=True);assert ck['krok']==1000
    net=MalyViT().cuda().eval();net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False);before=state_hash(net)
    import timm
    js(OUT/'ENVIRONMENT.json',dict(pid=os.getpid(),python=sys.executable,torch=torch.__version__,timm=timm.__version__,platform=platform.platform(),device=torch.cuda.get_device_name(),gpu_memory=torch.cuda.mem_get_info(),checkpoint=str(CKPT),checkpoint_sha256=sha(CKPT),model_state_sha256=before,runner_sha256=sha(HERE/'run.py'),model_source_sha256=sha(HERE/'model_snapshot.py')))
    Z=torch.empty(400,600,768);R=torch.empty(400,600,192);base=torch.empty(400,20,30,5);held={};err=0.
    h1=net.bloki[-1].norm2.register_forward_pre_hook(lambda m,a:held.update(r=a[0].detach()))
    h2=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:held.update(z=o.detach()))
    with torch.no_grad():
        for lo in range(0,400,4):
            x=torch.stack([image(r['path']) for r in rows[lo:lo+4]]).cuda();o,_=net(x);quick=tail(net,held['z'],held['r']);torch.testing.assert_close(o,quick,atol=2e-5,rtol=1e-5)
            err=max(err,float((o-quick).abs().max()));Z[lo:lo+4]=held['z'].cpu();R[lo:lo+4]=held['r'].cpu();base[lo:lo+4]=o.cpu()
            if lo%40==0:progress('cache_real_images',processed=lo+4,total=400)
    h1.remove();h2.remove()
    def ts(ix):return tuple(v[ix] for v in T)
    def forward(ix,d):return tail(net,intervene(Z[ix].cuda(),d,J,1.),R[ix].cuda())
    def outputs(ix,d):
        with torch.no_grad():return torch.cat([forward(ix[lo:lo+8],d).cpu() for lo in range(0,len(ix),8)])
    # Real full-forward hook, unchanged 640 channels, nonzero shared gradient, source loss agreement.
    probe=torch.full((128,),.01,device='cuda',requires_grad=True);ix=ids['train'][:2];z=Z[ix].cuda();edited=intervene(z,probe,J,1.)
    mask=torch.ones(768,device='cuda',dtype=torch.bool);mask[J]=False;assert torch.equal(edited[:,:,mask],z[:,:,mask])
    h=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:intervene(o,probe,J,1.));actual,_=net(torch.stack([image(rows[i]['path']) for i in ix]).cuda());h.remove()
    expected=forward(ix,probe);torch.testing.assert_close(actual,expected,atol=2e-5,rtol=1e-5)
    tt=tuple(v.cuda() for v in ts(ix));ls=losses(expected,tt);ref=torch.stack([strata(expected[k:k+1],*(v[k:k+1] for v in tt))[0] for k in range(2)])
    torch.testing.assert_close(ls,ref,atol=1e-5,rtol=1e-5);ls.mean().backward();assert torch.isfinite(probe.grad).all() and probe.grad.norm()>0
    # AP sanity: perfect and displaced rectangles.
    assert aps(np.ones((2,1)),np.ones((2,1))).min()==1 and aps(np.ones((2,1)),np.zeros((2,1))).max()==0
    js(OUT/'PREFLIGHT.json',dict(full_vs_tail_max_error=err,intervened_hook_error=float((actual-expected).abs().max()),source_loss_match=True,gradient_norm=float(probe.grad.norm()),unchanged640=True,all_tokens_broadcast=True,BN_eval=all(not m.training for m in net.modules()),model_frozen=all(not p.requires_grad for p in net.parameters())))
    del actual,expected,edited,z,probe,tt
    resume='--resume' in sys.argv
    if resume:
        frozen=read(OUT/'FREEZE.json');assert frozen['selection_completed_before_test'] and sha(OUT/'DIRECTION.npz')==frozen['artifact_sha256']
        progress('resume_frozen_evaluation',original_selection=frozen['selected'])
    else:progress('training_start',configs=6,steps_per_config=200)
    history=read(OUT/'HISTORY.json') if resume else [];candidates=frozen['candidates'] if resume else [];train_start=time.time()
    va=ids['val'];tr=ids['train']
    baseline_val=stats(base[va],gt[va],ts(va));js(OUT/'BASELINE_VAL.json',baseline_val)
    for seed in ([] if resume else CONFIG['seeds']):
        for lr in CONFIG['learning_rates']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);d=(torch.randn(128,device='cuda')*.001).requires_grad_(True);opt=torch.optim.Adam([d],lr=lr)
            bestval=float('inf');best=None;beststep=0
            for step in range(1,CONFIG['steps']+1):
                ix=rng.choice(tr,CONFIG['batch'],replace=False);o=forward(ix,d);tt=tuple(v.cuda() for v in ts(ix));dl=losses(o,tt).mean();reg=CONFIG['regularization']*d.square().mean();loss=dl+reg
                opt.zero_grad();loss.backward();assert torch.isfinite(d.grad).all();opt.step()
                with torch.no_grad():d.mul_(min(1.,CONFIG['delta_norm_limit']/max(float(d.norm()),1e-12)))
                h=dict(seed=seed,lr=lr,step=step,train_detection_loss=float(dl.detach()),regularization=float(reg.detach()),delta_norm=float(d.detach().norm()),elapsed_seconds=time.time()-START)
                if step%50==0:
                    val=outputs(va,d);h['val']=stats(val,gt[va],ts(va))
                    if h['val']['detection_loss']<bestval:bestval=h['val']['detection_loss'];best=d.detach().clone();beststep=step
                    js(OUT/'HISTORY.json',history+[h])
                history.append(h)
                if step==1 or step%25==0:progress('training',seed=seed,lr=lr,step=step,train_loss=h['train_detection_loss'],delta_norm=h['delta_norm'])
            norm=float(best.norm());assert norm>0
            np.savez(OUT/f'CANDIDATE_s{seed}_lr{lr}.npz',delta=best.cpu().numpy(),direction=(best/norm).cpu().numpy(),norm=norm,channels0=j)
            for scale in CONFIG['scales']:
                m=stats(outputs(va,best*scale),gt[va],ts(va));candidates.append(dict(seed=seed,lr=lr,step=beststep,scale=scale,norm=norm,metrics=m))
            progress('candidate_validated',seed=seed,lr=lr)
    js(OUT/'HISTORY.json',history);chosen=frozen['selected'] if resume else max(candidates,key=lambda c:rank(c['metrics'],c['scale']));raw=np.load(OUT/f"CANDIDATE_s{chosen['seed']}_lr{chosen['lr']}.npz")['delta'];norm=float(np.linalg.norm(raw));v=raw/norm;alpha=norm*chosen['scale'];delta=torch.tensor(alpha*v,device='cuda')
    if not resume:np.savez(OUT/'DIRECTION.npz',direction=v,kierunek=v,channels0=j,alpha=alpha,delta=delta.cpu().numpy(),trained_delta=raw,trained_norm=norm,effective_norm=abs(alpha))
    controls=frozen['controls'] if resume else []
    for seed in ([] if resume else CONFIG['random_seeds']):
        u=np.random.default_rng(seed).normal(size=128);u=(u/np.linalg.norm(u)).astype(np.float32);u=torch.tensor(u,device='cuda');grid=[]
        for scale in CONFIG['scales']:grid.append(dict(scale=scale,metrics=stats(outputs(va,u*abs(alpha)*scale),gt[va],ts(va))))
        sel=max(grid,key=lambda c:rank(c['metrics'],c['scale']));controls.append(dict(seed=seed,base_norm=abs(alpha),scale=sel['scale'],alpha=abs(alpha)*sel['scale'],direction=u.cpu().numpy(),validation_grid=grid))
    if not resume:js(OUT/'FREEZE.json',dict(selected=chosen,candidates=candidates,controls=controls,selection_completed_before_test=True,configuration=CONFIG,train_seconds=time.time()-train_start,artifact_sha256=sha(OUT/'DIRECTION.npz')))
    progress('selection_frozen_before_test',chosen_seed=chosen['seed'],lr=chosen['lr'],step=chosen['step'],scale=chosen['scale'],alpha=alpha)
    methods={'baseline':torch.zeros(128,device='cuda'),'shared':delta}
    for c in controls:methods[f"random{c['seed']}"]=torch.tensor(np.asarray(c['direction'],dtype=np.float32)*c['alpha'],device='cuda')
    metrics={};predictions={};dec={}
    for split in ['val','test']:
        ix=ids[split];metrics[split]={};predictions[split]={};dec[split]={}
        for name,d in methods.items():
            o=base[ix] if name=='baseline' else outputs(ix,d);metrics[split][name]=stats(o,gt[ix],ts(ix));predictions[split][name]=o.numpy();dec[split][name]=decoded(o,gt[ix])
        np.savez_compressed(OUT/f'DETECTIONS_{split}.npz',indices=ix,**predictions[split]);progress('evaluation',split=split,metrics=metrics[split])
    js(OUT/'METRICS.json',metrics)
    # Paired bootstrap: identical sampled whole films in both arms. Recompute ranked AP.
    boot={}
    for split in ['val','test']:
        ix=ids[split];films=np.array([rows[i]['film'] for i in ix]);names=np.unique(films);groups=[np.flatnonzero(films==f) for f in names];rng=np.random.default_rng(CONFIG['bootstrap_seed']);samples=[]
        for k in range(CONFIG['bootstrap_resamples']):
            take=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);a=[]
            for name in ['baseline','shared']:
                _,s,ov=dec[split][name];ap=aps(s[take],ov[take]);a.append([ap[0],ap.mean()])
            samples.append(np.array(a[1])-a[0])
            if (k+1)%250==0:progress('paired_film_bootstrap',split=split,resamples=k+1)
        samples=np.array(samples);np.save(OUT/f'BOOTSTRAP_{split}.npy',samples)
        boot[split]=dict(delta_AP50=metrics[split]['shared']['AP50']-metrics[split]['baseline']['AP50'],delta_AP50_95=metrics[split]['shared']['AP50_95']-metrics[split]['baseline']['AP50_95'],CI95=np.quantile(samples,[.025,.975],axis=0),columns=['AP50','AP50_95'],films=len(names),resamples=len(samples),seed=CONFIG['bootstrap_seed'],conditional_on_fixed_selected_intervention=True)
    js(OUT/'BOOTSTRAP.json',boot)
    # Fresh frozen model and persisted intervention: image-only inference, no labels in API.
    fresh=MalyViT().cuda().eval();fresh.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True)['net']);fresh.requires_grad_(False);freshbefore=state_hash(fresh)
    with np.load(OUT/'DIRECTION.npz') as saved:dd=torch.tensor(float(saved['alpha'])*saved['direction'],dtype=torch.float32,device='cuda');jj=torch.tensor(saved['channels0'],device='cuda')
    checks=[]
    for split in ['train','val','test']:
        ix=ids[split][:2];x=torch.stack([image(rows[i]['path']) for i in ix]).cuda()
        with torch.no_grad():baseline,_=fresh(x)
        torch.testing.assert_close(baseline.cpu(),base[ix],atol=2e-5,rtol=1e-5)
        h=fresh.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:intervene(o,dd,jj,1.))
        with torch.no_grad():actual,_=fresh(x)
        h.remove();expected=outputs(ix,dd);torch.testing.assert_close(actual.cpu(),expected,atol=2e-5,rtol=1e-5)
        checks.append(dict(split=split,indices=ix,max_error=float((actual.cpu()-expected).abs().max())))
    assert before==state_hash(net)==freshbefore==state_hash(fresh)
    js(OUT/'VERIFICATION.json',dict(fresh_reload_checks=checks,weights_and_BN_buffers_unchanged=True,state_sha256=before,no_labels_in_inference=True,checkpoint_unchanged=sha(CKPT)==read(OUT/'MANIFEST.json')['checkpoint_sha256'],tail_full_max_error=err,unchanged640=True,intervention_point='postGELU768, not complete model state'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for seed in CONFIG['seeds']:
        for lr in CONFIG['learning_rates']:
            hh=[h for h in history if h['seed']==seed and h['lr']==lr];ax[0].plot([h['step'] for h in hh],[h['train_detection_loss'] for h in hh],alpha=.5,label=f'{seed}/{lr}');vv=[h for h in hh if 'val' in h];ax[1].plot([h['step'] for h in vv],[h['val']['detection_loss'] for h in vv],label=f'{seed}/{lr}')
    ax[0].set_title('Train minibatch detection loss');ax[1].set_title('Validation detection loss');ax[1].legend();fig.tight_layout();fig.savefig(OUT/'CURVES.png');plt.close(fig)
    elapsed=time.time()-START;js(OUT/'COST.json',dict(seconds=elapsed,training_and_validation_seconds=time.time()-train_start,optimization_steps=1200,steps_per_configuration=200,configurations=6,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),pid=os.getpid(),device=torch.cuda.get_device_name()))
    lines=['# Wspólny kierunek nadzorowany detekcją: pilot','',f'Jedna delta[128] dodawana do wszystkich 600 tokenów każdego obrazu, tylko w channels0. Punkt interwencji postGELU768 ostatniego MLP; nie pełny stan modelu. Oryginalny k1000, zamrożone wagi i BN, niezmieniony residual.','',f'Wybrano seed {chosen["seed"]}, LR {chosen["lr"]}, krok {chosen["step"]}, skala {chosen["scale"]}, norma uczonej delty {norm:.6g}, alpha {alpha:.6g}; v ma normę jeden. Sześć konfiguracji po 200 kroków, batch 8. Focal objectness jednej klasy + 2 GIoU, regularizator 0.001 mean(delta²), limit normy 5. Skale dwustronne '+str(CONFIG['scales'])+'.','', '|Split|Wariant|AP50|AP50:95|IoU ramki o najwyższej confidence|Confidence|Loss|','|---|---|---|---|---|---|---|']
    for split,mm in metrics.items():
        for name,m in mm.items():lines.append(f'|{split}|{name}|{m["AP50"]:.6f}|{m["AP50_95"]:.6f}|{m["bbox_IoU_top_confidence"]:.6f}|{m["confidence_top"]:.6f}|{m["detection_loss"]:.6f}|')
    lines+=['','AP to jednoclasowa interpolacja 101-punktowa po 10 lokalnych szczytach na obraz; AP50:95 to średnia 10 progów. To jawny protokół tej głowy, nie pełny COCO evaluator. IoU i confidence dotyczą najwyżej ocenionej ramki wybranej bez GT.','', 'Kontrole: trzy izotropowe losowe kierunki o tej samej normie startowej co wybrana korekta; ich skale również dobierane tylko val. Finalne normy mogą różnić się po doborze skali i są zapisane w FREEZE.json. Bootstrap sparowany po całych filmach, warunkowy względem już wybranej korekty.']
    for split,b in boot.items():lines.append(f'{split}: delta AP50 {b["delta_AP50"]:.6f}, delta AP50:95 {b["delta_AP50_95"]:.6f}, CI95 kolumn [AP50, AP50:95]: {np.asarray(b["CI95"]).tolist()}.')
    lines+=['','Podział historyczny 240/75/85 obrazów i 69/23/24 filmów, bez overlap. Obrazy były używane rozwojowo i mogły uczyć ViT, a kanały wybrano historycznie. To pilot z rozdzieleniem filmów w tym zbiorze, bez niezależnego dowodu generalizacji. GT tylko loss i ocena, bez maski/ROI w inferencji.','', 'Wspólny kierunek działa jednakowo dla wszystkich obrazów. Oracle per-image optymalizuje osobno z GT danego obrazu; tutaj go nie wykonano. CAV to kierunek sondy rozróżniającej koncept; TCAV to analiza znaku pochodnej odpowiedzi wzdłuż CAV. Tutaj kierunek uczony jest bezpośrednio stratą detekcji, nie CAV ani TCAV. Jest to poprawa przez jawną interwencję, nie wyuczenie naturalnego latentu w wagach. Brak poprawy AP albo wynik słabszy od losowych kontroli jest pełnoprawnym wynikiem.','', f'Świeży reload, pełny ViT kontra szybka końcówka: PASS. Wagi i BN niezmienione. Czas {elapsed:.1f} s, PID {os.getpid()}, RTX 3070. Proces zakończył obliczenia.','', 'Reprodukcja: F:/Python/python.exe -B experiments/vit_kierunek_nadzorowany_detekcja128/run.py z SmallViTAblation. RUN.lock zapobiega dwóm kopiom. Wyczyścić zakończone własne wyniki przed nowym biegiem lub zachować je osobno. Wszystkie źródła i indeksy: MANIFEST.json; RNG i parametry: CONFIG.json; korekta: DIRECTION.npz; dobór przed testem: FREEZE.json; krzywe: HISTORY.json i CURVES.png.']
    report='\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(report,encoding='utf-8');(HERE/'WYNIKI_PILOT.md').write_text(report,encoding='utf-8')
    js(OUT/'ARTIFACT_HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.npz','.npy','.png','.md'] and p.name not in ['PROGRESS.json','ARTIFACT_HASHES.json']})
    progress('complete',metrics=metrics,seconds=elapsed)
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();progress('failed',traceback=traceback.format_exc());raise
