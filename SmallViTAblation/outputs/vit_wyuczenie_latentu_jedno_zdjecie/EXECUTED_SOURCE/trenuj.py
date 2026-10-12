"""Czy uczenie wag daje naturalny cel GELU? Agreement<90 lub błędna ramka obala sukces w budżecie."""
import os,sys,time,json,csv,math,platform,subprocess,traceback,copy,shutil
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import numpy as np,torch,timm
from model_snapshot import MalyViT,dekoduj,cele,strata
from dane import HERE,ROOT,SOURCE,OUT,load,sha
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik

CONFIG=dict(seed=20261005,steps=1000,lr=0.0003,min_lr=0.00003,retention=10.0,detection_weight=0.02,
            pause_seconds=0.03,agreement_threshold=90.0,iou_threshold=0.90,confidence_threshold=0.90,
            plateau_window=250,plateau_gain_pp=0.2,precision='float32',autocast=False,tf32=False)
ARCH=dict(name='MalyViT',dim=192,glebokosc=6,glowy=3,mlp=4.0,model_source_sha256=sha(HERE/'model_snapshot.py'))
def js(p,v):
    tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(12):
        try:os.replace(tmp,p);return True
        except OSError as e:
            if attempt<11:time.sleep(min(.02*2**attempt,.25));continue
            if p.name=='PROGRESS.json':
                print(f'PROGRESS_WRITE_SKIPPED {type(e).__name__}: {e}',file=sys.stderr,flush=True)
                try:tmp.unlink()
                except OSError:pass
                return False
            raise
def state(net):return {n:t.detach().cpu().clone() for n,t in net.state_dict().items()}
def iou(b,g):
    d=np.maximum(0,np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2]));inter=float(np.prod(d))
    return inter/(float(np.prod(b[2:]-b[:2]))+float(np.prod(g[2:]-g[:2]))-inter+1e-9)
def det(o,g):
    boxes,scores=dekoduj(o.detach().cpu());b=boxes[0,0].numpy()
    return dict(confidence=float(scores[0,0]),iou=iou(b,g),bbox_xyxy=b.tolist(),all_boxes_xyxy=boxes[0].tolist(),all_scores=scores[0].tolist())
def measure(z,o,target,J,K,g,targets):
    z=z[0];err=(z-target).square();m128=float(err[:,J].mean());m640=float(err[:,K].mean());energy=float(target[:,J].square().mean())
    a=z[:,J].flatten();b=target[:,J].flatten()
    gt_loss=float(strata(o,*targets)[0].detach())
    return dict(MSE128=m128,MSE640=m640,MSE768=float(err.mean()),agreement128=100*max(0,1-math.sqrt(m128/energy)),
                cosine128=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),GTloss=gt_loss,**det(o,g))
def successful(m):return m['agreement128']>=90 and m['iou']>=.9 and m['confidence']>=.9
def score(m):return (m['iou']>=.9 and m['confidence']>=.9,m['agreement128'])
def checkpoint(net,opt,names,stage,step,metrics,path):
    torch.save(dict(net=state(net),optimizer=opt.state_dict() if opt else None,trainable_parameters=names,
                    architecture=ARCH,torch_version=str(torch.__version__),timm_version=str(timm.__version__),
                    config=CONFIG,stage=stage,step=step,metrics=metrics,source_checkpoint_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt')),path)
def audit(net,original,names):
    now=state(net);changed=[n for n in now if not torch.equal(now[n],original[n])]
    assert set(changed)<=set(names),[n for n in changed if n not in names]
    bn=[n for n in now if 'running_' in n or 'num_batches_tracked' in n]
    return dict(changed_parameters=changed,allowed_parameters=names,frozen_unchanged=True,BN_buffers_unchanged=all(torch.equal(now[n],original[n]) for n in bn),
                every_tensor=[dict(name=n,changed=n in changed,allowed=n in names,max_abs_delta=float((now[n]-original[n]).abs().max())) for n in now])
def export(net,image,target,before,J,K,g,targets,tag,original,names,rgb):
    net.eval();cache={}
    def read(m,i,o):cache['z']=o
    h=net.bloki[-1].mlp.act.register_forward_hook(read)
    with torch.no_grad():o,_=net(image)
    h.remove();z=cache['z'];m=measure(z,o,target,J,K,g,targets)
    arr=z[0].cpu().numpy();j=J.cpu().numpy()
    np.savez_compressed(OUT/f'LATENT_NATURAL_{tag}.npz',natural=arr,before=before.cpu().numpy(),target=target.cpu().numpy(),channels0=j,grid_shape=[20,30],detector_map=o.cpu().numpy())
    with (OUT/f'VEKTORY_J128_{tag}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['token','row','col','channel0','before','target','natural','error'])
        aa=before.cpu().numpy();bb=target.cpu().numpy()
        for t in range(600):
            for c in j:w.writerow([t,t//30,t%30,int(c),float(aa[t,c]),float(bb[t,c]),float(arr[t,c]),float(arr[t,c]-bb[t,c])])
    js(OUT/f'AUDIT_{tag}.json',audit(net,original,names))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(8,5));ax.imshow(rgb)
    for box,color,label in [(g,'cyan','GT'),(m['bbox_xyxy'],'lime','Natural prediction')]:
        ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=2,label=label))
    ax.set_title(f"{tag}: agreement {m['agreement128']:.3f}% | confidence {m['confidence']:.6f} | IoU {m['iou']:.6f}");ax.legend();ax.axis('off');fig.tight_layout();fig.savefig(OUT/f'OVERLAY_{tag}.png',dpi=150);plt.close(fig)
    return m
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Completed results exist; refusing overwrite')
    start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu'
    if torch.cuda.is_available():
        wait_start=time.perf_counter()
        while time.perf_counter()-wait_start<60:
            free,total=torch.cuda.mem_get_info()
            if free>=1800*1024**2:device='cuda';break
            js(OUT/'PROGRESS.json',dict(status='waiting_for_memory',pid=os.getpid(),free_bytes=free));time.sleep(5)
        if device=='cuda':torch.cuda.set_per_process_memory_fraction(.22)
    meta,rgb,image,before,target,J,K,g=load(device)
    targets=tuple(t.to(device) for t in cele([g[None]],20,30))
    net=MalyViT().to(device).eval();ck=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False);original=state(net)
    cache={}
    def read(m,i,o):cache['z']=o
    hook=net.bloki[-1].mlp.act.register_forward_hook(read)
    with torch.no_grad():o,_=net(image)
    baseline_z=cache['z'][0].detach();torch.testing.assert_close(baseline_z,before,rtol=2e-5,atol=2e-5)
    base=measure(cache['z'],o,target,J,K,g,targets)
    assert abs(base['confidence']-meta['baseline']['confidence'])<2e-5
    assert abs(base['iou']-meta['baseline']['iou'])<2e-5
    js(OUT/'BASELINE_CHECK.json',dict(metrics=base,latent_max_abs_error=float((baseline_z-before).abs().max()),rtol=2e-5,atol=2e-5,
          checkpoint_matches=True,RGB_matches_saved=True,source_index=29,source_hashes={n:sha(SOURCE/n) for n in ['MODEL_VIT_K1000.pt','LATENT_PELNY.npz','ZDJECIE_ZRODLOWE.jpg','WEJSCIE_VIT.png']}))
    checkpoint(net,None,[],'before',0,base,OUT/'MODEL_before.pt')
    env=dict(config=CONFIG,architecture=ARCH,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
             python=sys.version,torch=str(torch.__version__),timm=timm.__version__,device=device,source=meta['record'],
             target_energy128=float(target[:,J].square().mean()),target_energy640=float(target[:,K].square().mean()),
             code_hashes={p.name:sha(p) for p in HERE.glob('*.py')})
    js(OUT/'RUN_CONFIG.json',env)
    energy128=target[:,J].square().mean().detach();energy640=target[:,K].square().mean().detach()
    records=[];summaries=[];allnames=[];global_best=None;global_rank=(-1,-1)
    for stage in ['T1_norm2_fc1','T2_attention_norm1']:
        names=[n for n,p in net.named_parameters() if n.startswith(('bloki.5.norm2.','bloki.5.mlp.fc1.')) or
               (stage.startswith('T2') and n.startswith(('bloki.5.norm1.','bloki.5.attn.')))]
        allnames=names;net.requires_grad_(False)
        for n,p in net.named_parameters():p.requires_grad_(n in names)
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=CONFIG['lr'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CONFIG['steps'],eta_min=CONFIG['min_lr'])
        stage_start=time.perf_counter();best_state=state(net);best_metric=base;best_step=0;best_opt=copy.deepcopy(opt.state_dict());best_rank=score(base)
        max_agree=base['agreement128'];agreements=[max_agree];reason='budget';step=0
        with torch.no_grad():o,_=net(image);m=measure(cache['z'],o,target,J,K,g,targets)
        row=dict(stage=stage,step=0,seconds=time.perf_counter()-start,lr=opt.param_groups[0]['lr'],**m)
        records.append(row);print(json.dumps(row),flush=True)
        for step in range(1,CONFIG['steps']+1):
            opt.zero_grad(set_to_none=True);o,_=net(image);z=cache['z'][0]
            l128=(z[:,J]-target[:,J]).square().mean()/energy128
            l640=(z[:,K]-target[:,K]).square().mean()/energy640
            dl=strata(o,*targets)[0];loss=l128+CONFIG['retention']*l640+CONFIG['detection_weight']*dl
            assert torch.isfinite(loss), 'nonfinite loss';loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
            opt.step();sched.step()
            with torch.no_grad():o,_=net(image);m=measure(cache['z'],o,target,J,K,g,targets)
            max_agree=max(max_agree,m['agreement128']);agreements.append(max_agree)
            if score(m)>best_rank:
                best_rank=score(m);best_state=state(net);best_metric=m;best_step=step;best_opt=copy.deepcopy(opt.state_dict())
                checkpoint(net,opt,names,stage,step,m,OUT/f'MODEL_{stage}_best_live.pt')
            row=dict(stage=stage,step=step,seconds=time.perf_counter()-start,lr=opt.param_groups[0]['lr'],loss_before_update=float(loss.detach()),
                     normalized_MSE128_before=float(l128.detach()),normalized_MSE640_before=float(l640.detach()),**m)
            records.append(row)
            js(OUT/'PROGRESS.json',dict(status='training',pid=os.getpid(),best_step=best_step,best_agreement=best_metric['agreement128'],**row))
            if step in (1,10,25,50,100,250,500,1000) or step%100==0:print(json.dumps(row),flush=True)
            if successful(m):reason='success';break
            if step>=500 and agreements[-1]-agreements[-251]<CONFIG['plateau_gain_pp']:reason='plateau';break
            time.sleep(CONFIG['pause_seconds'])
        checkpoint(net,opt,names,stage,step,m,OUT/f'MODEL_{stage}_final.pt')
        net.load_state_dict(best_state);opt.load_state_dict(best_opt)
        checkpoint(net,opt,names,stage,best_step,best_metric,OUT/f'MODEL_{stage}_best.pt')
        summaries.append(dict(stage=stage,steps=step,best_step=best_step,best=best_metric,final=m,stop_reason=reason,
                         seconds=time.perf_counter()-stage_start,trainable_parameters=names,
                         trainable_count=sum(p.numel() for p in net.parameters() if p.requires_grad),tail_gain_pp=agreements[-1]-agreements[max(0,len(agreements)-251)]))
        js(OUT/f'BIEG_{stage}.json',summaries[-1])
        if best_rank>global_rank:global_rank=best_rank;global_best=OUT/f'MODEL_{stage}_best.pt'
        base=best_metric
        if successful(best_metric) or stage.startswith('T2'):break
        if summaries[-1]['tail_gain_pp']>=CONFIG['plateau_gain_pp']:break
    hook.remove();cache.clear();net.requires_grad_(False)
    for tag,path in [('before',OUT/'MODEL_before.pt'),('best',global_best),('final',OUT/f'MODEL_{summaries[-1]["stage"]}_final.pt')]:
        saved=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(saved['net'])
        if tag!='before':shutil.copy2(path,OUT/f'MODEL_{tag}.pt')
        export(net,image,target,before,J,K,g,targets,tag,original,saved['trainable_parameters'],rgb)
    with (OUT/'HISTORY.csv').open('w',newline='',encoding='utf-8') as f:
        fields=sorted(set().union(*(r.keys() for r in records)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    js(OUT/'HISTORY.json',records)
    result=dict(status='training_complete_pending_fresh_reload',ziarno=CONFIG['seed'],baseline=records[0],runs=summaries,
                best_checkpoint=str(global_best),success=successful(torch.load(global_best,map_location='cpu',weights_only=False)['metrics']),
                elapsed_seconds=time.perf_counter()-start,optimizer='Adam',config=CONFIG,uses_source_GT=True,single_image_overfit=True,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    js(OUT/'RESULT.json',result);zapisz_wynik(__file__,'training',result)
    js(OUT/'PROGRESS.json',dict(status='training_complete_pending_fresh_reload',pid=os.getpid(),elapsed_seconds=result['elapsed_seconds'],success=result['success']))
    print('TRAINING_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        js(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),pid=os.getpid()));js(OUT/'PROGRESS.json',dict(status='failed',pid=os.getpid(),error=str(e)));raise
