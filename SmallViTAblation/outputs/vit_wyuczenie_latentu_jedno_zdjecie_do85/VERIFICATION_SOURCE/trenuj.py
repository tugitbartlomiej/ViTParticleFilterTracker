"""Czy naturalny GELU osiągnie agreement128>=85? Niższy wynik po budżecie/plateau obala osiągnięcie w tym biegu."""
import os,sys,time,json,csv,copy,shutil,traceback,subprocess,msvcrt
from datetime import datetime,timezone
import torch,numpy as np,timm
import aparat as a
from aparat import HERE,ROOT,SOURCE,OUT,PREVIOUS,RESUME,CONFIG,ARCH,js,load,sha,state,measure,successful,export,audit
from model_snapshot import MalyViT,cele,strata

def save(net,opt,names,step,metrics,path):
    weights=state(net);changed=[n for n in weights if not torch.equal(weights[n],a._REFERENCE[n])]
    assert set(changed)<=set(names)
    payload=dict(net=weights,optimizer=opt.state_dict(),trainable_parameters=names,changed_parameters=changed,
                 architecture=ARCH,torch_version=str(torch.__version__),timm_version=str(timm.__version__),
                 config=CONFIG,stage='T1_norm2_fc1_do85',step=step,metrics=metrics,source_checkpoint_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt'))
    tmp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp');torch.save(payload,tmp)
    for attempt in range(12):
        try:os.replace(tmp,path);return
        except OSError:
            if attempt==11:raise
            time.sleep(min(.02*2**attempt,.25))
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    # OS-held byte lock prevents concurrent writers, and releases automatically on process exit.
    lock=(OUT/'WRITER.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Existing completed/pending result; no overwrite')
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
    original=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'];a._REFERENCE=original
    resume=torch.load(RESUME,map_location='cpu',weights_only=False)
    assert resume['step']==2000 and resume['optimizer'] is not None and resume['architecture']==ARCH
    net=MalyViT().to(device).eval();net.load_state_dict(resume['net'],strict=True);net.requires_grad_(False)
    names=[n for n,p in net.named_parameters() if n.startswith(('bloki.5.norm2.','bloki.5.mlp.fc1.'))]
    assert names==resume['trainable_parameters']
    for n,p in net.named_parameters():p.requires_grad_(n in names)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=CONFIG['lr']);opt.load_state_dict(resume['optimizer'])
    assert len(opt.state)==4 and all(float(v['step'])==2000 for v in opt.state.values())
    assert all(p['lr']==3e-5 for p in opt.param_groups)
    cache={}
    def read(m,i,o):cache['z']=o
    hook=net.bloki[-1].mlp.act.register_forward_hook(read)
    with torch.no_grad():o,_=net(image)
    prev=np.load(PREVIOUS/'LATENT_NATURAL_final.npz')
    torch.testing.assert_close(cache['z'][0].cpu(),torch.from_numpy(prev['natural']),rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(o.cpu(),torch.from_numpy(prev['detector_map']),rtol=2e-5,atol=2e-5)
    base=measure(cache['z'],o,target,J,K,g,targets)
    for k in ['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss']:
        assert abs(base[k]-resume['metrics'][k])<2e-5,(k,base[k],resume['metrics'][k])
    baseline=dict(metrics=base,latent_max_abs_error=float(np.max(abs(cache['z'][0].detach().cpu().numpy()-prev['natural']))),initial_step=2000,
                  resume_checkpoint=str(RESUME),resume_checkpoint_sha256=sha(RESUME),previous_final_latent_sha256=sha(PREVIOUS/'LATENT_NATURAL_final.npz'),
                  source_hashes={n:sha(SOURCE/n) for n in ['MODEL_VIT_K1000.pt','LATENT_PELNY.npz','ZDJECIE_ZRODLOWE.jpg','WEJSCIE_VIT.png']})
    js(OUT/'BASELINE_CHECK.json',baseline)
    env=dict(config=CONFIG,architecture=ARCH,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
             torch=str(torch.__version__),timm=timm.__version__,device=device,source=meta['record'],resume_checkpoint_sha256=sha(RESUME),
             resume_optimizer_steps=[float(v['step']) for v in opt.state.values()],resume_lr=[p['lr'] for p in opt.param_groups],
             target_energy128=float(target[:,J].square().mean()),target_energy640=float(target[:,K].square().mean()),
             code_hashes={p.name:sha(p) for p in HERE.glob('*.py')})
    js(OUT/'RUN_CONFIG.json',env)
    snapshots=OUT/'EXECUTED_SOURCE';snapshots.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,snapshots/p.name)
    energy128=target[:,J].square().mean().detach();energy640=target[:,K].square().mean().detach()
    save(net,opt,names,2000,base,OUT/'MODEL_before.pt');save(net,opt,names,2000,base,OUT/'MODEL_best_live.pt')
    best=base;best_step=2000;max_agreements=[base['agreement128']];events=[];lowered_at=None;lowered_best=None;reason='engineering_budget52000'
    records=[dict(step=2000,seconds=time.perf_counter()-start,lr=opt.param_groups[0]['lr'],**base)]
    print(json.dumps(records[-1]),flush=True)
    hist=(OUT/'HISTORY.jsonl').open('a',encoding='utf-8');hist.write(json.dumps(records[-1])+'\n');hist.flush()
    for step in range(2001,52001):
        opt.zero_grad(set_to_none=True);o,_=net(image);z=cache['z'][0]
        l128=(z[:,J]-target[:,J]).square().mean()/energy128;l640=(z[:,K]-target[:,K]).square().mean()/energy640
        dl=strata(o,*targets)[0];loss=l128+CONFIG['retention']*l640+CONFIG['detection_weight']*dl
        assert torch.isfinite(loss),'nonfinite loss';loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad),'nonfinite gradient'
        opt.step()
        with torch.no_grad():o,_=net(image);m=measure(cache['z'],o,target,J,K,g,targets)
        assert all(np.isfinite(m[k]) for k in ['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss'])
        if m['agreement128']>best['agreement128']:
            best=m;best_step=step;save(net,opt,names,step,m,OUT/'MODEL_best_live.pt')
        max_agreements.append(best['agreement128'])
        row=dict(step=step,seconds=time.perf_counter()-start,lr=opt.param_groups[0]['lr'],loss_before_update=float(loss.detach()),
                 normalized_MSE128_before=float(l128.detach()),normalized_MSE640_before=float(l640.detach()),**m)
        records.append(row);hist.write(json.dumps(row)+'\n');hist.flush()
        if step%100==0:save(net,opt,names,step,m,OUT/'MODEL_latest.pt')
        if step%1000==0:
            save(net,opt,names,step,m,OUT/f'MODEL_chunk_{step:05d}.pt')
            js(OUT/f'BIEG_chunk_{step:05d}.json',dict(step=step,best_step=best_step,best=best,metrics=m,seconds=row['seconds'],lr=row['lr']))
        js(OUT/'PROGRESS.json',dict(status='training',pid=os.getpid(),best_step=best_step,best_agreement=best['agreement128'],target_agreement=85,**row))
        if step in (2001,2010,2025,2050) or step%100==0:print(json.dumps(row),flush=True)
        if successful(m):reason='agreement85_candidate_pending_reload';break
        if step>=4000 and lowered_at is None and max_agreements[-1]-max_agreements[-2001]<CONFIG['plateau_gain_pp']:
            old=opt.param_groups[0]['lr']
            for group in opt.param_groups:group['lr']=CONFIG['minimum_plateau_lr']
            lowered_at=step;lowered_best=best['agreement128']
            event=dict(step=step,old_lr=old,new_lr=CONFIG['minimum_plateau_lr'],reason='best agreement gain in last2000 <0.10pp',gain_pp=max_agreements[-1]-max_agreements[-2001]);events.append(event);js(OUT/'LR_EVENTS.json',events);print('LR_EVENT',json.dumps(event),flush=True)
        if lowered_at is not None and step-lowered_at>=4000 and best['agreement128']-lowered_best<CONFIG['stagnation_gain_pp']:
            reason='long_stagnation_after_lr_reduction';break
        time.sleep(CONFIG['pause_seconds'])
    hist.close();save(net,opt,names,step,m,OUT/'MODEL_final.pt');save(net,opt,names,step,m,OUT/'MODEL_latest.pt')
    shutil.copy2(OUT/'MODEL_best_live.pt',OUT/'MODEL_best.pt');js(OUT/'LR_EVENTS.json',events)
    hook.remove();cache.clear();net.requires_grad_(False)
    for tag in ('before','best','final'):
        saved=torch.load(OUT/f'MODEL_{tag}.pt',map_location='cpu',weights_only=False);net.load_state_dict(saved['net'])
        export(net,image,target,before,J,K,g,targets,tag,original,names,rgb)
    with (OUT/'HISTORY.csv').open('w',newline='',encoding='utf-8') as f:
        fields=sorted(set().union(*(r.keys() for r in records)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    js(OUT/'HISTORY.json',records)
    result=dict(status='training_complete_pending_fresh_reload',ziarno=CONFIG['seed'],baseline=records[0],best=best,final=m,best_step=best_step,final_step=step,
                new_updates=step-2000,stop_reason=reason,elapsed_seconds=time.perf_counter()-start,config=CONFIG,optimizer='Adam',resumed_optimizer=True,
                resumed_from_step=2000,uses_source_GT=True,single_image_overfit=True,success=False,candidate_agreement85=successful(best),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,lr_events=events)
    js(OUT/'RESULT.json',result);a.zapisz_wynik(__file__,'training',result)
    js(OUT/'PROGRESS.json',dict(status='training_complete_pending_fresh_reload',pid=os.getpid(),step=step,best=best))
    del net,opt;torch.cuda.empty_cache() if device=='cuda' else None
    # A separate process must confirm the primary criterion before success is declared.
    with (OUT/'VERIFY.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'VERIFY.stderr.log').open('w',encoding='utf-8') as stderr:
        ret=subprocess.run([sys.executable,'-B','-u',str(HERE/'sprawdz.py')],cwd=str(ROOT),stdout=stdout,stderr=stderr)
    assert ret.returncode==0,'Fresh reload failed; inspect VERIFY.stderr.log'
    print('COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        js(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),pid=os.getpid()));js(OUT/'PROGRESS.json',dict(status='failed',pid=os.getpid(),error=str(e)));raise
