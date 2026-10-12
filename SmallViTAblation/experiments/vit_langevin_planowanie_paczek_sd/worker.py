"""Czy kolejność czterech paczek wybrana aktualną prognozą Langevina zmniejsza
MSE J128 sztucznego celu sondy29 bardziej niż losowanie i równy cykl?
Brak przewagi obala użyteczność tej polityki w zadanym budżecie. Każda decyzja
powstaje przed aktualizacją z obecnych wag/AdamW/GT, jest zapisana i sealed.
To greedy receding-horizon1, nie dowód globalnej optymalności sekwencji.
Nie używa CAV, przyszłych gradientów ani podmiany latentu.
"""
import argparse,copy,csv,gc,hashlib,json,os,random,shutil,subprocess,time,traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import cv2
import model as M
import rownanie_original as RW
from dane import prepare,ROOT,OUT,SOURCE,PREV,SEEDS,sha
HERE=Path(__file__).resolve().parent
POLICIES=['controlled','random_choice','balanced_cycle']
SAVED={1,5,10,25,50,100,200,300,400,500}
START=time.time()
COST=dict(candidate_gradient_batches=0,candidate_probe_forwards=0,training_updates=0,training_image_exposures=0)

def js(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for _ in range(30):
        try:tmp.replace(p);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError(f'Atomic replace failed: {p}')

def progress(phase,**kw):
    value=dict(status='running',phase=phase,pid=os.getpid(),seconds=time.time()-START,cost=COST,**kw)
    js(OUT/'PROGRESS.json',value);print(phase,kw,flush=True)

def wait_gpu():
    deadline=time.monotonic()+5400
    while True:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        free=int(r.stdout.splitlines()[0]) if r.returncode==0 else 0
        if free>=2816:return
        progress('queued_shared_GPU',free_mib=free,required_free_mib=2816)
        if time.monotonic()>deadline:raise TimeoutError('Full32 BN cannot fit; bounded90minute queue')
        time.sleep(10)

def main(args):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Completed result already exists')
    if not args.smoke and (OUT/'RUN_ACTIVE.json').exists():
        prior=json.loads((OUT/'RUN_ACTIVE.json').read_text())
        import psutil
        if psutil.pid_exists(prior['pid']):raise RuntimeError('Same-family live worker exists; refuse duplicate')
    (OUT/'PID.txt').write_text(str(os.getpid()))
    torch.set_num_threads(4);torch.manual_seed(20261006);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    images,boxes,selection=prepare();progress('inputs_verified_CPU')
    cp=PREV/'MODEL_AUTHENTIC_K1000.pt'
    provenance=json.loads((PREV/'CHECKPOINT_VALIDATION.json').read_text(encoding='utf-8'))
    assert sha(cp)==provenance['authentic_checkpoint_sha256']
    st=torch.load(cp,map_location='cpu',weights_only=True)
    ref=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)
    assert st['krok']==1000 and 'opt' in st and st['net'].keys()==ref['net'].keys()
    assert all(torch.equal(st['net'][n],ref['net'][n]) for n in ref['net'])
    shutil.copy2(cp,OUT/'MODEL_ORIGINAL_ADAMW_K1000.pt')
    js(OUT/'CHECKPOINT_VALIDATION.json',dict(status='passed',original_SD_k1000_all_weights_equal=True,
       authentic_optimizer_sha256=sha(cp),original_model_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt'),
       optimizer_groups=st['opt']['param_groups'],new_single_image_fit_used=False,previous_provenance=provenance))
    with np.load(SOURCE/'LATENT_PELNY.npz') as f:
        target=f['after'].astype(np.float64);before=f['before'].copy();J=f['channels0'].copy()
    assert target.shape==before.shape==(600,768) and len(J)==128
    other=np.array([i for i in range(768) if i not in set(J)])
    assert len(other)==640
    im=cv2.imread(str(OUT/'SONDA_ORIGINAL.jpg'))
    rgb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    np.save(OUT/'SONDA_RGB_480x320.npy',rgb)
    x,y,w,h=selection['record']['bbox_xywh'];GT=np.array([x,y,x+w,y+h],np.float64)
    wait_gpu();probe=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255
    ARM_NAMES=list(map(str,SEEDS))
    def make(state):
        net=M.MalyViT().cuda();net.load_state_dict(state['net']);net.zapisuj_latent(True)
        opt=torch.optim.AdamW([dict(params=[p for p in net.parameters() if p.ndim>=2],weight_decay=.05),dict(params=[p for p in net.parameters() if p.ndim<2],weight_decay=0)],lr=5e-4,betas=(.9,.999))
        opt.load_state_dict(copy.deepcopy(state['opt']))
        return SimpleNamespace(net=net,opt=opt,dev='cuda',_lr=lambda:5e-4)
    def detector(o):
        b,s=M.dekoduj(o);bb=b[0].cpu().numpy().astype(np.float64)
        inter=np.prod(np.maximum(np.minimum(bb[:,2:],GT[2:])-np.maximum(bb[:,:2],GT[:2]),0),axis=1)
        area=np.prod(np.maximum(bb[:,2:]-bb[:,:2],0),axis=1)
        iou=inter/np.maximum(area+np.prod(GT[2:]-GT[:2])-inter,1e-30)
        return dict(confidence=float(s[0,0]),iou=float(iou[0]),max_iou_top10=float(iou.max()))
    def read(u,delta=None):
        u.net.eval();params=dict(u.net.named_parameters());buffers=dict(u.net.named_buffers())
        if delta is not None:params={n:p+delta[n] for n,p in params.items()}
        with torch.no_grad():
            o,z=torch.func.functional_call(u.net,(params,buffers),(probe,))
            a=z[0].float().cpu().numpy().copy();d=detector(o)
        u.net._latent=None
        return a,d
    u=make(st);z0,d0=read(u);np.testing.assert_allclose(z0,before,rtol=1e-4,atol=3e-5)
    initial={}
    target_rms=float(np.sqrt(np.mean(target[:,J]**2)))
    assert target_rms>0
    def measure(z):
        z=np.asarray(z,np.float64);m={}
        for name,ix in [('128',J),('640',other),('768',np.arange(768))]:
            a=z[:,ix];b=target[:,ix]
            m['mse'+name]=float(np.mean((a-b)**2))
            m['gain'+name]=1-m['mse'+name]/initial['mse'+name] if initial.get('mse'+name,0)>0 else 0.0
            m['cosine'+name]=float(np.sum(a*b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-30))
        m['value_agreement128']=float(np.clip(1-np.sqrt(m['mse128'])/target_rms,0,1))
        return m
    initial=measure(z0)
    np.savez_compressed(OUT/'BASELINE.npz',before=z0,target=target,channels0=J,other640=other)
    js(OUT/'INITIAL.json',dict(**initial,**d0,exact_before_max_abs_error=float(np.max(abs(z0-before))),
       value_agreement_definition='clip(1-RMSE(z_J128,target_J128)/RMS(target_J128),0,1)',target_rms128=target_rms,
       agreement_threshold=.85,error_reduction_thresholds=[.05,.1,.25]))
    def batch(arm):
        x=torch.from_numpy(images[arm]).permute(0,3,1,2).float().cuda()/255
        c=tuple(t.cuda() for t in M.cele([np.asarray(b,np.float32).reshape(-1,4) for b in boxes[arm]],20,30))
        return x,c
    def candidate(u,arm,current):
        x,c=batch(arm)
        g=RW.gradient_tensorow(u,x,*c);u.net._latent=None
        new,ped=RW.czesci_kroku(u,g)
        zn,_=read(u,new);zp,_=read(u,ped)
        sn=zn-current;sp=zp-current;force=sn+sp
        predicted=current+force
        desired=target[:,J]-current[:,J];f=force[:,J].astype(np.float64)
        dot=float(np.sum(f*desired));den=float(np.linalg.norm(f)*np.linalg.norm(desired))
        info=dict(arm=arm,predicted=measure(predicted),alignment_dot128=dot,
          alignment_cosine128=dot/max(den,1e-30),coordinate_toward_fraction128=float(np.mean(f*desired>0)),
          new_norm128=float(np.linalg.norm(sn[:,J])),momentum_norm128=float(np.linalg.norm(sp[:,J])),
          force_sha256=hashlib.sha256(force.tobytes()).hexdigest())
        COST['candidate_gradient_batches']+=1;COST['candidate_probe_forwards']+=2
        del x,c,g,new,ped;gc.collect()
        return info,force,sn,sp
    torch.cuda.reset_peak_memory_stats()
    base_buffers={n:b.clone() for n,b in u.net.named_buffers()}
    base_params={n:p.detach().clone() for n,p in u.net.named_parameters()}
    candidates=[]
    for arm in ARM_NAMES:
        info,force,sn,sp=candidate(u,arm,z0);candidates.append(info)
        progress('initial_candidate_evaluated',arm=arm,predicted_mse128=info['predicted']['mse128'])
    assert all(torch.equal(base_buffers[n],b) for n,b in u.net.named_buffers())
    assert all(torch.equal(base_params[n],p) for n,p in u.net.named_parameters())
    assert all(p.grad is None for p in u.net.parameters())
    js(OUT/'SMOKE.json',dict(status='passed',full32=True,original_weights_and_before_match=True,
       candidate_buffers_params_unchanged=True,candidates=candidates,selected=min(candidates,key=lambda a:a['predicted']['mse128'])['arm'],
       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    del u,base_params,base_buffers,force,sn,sp;gc.collect();torch.cuda.empty_cache()
    if args.smoke:progress('smoke_passed');return
    js(OUT/'RUN_ACTIVE.json',dict(pid=os.getpid(),steps=args.steps))
    budget=args.steps;assert budget in (100,500)
    snapshots={p:st for p in POLICIES};series={p:[dict(step=0,**initial,**d0)] for p in POLICIES}
    schedules={p:[] for p in POLICIES};best={p:dict(step=0,metrics=initial,detector=d0) for p in POLICIES}
    rng=random.Random(20261006)
    streams={};writers={};latents={};seals={p:[] for p in POLICIES}
    state_dir=OUT/'checkpoints';state_dir.mkdir(exist_ok=True)
    for p in POLICIES:
        (OUT/p/'decisions').mkdir(parents=True,exist_ok=True)
        (OUT/p/'forces').mkdir(exist_ok=True)
        latents[p]=np.lib.format.open_memmap(OUT/p/'TRAJECTORY.npy',mode='w+',dtype=np.float32,shape=(budget+1,600,768));latents[p][0]=z0
        streams[p]=open(OUT/p/'SCHEDULE.csv','w',newline='',encoding='utf-8')
        fields=['step','arm','checkpoint_step_before','predicted_mse128','actual_mse128','actual_mse640','actual_mse768','gain128','gain768','value_agreement128','confidence','iou','forecast_error_mse128','force_skill128_percent','force_alignment_cosine128','seal_sha256']
        writers[p]=csv.DictWriter(streams[p],fieldnames=fields);writers[p].writeheader()
        np.savez_compressed(OUT/p/'BEST_LATENT.npz',natural=z0,target=target,channels0=J)
        torch.save(dict(net=st['net'],opt=st['opt'],krok=1000,policy=p,best_step=0),state_dir/f'BEST_{p}.pt')
    stages=[100]+([500] if budget==500 else [])
    for stage in stages:
        for policy in POLICIES:
            gc.collect();torch.cuda.empty_cache();wait_gpu()
            u=make(snapshots[policy]);current=latents[policy][len(series[policy])-1].copy()
            start_step=len(series[policy])
            progress('closed_loop_stage',policy=policy,start_step=start_step,end_step=stage)
            for step in range(start_step,stage+1):
                steps_before={float(v['step']) for v in u.opt.state.values()}
                assert steps_before=={1000.0+step-1}
                candidates=[];chosen_data=None
                if policy=='controlled':
                    for arm in ARM_NAMES:
                        info,force,sn,sp=candidate(u,arm,current)
                        candidates.append(info)
                        if chosen_data is None or info['predicted']['mse128']<chosen_data[0]['predicted']['mse128']:
                            chosen_data=(info,force,sn,sp)
                else:
                    arm=rng.choice(ARM_NAMES) if policy=='random_choice' else ARM_NAMES[(step-1)%4]
                    chosen_data=candidate(u,arm,current);candidates=[chosen_data[0]]
                choice,force,sn,sp=chosen_data;arm=choice['arm'];pred=current+force
                # Each immutable decision is sealed before the genuine optimizer update.
                decision=dict(policy=policy,step=step,forecast_origin_step=step-1,checkpoint_step_before=1000+step-1,
                    current=measure(current),candidate_predictions=candidates,selected_arm=arm,
                    selection='argmin current one-step Langevin MSE128' if policy=='controlled' else policy,
                    selected_force_sha256=choice['force_sha256'],future_information_used=False,
                    forecast_horizon=1,coefficients=dict(nowa=1,ped=1),
                    omitted_terms='weight decay, BN buffer evolution, nonlinear interaction; evaluated by actual next state')
                decision_path=OUT/policy/'decisions'/f'{step:04d}.json';js(decision_path,decision)
                seal=sha(decision_path);seals[policy].append(seal)
                with (OUT/policy/'FORECAST_SEALS.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(step=step,sha256=seal,created_before_update=True))+'\n');f.flush();os.fsync(f.fileno())
                if step in SAVED:
                    np.savez_compressed(OUT/policy/'forces'/f'{step:04d}.npz',new=sn,momentum=sp,total=force,
                       before=current,predicted=pred,target=target,channels0=J)
                x,c=batch(arm);u.net.train();u.opt.zero_grad(set_to_none=True)
                for group in u.opt.param_groups:group['lr']=5e-4
                with torch.autocast('cuda',dtype=torch.bfloat16):output,_=u.net(x)
                loss,_=M.strata(output,*c);loss.backward();norm=torch.nn.utils.clip_grad_norm_(u.net.parameters(),1.0);u.opt.step()
                actual,d=read(u);m=measure(actual)
                delta=actual[:,J]-current[:,J];prediction_delta=force[:,J]
                error=float(np.mean((actual[:,J]-pred[:,J])**2))
                skill=float(100*(1-np.sum((delta-prediction_delta).astype(np.float64)**2)/max(float(np.sum(delta.astype(np.float64)**2)),1e-30)))
                metric=dict(step=step,arm=arm,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
                    forecast_mse128=choice['predicted']['mse128'],forecast_error_mse128=error,
                    force_skill128_percent=skill,force_alignment_cosine128=choice['alignment_cosine128'],**m,**d)
                series[policy].append(metric);schedules[policy].append(arm);latents[policy][step]=actual
                COST['training_updates']+=1;COST['training_image_exposures']+=32
                writers[policy].writerow(dict(step=step,arm=arm,checkpoint_step_before=1000+step-1,
                    predicted_mse128=choice['predicted']['mse128'],actual_mse128=m['mse128'],actual_mse640=m['mse640'],actual_mse768=m['mse768'],
                    gain128=m['gain128'],gain768=m['gain768'],value_agreement128=m['value_agreement128'],confidence=d['confidence'],iou=d['iou'],
                    forecast_error_mse128=error,force_skill128_percent=skill,force_alignment_cosine128=choice['alignment_cosine128'],seal_sha256=seal))
                if m['mse128']<best[policy]['metrics']['mse128']:
                    best[policy]=dict(step=step,metrics=m,detector=d)
                    tmp=state_dir/f'BEST_{policy}.pt.tmp'
                    torch.save(dict(net=u.net.state_dict(),opt=u.opt.state_dict(),krok=1000+step,policy=policy,best_step=step),tmp)
                    tmp.replace(state_dir/f'BEST_{policy}.pt')
                    np.savez_compressed(OUT/policy/'BEST_LATENT.npz',natural=actual,target=target,channels0=J)
                    js(OUT/policy/'BEST.json',best[policy])
                if step==1 or step%5==0:
                    streams[policy].flush();latents[policy].flush()
                    js(OUT/policy/'ACTUAL.json',series[policy]);js(OUT/policy/'BEST.json',best[policy])
                    progress('closed_loop_updates',policy=policy,update=step,stage=stage,selected_arm=arm,
                        mse128=m['mse128'],gain128=m['gain128'],best_step=best[policy]['step'],
                        best_gain128=best[policy]['metrics']['gain128'],agreement128=m['value_agreement128'],confidence=d['confidence'],iou=d['iou'])
                current=actual
                del x,c,output,loss,chosen_data,force,sn,sp,pred
            torch.save(dict(net=u.net.state_dict(),opt=u.opt.state_dict(),krok=1000+stage,policy=policy),state_dir/f'{policy}_k{1000+stage}.pt')
            snapshots[policy]=torch.load(state_dir/f'{policy}_k{1000+stage}.pt',map_location='cpu',weights_only=True)
            latents[policy].flush();streams[policy].flush();js(OUT/policy/'ACTUAL.json',series[policy])
            js(OUT/policy/'BEST.json',best[policy])
            del u;gc.collect();torch.cuda.empty_cache()
        stage_result={p:dict(final=series[p][-1],best=best[p],selection_counts=dict(Counter(schedules[p])),updates=stage) for p in POLICIES}
        js(OUT/f'RESULT_{stage}.json',dict(status='complete_budget_stage',updates_per_policy=stage,policies=stage_result,
            sampling='every optimizer update measured; checkpoint saves are only save frequency'))
        progress('fair_budget_stage_complete',updates_each=stage,best_gain128={p:best[p]['metrics']['gain128'] for p in POLICIES})
    results={};reload_checks={}
    for policy in POLICIES:
        streams[policy].close();latents[policy].flush()
        cpbest=torch.load(state_dir/f'BEST_{policy}.pt',map_location='cpu',weights_only=True)
        gc.collect();torch.cuda.empty_cache();wait_gpu();u=make(cpbest)
        reloaded,d=read(u);expected=latents[policy][best[policy]['step']]
        difference=float(np.max(np.abs(reloaded-expected)))
        assert difference<3e-5
        assert np.isclose(measure(reloaded)['mse128'],best[policy]['metrics']['mse128'],rtol=1e-8)
        assert np.isclose(d['confidence'],best[policy]['detector']['confidence'],atol=1e-6)
        reload_checks[policy]=dict(max_latent_abs_error=difference,optimizer_step=cpbest['krok'],best_step=best[policy]['step'],natural_detector=d)
        for step,seal in enumerate(seals[policy],1):assert sha(OUT/policy/'decisions'/f'{step:04d}.json')==seal
        hits={str(r):next((m['step'] for m in series[policy] if m['gain128']>=r),None) for r in [.05,.1,.25]}
        agreement_hit=next((m['step'] for m in series[policy] if m['value_agreement128']>=.85),None)
        head_change=float(sum((p.detach().cpu()-ref['net'][n]).square().sum() for n,p in u.net.named_parameters() if n.startswith('glowa.')))
        prefix=schedules[policy][:best[policy]['step']]
        compressed=[]
        for arm in prefix:
            if compressed and compressed[-1]['arm']==arm:compressed[-1]['repetitions']+=1
            else:compressed.append(dict(arm=arm,repetitions=1))
        js(OUT/policy/'BEST_SEQUENCE.json',dict(best_step=best[policy]['step'],sequence=prefix,run_length_encoding=compressed))
        final_head_change=float(sum((snapshots[policy]['net'][n]-ref['net'][n]).square().sum() for n in ref['net'] if n.startswith('glowa.') and n in dict(u.net.named_parameters())))
        results[policy]=dict(final=series[policy][-1],best=best[policy],updates=budget,images=32*budget,
           schedule=schedules[policy],selection_counts=dict(Counter(schedules[policy])),
           MSE128_reduction_threshold_steps=hits,value_agreement85_step=agreement_hit,
           best_head_delta_squared=head_change,final_head_delta_squared=final_head_change,
           median_one_step_force_skill128_percent=float(np.median([m['force_skill128_percent'] for m in series[policy][1:]])),
           mean_forecast_error_mse128=float(np.mean([m['forecast_error_mse128'] for m in series[policy][1:]])))
        del cpbest,u;gc.collect();torch.cuda.empty_cache()
    js(OUT/'BEST_RELOAD_VERIFICATION.json',dict(status='passed',policies=reload_checks,all_decision_seals_verified=True))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for p in POLICIES:
        rows=series[p];steps=range(budget+1)
        axs[0,0].plot(steps,[m['mse128'] for m in rows],label=p)
        axs[0,1].plot(steps,[100*m['value_agreement128'] for m in rows],label=p)
        axs[1,0].plot(steps,[m['mse640'] for m in rows],label=p)
        axs[1,1].plot(steps,[m['iou'] for m in rows],label=p)
    for ax,title in zip(axs.flat,['Artificial target MSE J128','Defined value agreement J128 [%]','Other640 MSE','Natural original probe top1 IoU']):
        ax.set(title=title,xlabel='Actual optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'CONTROL_COMPARISON.png',dpi=160);plt.close(fig)
    result=dict(status='complete',ziarno=20261006,initial=dict(**initial,**d0),policies=results,budget_updates_each=budget,
       total_updates=3*budget,total_training_images=3*32*budget,cost=COST,seconds=time.time()-START,
       coefficients=dict(nowa=1,ped=1),control_horizon=1,forecast_information='present model/AdamW/current candidate GT only; each decision sealed before its genuine update',
       original_SD_checkpoint=True,latent_intervention=False,CAV_used=False,
       value_agreement_definition='clip(1-RMSE(zJ,targetJ)/RMS(targetJ),0,1)',
       nochange=dict(**initial,**d0),limitations=['Greedy one-step policy, not global optimum over all sequences',
           'Single probe and four batches; no generalization claim','Conditional forecasts omit weight decay and actual BN changes',
           'Same actual update budgets; controller has4 candidate gradient queries per update, controls1',
           'Member permutation does not change symmetric averaged batch loss mathematically; minor floating reductions may differ',
           'No epochs inferred;32 images per update; checkpoint save frequency not learning unit'])
    import sys;sys.path.insert(0,str(ROOT));import src.narzedzia.deklaracja_wynikow as declaration
    declaration._brudne_drzewo=lambda:True;declaration.zapisz_wynik(__file__,'planowanie500' if budget==500 else 'planowanie100',result)
    js(OUT/'RESULT.json',result)
    report=['# Planowanie kolejności czterech paczek SD','',
      'Poprzedni eksperyment powtarzał niezależne paczki i ekstrapolował siłę początkową; nie szukał sekwencji. Tutaj po każdej rzeczywistej aktualizacji ponownie liczymy siły czterech kandydatów i wybieramy najmniejsze przewidywane MSE J128. To greedy kontrola z horyzontem1. Bez CAV i podmiany latentu. Nie deklarujemy globalnego optimum sekwencji.','',
      'Oryginalny autentyczny checkpoint k1000/AdamW i ta sama sonda29. Prawdziwe GT tych samych czterech paczek32, naturalny odczyt sondy. Cel after jest sztuczny, odległość do niego oceniana w J128; inne640 i detektor osobno. Pęd zachowany po każdym kroku, przy przejściu100->500 również.','',
      '| Polityka | najlepszy krok | redukcja MSE128 w najlepszym stanie | MSE128 najlepsze | redukcja końcowa | IoU w najlepszym stanie |','|---|---:|---:|---:|---:|---:|']
    for p,a in results.items():
        b=a['best'];report.append(f"| {p} | {b['step']} | {100*b['metrics']['gain128']:.5f}% | {b['metrics']['mse128']:.6f} | {100*a['final']['gain128']:.5f}% | {b['detector']['iou']:.4f} |")
    report+=['',f'Budżet: {budget} aktualizacji i {32*budget} obrazów na politykę, razem{3*budget} aktualizacji. Etap100 zachowany w RESULT_100.json. Liczba checkpointów to częstość zapisu, nie liczba jednostek uczenia. Brak nazw epok pełnego datasetu.','',
       'Progi redukcji MSE128:5/10/25 procent. Zgodność wartości85 procent jest osobnym progiem: clip(1-RMSE/RMS celu,0,1), a nie cosine, AP, skill prognozy lub procent redukcji MSE. null w wynikach oznacza brak osiągnięcia w skończonym budżecie; nie ekstrapolujemy czasu do celu poza budżet.','',
       'Kolejność pomiędzy paczkami zmienia pamięć AdamW i ścieżkę wag. Permutacja członków jednej paczki nie zmienia matematycznie uśrednionej symetrycznej straty ani statystyk BN bez stochastyki i przetwarzania zależnego od kolejności. Drobne różnice redukcji zmiennoprzecinkowych nie uzasadniają szukania32! permutacji.','',
       'Wybory i metryki: SCHEDULE.csv. Przed każdym wykonanym krokiem decyzja jest sealed w decisions/ oraz FORECAST_SEALS.jsonl. Rzeczywiste TRAJECTORY.npy, ACTUAL.json, siły pełnej siatki na zadeklarowanych krokach, checkpointy z pamięcią optymalizatora i naturalna weryfikacja najlepszego checkpointu są zachowane.','',
       f'Koszt obliczeń: {json.dumps(COST)}; {time.time()-START:.1f} sekund. Wybór kontrolowany wymaga4 gradientów kandydatów na krok, losowy/cykl1. Każda polityka ma ten sam budżet realnych aktualizacji, nie ten sam koszt prognoz. Ścisły nochange ma0 aktualizacji. Żadnych przyszłych gradientów w zapowiedzi decyzji.','',
       'Najlepszy zaobserwowany stan to osiągnięty wynik tej polityki, nie dowód osiągalnego globalnego minimum. Głowa detektora jest uczona, dlatego kierunek latentu i IoU/pewność nie są utożsamiane.']
    text='\n'.join(report)+'\n';(OUT/'REPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI.md').write_text(text,encoding='utf-8')
    for name in ('worker.py','model.py','rownanie_original.py','dane.py','PLAN.md','README.md'):shutil.copy2(HERE/name,OUT/f'SOURCE_{name}')
    js(OUT/'HASHES.json',{str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('HASHES.json','stdout.log','stderr.log','JOB.json','PID.txt','PROGRESS.json','RUN_ACTIVE.json') and not p.name.endswith('.tmp')})
    js(OUT/'PROGRESS.json',dict(status='complete',phase='complete',pid=os.getpid(),cost=COST,seconds=time.time()-START))
    print('COMPLETE',json.dumps({p:a['best'] for p,a in results.items()}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=500);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:main(args)
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);js(OUT/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()))
        js(OUT/'PROGRESS.json',dict(status='failed',phase='failed',error=str(e),pid=os.getpid()));raise
