"""Czy cztery paczki32 kierują oryginalną sondę29 ku sztucznemu celowi SD?
Brak zysku ponad nochange/random32 obala korzystny kierunek tych paczek.
Prognoza origin0 jest warunkowa na znanej początkowej paczce i stanie AdamW;
stała siła do100 to założenie, nie znajomość przyszłych gradientów. Transfer
równania ze średnich tokenów narzędzia do600x768 wymaga walidacji kroku1.
"""
import copy, csv, gc, hashlib, io, json, os, random, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs'/HERE.name
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
OLD=ROOT/'outputs/vit_langevin_paczka32_cel_sztuczny'
RETR=ROOT/'outputs/vit_sd_cztery_obrazy_paczki32'
SEEDS=[2026100511,2026100523,2026100537,2026100549]
ARMS=[str(x) for x in SEEDS]+['random32']
HORIZONS=[1,5,10,25,50,100]
THRESHOLDS=[.05,.10,.25,.85,.90]
START=time.time()
COST=dict(initial_gradient_batches=0,functional_probe_forwards_total=0,actual_updates=0,actual_training_images=0,actual_probe_forwards=0)


def js(path,obj):
    # Windows readers occasionally retain handles; retry replacing, never truncate target.
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(30):
        try:
            temp.replace(path);return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError(f'Atomic replace repeatedly refused: {path}')


def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def progress(phase,**kw):
    obj=dict(status='running',phase=phase,pid=os.getpid(),seconds=time.time()-START,cost=dict(COST),**kw)
    js(OUT/'PROGRESS.json',obj)
    print(phase,kw,flush=True)


def prepare():
    import cv2,numpy as np
    cv2.setNumThreads(1)
    cachepath=OUT/'PREPARATION_CACHE.json'
    if cachepath.exists():
        cached=json.loads(cachepath.read_text(encoding='utf-8'))
        assert all(sha(Path(k))==v for k,v in cached['hashes'].items())
        progress('reusing_hash_verified_GT_CPU')
        images={arm:np.load(OUT/f'IMAGES_{arm}.npy') for arm in ARMS}
        boxes={arm:[r['boxes_xyxy_480x320'] for r in json.loads((OUT/f'GROUNDTRUTH_{arm}.json').read_text(encoding='utf-8'))] for arm in ARMS}
        assert all(len(images[a])==len(boxes[a])==32 for a in ARMS)
        return images,boxes,json.loads((SOURCE/'SELECTION.json').read_text(encoding='utf-8'))
    progress('preparing_exact_manifests_and_GT_CPU')
    result=json.loads((RETR/'RESULT.json').read_text(encoding='utf-8-sig'))
    assert result['status']=='complete' and result['unique_selected_pixels']==128
    membership=json.loads((RETR/'TRAIN_MEMBERSHIP.json').read_text(encoding='utf-8-sig'))
    entries=membership['archives'];valid={e['path']:set(e['members']) for e in entries}
    selections={}
    for seed in SEEDS:
        selections[str(seed)]=json.loads((RETR/str(seed)/'TOP32.json').read_text(encoding='utf-8-sig'))['selected']
        assert len(selections[str(seed)])==32
    pool=[dict(archive=e['path'],member=n,film=e['film'],split='train') for e in entries for n in e['members']]
    assert len(pool)==310257
    selections['random32']=random.Random(20261005).sample(pool,32)
    js(OUT/'SELECTION.json',dict(arms=selections,selected_retrieval_count=result['ranking_denominator'],
        selected_retrieval_partial=True,random_pool_count=len(pool),random_seed=20261005,
        random_method='uniform without replacement over full known train membership; fixed before forces',
        manifest_hashes={str(s):sha(RETR/str(s)/'TOP32.json') for s in SEEDS},membership_sha256=sha(RETR/'TRAIN_MEMBERSHIP.json')))
    folder=OUT/'input';folder.mkdir(exist_ok=True)
    cache={};audit={};images={};boxes={}
    global_hashes=set()
    for arm,rows in selections.items():
        aa=[];ims=[];bb=[];pixels=set()
        for index,r in enumerate(rows):
            assert r['split']=='train' and r['member'] in valid[r['archive']]
            archive=r['archive']
            if archive not in cache:
                with zipfile.ZipFile(archive) as z:
                    member=next(n for n in z.namelist() if n.endswith('/annotation.csv'))
                    rawann=z.read(member)
                (folder/f"{r['film']}_annotation.csv").write_bytes(rawann)
                cache[archive]={x['filename']:x for x in csv.DictReader(io.StringIO(rawann.decode('utf-8-sig')))}
                audit[archive]=dict(annotation_member=member,sha256=hashlib.sha256(rawann).hexdigest())
            with zipfile.ZipFile(archive) as z:
                raw=z.read(r['member']);crc=z.getinfo(r['member']).CRC
            if arm!='random32':
                assert hashlib.sha256(raw).hexdigest()==r['original_sha256']
                assert Path(r['original_photo']).read_bytes()==raw
            native=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            assert native is not None
            rgb=cv2.cvtColor(native,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
            digest=hashlib.sha256();digest.update(str(rgb.shape).encode('ascii'));digest.update(rgb.tobytes());ph=digest.hexdigest()
            if arm!='random32':assert ph==r['pixel_sha256']
            assert ph not in pixels
            pixels.add(ph)
            if arm!='random32':global_hashes.add(ph)
            ann=cache[archive][Path(r['member']).name]
            rawboxes=[];scaled=[]
            for k,v in ann.items():
                if not k.endswith('_bbox') or k in ('cornea_bbox','pupil_bbox') or v in ('','None','null','nan'):continue
                b=json.loads(v)
                if b is None or len(b)!=4 or b[2]<=0 or b[3]<=0:continue
                x,y,bw,bh=b;rawboxes.append(dict(column=k,xywh=b))
                scaled.append([x*480/w,y*320/h,(x+bw)*480/w,(y+bh)*320/h])
            ims.append(cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA));bb.append(scaled)
            aa.append(dict(**r,pixel_verified_sha256=ph,zip_crc=crc,native_shape=[h,w],scale_xy=[480/w,320/h],annotation=ann,raw_boxes=rawboxes,boxes_xyxy_480x320=scaled))
        images[arm]=np.stack(ims);boxes[arm]=bb
        np.save(OUT/f'IMAGES_{arm}.npy',images[arm]);js(OUT/f'GROUNDTRUTH_{arm}.json',aa)
        progress('prepared_GT',arm=arm,images=32,positive_images=sum(bool(b) for b in bb),negative_images=sum(not b for b in bb))
    assert len(global_hashes)==128
    js(OUT/'ARCHIVE_AUDIT.json',audit)
    selection=json.loads((SOURCE/'SELECTION.json').read_text(encoding='utf-8-sig'))
    assert selection['source_index']==29 and selection['record']['frame_id'].endswith('/00001926')
    shutil.copy2(SOURCE/'ZDJECIE_ZRODLOWE.jpg',folder/'SONDA_ORIGINAL.jpg')
    assert sha(folder/'SONDA_ORIGINAL.jpg')==selection['record']['sha256']
    return images,boxes,selection


def wait_gpu():
    start=time.monotonic()
    while True:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        free=int(r.stdout.strip().splitlines()[0]) if r.returncode==0 else 0
        if free>=2816:return free
        progress('queued_GPU_preserving_other_jobs',free_mib=free,required_free_mib=2816)
        if time.monotonic()-start>5400:raise TimeoutError('No room for full32 BN protocol for90minutes')
        time.sleep(10)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Existing completed result; no overwrite')
    (OUT/'PID.txt').write_text(str(os.getpid()))
    import cv2,numpy as np,torch
    import model as M
    import rownanie_original as RW
    torch.set_num_threads(4);torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    imgs,boxes,selection=prepare()
    progress('verify_original_SD_checkpoint_and_AdamW_CPU')
    prior=json.loads((OLD/'OPTIMIZER_PROVENANCE.json').read_text(encoding='utf-8-sig'))
    cp=OLD/'MODEL_AUTHENTIC_K1000.pt'
    assert sha(cp)==prior['sha256'] and prior['remote_hash_verified']
    shutil.copy2(cp,OUT/'MODEL_AUTHENTIC_K1000.pt')
    st=torch.load(cp,map_location='cpu',weights_only=True)
    reference=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)
    assert 'opt' in st and st['krok']==1000
    assert st['net'].keys()==reference['net'].keys()
    assert all(torch.equal(t,reference['net'][n]) for n,t in st['net'].items())
    js(OUT/'CHECKPOINT_VALIDATION.json',dict(original_SD_source_checkpoint=str(SOURCE/'MODEL_VIT_K1000.pt'),
        reference_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt'),authentic_checkpoint_sha256=sha(cp),
        original_optimizer_provenance=prior,step=1000,identical_all_model_tensors=True,
        optimizer_groups=st['opt']['param_groups'],optimizer_states=len(st['opt']['state']),
        new_single_image_Lorentz_checkpoint_used=False))
    coeffpath=ROOT/'experiments/maly_vit_tozsamosc_jeden_krok/wyniki/TOZSAMOSC_JEDEN_KROK.json'
    coefficients=json.loads(coeffpath.read_text(encoding='utf-8-sig'))['wspolczynniki_s47']['paczka + pęd']
    assert coefficients==json.loads((OLD/'COEFFICIENTS.json').read_text(encoding='utf-8-sig'))['coefficients']
    js(OUT/'COEFFICIENTS.json',dict(original=dict(nowa=1,ped=1),historical=coefficients,source=str(coeffpath),
        sha256=sha(coeffpath),equation_code_sha256=sha(HERE/'rownanie_original.py'),
        transfer='coefficients fitted to pooled tool tokens;600x768 extension tested at first actual step'))
    with np.load(SOURCE/'LATENT_PELNY.npz') as f:
        target=f['after'].astype(np.float64);before=f['before'].copy();J=f['channels0'].copy()
    assert target.shape==before.shape==(600,768) and len(J)==128
    im=cv2.imread(str(OUT/'input/SONDA_ORIGINAL.jpg'))
    rgb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    np.save(OUT/'SONDA_RGB_480x320.npy',rgb)
    x,y,w,h=selection['record']['bbox_xywh'];gt=np.array([x,y,x+w,y+h],np.float64)
    js(OUT/'SONDA_PROVENANCE.json',dict(selection=selection,original_image_sha256=sha(OUT/'input/SONDA_ORIGINAL.jpg'),
        resized_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),gt_bbox_xyxy=gt.tolist(),
        target_sha256=sha(SOURCE/'LATENT_PELNY.npz'),preprocess='RGB480x320 INTER_AREA; identical original probe29'))
    wait_gpu();progress('verified_weights_start_probe_and_forces_GPU')
    probe=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255

    def make():
        net=M.MalyViT().cuda();net.load_state_dict(st['net']);net.zapisuj_latent(True)
        opt=torch.optim.AdamW([dict(params=[p for p in net.parameters() if p.ndim>=2],weight_decay=.05),dict(params=[p for p in net.parameters() if p.ndim<2],weight_decay=0)],lr=5e-4,betas=(.9,.999))
        opt.load_state_dict(copy.deepcopy(st['opt']))
        return SimpleNamespace(net=net,opt=opt,dev='cuda',_lr=lambda:5e-4)

    def detector(output):
        b,s=M.dekoduj(output)
        bb=b[0].cpu().numpy().astype(np.float64);scores=s[0].cpu().numpy()
        lo=np.maximum(bb[:,:2],gt[:2]);hi=np.minimum(bb[:,2:],gt[2:])
        inter=np.prod(np.maximum(hi-lo,0),axis=1)
        area=np.prod(np.maximum(bb[:,2:]-bb[:,:2],0),axis=1)
        iou=inter/np.maximum(area+np.prod(gt[2:]-gt[:2])-inter,1e-30)
        return dict(confidence=float(scores[0]),iou=float(iou[0]),max_iou_top10=float(iou.max()),bbox_xyxy=bb[0].tolist())

    def read_probe(u,delta=None,buffers=None):
        u.net.eval();p=dict(u.net.named_parameters());b=dict(u.net.named_buffers()) if buffers is None else buffers
        if delta is not None:p={n:v+delta[n] for n,v in p.items()}
        with torch.no_grad():
            output,z=torch.func.functional_call(u.net,(p,b),(probe,))
            natural=z[0].float().cpu().numpy().copy();d=detector(output)
        u.net._latent=None
        COST['functional_probe_forwards_total']+=1
        return natural,d

    u=make();z0,det0=read_probe(u);np.testing.assert_allclose(z0,before,rtol=1e-4,atol=3e-5)
    js(OUT/'BASELINE_VALIDATION.json',dict(status='passed',latent_before_max_abs_error=float(np.max(np.abs(z0-before))),
        natural_detector=det0,reference_detector=json.loads((SOURCE/'RESULT.json').read_text(encoding='utf-8-sig'))['baseline'],
        probe_RGB_shape=list(rgb.shape),initial_step=1000))
    np.savez_compressed(OUT/'BASELINE.npz',before=z0,target=target,channels0=J)
    initial={}

    def metrics(z):
        z=np.asarray(z,np.float64);out={}
        for name,ix in [('768',np.arange(768)),('128',J)]:
            a=z[:,ix];b=target[:,ix];d=a-z0[:,ix];desired=b-z0[:,ix]
            dot=float(np.sum(d*desired));norm=float(np.linalg.norm(d)*np.linalg.norm(desired))
            out['mse'+name]=float(np.mean((a-b)**2))
            out['cosine'+name]=float(np.sum(a*b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-30))
            out['alignment_dot'+name]=dot;out['alignment_cosine'+name]=dot/max(norm,1e-30)
            out['distance_gain'+name]=1-out['mse'+name]/initial.get('mse'+name,out['mse'+name])
            active=np.abs(desired)>1e-10
            out['coordinate_toward_fraction'+name]=float(np.mean((d*desired)[active]>0))
            out['norm_delta'+name]=float(np.linalg.norm(d))
        return out

    initial=metrics(z0);js(OUT/'INITIAL_METRICS.json',dict(**initial,**det0))
    def batch(arm):
        x=torch.from_numpy(imgs[arm]).permute(0,3,1,2).float().cuda()/255
        c=tuple(t.cuda() for t in M.cele([np.array(b,np.float32).reshape(-1,4) for b in boxes[arm]],20,30))
        return x,c
    forecasts={};forces={};forecast_rows=[];virtual_det={};force_summaries={}
    arms=ARMS[:1] if '--smoke' in sys.argv else ARMS
    for arm in arms:
        progress('initial_conditional_forces',arm=arm,forecast_origin=0)
        x,c=batch(arm);torch.cuda.reset_peak_memory_stats()
        params0={n:p.detach().clone() for n,p in u.net.named_parameters()}
        buffers0={n:b.detach().clone() for n,b in u.net.named_buffers()}
        g=RW.gradient_tensorow(u,x,*c);COST['initial_gradient_batches']+=1;u.net._latent=None
        new,mom=RW.czesci_kroku(u,g)
        wd={}
        for group in u.opt.param_groups:
            for p in group['params']:
                n=next(n for n,v in u.net.named_parameters() if v is p)
                wd[n]=-5e-4*group['weight_decay']*p.detach()
        zn,dn=read_probe(u,new);zm,dm=read_probe(u,mom);zw,dw=read_probe(u,wd)
        combined={n:new[n]+mom[n]+wd[n] for n in new}
        exact,dc=read_probe(u,combined)
        sn=zn-z0;sm=zm-z0;sw=zw-z0
        f=sn+sm;ff=coefficients['nowa']*sn+coefficients['ped']*sm
        forces[arm]=f
        force_summaries[arm]=dict(forecast_origin=0,predicted_step1=metrics(z0+f),historically_fitted_step1=metrics(z0+ff),combined_parameter_step_detector=dc)
        js(OUT/'INITIAL_FORCE_SUMMARY.json',dict(initial=initial,arms=force_summaries,prospective_only=True))
        progress('first_step_force_saved',arm=arm,predicted_step1=force_summaries[arm]['predicted_step1'])
        virtual_det[arm]=dict(new_only=dn,momentum_only=dm,weight_decay_only=dw,combined_current_parameter_step=dc,
                             future_extrapolated_latent_detector=None,reason='extrapolated latent alone does not determine full detector head state')
        np.savez_compressed(OUT/f'FORCE_{arm}.npz',nowa=sn,ped=sm,weight_decay=sw,original_force=f,fitted_force=ff,
                            exact_current_parameter_step=exact-z0,before=z0,target=target,channels0=J)
        assert all(torch.equal(params0[n],p) for n,p in u.net.named_parameters())
        assert all(torch.equal(buffers0[n],b) for n,b in u.net.named_buffers())
        assert all(p.grad is None for p in u.net.parameters())
        forecasts[arm]={}
        for method,force in [('original',f),('historically_fitted',ff)]:
            trajectory=[]
            for h in range(101):
                m=metrics(z0+h*force);trajectory.append(dict(step=h,**m))
                forecast_rows.append(dict(arm=arm,method=method,forecast_origin=0,update=h,**m))
            hits={dim:{str(r):next((m['step'] for m in trajectory if m['distance_gain'+dim]>=r),None) for r in THRESHOLDS} for dim in ('128','768')}
            forecasts[arm][method]=dict(metrics=trajectory,hitting_time_updates=hits,horizon=100,origin=0,
                assumption='constant initial latent force repeated; fixed known batch; no future gradients or observed future latents',
                no_hit_semantics='null means no crossing by100; no extrapolation beyond100',
                detector_forecast=None)
        if '--smoke' in sys.argv:
            js(OUT/'SMOKE.json',dict(status='passed',arm=arm,checkpoint_match=True,before_match=True,
                parameters_and_BN_unchanged=True,batch_size=32,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,required_free_mib=2816,baseline_detector=det0))
        del x,c,g,new,mom,wd,combined,params0,buffers0
        gc.collect();torch.cuda.empty_cache()
    del u;gc.collect();torch.cuda.empty_cache()
    if '--smoke' in sys.argv:
        progress('smoke_passed');return
    for h in range(101):forecast_rows.append(dict(arm='nochange',method='nochange',forecast_origin=0,update=h,**initial))
    with open(OUT/'FORECAST.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(forecast_rows[0]));w.writeheader();w.writerows(forecast_rows)
    js(OUT/'FORECAST.json',dict(initial=initial,arms=forecasts,thresholds=THRESHOLDS,
        forecast_saved_before_actual_training=True,information='original weights/optimizer, current candidate GT atorigin0 only',
        epoch_definition='one update repeats same32 photos;100updates not full-dataset epochs',virtual_probe_detector=virtual_det))
    forecast_sha=sha(OUT/'FORECAST.json');js(OUT/'FORECAST_SEAL.json',dict(sha256=forecast_sha,saved_before_first_update=True,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    progress('forecasts_sealed_before_actual_training',forecast_sha256=forecast_sha)
    actual={};validation={};comparison=[]
    for arm in ARMS:
        gc.collect();torch.cuda.empty_cache();wait_gpu()
        u=make();x,c=batch(arm);buf0={n:b.detach().clone() for n,b in u.net.named_buffers()}
        series=[dict(step=0,**initial,**det0)]
        latents=np.lib.format.open_memmap(OUT/f'TRAJECTORY_{arm}.npy',mode='w+',dtype=np.float32,shape=(101,600,768));latents[0]=z0
        for step in range(1,101):
            u.net.train();u.opt.zero_grad(set_to_none=True)
            for group in u.opt.param_groups:group['lr']=5e-4
            with torch.autocast('cuda',dtype=torch.bfloat16):o,_=u.net(x)
            loss,_=M.strata(o,*c);loss.backward();norm=torch.nn.utils.clip_grad_norm_(u.net.parameters(),1.0);u.opt.step()
            COST['actual_updates']+=1;COST['actual_training_images']+=32
            z,d=read_probe(u);COST['actual_probe_forwards']+=1
            m=metrics(z);series.append(dict(step=step,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),**m,**d));latents[step]=z
            if step==1:
                fixed,_=read_probe(u,buffers=buf0);pred=z0+forces[arm]
                def validate(truth):
                    answer={}
                    for dim,ix in [('768',np.arange(768)),('128',J)]:
                        actual_delta=(truth-z0)[:,ix].astype(np.float64);pd=forces[arm][:,ix].astype(np.float64)
                        ss=float(np.sum(actual_delta**2));sse=float(np.sum((actual_delta-pd)**2))
                        active=np.abs(actual_delta)>1e-10
                        answer['skill_percent'+dim]=100*(1-sse/max(ss,1e-30))
                        answer['coordinate_sign_agreement'+dim]=float(np.mean(np.sign(actual_delta[active])==np.sign(pd[active])))
                        answer['delta_cosine'+dim]=float(np.sum(actual_delta*pd)/max(float(np.linalg.norm(actual_delta)*np.linalg.norm(pd)),1e-30))
                    return answer
                validation[arm]=dict(actual_BN=validate(z),fixed_initial_BN=validate(fixed),predicted=metrics(pred),actual=m,
                    exact_virtual_first_step_mse=float(np.mean((fixed-(z0+np.load(OUT/f'FORCE_{arm}.npz')['exact_current_parameter_step']))**2)))
                js(OUT/'FIRST_STEP_VALIDATION.json',validation)
            if step in HORIZONS:
                for method in ('original','historically_fitted'):
                    p=forecasts[arm][method]['metrics'][step]
                    row=dict(arm=arm,method=method,forecast_origin=0,update=step,
                             actual_confidence=d['confidence'],actual_iou=d['iou'])
                    for dim,ix in [('768',np.arange(768)),('128',J)]:
                        predicted=z0+step*(forces[arm] if method=='original' else np.load(OUT/f'FORCE_{arm}.npz')['fitted_force'])
                        pd=(predicted-z0)[:,ix];ad=(z-z0)[:,ix]
                        row.update({f'predicted_mse{dim}':p['mse'+dim],f'actual_mse{dim}':m['mse'+dim],
                            f'predicted_gain{dim}':p['distance_gain'+dim],f'actual_gain{dim}':m['distance_gain'+dim],
                            f'forecast_latent_error_mse{dim}':float(np.mean((predicted[:,ix]-z[:,ix])**2)),
                            f'coordinate_sign_agreement{dim}':float(np.mean(np.sign(pd)==np.sign(ad)))})
                    comparison.append(row)
                np.savez_compressed(OUT/f'LATENT_{arm}_k{1000+step}.npz',natural=z,target=target,channels0=J,detector_metrics=np.array([d['confidence'],d['iou']]))
            js(OUT/f'ACTUAL_{arm}.json',series)
            if step==1 or step%5==0:
                latents.flush();progress('actual_training',arm=arm,update=step,total_updates=100,mse128=m['mse128'],mse768=m['mse768'],confidence=d['confidence'],iou=d['iou'])
            del o,loss
        latents.flush();del latents
        head_delta=float(sum((p.detach().cpu()-st['net'][n]).square().sum() for n,p in u.net.named_parameters() if n.startswith('glowa.')))
        assert head_delta>0
        actual[arm]=dict(metrics=series,hitting_time_updates={dim:{str(r):next((m['step'] for m in series if m['distance_gain'+dim]>=r),None) for r in THRESHOLDS} for dim in ('128','768')},head_weight_delta_squared=head_delta,
            detector_threshold_observed={name:{str(t):next((m['step'] for m in series if m[name]>=t),None) for t in (.85,.9)} for name in ('confidence','iou')})
        torch.save(dict(net=u.net.state_dict(),opt=u.opt.state_dict(),krok=1100,arm=arm),OUT/f'MODEL_{arm}_k1100.pt')
        del u,x,c,buf0;gc.collect();torch.cuda.empty_cache()
    actual['nochange']=dict(metrics=[dict(step=h,**initial,**det0) for h in range(101)],hitting_time_updates={dim:{str(r):None for r in THRESHOLDS} for dim in ('128','768')},actual_updates=0)
    with open(OUT/'FORECAST_VS_ACTUAL.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    with open(OUT/'ACTUAL_ALL.csv','w',newline='',encoding='utf-8') as f:
        rows=[dict(arm=arm,**r) for arm,data in actual.items() for r in data['metrics']]
        w=csv.DictWriter(f,fieldnames=list(rows[1]),extrasaction='ignore');w.writeheader();w.writerows(rows)
    assert sha(OUT/'FORECAST.json')==forecast_sha
    assert all(torch.equal(st['net'][n],reference['net'][n]) for n in reference['net'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for dim,ax in zip(('128','768'),axs):
        for arm in ARMS:
            line=ax.plot(range(101),[m['mse'+dim] for m in actual[arm]['metrics']],label=arm)[0]
            ax.plot(range(101),[m['mse'+dim] for m in forecasts[arm]['original']['metrics']],ls='--',color=line.get_color(),alpha=.65)
        ax.axhline(initial['mse'+dim],color='black',ls=':',label='nochange')
        ax.set(title=f'MSE{dim}: solid actual, dashed initial-force forecast',xlabel='Updates: repeat fixed32',ylabel='Raw squared distance / coordinate');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'DISTANCE_FORECAST_ACTUAL.png',dpi=160);plt.close(fig)
    result=dict(status='complete',ziarno=20261005,initial=dict(**initial,**det0),forecast=forecasts,actual=actual,
        first_step_validation=validation,original_SD_source_checkpoint=True,authentic_optimizer=True,
        forecast_origin=0,horizons=HORIZONS,actual_updates_each=100,actual_updates_total=500,cost=COST,
        seconds=time.time()-START,epoch_definition='100 repetitions of fixed32, not full-dataset epochs',
        latent_intervention=False,all_weights_trainable=True,BN='normal full32 train; virtual gradients use copied buffers',
        prior_context='previous different selected32: MSE76814.3202 ->14.3605 at50; not assumed for new arms',
        limitations=['One original probe; no generalization inference','Historical pooled-tool fit transferred to600x768','Fixed initial-force forecast is conditional, no future-gradient information','Latent direction is not detection quality; detector head changes','Selected128 from deterministic partial retrieval prefix, random32 from full train'])
    sys.path.insert(0,str(ROOT));import src.narzedzia.deklaracja_wynikow as declaration
    declaration._brudne_drzewo=lambda:True
    declaration.zapisz_wynik(__file__,'cztery_paczki100',result)
    js(OUT/'RESULT.json',result)
    lines=['# Cztery paczki SD: prognoza i100 aktualizacji','',
        'Punkt wyjścia to oryginalny checkpoint SD k1000 i jego autentyczny stan AdamW. Sonda29: niezmienione zdjęcie TR_0154/1926. Cel: after z oryginalnego LATENT_PELNY, nie naturalny latent obrazów SD. Wszystkie32 zdjęcia każdego ramienia mają prawdziwe GT z ZIP.','',
        'Prognozy zapisane i zahashowane przed aktualizacjami. Główna suma S_nowa+S_ped, dodatkowo historyczne współczynniki ze źródła. Transfer do600x768 oceniony na kroku1. Horyzont1/5/10/25/50/100: stała siła początkowa, brak przyszłych gradientów. Jeden krok powtarza paczkę32;100 nie oznacza100 epok pełnego zbioru.','',
        '| Ramię | prognoza gain128 na100 | rzeczywisty gain128 | prognoza gain768 | rzeczywisty gain768 | IoU sondy | pewność sondy |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=actual[arm]['metrics'][-1];p=forecasts[arm]['original']['metrics'][-1]
        lines.append(f"| {arm} | {100*p['distance_gain128']:.4f}% | {100*a['distance_gain128']:.4f}% | {100*p['distance_gain768']:.4f}% | {100*a['distance_gain768']:.4f}% | {a['iou']:.4f} | {a['confidence']:.4f} |")
    lines+=['','Gain dodatni oznacza redukcję MSE, ujemny pogorszenie. Nochange gain=0. Wyniki ujemne są zachowane. Progi5/10/25/85/90 procent redukcji błędu: null oznacza brak przekroczenia do100, bez wymyślonych czasów ani epok. Obserwowane progi IoU/pewności są oddzielne od prognozy latentu.','',
        'Pełne wyniki progów i kierunku: RESULT.json, FORECAST_VS_ACTUAL.csv. Dokładne trajektorie naturalnych latentów: TRAJECTORY_*.npy; checkpointy mają również optymalizator. Głowa detektora zmieniała wagi w każdym ramieniu. Wirtualny krok na parametrach daje naturalną odpowiedź sondy tylko dla kroku1; ekstrapolowany latent nie wyznacza przyszłej głowy, więc dalszej pewności z niego nie wymyślamy.','',
        f"Koszt: {json.dumps(COST)}; {time.time()-START:.1f} sekund.500 rzeczywistych aktualizacji po32 zdjęcia. Kontrola nochange ma0 aktualizacji. Żadne cudze zadanie nie zostało zatrzymane.",'',
        'Poprzednia, inna paczka32 pogorszyła MSE768 z14.3202 do14.3605 po50. To kontekst, nie założenie obecnego wyniku. Test jednej sondy nie dowodzi generalizacji ani globalnej nieosiągalności celu po dalszym uczeniu. Brak augmentacji zadeklarowano; zachowano zwykły BN pełnej paczki32.']
    report='\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf-8');(HERE/'WYNIKI.md').write_text(report,encoding='utf-8')
    for name in ('worker.py','model.py','rownanie_original.py','PLAN.md','README.md'):
        shutil.copy2(HERE/name,OUT/f'SOURCE_{name}')
    js(OUT/'HASHES.json',{str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('HASHES.json','stdout.log','stderr.log','smoke_stdout.log','smoke_stderr.log','PROGRESS.json','JOB.json','PID.txt')})
    progress('complete');js(OUT/'PROGRESS.json',dict(status='complete',phase='complete',pid=os.getpid(),seconds=time.time()-START,cost=COST))


if __name__=='__main__':
    try:run()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        js(OUT/'FAILURE.json',dict(status='failed',error=str(e),traceback=traceback.format_exc()))
        js(OUT/'PROGRESS.json',dict(status='failed',error=str(e),pid=os.getpid()))
        raise
