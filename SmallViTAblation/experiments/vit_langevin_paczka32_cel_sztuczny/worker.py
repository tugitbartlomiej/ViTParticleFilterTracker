"""Sprawdza siłę wybranej paczki32 wobec celu tej samej sondy; obala ją brak poprawy ponad kontrolę.
Prognoza powstaje przed treningiem, oryginalne pchnięcia AdamW, bez podmiany latentu.
"""
import csv, hashlib, heapq, io, json, os, random, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs'/HERE.name
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
RETR=ROOT/'outputs/vit_sd_obraz_najblizszy'
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pelny/k1000.pt'
SEED=20261005
START=time.time()
def js(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');temp.replace(path)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def progress(status,**kw):
    js(OUT/'PROGRESS.json',dict(status=status,pid=os.getpid(),seconds=time.time()-START,**kw))
    print(status,kw,flush=True)
def prepare():
    if all((OUT/f'GROUNDTRUTH_{a}.json').exists() and (OUT/f'IMAGES_{a}.npy').exists() for a in ('selected32','random32')) and (OUT/'SONDA_PROVENANCE.json').exists():
        import numpy as np
        progress('reusing_verified_preparation_cpu')
        images={a:np.load(OUT/f'IMAGES_{a}.npy') for a in ('selected32','random32')}
        boxes={a:[r['boxes_xyxy_480x320'] for r in json.loads((OUT/f'GROUNDTRUTH_{a}.json').read_text())] for a in images}
        for a in images:assert len(images[a])==len(boxes[a])==32
        return images,boxes
    progress('preparing_selection_cpu')
    assert json.loads((RETR/'RESULT.json').read_text())['status']=='complete'
    heap=[]; reservoir=[];rng=random.Random(SEED);n=0
    with open(RETR/'RANKING_CALY_ZBIOR.csv',encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):
            if r['split']!='train':continue
            n+=1;key=(float(r['mse768']),int(r['ordinal']))
            item=(-key[0],-key[1],r)
            if len(heap)<64:heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
            if len(reservoir)<32:reservoir.append(r)
            else:
                j=rng.randrange(n)
                if j<32:reservoir[j]=r
    seen=set();selected=[]
    for _,_,r in sorted(heap,key=lambda x:(-x[0],-x[1])):
        key=(r['archive'],r['member'])
        if key not in seen:seen.add(key);selected.append(r)
        if len(selected)==32:break
    assert len(selected)==32 and len({(r['archive'],r['member']) for r in reservoir})==32
    assert selected[0]['film']=='TR_0006_S1_P03' and selected[0]['member'].endswith('_001193.jpg')
    js(OUT/'SELECTION.json',dict(selected=selected,random32=reservoir,train_records=n,method='full CSV sorted mse768,ordinal; unique archive/member',seed=SEED,ranking_sha256=sha(RETR/'RANKING_CALY_ZBIOR.csv')))
    progress('fetching_actual_images_groundtruth_cpu',selected_count=32,random_count=32)
    import cv2,numpy as np
    cv2.setNumThreads(1)
    cache={}; audit={};images={};boxes={}
    for arm,rows in [('selected32',selected),('random32',reservoir)]:
        folder=OUT/'input'/arm;folder.mkdir(parents=True,exist_ok=True)
        ims=[];bb=[];aa=[]
        for i,r in enumerate(rows):
            archive=r['archive']
            if archive not in cache:
                with zipfile.ZipFile(archive) as z:
                    member=next(n for n in z.namelist() if n.endswith('/annotation.csv'))
                    raw=z.read(member)
                dest=OUT/'input'/f"{r['film']}_annotation.csv";dest.write_bytes(raw)
                cache[archive]={x['filename']:x for x in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))}
                audit[archive]=dict(annotation_member=member,annotation_sha256=hashlib.sha256(raw).hexdigest())
            with zipfile.ZipFile(archive) as z:
                raw=z.read(r['member']);info=z.getinfo(r['member'])
            dest=folder/f'{i:02d}.jpg';dest.write_bytes(raw)
            native=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);assert native is not None
            h,w=native.shape[:2];ann=cache[archive][Path(r['member']).name]
            rawboxes=[];scaled=[]
            for k,v in ann.items():
                if not k.endswith('_bbox') or k in ('cornea_bbox','pupil_bbox') or v in ('','None','null','nan'):continue
                b=json.loads(v)
                if b is None or len(b)!=4 or b[2]<=0 or b[3]<=0:continue
                x,y,bw,bh=b;rawboxes.append(dict(column=k,xywh=b))
                scaled.append([x*480/w,y*320/h,(x+bw)*480/w,(y+bh)*320/h])
            ims.append(cv2.resize(cv2.cvtColor(native,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA))
            bb.append(scaled)
            aa.append(dict(**r,local_image=str(dest),sha256=sha(dest),zip_crc=info.CRC,native_shape=[h,w],scale_xy=[480/w,320/h],annotation=ann,raw_boxes=rawboxes,boxes_xyxy_480x320=scaled))
        images[arm]=np.stack(ims);boxes[arm]=bb
        np.save(OUT/f'IMAGES_{arm}.npy',images[arm]);js(OUT/f'GROUNDTRUTH_{arm}.json',aa)
    js(OUT/'ARCHIVE_AUDIT.json',audit)
    original=json.loads((SOURCE/'SELECTION.json').read_text())
    assert original['source_index']==29 and original['record']['frame_id'].endswith('/00001926')
    shutil.copy2(SOURCE/'ZDJECIE_ZRODLOWE.jpg',OUT/'input/SONDA_ORIGINAL.jpg')
    assert sha(OUT/'input/SONDA_ORIGINAL.jpg')==original['record']['sha256']
    js(OUT/'SONDA_PROVENANCE.json',dict(selection=original,image_sha256=sha(OUT/'input/SONDA_ORIGINAL.jpg'),target_sha256=sha(SOURCE/'LATENT_PELNY.npz'),model_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt')))
    return images,boxes
def checkpoint():
    progress('fetching_authentic_optimizer_cpu',remote=REMOTE)
    cp=OUT/'MODEL_AUTHENTIC_K1000.pt'
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','eden-cluster',f'sha256sum {REMOTE}'],capture_output=True,text=True,timeout=45)
    (OUT/'SSH_CHECKSUM.log').write_text(r.stdout+r.stderr,encoding='utf8')
    if r.returncode:raise RuntimeError('Authentic optimizer unavailable; see SSH_CHECKSUM.log')
    expected=r.stdout.split()[0]
    batch=OUT/'FETCH_CHECKPOINT.sftp'
    batch.write_text(f'get {REMOTE} "{cp.as_posix()}"\n',encoding='utf8')
    if not cp.exists() or sha(cp)!=expected:
        r=subprocess.run(['sftp','-o','BatchMode=yes','-o','ConnectTimeout=10','-b',str(batch),'eden-cluster'],capture_output=True,text=True,timeout=180)
        (OUT/'SFTP.log').write_text(r.stdout+r.stderr,encoding='utf8')
        if r.returncode:raise RuntimeError('Authentic optimizer download failed; see SFTP.log')
    assert sha(cp)==expected
    js(OUT/'OPTIMIZER_PROVENANCE.json',dict(remote=REMOTE,host='eden-cluster',local=str(cp),sha256=expected,remote_hash_verified=True))
    return cp
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Existing result; refusing overwrite')
    (OUT/'PID.txt').write_text(str(os.getpid()),encoding='utf8')
    js(OUT/'WORKER_IDENTITY.json',dict(pid=os.getpid(),python=sys.executable,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),family=HERE.name))
    imgs,boxes=prepare();cp=checkpoint()
    progress('loading_and_verifying_authentic_checkpoint_cpu')
    import cv2,numpy as np,torch
    import model as M
    import rownanie_original as RW
    from types import SimpleNamespace
    torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    st=torch.load(cp,map_location='cpu',weights_only=True)
    reference=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)
    assert 'opt' in st and st['krok']==1000
    assert st['net'].keys()==reference['net'].keys()
    assert all(torch.equal(t,reference['net'][n]) for n,t in st['net'].items()),'Wrong checkpoint identity'
    js(OUT/'CHECKPOINT_VALIDATION.json',dict(keys=list(st),step=st['krok'],identical_model_tensors=True,optimizer_states=len(st['opt']['state']),optimizer_groups=st['opt']['param_groups']))
    coeffpath=ROOT/'experiments/maly_vit_tozsamosc_jeden_krok/wyniki/TOZSAMOSC_JEDEN_KROK.json'
    coeff=json.loads(coeffpath.read_text())['wspolczynniki_s47']['paczka + pęd']
    js(OUT/'COEFFICIENTS.json',dict(coefficients=coeff,source=str(coeffpath),sha256=sha(coeffpath),original_unfitted=dict(nowa=1,ped=1),fit_transfer='Historical mean tool tokens; full-token grid extension requires validation'))
    sd=ROOT/'outputs/vit_sd_cztery_szumy'
    smoke=json.loads((OUT/'GPU_MEMORY_SMOKE.json').read_text()) if (OUT/'GPU_MEMORY_SMOKE.json').exists() else {}
    threshold=smoke['recommended_threshold_mib'] if smoke.get('status')=='passed' else 6500
    js(OUT/'GPU_QUEUE_THRESHOLD.json',dict(required_free_mib=threshold,basis=smoke,keeps_full32_and_original_BN=True))
    while True:
        sp=json.loads((sd/'PROGRESS.json').read_text()) if (sd/'PROGRESS.json').exists() else {}
        complete=sp.get('status')=='complete' or (sd/'RESULT.json').exists()
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
        free=int(r.stdout.strip().splitlines()[0]) if r.returncode==0 else 0
        if complete and free>=threshold:break
        progress('queued_gpu',sd_status=sp.get('status','not_started'),free_mib=free,required_free_mib=threshold,prepared_images=64,authentic_optimizer_verified=True)
        time.sleep(20)
    dev='cuda';progress('computing_current_batch_forces',device=dev)
    zfile=np.load(SOURCE/'LATENT_PELNY.npz');target=zfile['after'].astype(np.float64);before=zfile['before'];J=zfile['channels0']
    assert target.shape==(600,768) and len(J)==128
    im=cv2.imread(str(OUT/'input/SONDA_ORIGINAL.jpg'));rgb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    probe=torch.from_numpy(rgb).permute(2,0,1)[None].float().to(dev)/255
    def make():
        net=M.MalyViT().to(dev);net.load_state_dict(st['net']);net.zapisuj_latent(True)
        opt=torch.optim.AdamW([dict(params=[p for p in net.parameters() if p.ndim>=2],weight_decay=.05),dict(params=[p for p in net.parameters() if p.ndim<2],weight_decay=0)],lr=5e-4,betas=(.9,.999))
        opt.load_state_dict(st['opt'])
        return SimpleNamespace(net=net,opt=opt,dev=dev,_lr=lambda:5e-4)
    def latent(u,delta=None,buffers=None):
        u.net.eval()
        p=dict(u.net.named_parameters());b=dict(u.net.named_buffers()) if buffers is None else buffers
        if delta is not None:p={n:v+delta[n] for n,v in p.items()}
        with torch.no_grad():torch.func.functional_call(u.net,(p,b),(probe,));return u.net._latent[0].float().cpu().numpy().copy()
    def metrics(z):
        z=np.asarray(z,np.float64);out={}
        for name,ix in [('768',np.arange(768)),('128',J)]:
            a=z[:,ix];b=target[:,ix];out['mse'+name]=float(np.mean((a-b)**2));out['cosine'+name]=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        return out
    data={arm:(torch.from_numpy(x).permute(0,3,1,2).float().to(dev)/255,tuple(t.to(dev) for t in M.cele([np.array(b,dtype=np.float32).reshape(-1,4) for b in boxes[arm]],20,30))) for arm,x in imgs.items()}
    u=make();z0=latent(u);np.testing.assert_allclose(z0,before,rtol=1e-4,atol=3e-5)
    initial=metrics(z0);js(OUT/'INITIAL_METRICS.json',initial)
    forecasts={};forces={};rows=[]
    for arm,(x,c) in data.items():
        g=RW.gradient_tensorow(u,x,*c);new,mom=RW.czesci_kroku(u,g)
        sn=latent(u,new)-z0;sm=latent(u,mom)-z0
        f=sn+sm;ff=coeff['nowa']*sn+coeff['ped']*sm;forces[arm]=f
        np.savez_compressed(OUT/f'FORCE_{arm}.npz',nowa=sn,ped=sm,original_force=f,fitted_force=ff,channels0=J)
        forecasts[arm]={}
        for method,force in [('original',f),('historically_fitted',ff)]:
            trajectory=[]
            for h in range(51):
                m=metrics(z0+h*force);trajectory.append(m);rows.append(dict(arm=arm,method=method,step=h,pool32_epochs=h,**m))
            hits={}
            for dim in ('128','768'):
                hits[dim]={str(r):next((h for h,m in enumerate(trajectory) if m['mse'+dim]<=(1-r)*initial['mse'+dim]),None) for r in (.1,.25,.5)}
            forecasts[arm][method]=dict(hitting_time=hits,horizon=50,no_hit_semantics='null = no hit within50',assumption='same initial latent force each future step, frozen initial weights for all force calculations; no future gradients',metrics=trajectory)
        del g,new,mom
    for h in range(51):rows.append(dict(arm='nochange',method='nochange',step=h,pool32_epochs=h,**initial))
    with open(OUT/'FORECAST.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    js(OUT/'FORECAST.json',dict(initial=initial,arms=forecasts,thresholds=[.1,.25,.5],forecast_saved_before_training=True,epoch_definition='one update on fixed32 pool; not full-dataset epoch'))
    del u;torch.cuda.empty_cache()
    actual={};validation={}
    for arm,(x,c) in data.items():
        u=make();buf0={n:b.detach().clone() for n,b in u.net.named_buffers()};a=[]
        for step in range(1,51):
            u.net.train();u.opt.zero_grad(set_to_none=True)
            for group in u.opt.param_groups:group['lr']=5e-4
            with torch.autocast('cuda',dtype=torch.bfloat16):o,_=u.net(x)
            loss,_=M.strata(o,*c);loss.backward();norm=torch.nn.utils.clip_grad_norm_(u.net.parameters(),1.0);u.opt.step()
            z=latent(u);m=metrics(z);a.append(dict(step=step,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),**m))
            if step==1:
                fixed=latent(u,buffers=buf0);pred=z0+forces[arm]
                def skill(truth):
                    dz=truth-z0;return float(100*(1-np.sum((truth-pred).astype(np.float64)**2)/max(np.sum(dz.astype(np.float64)**2),1e-30)))
                validation[arm]=dict(full_with_BN_skill_percent=skill(z),fixed_old_BN_skill_percent=skill(fixed),predicted=metrics(pred),actual_full_BN=m,actual_fixed_BN=metrics(fixed))
                js(OUT/'FIRST_STEP_VALIDATION.json',validation)
            if step in (1,5,10,25,50):np.savez_compressed(OUT/f'LATENT_{arm}_k{step:03d}.npz',natural=z,original_target=target,channels0=J)
            js(OUT/f'ACTUAL_{arm}.json',a)
            progress('training_copy',arm=arm,step=step,total_steps=50,mse128=m['mse128'],mse768=m['mse768'])
            del o,loss
        actual[arm]=dict(metrics=a,hitting_time={dim:{str(r):next((m['step'] for m in a if m['mse'+dim]<=(1-r)*initial['mse'+dim]),None) for r in (.1,.25,.5)} for dim in ('128','768')})
        torch.save(dict(net=u.net.state_dict(),opt=u.opt.state_dict(),krok=1050),OUT/f'MODEL_{arm}_k1050.pt');del u;torch.cuda.empty_cache()
    result=dict(status='complete',ziarno=SEED,initial=initial,forecast=forecasts,actual=actual,first_step_validation=validation,authentic_optimizer=True,original_source_identity=True,latent_intervention=False,architecture_unchanged=True,protocol='fixed32 unaugmented batch, original loss, full train mode BN, original AdamW continuation',limitations=['Single original probe; no generalization inference','Historical coefficients fitted to pooled tool tokens transferred to full grid','Constant force is prospective assumption, not future actual gradients','Frozen checkpoint reachability bound is not global after weights update'])
    sys.path.insert(0,str(ROOT));import src.narzedzia.deklaracja_wynikow as declaration
    declaration._brudne_drzewo=lambda:True
    declaration.zapisz_wynik(__file__,'batch32',result)
    js(OUT/'RESULT.json',result)
    report='# Wynik paczki32\n\nAutentyczny stan AdamW k1000; sonda29 TR_0154/1926, bez ingerencji w latent. Prognoza stałej początkowej siły zapisana przed treningiem. Jedna epoka puli32 to jeden krok, nie epoka pełnego zbioru.\n\n'+json.dumps(dict(initial=initial,first_step=validation,forecast_hits={a:{k:v['hitting_time'] for k,v in d.items()} for a,d in forecasts.items()},actual_hits={a:d['hitting_time'] for a,d in actual.items()}),ensure_ascii=False,indent=2)+'\n\nnull oznacza brak trafienia w horyzoncie50. Nie dowodzi to globalnej niemożliwości osiągnięcia celu po zmianie wag. Współczynniki historyczne dopasowano do średnich narzędzia, transfer na wszystkie600 tokenów oceniono pierwszym krokiem. Test pojedynczej sondy, bez wnioskowania o generalizacji.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf8');(HERE/'WYNIKI_BATCH32.md').write_text(report,encoding='utf8')
    js(OUT/'HASHES.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('HASHES.json','stdout.log','stderr.log','PROGRESS.json')})
    progress('complete')
if __name__=='__main__':
    try:run()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        js(OUT/'BLOCKER.json',dict(error=str(e),traceback=traceback.format_exc(),exact_continuation_not_claimed=True))
        (OUT/'REPORT.md').write_text('# Bieg zatrzymany\n\n'+str(e)+'\n\nNie deklarujemy autentycznej kontynuacji ani wyniku prognozy/treningu po błędzie. Szczegóły w BLOCKER.json.\n',encoding='utf8')
        progress('blocked',error=str(e));raise
