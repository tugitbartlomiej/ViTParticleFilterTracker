"""Buduje pilot TRAIN CLS DINO. Niezgodny split/GT/mapping lub QA obala indeks.
Nie estymuje generalizacji, jakości detektora ani zgodności sztucznego latentu.
"""
import argparse, ast, csv, gc, hashlib, io, json, math, os, sys, time, traceback, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
import torch, torchvision, faiss
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = ROOT/'outputs/vit_dino_faiss_35000'
SPLIT = ROOT/'feature_cache/eden_static36/data/tracking_static36/split_manifest.json'
ARCHIVES = ROOT.parent/'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
SPEC = json.loads((EXP/'DINO_SPEC.json').read_text())
START = time.perf_counter()
TOTAL = 35000
CHUNK = 1000
CONFIG_HASH = None
BATCH = 16
AMP = False
PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def js(p, d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(p)

def progress(stage, done=0, **kw):
    elapsed=time.perf_counter()-START
    js(OUT/'PROGRESS.json', dict(pid=os.getpid(),stage=stage,status='running',processed=done,total=TOTAL,elapsed=elapsed,rate=kw.get('images_per_second'),ETA=kw.get('eta_seconds'),config_hash=CONFIG_HASH,**kw))

def box(s):
    try:
        a=ast.literal_eval(s) if isinstance(s,str) else s
        if len(a)!=4: return None
        a=list(map(float,a)); x,y,w,h=a
        if not all(math.isfinite(v) for v in a) or x<0 or y<0 or w<=0 or h<=0: return None
        return a
    except (ValueError,TypeError,SyntaxError): return None

def inventory_one(p):
    with zipfile.ZipFile(p) as z:
        names=z.namelist(); images=[n for n in names if n.lower().endswith(('.jpg','.jpeg','.png'))]
        byname={}
        for n in images: byname.setdefault(Path(n).name,[]).append(n)
        cn=[n for n in names if Path(n).name=='annotation.csv']
        if len(cn)!=1: return [],dict(film=p.stem,images=len(images),eligible=0,error='missing/ambiguous annotation.csv')
        raw=z.read(cn[0]); annotation_hash=hashlib.sha256(raw).hexdigest(); rows=[]; seen=set(); invalid=0
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        for a in reader:
            fn=a.get('filename',''); matches=byname.get(fn,[])
            if len(matches)!=1: invalid+=1; continue
            member=matches[0]
            if member in seen: continue
            gt=[]; aux=[]
            for k,v in a.items():
                if k and k.endswith('_bbox'):
                    b=box(v)
                    if b is not None:
                        dst=aux if k in ('cornea_bbox','pupil_bbox') else gt
                        dst.append(dict(category=k[:-5],bbox_xywh=b))
            if not gt: invalid+=1; continue
            try: frame=int(a['frame_number'])
            except (ValueError,KeyError): invalid+=1; continue
            if frame!=int(Path(fn).stem.rsplit('_',1)[-1]): invalid+=1; continue
            seen.add(member)
            info=z.getinfo(member)
            rows.append(dict(id=p.stem+'/'+str(frame),split='TRAIN',film=p.stem,frame=frame,time_unit='frames',fps=None,archive=str(p.resolve()),member=member,realpath=None,gt=gt,aux_gt=aux,gt_source=cn[0],gt_source_sha256=annotation_hash,zip_crc=info.CRC,zip_size=info.file_size))
        rows.sort(key=lambda r:(r['frame'],r['member']))
        return rows,dict(film=p.stem,archive=str(p.resolve()),archive_bytes=p.stat().st_size,archive_mtime_ns=p.stat().st_mtime_ns,images=len(images),eligible=len(rows),invalid_gt_or_mapping=invalid,annotation_sha256=annotation_hash)

def quotas(cap, n):
    q={f:0 for f in sorted(cap)}
    while n:
        active=[f for f in q if q[f]<cap[f]]
        if not active: break
        step=max(1,n//len(active))
        for f in active:
            k=min(step,cap[f]-q[f],n); q[f]+=k; n-=k
    return q,n

def read_image(r):
    with zipfile.ZipFile(r['archive']) as z: raw=z.read(r['member'])
    with Image.open(io.BytesIO(raw)) as im:
        im.load(); rgb=im.convert('RGB')
    return rgb,raw

def valid_image(r):
    try:
        im,raw=read_image(r); w,h=im.size
        if any(b['bbox_xywh'][0]+b['bbox_xywh'][2]>w+1e-4 or b['bbox_xywh'][1]+b['bbox_xywh'][3]>h+1e-4 for b in r['gt']): raise ValueError('bbox outside decoded image')
        return dict(r,width=w,height=h,image_sha256=hashlib.sha256(raw).hexdigest(),pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()),None
    except Exception as e: return None,str(e)

def select_film(args):
    film,rows,q=args; selected=[]; used=set(); defects=[]
    if not rows or not q: return selected,defects
    targets=np.linspace(rows[0]['frame'],rows[-1]['frame'],q) if q>1 else [(rows[0]['frame']+rows[-1]['frame'])/2]
    with zipfile.ZipFile(rows[0]['archive']) as z:
        for target in targets:
            # Stable nearest-frame replacement including previously discovered defects.
            for j in sorted(range(len(rows)),key=lambda j:(abs(rows[j]['frame']-target),rows[j]['frame'],rows[j]['member'])):
                if j in used: continue
                used.add(j); r=rows[j]
                try:
                    raw=z.read(r['member'])
                    with Image.open(io.BytesIO(raw)) as im: im.load(); im=im.convert('RGB')
                    w,h=im.size
                    if any(b['bbox_xywh'][0]+b['bbox_xywh'][2]>w+1e-4 or b['bbox_xywh'][1]+b['bbox_xywh'][3]>h+1e-4 for b in r['gt']): raise ValueError('bbox outside decoded image')
                    selected.append(dict(r,width=w,height=h,target_frame=float(target),image_sha256=hashlib.sha256(raw).hexdigest(),pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
                    break
                except Exception as e: defects.append(dict(film=film,member=r['member'],error=str(e)))
    return selected,defects

def prepare():
    t=time.perf_counter(); groups=json.loads(SPLIT.read_text())['groups']
    paths=[p for p in sorted(ARCHIVES.glob('*.zip')) if groups.get(p.stem)=='train']
    pools={}; inv=[]
    csv.field_size_limit(32*1024*1024)
    with ThreadPoolExecutor(4) as ex:
        for k,(rows,stat) in enumerate(ex.map(inventory_one,paths)):
            pools[stat['film']]=rows; inv.append(stat)
            progress('inventory',films_scanned=k+1,films_total=len(paths))
    js(OUT/'INVENTORY.json',dict(split_manifest=str(SPLIT),split_sha256=sha(SPLIT),train_images=sum(i['images'] for i in inv),eligible_gt=sum(len(v) for v in pools.values()),films=inv))
    q,short=quotas({f:len(r) for f,r in pools.items()},TOTAL)
    selected=[]; defects=[]
    with ThreadPoolExecutor(6) as ex:
        for k,(s,d) in enumerate(ex.map(select_film,[(f,r,q[f]) for f,r in sorted(pools.items())])):
            selected.extend(s); defects.extend(d); progress('validate_images',len(selected),films_validated=k+1)
    # Redistribute only if full decoding reduces one film's original quota.
    ids={r['id'] for r in selected}; bad={(d['film'],d['member']) for d in defects}
    while len(selected)<TOTAL:
        cap={f:sum(r['id'] not in ids and (f,r['member']) not in bad for r in rows) for f,rows in pools.items()}
        add,missing=quotas(cap,TOTAL-len(selected))
        if not sum(add.values()): break
        before=len(selected)
        for f in sorted(pools):
            remaining=[r for r in pools[f] if r['id'] not in ids and (f,r['member']) not in bad]
            s,d=select_film((f,remaining,add[f])); selected.extend(s); defects.extend(d)
            ids.update(r['id'] for r in s); bad.update((d['film'],d['member']) for d in d)
        if len(selected)==before: break
    selected.sort(key=lambda r:(r['film'],r['frame'],r['member']))
    assert len({(r['archive'],r['member']) for r in selected})==len(selected)==len({r['id'] for r in selected})
    tmp=OUT/'manifest.jsonl.tmp'
    with tmp.open('w',encoding='utf-8') as f:
        for row,r in enumerate(selected): r['row']=row; f.write(json.dumps(r,ensure_ascii=False)+'\n')
    tmp.replace(OUT/'manifest.jsonl')
    coverage=[]
    for film,rows in sorted(pools.items()):
        ss=[r['frame'] for r in selected if r['film']==film]
        coverage.append(dict(film=film,eligible=len(rows),quota=q[film],selected=len(ss),eligible_min=rows[0]['frame'] if rows else None,eligible_max=rows[-1]['frame'] if rows else None,selected_min=min(ss) if ss else None,selected_max=max(ss) if ss else None,max_gap=max(np.diff(ss)).item() if len(ss)>1 else None))
    js(OUT/'COVERAGE.json',dict(total=len(selected),films_with_selection=sum(c['selected']>0 for c in coverage),coverage=coverage,defects=defects,manifest_sha256=sha(OUT/'manifest.jsonl'),preparation_seconds=time.perf_counter()-t,pixel_unique=len({r['pixel_sha256'] for r in selected}),time_unit='frame index, FPS unavailable'))
    if len(selected)!=TOTAL: raise RuntimeError(f'Only {len(selected)} valid TRAIN images, need {TOTAL}')
    return selected

def load_model(device):
    assert sha(SPEC['weights'])==SPEC['weights_sha256']
    assert sha(Path(SPEC['hub_repo'])/'dinov2/models/vision_transformer.py')==SPEC['vision_transformer_source_sha256']
    model=torch.hub.load(SPEC['hub_repo'],'dinov2_vits14',source='local',pretrained=False)
    model.load_state_dict(torch.load(SPEC['weights'],map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)

def gpu_wait(done=0):
    while True:
        free,_=torch.cuda.mem_get_info()
        if free>=900*1024**2: return
        progress('waiting_vram',done,free_vram_bytes=free)
        torch.cuda.empty_cache(); time.sleep(15)

def feature(model,x,amp=False):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
        f=model.forward_features(x.to('cuda'))['x_norm_clstoken']
    f=torch.nn.functional.normalize(f.float(),dim=1)
    a=f.cpu().numpy().astype('float32'); assert np.isfinite(a).all()
    return a

def decode(r):
    im,raw=read_image(r)
    if hashlib.sha256(raw).hexdigest()!=r['image_sha256']: raise RuntimeError('image changed after validation')
    return PRE(im)

def extract(model,rows):
    global BATCH
    t=time.perf_counter(); done=0; fresh=0
    chunks=OUT/'chunks'; chunks.mkdir(exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        for start in range(0,len(rows),CHUNK):
            stop=min(len(rows),start+CHUNK); p=chunks/f'{start:06d}.npy'; meta=p.with_suffix('.json')
            if p.exists() and meta.exists():
                m=json.loads(meta.read_text()); a=np.load(p)
                assert m['config_hash']==CONFIG_HASH and m['sha256']==sha(p) and a.shape==(stop-start,384) and np.isfinite(a).all()
                done=stop; continue
            tensors=list(ex.map(decode,rows[start:stop])); output=[]; pos=0
            while pos<len(tensors):
                gpu_wait(done+pos)
                try:
                    output.append(feature(model,torch.stack(tensors[pos:pos+BATCH]),AMP)); pos+=len(output[-1])
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    if BATCH>1: BATCH=max(1,BATCH//2); print('BATCH_REDUCED',BATCH,flush=True)
                    else: progress('waiting_vram',done+pos); time.sleep(15)
            a=np.concatenate(output); assert a.shape==(stop-start,384)
            tmp=p.with_suffix('.tmp');
            with tmp.open('wb') as f: np.save(f,a); f.flush(); os.fsync(f.fileno())
            tmp.replace(p); js(meta,dict(config_hash=CONFIG_HASH,start=start,stop=stop,sha256=sha(p),batch=BATCH))
            done=stop; fresh+=len(a); dt=time.perf_counter()-t; rate=fresh/dt
            progress('extract',done,images_per_second=rate,eta_seconds=(len(rows)-done)/rate,completed_chunks=(done+CHUNK-1)//CHUNK,extraction_elapsed=dt,batch=BATCH,allocated_vram=torch.cuda.memory_allocated(),reserved_vram=torch.cuda.memory_reserved())
            print('CHUNK',done,'rate',rate,'ETA',(len(rows)-done)/rate,flush=True)
            if fresh==CHUNK: js(OUT/'PILOT.json',dict(images=fresh,seconds=dt,rate=rate,eta_seconds=(len(rows)-done)/rate,warmup_excluded=True,includes_cpu_decode_and_atomic_chunk_write=True))
    p=OUT/'features.npy'; tmp=OUT/'features.tmp.npy'
    f=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(len(rows),384))
    for start in range(0,len(rows),CHUNK):
        a=np.load(chunks/f'{start:06d}.npy'); f[start:start+len(a)]=a
    f.flush(); del f; tmp.replace(p)
    return time.perf_counter()-t

def export_query(model,index,rows,image,topk=32,device='cuda'):
    image=Path(image); folder=OUT/'queries'/image.stem; folder.mkdir(parents=True,exist_ok=True)
    with Image.open(image) as im: im.load(); q=PRE(im.convert('RGB')).unsqueeze(0)
    v=feature(model,q,AMP); np.save(folder/'query_feature.npy',v)
    scores,ids=index.search(v,topk); ranking=[]
    sheet=Image.new('RGB',(4*240,8*188),'white'); draw=ImageDraw.Draw(sheet)
    for rank,(score,rid) in enumerate(zip(scores[0],ids[0]),1):
        r=rows[int(rid)]; ranking.append(dict(rank=rank,score=float(score),**r))
        im,_=read_image(r); im.save(folder/f'{rank:02d}_row{rid}.jpg',quality=95)
        im.thumbnail((236,156)); x=((rank-1)%4)*240; y=((rank-1)//4)*188
        sheet.paste(im,(x,y)); draw.text((x,y+156),f'{rank}: {float(score):.6f} row={rid}',fill='black'); draw.text((x,y+171),r['film']+' f'+str(r['frame']),fill='black')
    sheet.save(folder/'CONTACT_SHEET.jpg',quality=95)
    js(folder/'RANKING.json',dict(query=str(image.resolve()),query_sha256=sha(image),config_hash=CONFIG_HASH,meaning='DINO cosine similarity only; no detector confidence or synthetic latent agreement',results=ranking))
    with (folder/'RANKING.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['rank','score','row','id','film','frame','archive','member','gt']); w.writeheader()
        for r in ranking: w.writerow({k:json.dumps(r[k]) if k=='gt' else r[k] for k in w.fieldnames})
    return dict(query=str(image),top1=ranking[0],folder=str(folder)),v

def finish(rows,extract_seconds,model):
    t=time.perf_counter(); faiss.omp_set_num_threads(4)
    f=np.load(OUT/'features.npy',mmap_mode='r'); norms=np.linalg.norm(f,axis=1)
    assert len(rows)==TOTAL and len({r['id'] for r in rows})==TOTAL
    assert all(r['row']==i and r['split']=='TRAIN' for i,r in enumerate(rows))
    assert np.isfinite(f).all() and np.max(np.abs(norms-1))<2e-5
    index=faiss.IndexFlatIP(384); index.add(np.asarray(f)); faiss.write_index(index,str(OUT/'index.faiss'))
    index=faiss.read_index(str(OUT/'index.faiss')); assert index.ntotal==len(rows)==f.shape[0]
    checks=[]
    for rid in np.linspace(0,TOTAL-1,12).astype(int):
        v=np.asarray(f[rid:rid+1]); scores,ids=index.search(v,32); brute=np.asarray(f)@v[0]
        expected=np.sort(brute)[-32:][::-1]
        assert np.allclose(scores[0],expected,atol=2e-6)
        assert brute[rid]>=brute.max()-2e-6
        assert abs(scores[0,0]-brute[rid])<2e-6
        checks.append(dict(row=int(rid),self_rank_or_tie=True,score=float(scores[0,0]),top32_bruteforce_max_error=float(np.max(np.abs(scores[0]-expected))),tie_count=int(np.sum(np.abs(brute-brute.max())<2e-6))))
    queries=[ROOT/f'outputs/vit_sd_cztery_szumy/OBRAZ_{s}_prowadzony.png' for s in [2026100511,2026100523,2026100537,2026100549]]
    extra=ROOT/'outputs/vit_sd_szum_nowy_checkpoint/OBRAZ_SD.png'
    if extra.exists(): queries.append(extra)
    result=[]; features=[]
    for p in queries:
        if not p.exists(): raise FileNotFoundError(p)
        r,v=export_query(model,index,rows,p); result.append(r); features.append(v)
    del model; gc.collect(); torch.cuda.empty_cache(); gpu_wait()
    fresh=load_model('cuda'); stability=[]
    for p,v in zip(queries,features):
        with Image.open(p) as im: x=PRE(im.convert('RGB')).unsqueeze(0)
        again=feature(fresh,x,AMP); err=float(np.max(np.abs(again-v))); assert err<2e-5
        s,i=index.search(v,32); s2,i2=index.search(again,32)
        assert np.allclose(s,s2,atol=2e-5)
        stability.append(dict(query=str(p),max_abs_error=err,same_top32_ids=bool(np.array_equal(i,i2))))
    js(OUT/'QA.json',dict(status='passed',rows=TOTAL,index_ntotal=index.ntotal,finite=True,norm_min=float(norms.min()),norm_max=float(norms.max()),bruteforce=checks,fresh_model_query_stability=stability))
    cov=json.loads((OUT/'COVERAGE.json').read_text()); meta=dict(status='complete',pid=os.getpid(),rows=TOTAL,dimension=384,films=cov['films_with_selection'],config_hash=CONFIG_HASH,manifest_sha256=sha(OUT/'manifest.jsonl'),features_sha256=sha(OUT/'features.npy'),index_sha256=sha(OUT/'index.faiss'),index_bytes=(OUT/'index.faiss').stat().st_size,preparation_seconds=cov['preparation_seconds'],extraction_seconds=extract_seconds,finalization_seconds=time.perf_counter()-t,total_wall_seconds=time.perf_counter()-START,torch=torch.__version__,torchvision=torchvision.__version__,faiss=faiss.__version__,numpy=np.__version__,queries=result)
    js(OUT/'METADATA.json',meta)
    lines=['# Indeks DINO i FAISS: gotowy','',f"Dokładnie {TOTAL} niepowtórzonych ID TRAIN, {meta['films']} filmów. Globalny CLS 384, L2 w FP32, CPU IndexFlatIP. Indeks {meta['index_bytes']} bajtów.",'',f"Czas przygotowania: {meta['preparation_seconds']:.1f} s. Ekstrakcja: {extract_seconds:.1f} s. Finalizacja: {meta['finalization_seconds']:.1f} s. Cały bieżący proces: {meta['total_wall_seconds']:.1f} s.",'','Pilot i ETA z rzeczywistego pomiaru: PILOT.json. Pokrycie wszystkich filmów i zakresów klatek: COVERAGE.json. FPS niedostępne; czas raportowany jako numer klatki. CONFIG.json i DINO_SPEC.json ustalają wagi i preprocessing. QA.json: świeży reload indeksu, brute-force top32, self-query z remisami oraz ponownie wczytany model dla zapytań.','', '## Rankingi','']
    for r in result: lines.append(f"- {Path(r['query']).name}: top1 {r['top1']['id']}, cosine {r['top1']['score']:.6f}; pełne top32, GT i stykówka w queries/{Path(r['query']).stem}.")
    lines+=['','## Ograniczenia','','Pilotaż 35000 zdjęć, a nie cały TRAIN. Brak held-out oceny i treningu. Globalny descriptor może preferować tło. Obraz SD nie jest automatycznie naturalnym obiektem; cosine nie jest confidence ani miarą zgodności z pełnym latentu ViT 600x768. Podobieństwo nie dowodzi korzyści z uczenia.']
    (OUT/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    js(OUT/'PROGRESS.json',dict(status='complete',stage='complete',pid=os.getpid(),processed=TOTAL,total=TOTAL,elapsed=meta['total_wall_seconds'],rate=TOTAL/extract_seconds,ETA=0,config_hash=CONFIG_HASH))
    print('COMPLETE',meta['total_wall_seconds'],flush=True)

def main():
    global CONFIG_HASH,AMP
    OUT.mkdir(parents=True,exist_ok=True)
    # Lock held by Windows OS, automatically released after crash. No duplicate worker.
    import msvcrt
    lock=open(OUT/'RUN.lock','a+b'); lock.seek(0)
    if (OUT/'RUN.lock').stat().st_size==0: lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    progress('starting'); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    if (OUT/'manifest.jsonl').exists():
        rows=[json.loads(s) for s in (OUT/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==TOTAL
        inv=json.loads((OUT/'INVENTORY.json').read_text()); assert inv['split_sha256']==sha(SPLIT)
        groups=json.loads(SPLIT.read_text())['groups']; assert all(groups.get(r['film'])=='train' for r in rows)
        for a in inv['films']:
            p=Path(a['archive']); assert p.stat().st_size==a['archive_bytes'] and p.stat().st_mtime_ns==a['archive_mtime_ns']
    else: rows=prepare()
    gpu_wait(); model=load_model('cuda'); progress('precision_control')
    sample=rows[::max(1,len(rows)//117)][:117]
    # Each film is represented: first selected frame, plus temporal endpoints for a balanced check.
    sample=[]
    for film in sorted({r['film'] for r in rows}):
        rr=[r for r in rows if r['film']==film]; sample.extend([rr[0],rr[len(rr)//2],rr[-1]])
    x=torch.stack([decode(r) for r in sample[:8]])
    feature(model,x,False); feature(model,x,True); torch.cuda.synchronize()
    worst=0.; cosine_min=1.; errors=[]
    for s in range(0,len(sample),8):
        x=torch.stack([decode(r) for r in sample[s:s+8]])
        fp=feature(model,x,False); mp=feature(model,x,True)
        err=float(np.max(np.abs(fp-mp))); cos=float(np.min(np.sum(fp*mp,axis=1))); worst=max(worst,err); cosine_min=min(cosine_min,cos)
        errors.append(dict(start=s,abs_error=err,min_cosine=cos))
    AMP=False  # Shared translator targets are FP32: keep precisely the same forward precision.
    control=dict(samples=len(sample),films=len({r['film'] for r in sample}),temporal_positions=['start','middle','end'],max_abs_error=worst,min_cosine=cosine_min,threshold_abs=.005,threshold_cosine=.9999,selected_precision='amp_fp16' if AMP else 'fp32',details=errors)
    js(OUT/'PRECISION_QA.json',control)
    translator=ROOT/'outputs/vit_translator_latent_dino/CONFIG.json'
    shared=json.loads(translator.read_text())
    assert shared['dino_model']==SPEC['model'] and shared['dino_weights_sha256']==SPEC['weights_sha256']
    config=dict(spec=SPEC,manifest_sha256=sha(OUT/'manifest.jsonl'),split_sha256=sha(SPLIT),precision='amp_fp16' if AMP else 'fp32',normalization='float32 L2',translator_config=str(translator),translator_config_sha256=sha(translator),translator_representation_compatibility='same model, weights, RGB shorter256 bicubic center224 ImageNet normalization, L2 CLS384, FP32 forward; no claims of translator reliability',total=TOTAL,chunk_size=CHUNK,initial_batch=BATCH,vram_allocator_limit_bytes=2*1024**3,faiss='CPU IndexFlatIP',faiss_version=faiss.__version__,numpy_version=np.__version__,worker_sha256=sha(__file__))
    CONFIG_HASH=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (OUT/'CONFIG.json').exists(): assert json.loads((OUT/'CONFIG.json').read_text())['config_hash']==CONFIG_HASH
    js(OUT/'CONFIG.json',dict(config,config_hash=CONFIG_HASH))
    print('MANIFEST_READY',len(rows),'CONFIG',CONFIG_HASH,'AMP',AMP,flush=True)
    seconds=extract(model,rows); finish(rows,seconds,model)

if __name__=='__main__':
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True); js(OUT/'ERROR.json',dict(pid=os.getpid(),error=str(e),traceback=traceback.format_exc(),elapsed=time.perf_counter()-START))
        traceback.print_exc(); sys.exit(1)
