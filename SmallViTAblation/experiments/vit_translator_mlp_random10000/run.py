"""Warm-start MLP with full annotated TRAIN pool; frozen ViT and DINO online."""
import os,sys,time,json,traceback,hashlib,math,gc,csv
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch,cv2
from torch.nn import functional as F
from dane import ROOT,SPLIT,ARCHIVES,inventory,sample,Reader,order
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs'/HERE.name
sys.path.insert(0,str(ROOT/'experiments/vit_translator_mlp_pelny_latent'))
from model import load
sys.path.insert(0,str(ROOT/'experiments/vit_kierunek_nadzorowany_detekcja128'))
from model_snapshot import MalyViT
START=time.time();STEP=0
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(name,o):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');replace(t,p)
def replace(source,dest):
    # Windows readers and antivirus can temporarily deny rename of an open file.
    for attempt in range(30):
        try:os.replace(source,dest);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)
def progress(stage,**kw):
    o=dict(stage=stage,pid=os.getpid(),step=STEP,elapsed_seconds=time.time()-START,**kw);js('PROGRESS.json',o);print(json.dumps(o),flush=True)
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);replace(tmp,path)
def metric(p,y):
    sim=p@y.T;rank=1+(sim>np.diag(sim)[:,None]).sum(1)
    return dict(cosine=float((p*y).sum(1).mean()),top1=float((rank==1).mean()),top5=float((rank<=5).mean()),median_rank=float(np.median(rank)),gallery_size=len(y))
def encoders(config):
    vit=MalyViT().cuda().eval().requires_grad_(False)
    assert sha(config['vit_checkpoint'])==config['vit_checkpoint_sha256']
    vit.load_state_dict(torch.load(config['vit_checkpoint'],map_location='cpu',weights_only=True)['net']);vit.zapisuj_latent(True)
    weights='C:/Users/bartl/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
    assert sha(weights)==config['dino_weights_sha256']
    dino=torch.hub.load('C:/Users/bartl/.cache/torch/hub/facebookresearch_dinov2_main','dinov2_vits14',source='local',pretrained=False)
    dino.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));dino=dino.cuda().eval().requires_grad_(False)
    return vit,dino
def encoded(reader,rows,vit,dino):
    valid=[];bad=[];v=[];d=[]
    for r in rows:
        try:
            vi,di=reader.read(r);v.append(vi);d.append(di);valid.append(r)
        except Exception as e:bad.append(dict(id=r['id'],error=str(e)))
    if not valid:return None,None,[],bad
    with torch.no_grad():
        _,z=vit(torch.stack(v).cuda());z=z.detach()
        y=F.normalize(dino.forward_features(torch.stack(d).cuda())['x_norm_clstoken'].float(),dim=1)
    return z,y,valid,bad
@torch.no_grad()
def evaluate(model,reader,rows,vit,dino):
    model.eval();p=[];y=[];ids=[];bad=[]
    for i in range(0,len(rows),4):
        z,target,valid,errors=encoded(reader,rows[i:i+4],vit,dino);bad.extend(errors)
        if z is None:continue
        p.append(model(z)[0].cpu().numpy());y.append(target.cpu().numpy());ids.extend(r['id'] for r in valid)
    if not p:raise RuntimeError('No valid evaluation images')
    p=np.concatenate(p);y=np.concatenate(y)
    return metric(p,y),p,y,ids,bad
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'RUN.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:run()
    except Exception:js('FAILED.json',dict(step=STEP,traceback=traceback.format_exc()));raise
    finally:lock.unlink(missing_ok=True)
def run():
    global STEP
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();csv.field_size_limit(32*1024*1024)
    config=json.loads((ROOT/'outputs/vit_translator_mlp_pelny_latent/CONFIG.json').read_text(encoding='utf-8'))
    initial=ROOT/'outputs/vit_translator_mlp_calydataset/MODEL_latest.pt'
    config.update(source_mlp=str(initial),source_mlp_sha256=sha(initial),split_sha256=sha(SPLIT),epochs=3,batch=8,learning_rate=.0001,weight_decay=.01,validate_every=500,checkpoint_every=500,validation_images=512,test_images=512,early_stop_patience=8,early_stop_minimum_epochs=1,bank_size=512,seed=101,normalization='retain original warm-start MLP train buffers',optimizer='continue AdamW from full-dataset step500 checkpoint',dataset_limit=10000,selection_seed=20261005,no_saved_vit_latents=True,all_full_latents_computed_online=True,no_latent_pooling=True,no_SD=True)
    js('CONFIG.json',config)
    manifest=OUT/'MANIFEST.jsonl'
    full_manifest=ROOT/'outputs/vit_translator_mlp_calydataset/MANIFEST.jsonl'
    if manifest.exists():rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    else:
        rng=np.random.default_rng(config['selection_seed']);reservoir=[];held=[];seen=0
        with full_manifest.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['split']=='train':
                    seen+=1
                    if len(reservoir)<10000:reservoir.append(r)
                    else:
                        j=int(rng.integers(seen))
                        if j<10000:reservoir[j]=r
                else:held.append(r)
        assert seen==283577 and len(reservoir)==10000
        rows=reservoir+sample([r for r in held if r['split']=='val'],512)+sample([r for r in held if r['split']=='test'],512)
        tmp=manifest.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        replace(tmp,manifest)
        js('SELECTION.json',dict(method='uniform reservoir sample without replacement, Algorithm R',seed=config['selection_seed'],selected=10000,source_eligible_train=seen,source_manifest_sha256=sha(full_manifest),initial_checkpoint_step=500,prior_training_exposures=4000))
    train=[r for r in rows if r['split']=='train'];val=sample([r for r in rows if r['split']=='val'],512);test=sample([r for r in rows if r['split']=='test'],512)
    films={s:{r['film'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not(films['train']&films['val'] or films['train']&films['test'] or films['val']&films['test'])
    assert len(train)==10000 and len(val)>0 and len(test)>0
    js('DATASET.json',dict(train_eligible=len(train),validation_selected=len(val),test_selected=len(test),films={k:sorted(v) for k,v in films.items()},manifest_sha256=sha(manifest),note='TRAIN eligibility from CSV and unique archive members; image decode and bbox bounds checked online. Canonical films held out from translator training; ViT may have previously seen these data.'))
    js('EVAL_MANIFEST.json',dict(val=val,test=test));progress('loading_models',train_eligible=len(train))
    model,old=load(initial,'cuda');hidden=old['hidden'];initial_optimizer=old['optimizer'];del old
    vit,dino=encoders(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    optimizer.load_state_dict(initial_optimizer);del initial_optimizer
    latest=OUT/'MODEL_latest.pt';best_score=(-1,-1,-1);no_improvement=0;history=[];epoch0=0;position0=0;valid_seen=0;skipped=0
    if latest.exists():
        ck=torch.load(latest,map_location='cuda',weights_only=True)
        assert ck['config']['source_mlp_sha256']==config['source_mlp_sha256'] and ck['config']['split_sha256']==config['split_sha256']
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);STEP=ck['step'];epoch0=ck['epoch'];position0=ck['position'];valid_seen=ck['valid_seen'];best_score=tuple(ck['best_score']);no_improvement=ck['no_improvement']
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        history=json.loads((OUT/'HISTORY.json').read_text()) if (OUT/'HISTORY.json').exists() else []
        del ck
    reader=Reader();errors=open(OUT/'INVALID_IMAGES.jsonl','a',encoding='utf-8');bank=None;bank_ids=[]
    def snapshot(epoch,position):
        return dict(hidden=hidden,model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,step=STEP,epoch=epoch,position=position,valid_seen=valid_seen,best_score=list(best_score),no_improvement=no_improvement,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    if not (OUT/'BASELINE_VAL.json').exists():
        progress('baseline_validation');m,_,_,_,_=evaluate(model,reader,val,vit,dino);js('BASELINE_VAL.json',m)
    ttrain=time.time();initial_seen=valid_seen;stopping=False
    for epoch in range(epoch0,config['epochs']):
        sequence=order(train,config['seed']+epoch);position=position0 if epoch==epoch0 else 0
        while position<len(sequence):
            ix=sequence[position:position+config['batch']];batch_rows=[train[i] for i in ix]
            z,y,valid,bad=encoded(reader,batch_rows,vit,dino);position+=len(ix)
            for b in bad:errors.write(json.dumps(b)+'\n')
            skipped+=len(bad)
            if z is None:continue
            model.train();pred,residual=model(z);target=(y-model.y_mean)/model.y_scale
            cosine=1-(pred*y).sum(1).mean();mse=F.mse_loss(residual,target)
            candidates=y if bank is None else torch.cat([y,bank],0)
            logits=pred@candidates.T/.07
            if bank is not None:
                for i,r in enumerate(valid):
                    for j,oldid in enumerate(bank_ids):
                        if r['id']==oldid:logits[i,len(y)+j]=-1e4
            contrast=F.cross_entropy(logits,torch.arange(len(y),device='cuda'))
            loss=cosine+.1*mse+.1*contrast;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();STEP+=1;valid_seen+=len(valid)
            bank=y.detach() if bank is None else torch.cat([bank,y.detach()],0)[-config['bank_size']:]
            bank_ids=(bank_ids+[r['id'] for r in valid])[-config['bank_size']:]
            if STEP%25==0 or STEP==1:
                rate=(valid_seen-initial_seen)/max(time.time()-ttrain,1e-9)
                progress('training',epoch=epoch+1,epochs=config['epochs'],valid_images_seen=valid_seen,train_eligible=len(train),position=position,skipped_images=skipped,loss=float(loss.detach()),images_per_second=rate,estimated_one_pass_seconds=len(train)/rate if rate else None,allocated_vram=torch.cuda.memory_allocated())
            if STEP%config['validate_every']==0 or position==len(sequence):
                m,_,_,_,bad_eval=evaluate(model,reader,val,vit,dino);score=(m['top1'],m['top5'],m['cosine'])
                history.append(dict(step=STEP,epoch=epoch+1,valid_images_seen=valid_seen,validation=m));js('HISTORY.json',history)
                if score>best_score:
                    best_score=score;no_improvement=0;save(OUT/'MODEL_best.pt',snapshot(epoch,position));js('BEST.json',history[-1])
                else:no_improvement+=1
                progress('validation',epoch=epoch+1,validation=m,no_improvement=no_improvement)
                if epoch>=config['early_stop_minimum_epochs'] and no_improvement>=config['early_stop_patience']:stopping=True
            if STEP%config['checkpoint_every']==0 or position==len(sequence) or stopping:save(latest,snapshot(epoch,position))
            if stopping:break
        position0=0
        if stopping:break
    save(OUT/'MODEL_final.pt',snapshot(epoch,position));errors.close();reader.close()
    del model,optimizer;torch.cuda.empty_cache();model,ck=load(OUT/'MODEL_best.pt','cuda');reader=Reader()
    progress('final_test');m,p,y,ids,bad=evaluate(model,reader,test,vit,dino)
    js('RESULT.json',dict(test=m,best=json.loads((OUT/'BEST.json').read_text()),valid_training_images_seen=valid_seen,skipped_images=skipped,steps=STEP,early_stopped=stopping,test_errors=bad))
    reader.close();progress('complete',test=m,valid_images_seen=valid_seen)
if __name__=='__main__':main()
