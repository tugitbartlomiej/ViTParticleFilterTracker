"""Train nonlinear full-latent translator with validation-only selection."""
import os,sys,time,json,traceback,hashlib
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import FullLatentMLP,load
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'outputs/vit_translator_latent_dino';OUT=ROOT/'outputs'/HERE.name
START=time.time()
def write(name,obj):
    p=OUT/name;q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(q,p)
def progress(stage,**kw):
    o=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kw);write('PROGRESS.json',o);print(json.dumps(o),flush=True)
def metric(pred,target):
    scores=pred@target.T;rank=1+(scores>np.diag(scores)[:,None]).sum(1)
    return dict(cosine=float((pred*target).sum(1).mean()),MSE=float(((pred-target)**2).mean()),top1=float((rank==1).mean()),top5=float((rank<=5).mean()),median_rank=float(np.median(rank)),gallery_size=len(target))
@torch.inference_mode()
def predict(model,z,ix,device):
    model.eval();return np.concatenate([model(torch.tensor(np.array(z[rows]),device=device))[0].cpu().numpy() for rows in np.array_split(ix,max(1,int(np.ceil(len(ix)/8))))])
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'RUN.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:run()
    except Exception:
        write('FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:lock.unlink(missing_ok=True)
def run():
    torch.set_num_threads(4);device='cuda';assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=True
    manifest=json.loads((SOURCE/'MANIFEST.json').read_text(encoding='utf-8'))
    cfg=json.loads((SOURCE/'CONFIG.json').read_text(encoding='utf-8'))
    rows=manifest['records'];ids={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','val','test']}
    films={s:{rows[i]['film'] for i in ix} for s,ix in ids.items()}
    assert not(films['train']&films['val'] or films['train']&films['test'] or films['val']&films['test'])
    cfg.update(architecture='flatten full 600x768 -> hidden32 or64 ->256 ->384, GELU and dropout; no pooling/PCA/SD',hidden_sizes=[32,64],seed=101,steps=800,batch=16,learning_rate=.0003,weight_decay=.01,validation_every=50,early_stop_checks=6,loss='cosine + .1 standardized target residual MSE + .1 InfoNCE(all240 training DINO features, temperature .07)',selection='validation top1, then top5, then cosine; test only after selecting',checkpoint_specific=True)
    write('CONFIG.json',cfg);write('MANIFEST.json',manifest)
    z=np.load(SOURCE/'vit.npy',mmap_mode='r');y=np.load(SOURCE/'dino.npy');assert z.shape==(400,600,768) and y.shape==(400,384)
    tr,va,te=[ids[s] for s in ['train','val','test']]
    train_x=np.array(z[tr]).reshape(len(tr),-1)
    mean=torch.tensor(train_x.mean(0),device=device);scale=torch.tensor(np.maximum(train_x.std(0),.1),device=device);del train_x
    yt=torch.tensor(y[tr],device=device);ym=yt.mean(0);ys=yt.std(0).clamp(min=.01)
    history=[];candidates=[]
    for hidden in cfg['hidden_sizes']:
        torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);model=FullLatentMLP(hidden).to(device)
        with torch.no_grad():model.x_mean.copy_(mean);model.x_scale.copy_(scale);model.y_mean.copy_(ym);model.y_scale.copy_(ys)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        best_rank=(-1,-1,-1);checks=0;path=OUT/f'MLP_hidden{hidden}.pt';order=np.array([],dtype=int);position=0
        progress('training',hidden=hidden,parameters=sum(p.numel() for p in model.parameters()))
        for step in range(1,cfg['steps']+1):
            if position+cfg['batch']>len(order):order=rng.permutation(len(tr));position=0
            local=order[position:position+cfg['batch']];position+=cfg['batch'];ix=tr[local]
            x=torch.tensor(np.array(z[ix]),device=device);target=yt[local];model.train();pred,residual=model(x)
            cos=1-(pred*target).sum(1).mean();res=F.mse_loss(residual,(target-ym)/ys)
            contrast=F.cross_entropy(pred@yt.T/.07,torch.tensor(local,device=device))
            loss=cos+.1*res+.1*contrast;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%cfg['validation_every']==0:
                m=metric(predict(model,z,va,device),y[va]);ranking=(m['top1'],m['top5'],m['cosine'])
                history.append(dict(hidden=hidden,step=step,loss=float(loss.detach()),validation=m))
                if ranking>best_rank:
                    best_rank=ranking;checks=0;best=dict(hidden=hidden,step=step,validation=m,path=str(path))
                    temp=path.with_suffix('.tmp');torch.save(dict(hidden=hidden,model=model.state_dict(),step=step,config=cfg),temp);os.replace(temp,path)
                else:checks+=1
                write('HISTORY.json',history);progress('training',hidden=hidden,step=step,validation=m,early_stop_checks=checks)
                if checks>=cfg['early_stop_checks']:break
        candidates.append(best);del optimizer,model,x,pred,residual,loss;torch.cuda.empty_cache()
    chosen=max(candidates,key=lambda b:(b['validation']['top1'],b['validation']['top5'],b['validation']['cosine']))
    model,checkpoint=load(chosen['path'],device);torch.save(checkpoint,OUT/'TRANSLATOR_MLP.pt')
    train_pred=predict(model,z,tr,device);test_pred=predict(model,z,te,device)
    linear=json.loads((SOURCE/'RESULT.json').read_text(encoding='utf-8'))
    result=dict(chosen=chosen,candidates=candidates,train=metric(train_pred,y[tr]),test=metric(test_pred,y[te]),linear_test=linear['test'],mean_baseline_test=linear['mean_baseline_test'],historical_development_set=True,independent_generalization=False)
    target_path=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz';target=np.load(target_path)
    with torch.inference_mode():
        artificial=model(torch.tensor(target['after'][None],device=device))[0].cpu().numpy()
        natural=model(torch.tensor(target['before'][None],device=device))[0].cpu().numpy()
    np.save(OUT/'DINO_CEL_SZTUCZNY.npy',artificial);np.save(OUT/'DINO_CEL_PRZED.npy',natural)
    np.savez(OUT/'TEST_PREDICTIONS.npz',indices=te,predicted=test_pred,actual=y[te])
    similarity=y@artificial[0];order=np.argsort(-similarity)[:32]
    write('KANDYDACI_CEL_SZTUCZNY.json',dict(scope='400 pilot originals, not35000 index',before_after_cosine=float(artificial[0]@natural[0]),candidates=[dict(index=int(i),path=manifest['paths'][i],cosine=float(similarity[i])) for i in order]))
    # Actual independent serialization check, on CPU without autocast or a GPU model.
    del model;torch.cuda.empty_cache();cpu,_=load(OUT/'TRANSLATOR_MLP.pt','cpu')
    with torch.inference_mode():check=cpu(torch.tensor(np.array(z[te[:2]])))[0].numpy()
    np.testing.assert_allclose(check,test_pred[:2],atol=2e-4,rtol=2e-3)
    result['verification']=dict(fresh_cpu_reload_passed=True,max_reload_error=float(np.max(abs(check-test_pred[:2]))),full_input_shape=[600,768],no_pooling=True,output_dim=384,test_not_used_for_selection=True)
    write('RESULT.json',result)
    (OUT/'REPORT.md').write_text('# MLP pełny latent ViT → DINO\n\nWszystkie 460800 wartości wchodzą do pierwszej warstwy gęstej. Nie uśredniamy tokenów i nie używamy SD. Trenujemy tylko translator; checkpoint ViT i DINO pozostają zamrożone. Normalizacja wyliczona tylko na train, wybór rozmiaru MLP i kroku na val. Test na85 obrazach z innych filmów wykonany po wyborze. Historyczne dane rozwojowe nie są niezależnym testem generalizacji.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nCosine nie jest procentem zgodności wartości. Trafienie top1/top5 dotyczy galerii85 zdjęć. Duża różnica train/test oznacza przeuczenie. Translacja sztucznego celu to ekstrapolacja i wymaga weryfikacji znalezionych zdjęć w ViT; sama nie dowodzi korzyści dla uczenia.\n',encoding='utf-8')
    progress('complete',test=result['test'])
if __name__=='__main__':main()
