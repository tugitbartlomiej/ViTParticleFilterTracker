"""Czy wspólna regulacja J128 przekroczy AP50=0.90? Wynik testu <=0.90 obala ten cel.

Teacher powstaje tylko na train. Dobór na val, test po zamrożeniu wyboru.
"""
import argparse,json,math,shutil,sys,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
from dane import (load_data,SEED,HERE,ROOT,OUT,INPUT,js,sha,state_hash,image_tensor,
                  metrics,objective,film_weights,dekoduj,iou)
from regulator import Regulator,predict_tail,from_changed,full_with_regulator,FLOOR
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik

def config(name,kind,lr,width=64,distill=0,strong=True,coords=False):
    return dict(name=name,kind=kind,lr=lr,width=width,distill=distill,strong=strong,coords=coords)

CONFIGS=[config('affine001','affine',.001),config('affine003','affine',.003),
         config('local128_001','local',.001,128),config('local128_003','local',.003,128),
         config('local128_distill','local',.001,128,.3),
         config('spatial64_focal','spatial',.001,strong=False),
         config('spatial64_strong','spatial',.001),
         config('spatial64_distill001','spatial',.001,distill=.3),
         config('spatial64_distill003','spatial',.003,distill=.3),
         config('spatial128_distill001','spatial',.001,128,.3),
         config('spatial128_distill003','spatial',.003,128,.3),
         config('spatial64_coordinates','spatial',.001,distill=.3,coords=True)]

def score(m):return m['AP50'],m['AP50_95'],-m['loss']
def save_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def objective_ids(o,d,ids,strong=True):
    return objective(o,tuple(t[ids] for t in d['T']),d['centers'][ids],d['center_log'][ids],strong)

@torch.no_grad()
def predict(d,reg,ids):
    reg.eval();parts=[]
    for lo in range(0,len(ids),16):
        ix=ids[lo:lo+16];parts.append(predict_tail(d['net'],d['J'],d['Z'][ix],d['Q'][ix],reg).cpu())
    return torch.cat(parts)

def checks(d):
    results=[];ids=d['sets']['train_original'][:2]
    x=torch.stack([image_tensor(d['rows'][i]) for i in ids])
    for kind in ('affine','local','spatial'):
        torch.manual_seed(SEED)
        reg=Regulator(config('check',kind,.001),d['stats']).eval()
        z=d['Z'][ids];q=d['Q'][ids]
        torch.testing.assert_close(predict_tail(d['net'],d['J'],z,q,reg),d['base'][ids].cuda(),rtol=2e-5,atol=2e-5)
        with torch.no_grad():
            bias=reg.bias if kind=='affine' else reg.mapping[-1].bias
            bias.copy_(torch.linspace(-.03,.03,128,device='cuda'))
            if kind!='affine':reg.mapping[-1].weight.normal_(0,.001)
        real,cache,flags=full_with_regulator(d['net'],d['J'],x,reg)
        fast=predict_tail(d['net'],d['J'],cache['z'],cache['q'],reg)
        torch.testing.assert_close(real,fast,rtol=2e-5,atol=2e-5)
        params=list(reg.parameters())
        ga=torch.autograd.grad(objective_ids(real,d,ids).mean(),params)
        gb=torch.autograd.grad(objective_ids(fast,d,ids).mean(),params)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-3,atol=2e-5)
        assert sum(float(g.abs().sum()) for g in gb)>0 and all(torch.isfinite(g).all() for g in gb)
        with torch.no_grad():torch.testing.assert_close(reg(z[:1],q[:1]),reg(z,q)[:1],rtol=2e-5,atol=2e-5)
        assert flags['unchanged640'] and flags['min_activation']>=FLOOR-1e-6
        results.append(dict(kind=kind,forward_error=float((real-fast).abs().max().detach()),
                            gradient_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb)),**flags))
    assert state_hash(d['net'])==d['provenance']['weights_hash']
    val=d['sets']['val'];m=metrics(d['base'][val],d['T'],d['gt'],d['rows'],val)
    scores,tp=tp_table(d['base'][val],[d['gt'][i] for i in val]);a=fast_ap(scores,tp)
    assert abs(a[0]-m['AP50'])<1e-7 and abs(a.mean()-m['AP50_95'])<1e-7
    js(OUT/'PREFLIGHT.json',dict(status='pass',models=results,provenance=d['provenance']))
    print('PREFLIGHT PASS',json.dumps(results),flush=True)

def teacher_train(d):
    ids=d['sets']['train'];train_original=d['sets']['train_original']
    # Only original TRAIN entries of the old oracle are loaded as warm starts.
    warm_path=ROOT/'outputs/vit_latent128_maksymalizacja/TARGETS_FULL400.npz'
    with np.load(warm_path) as old:
        assert np.array_equal(old['channels0'],d['J'].cpu().numpy())
        sources=old['source_indices'];targets=old['target'][train_original].copy()
        assert np.array_equal(sources[train_original],[d['rows'][i]['index'] for i in train_original])
    warm={i:targets[k] for k,i in enumerate(train_original)}
    teacher=torch.zeros(len(d['rows']),600,128,dtype=torch.float32)
    maps=d['base'].clone();hist=[];t0=time.perf_counter()
    for lo in range(0,len(ids),16):
        ix=ids[lo:lo+16];z=d['Z'][ix];q=d['Q'][ix]
        start=torch.stack([torch.from_numpy(warm[i]).cuda() if i in warm else z[k].clone() for k,i in enumerate(ix)])
        changed=start.clamp(min=FLOOR).detach().requires_grad_()
        opt=torch.optim.Adam([changed],lr=.05)
        with torch.no_grad():
            baseline_loss=objective_ids(from_changed(d['net'],d['J'],z,q,z),d,ix)
            initial_loss=objective_ids(from_changed(d['net'],d['J'],z,q,changed),d,ix)
            use=initial_loss<baseline_loss
            best=torch.where(use[:,None,None],changed,z).clone();bestloss=torch.minimum(initial_loss,baseline_loss)
        for step in range(600):
            o=from_changed(d['net'],d['J'],z,q,changed);loss=objective_ids(o,d,ix)
            assert torch.isfinite(loss).all();opt.zero_grad();loss.mean().backward()
            assert torch.isfinite(changed.grad).all();opt.step()
            with torch.no_grad():
                changed.clamp_(min=FLOOR)
                current=objective_ids(from_changed(d['net'],d['J'],z,q,changed),d,ix)
                better=current<bestloss;best[better]=changed[better];bestloss=torch.minimum(bestloss,current)
        with torch.no_grad():maps[ix]=from_changed(d['net'],d['J'],z,q,best).cpu();teacher[ix]=best.cpu()
        row=dict(done=lo+len(ix),total=len(ids),mean_objective=float(bestloss.mean()),seconds=time.perf_counter()-t0)
        hist.append(row);js(OUT/'PROGRESS.json',dict(status='teacher_train_only',**row))
        print('TEACHER',json.dumps(row),flush=True)
    m=metrics(maps[train_original],d['T'],d['gt'],d['rows'],train_original)
    torch.save(dict(targets=teacher,train_ids=ids,channels=d['J'].cpu(),warm_sha=sha(warm_path)),OUT/'TEACHER_TRAIN.pt')
    np.savez_compressed(OUT/'TEACHER_DETECTIONS.npz',train_original=np.asarray(train_original),maps=maps[train_original].numpy())
    result=dict(ziarno=SEED,status='complete',uses_labels='train_only',metrics_original_train=m,
                steps=600,lr=.05,history=hist,warm_sha=sha(warm_path),val_test_targets_used=False,
                train_ids=ids,provenance=d['provenance'])
    js(OUT/'TEACHER_RESULT.json',result);zapisz_wynik(__file__,'oracle_train',result)
    print('TEACHER COMPLETE',json.dumps(m),flush=True)
    return teacher

def tp_table(o,gt):
    boxes,scores=dekoduj(o);boxes=boxes.numpy();scores=scores.numpy()
    assert all(len(g)==1 for g in gt)
    overlaps=np.stack([iou(b,g[0]) for b,g in zip(boxes,gt)])
    hit=overlaps[:,:,None]>=np.arange(.5,.951,.05)[None,None,:]
    first=hit & (np.cumsum(hit,axis=1)==1)
    return scores,first

def fast_ap(scores,tp):
    order=np.argsort(-scores.reshape(-1),kind='stable');t=tp.reshape(-1,10)[order]
    c=np.cumsum(t,axis=0);rec=c/len(scores);prec=c/np.arange(1,len(c)+1)[:,None]
    prec=np.maximum.accumulate(prec[::-1],axis=0)[::-1];result=[]
    for k in range(10):
        ix=np.searchsorted(rec[:,k],np.linspace(0,1,101));valid=ix<len(c)
        result.append(float(np.where(valid,prec[np.minimum(ix,len(c)-1),k],0).mean()))
    return np.asarray(result)

def paired_ci(pred,base,d,ids,n=1000):
    ss,tt=tp_table(pred,[d['gt'][i] for i in ids]);sb,tb=tp_table(base,[d['gt'][i] for i in ids])
    # Verify vectorized AP against the historical implementation on actual data.
    mp=metrics(pred,d['T'],d['gt'],d['rows'],ids);mb=metrics(base,d['T'],d['gt'],d['rows'],ids)
    for scores,tp,m in ((ss,tt,mp),(sb,tb,mb)):
        a=fast_ap(scores,tp);assert abs(a[0]-m['AP50'])<1e-7 and abs(a.mean()-m['AP50_95'])<1e-7
    films=sorted({d['rows'][i]['film'] for i in ids})
    groups={f:[k for k,i in enumerate(ids) if d['rows'][i]['film']==f] for f in films}
    rng=np.random.default_rng(SEED+73);values=[]
    for _ in range(n):
        chosen=rng.choice(films,len(films),replace=True);ix=[k for f in chosen for k in groups[f]]
        delta=fast_ap(ss[ix],tt[ix])-fast_ap(sb[ix],tb[ix]);values.append((delta[0],delta.mean()))
    a=np.asarray(values)
    return {k:np.quantile(a[:,i],[.025,.975]).tolist() for i,k in enumerate(('AP50','AP50_95'))}

def report(result):
    lines=['# Przestrzenne ustawianie J128: wynik','',
           'Wspólne parametry regulatora; checkpoint k1000 i640 pozostałych kanałów zamrożone. Wybór na val przed testem. Benchmark historycznie oglądany, wynik eksploracyjny.','',
           '| Wariant | Split | AP50 | AP50:95 | Loss |','|---|---|---:|---:|---:|']
    for name,parts in result['metrics'].items():
        for split,m in parts.items():lines.append(f"| {name} | {split} | {m['AP50']:.6f} | {m['AP50_95']:.6f} | {m['loss']:.6f} |")
    lines+=['',f"Wybrany regulator: {result['selected']['config']['name']}, epoka {result['selected']['epoch']}.",
            f"Cel AP50>0,90 na test: {result['goal_over_90_test']}.",
            f"Czas kampanii: {result['seconds']:.1f} s.",'',
            'Oracle train ma własny raport TEACHER_RESULT.json i wykorzystuje etykiety tylko treningu. Jego wynik nie jest skutecznością regulatora na nowych obrazach. Nie znaleziono naturalnych zdjęć realizujących zadane latenty. Nie wykazano optimum globalnego ani domknięcia tożsamości.','',
            '## Porównanie i ograniczenia','',
            'Podłoga używa tych samych obrazów, modelu, dekodera i AP. Regulator warunkowy przewiduje korektę z cech, a static używa wspólnych skal/przesunięć. Przedziały ufności resamplują całe filmy i nie usuwają wpływu wielokrotnego oglądania starego benchmarku.','',
            '## Przedziały 95% różnicy wobec baseline','',
            '| Wariant | Metryka | Dolna | Górna |','|---|---|---:|---:|']
    for tag,intervals in result['CI95'].items():
        for metric,(lo,hi) in intervals.items():lines.append(f'| {tag} | {metric} | {lo:.6f} | {hi:.6f} |')
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI_SEARCH.md').write_text(text,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['preflight','sweep'],required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    t0=time.perf_counter();d=load_data(augment=args.phase=='sweep');checks(d)
    if args.phase=='preflight':return
    assert not (OUT/'RESULT.json').exists(),'Existing completed result'
    archive=OUT/'EXECUTED_SOURCE';archive.mkdir(exist_ok=True)
    for suffix in ('*.py','*.ps1','*.md','WYNIKI.json'):
        for file in HERE.glob(suffix):shutil.copy2(file,archive/file.name)
    js(OUT/'CONFIG.json',dict(configs=CONFIGS,seed=SEED,epochs=160,batch=16,patience_checks=6,minimum_epochs=60,
                             sources={f.name:sha(f) for f in archive.iterdir()},provenance=d['provenance']))
    teacher=teacher_train(d)
    train=d['sets']['train'];val=d['sets']['val'];base_val=metrics(d['base'][val],d['T'],d['gt'],d['rows'],val)
    weights=torch.zeros(len(d['rows']),device='cuda');weights[train]=torch.tensor(film_weights(d['rows'],train),device='cuda')*len(train)
    candidates=[]
    for ci,cfg in enumerate(CONFIGS):
        torch.manual_seed(SEED);reg=Regulator(cfg,d['stats']);opt=torch.optim.AdamW(reg.parameters(),lr=cfg['lr'],weight_decay=.01)
        best=dict(epoch=0,metrics=base_val);state=save_state(reg);history=[dict(epoch=0,val=base_val)]
        folder=OUT/cfg['name'];folder.mkdir(exist_ok=True);rng=np.random.default_rng(SEED);stale=0
        for epoch in range(1,161):
            reg.train();lr=cfg['lr']*min(1.,epoch/5)*(.05+.95*(1+math.cos(math.pi*(epoch-1)/160))/2)
            for group in opt.param_groups:group['lr']=lr
            order=rng.permutation(train).tolist();train_loss=[]
            for lo in range(0,len(order),16):
                ids=order[lo:lo+16];z=d['Z'][ids];q=d['Q'][ids];changed=reg(z,q)
                output=from_changed(d['net'],d['J'],z,q,changed)
                ll=objective_ids(output,d,ids,cfg['strong'])
                if cfg['distill']:
                    target=teacher[ids].cuda();spatial=1+4*(d['T'][2][ids].flatten(1)>0).float()
                    distance=F.smooth_l1_loss(changed/reg.z_sd,target/reg.z_sd,reduction='none').mean(-1)
                    ll=ll+cfg['distill']*(distance*spatial).mean(1)
                penalty=((changed-z)/reg.z_sd).square().mean()
                loss=(ll*weights[ids]).mean()+1e-5*penalty
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reg.parameters())
                torch.nn.utils.clip_grad_norm_(reg.parameters(),5);opt.step();train_loss.append(float(loss.detach()))
            if epoch%10==0:
                pred=predict(d,reg,val);m=metrics(pred,d['T'],d['gt'],d['rows'],val)
                history.append(dict(epoch=epoch,val=m,train_objective=float(np.mean(train_loss)),lr=lr))
                if score(m)>score(best['metrics']):best=dict(epoch=epoch,metrics=m);state=save_state(reg);stale=0
                else:stale+=1
                js(folder/'HISTORY.json',history)
                torch.save(dict(config=cfg,epoch=best['epoch'],state=state,channels=d['J'].cpu(),
                                stats={k:v.cpu() for k,v in d['stats'].items()},provenance=d['provenance']),folder/'model_best.pt')
                js(OUT/'PROGRESS.json',dict(status='training',configuration=ci+1,configurations=len(CONFIGS),name=cfg['name'],
                                          epoch=epoch,maximum_epochs=160,best_validation=best,seconds=time.perf_counter()-t0))
                print('VAL',cfg['name'],epoch,json.dumps(m),flush=True)
                if epoch>=60 and stale>=6:break
        candidate=dict(config=cfg,epoch=best['epoch'],val=best['metrics'],stopped_epoch=epoch,parameters=sum(p.numel() for p in reg.parameters()))
        candidates.append(candidate);js(folder/'CANDIDATE.json',candidate)
    selected=max(candidates,key=lambda c:score(c['val']));static=max([c for c in candidates if c['config']['kind']=='affine'],key=lambda c:score(c['val']))
    js(OUT/'SELECTION.json',dict(status='frozen_before_test',selected=selected,static=static,candidates=candidates,test_used=False))
    all_ids=list(range(400));predictions={'baseline':d['base'][:400]};verify=[]
    for tag,candidate in [('selected',selected),('static',static)]:
        saved=torch.load(OUT/candidate['config']['name']/'model_best.pt',map_location='cpu',weights_only=False)
        reg=Regulator(saved['config'],d['stats']).eval();reg.load_state_dict(saved['state']);predictions[tag]=predict(d,reg,all_ids)
        ix=[d['sets']['train_original'][0],val[0],d['sets']['test'][0]]
        x=torch.stack([image_tensor(d['rows'][i]) for i in ix])
        with torch.no_grad():
            real,_,flags=full_with_regulator(d['net'],d['J'],x,reg);expected=predictions[tag][ix].cuda()
            torch.testing.assert_close(real,expected,rtol=4e-5,atol=4e-5);assert flags['unchanged640']
            verify.append(dict(tag=tag,max_error=float((real-expected).abs().max()),**flags))
        torch.save(saved,OUT/f'model_{tag}.pt')
    sets={'train':d['sets']['train_original'],'val':val,'test':d['sets']['test'],'all':all_ids}
    mm={tag:{s:metrics(pred[ids],d['T'],d['gt'],d['rows'],ids) for s,ids in sets.items()} for tag,pred in predictions.items()}
    unchanged=state_hash(d['net'])==d['provenance']['weights_hash'];assert unchanged
    js(OUT/'VERIFICATION.json',dict(full_forward=verify,frozen_weights_buffers=unchanged))
    np.savez_compressed(OUT/'DETECTIONS.npz',**{k:v.numpy() for k,v in predictions.items()})
    ids=sets['test'];ci={tag:paired_ci(predictions[tag][ids],predictions['baseline'][ids],d,ids) for tag in ('selected','static')}
    result=dict(status='complete',ziarno=SEED,selected=selected,static=static,candidates=candidates,metrics=mm,CI95=ci,
                goal_over_90_test=mm['selected']['test']['AP50']>.90,provenance=d['provenance'],
                weights_buffers_unchanged=unchanged,test_used_for_selection=False,seconds=time.perf_counter()-t0)
    js(OUT/'RESULT.json',result);report(result);zapisz_wynik(__file__,'search_spatial',result)
    js(OUT/'PROGRESS.json',dict(status='complete',selected=selected['config']['name'],metrics=mm,seconds=result['seconds']))
    print('COMPLETE',json.dumps(mm),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        js(OUT/'FAILURE.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()))
        traceback.print_exc();raise
