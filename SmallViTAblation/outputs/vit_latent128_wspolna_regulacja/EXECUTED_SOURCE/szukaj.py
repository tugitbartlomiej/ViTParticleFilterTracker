"""Czy wspólna regulacja poprawia AP na innych filmach? Brak poprawy obala pilota.

Dobór tylko na val, test po zapisaniu i zamrożeniu wyboru. Bez parametrów per obraz.
"""
import argparse,csv,json,math,shutil,sys,time,traceback
from pathlib import Path
import numpy as np,torch
from dane import (HERE,ROOT,OUT,load,js,sha,state_hash,losses,metrics,film_weights,image_tensor,ap_at)
from model_snapshot import dekoduj
from regulator import Regulator,tail_fast,hook_for
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik

SEED=20261005
CONFIGS=[dict(name='shift_lr001',kind='shift',lr=.01,penalty=.001),
         dict(name='affine_lr0003',kind='affine',lr=.003,penalty=.001),
         dict(name='affine_lr001',kind='affine',lr=.01,penalty=.001),
         dict(name='affine_lr003',kind='affine',lr=.03,penalty=.001),
         dict(name='affine_lr001_reg',kind='affine',lr=.01,penalty=.01),
         dict(name='adapter16_lr0003',kind='adapter',lr=.003,penalty=.001),
         dict(name='adapter16_lr001',kind='adapter',lr=.01,penalty=.001)]

def key(m):return m['AP50_95'],m['AP50'],-m['loss']
def save_state(r):return {k:v.detach().cpu().clone() for k,v in r.state_dict().items()}

def checks(net,rows,sets,J,T,gt,H,Q,base,mu,sd,prov):
    results=[]
    for kind in ('shift','affine','adapter'):
        torch.manual_seed(SEED);reg=Regulator(kind,mu,sd)
        ids=sets['train'][:2];x=torch.stack([image_tensor(rows[i]) for i in ids])
        zero=tail_fast(net,J,H[ids],Q[ids],reg)
        torch.testing.assert_close(zero,base[ids].cuda(),rtol=2e-5,atol=2e-5)
        with torch.no_grad():
            reg.bias.copy_(torch.linspace(-.03,.03,128,device='cuda'))
            if kind=='affine':reg.scale.copy_(torch.linspace(-.02,.02,128,device='cuda'))
        flags={};mask=torch.ones(768,dtype=torch.bool,device='cuda');mask[J]=False
        def hook(m,a,o):
            changed=hook_for(J,reg)(m,a,o)
            flags['unchanged640']=torch.equal(o[:,:,mask],changed[:,:,mask])
            flags['min_gelu']=float(changed[:,:,J].min().detach())
            return changed
        handle=net.bloki[-1].mlp.act.register_forward_hook(hook)
        real,_=net(x);handle.remove()
        cheap=tail_fast(net,J,H[ids],Q[ids],reg)
        torch.testing.assert_close(cheap,real,rtol=2e-5,atol=2e-5)
        target=tuple(t[ids] for t in T);params=list(reg.parameters())
        ga=torch.autograd.grad(losses(real,target).mean(),params)
        gb=torch.autograd.grad(losses(cheap,target).mean(),params)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
        assert all(torch.isfinite(g).all() for g in gb) and sum(float(g.abs().sum()) for g in gb)>0
        assert flags['unchanged640'] and flags['min_gelu']>=-.169973
        # No batch-dependent rule: same token responds identically alone/in group.
        with torch.no_grad():torch.testing.assert_close(reg(H[ids[:1]]),reg(H[ids])[:1],rtol=1e-5,atol=1e-6)
        results.append(dict(kind=kind,parameters=sum(p.numel() for p in params),forward_error=float((cheap-real).abs().max().detach()),
                            gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb)),**flags))
    assert state_hash(net)==prov['weights_hash']
    result=dict(status='pass',models=results,provenance=prov,no_test_labels_in_training=True)
    js(OUT/'PREFLIGHT.json',result);print('PREFLIGHT PASS',json.dumps(results),flush=True)
    return result

def predict(net,J,H,Q,reg,ids):
    p=[]
    with torch.no_grad():
        for lo in range(0,len(ids),16):
            ix=ids[lo:lo+16];p.append(tail_fast(net,J,H[ix],Q[ix],reg).cpu())
    return torch.cat(p)

def bootstrap(o1,o0,T,gt,rows,ids,n=500):
    # Paired resampling of complete films. Duplicated draws receive new image IDs.
    b1,s1=dekoduj(o1);b0,s0=dekoduj(o0)
    p1=list(zip(b1.numpy(),s1.numpy()));p0=list(zip(b0.numpy(),s0.numpy()));gg=[gt[i] for i in ids]
    films=sorted({rows[i]['film'] for i in ids});local={f:[j for j,i in enumerate(ids) if rows[i]['film']==f] for f in films}
    t=tuple(v[ids].cpu() for v in T);dl=(losses(o1,t)-losses(o0,t)).numpy()
    rng=np.random.default_rng(SEED+71);out=[]
    for k in range(n):
        draw=rng.choice(films,len(films),replace=True);ix=[j for f in draw for j in local[f]]
        aa=[p1[j] for j in ix];bb=[p0[j] for j in ix];g=[gg[j] for j in ix]
        dd=[ap_at(aa,g,float(th))-ap_at(bb,g,float(th)) for th in np.arange(.5,.951,.05)]
        out.append([dd[0],float(np.mean(dd)),float(np.mean([dl[local[f]].mean() for f in draw]))])
        if (k+1)%100==0:print('BOOTSTRAP',k+1,flush=True)
    a=np.asarray(out)
    return {k:np.quantile(a[:,j],[.025,.975]).tolist() for j,k in enumerate(('AP50','AP50_95','loss'))}

def report(result):
    lines=['# Wspólna regulacja J128: wynik pilota','',
           'Zamrożony ViT k1000, stałe J128 z k500. Jeden wspólny regulator na wszystkie obrazy i tokeny. Parametry wybierane na val; żadnych etykiet testu w interwencji.','',
           '| Metoda | Split | AP50 | AP50:95 | Loss (po filmach) |','|---|---|---:|---:|---:|']
    for name,parts in result['metrics'].items():
        for split,m in parts.items():lines.append(f"| {name} | {split} | {m['AP50']:.6f} | {m['AP50_95']:.6f} | {m['loss']:.6f} |")
    lines+=['',f"Wybór ogólny: {result['selected']['config']['name']}, epoka {result['selected']['epoch']}.",
            f"Najlepszy statyczny regulator: {result['selected_static']['config']['name']}, epoka {result['selected_static']['epoch']}.",
            f"Bramka pilota (AP50 i AP50:95 test > baseline): {result['gate_passed']}.",
            f"Czas całości: {result['seconds']:.1f} s; urządzenie: {result['provenance']['device']}.",'',
            '## Znaczenie i ograniczenia','',
            'Stały regulator to skale/przesunięcia wspólne dla wszystkich danych. Adapter jest osobną rodziną: wspólna funkcja zależna od wartości tokenu, bez identyfikatora obrazu i GT. Wagi bazowego ViT i pozostałe640 kanałów niezmienione. Wybrane wartości regulatora zapisano w modelach; KANALY_STATYCZNE.csv opisuje jego skale i przesunięcia.',
            'Nie jest to ten sam rodzaj wyniku co oracle AP50=0.884748 z etykietą każdego obrazu. Nie gwarantuje optimum globalnego, pełnej tożsamości narzędzia ani istnienia zdjęcia odtwarzającego dany latent. Historyczne grupy były wcześniej oglądane; to pilot jednego seeda, nie niezależne potwierdzenie końcowe.','',
            '## Kontrole','',
            'PREFLIGHT.json: zgodność szybkiego tail z pełnym forwardem i gradientami, zakres GELU, niezmienność640 kanałów, rozłączność filmów. VERIFICATION.json: kontrola wybranej interwencji w pełnym forwardzie na nowych przykładach, hash wag i buforów.',
            'SELECTION.json powstaje przed obliczeniem metryk testu. Historia wszystkich konfiguracji i wyniki negatywne zachowane. Przedziały ufności resamplują całe filmy i są warunkowe względem już wybranej konfiguracji; nie usuwają efektu wielokrotnego oglądania historycznego benchmarku.']
    if 'paired_film_CI95' in result:
        lines+=['','## Przedziały ufności różnicy względem baseline','',
                '| Regulator | Metryka | Dolna granica 95% | Górna granica 95% |','|---|---|---:|---:|']
        for tag,intervals in result['paired_film_CI95'].items():
            for metric,(lo,hi) in intervals.items():lines.append(f'| {tag} | {metric} | {lo:.6f} | {hi:.6f} |')
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI.md').write_text(text,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','sweep'],required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();net,rows,sets,J,T,gt,H,Q,base,mu,sd,prov=load()
    js(OUT/'PROGRESS.json',dict(status='loaded',phase=args.phase,provenance=prov))
    pre=checks(net,rows,sets,J,T,gt,H,Q,base,mu,sd,prov)
    if args.phase=='preflight':return
    if (OUT/'RESULT.json').exists():raise RuntimeError('Completed output exists; refusing to overwrite')
    executed=OUT/'EXECUTED_SOURCE';executed.mkdir(exist_ok=True)
    for file in list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(HERE.glob('*.ps1'))+[HERE/'WYNIKI.json']:shutil.copy2(file,executed/file.name)
    js(OUT/'CONFIG.json',dict(configs=CONFIGS,epochs=120,batch=16,seed=SEED,provenance=prov,
                             selection='max val AP50_95, then AP50, then min film-weighted loss',
                             executed_sources={f.name:sha(f) for f in executed.iterdir()}))
    base_val=metrics(base[sets['val']],T,gt,rows,sets['val'])
    train_ids=sets['train'];val_ids=sets['val'];wmap=torch.zeros(len(rows),device='cuda');wmap[train_ids]=torch.as_tensor(film_weights(rows,train_ids),device='cuda')*len(train_ids)
    candidates=[]
    for ci,cfg in enumerate(CONFIGS):
        torch.manual_seed(SEED);reg=Regulator(cfg['kind'],mu,sd);opt=torch.optim.Adam(reg.parameters(),lr=cfg['lr'])
        best=dict(epoch=0,metrics=base_val);beststate=save_state(reg);history=[dict(epoch=0,val=base_val)]
        rng=np.random.default_rng(SEED);t0=time.perf_counter();folder=OUT/cfg['name'];folder.mkdir(exist_ok=True)
        for epoch in range(1,121):
            lr=cfg['lr']*(.1+.9*(1+math.cos(math.pi*(epoch-1)/120))/2)
            for g in opt.param_groups:g['lr']=lr
            order=rng.permutation(train_ids).tolist();ll=[]
            for lo in range(0,len(order),16):
                ids=order[lo:lo+16];h=H[ids];o=tail_fast(net,J,h,Q[ids],reg)
                loss=(losses(o,tuple(t[ids] for t in T))*wmap[ids]).mean()+cfg['penalty']*reg.delta(h).square().mean()
                assert torch.isfinite(loss);opt.zero_grad();loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reg.parameters())
                torch.nn.utils.clip_grad_norm_(reg.parameters(),5.);opt.step();ll.append(float(loss.detach()))
            if epoch%10==0:
                pred=predict(net,J,H,Q,reg,val_ids);m=metrics(pred,T,gt,rows,val_ids)
                row=dict(epoch=epoch,train_objective=float(np.mean(ll)),val=m,lr=lr,seconds=time.perf_counter()-t0);history.append(row)
                if key(m)>key(best['metrics']):best=dict(epoch=epoch,metrics=m);beststate=save_state(reg)
                js(folder/'HISTORY.json',history);torch.save(dict(config=cfg,epoch=best['epoch'],state=beststate,channels=J.cpu(),provenance=prov),folder/'model_best.pt')
                js(OUT/'PROGRESS.json',dict(status='training',configuration=ci+1,configurations=len(CONFIGS),name=cfg['name'],epoch=epoch,epochs=120,best_validation=best,seconds=time.perf_counter()-start))
                print('VAL',cfg['name'],epoch,json.dumps(m),flush=True)
        candidate=dict(config=cfg,epoch=best['epoch'],val=best['metrics'],seconds=time.perf_counter()-t0,parameters=sum(p.numel() for p in reg.parameters()))
        candidates.append(candidate);js(folder/'CANDIDATE.json',candidate)
    selected=max(candidates,key=lambda c:key(c['val']))
    selected_static=max([c for c in candidates if c['config']['kind']!='adapter'],key=lambda c:key(c['val']))
    selection=dict(status='frozen_before_test',selected=selected,selected_static=selected_static,candidates=candidates,test_used_for_selection=False)
    js(OUT/'SELECTION.json',selection)
    all_ids=list(range(len(rows)));predictions={'baseline':base};models={};verification=[]
    for tag,candidate in [('selected',selected),('static',selected_static)]:
        cfg=candidate['config'];reg=Regulator(cfg['kind'],mu,sd);ck=torch.load(OUT/cfg['name']/'model_best.pt',map_location='cpu',weights_only=False);reg.load_state_dict(ck['state']);models[tag]=reg
        predictions[tag]=predict(net,J,H,Q,reg,all_ids)
        ix=[sets['train'][0],sets['val'][0],sets['test'][0]]
        with torch.no_grad():
            handle=net.bloki[-1].mlp.act.register_forward_hook(hook_for(J,reg));actual,_=net(torch.stack([image_tensor(rows[i]) for i in ix]));handle.remove()
            expected=predictions[tag][ix].cuda();torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-5)
            verification.append(dict(tag=tag,max_abs_error=float((actual-expected).abs().max())))
    metrics_all={tag:{s:metrics(pred[ids],T,gt,rows,ids) for s,ids in {**sets,'all':all_ids}.items()} for tag,pred in predictions.items()}
    static=models['static'];sc=(1+static.scale.detach().cpu().numpy()) if hasattr(static,'scale') else np.ones(128)
    bb=(static.bias*static.sd).detach().cpu().numpy()
    with (OUT/'KANALY_STATYCZNE.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['channel0','scale_pre_gelu','shift_pre_gelu'])
        writer.writerows(zip(J.cpu().numpy().tolist(),sc.tolist(),bb.tolist()))
    np.savez_compressed(OUT/'DETECTIONS.npz',**{k:v.numpy() for k,v in predictions.items()},channels=J.cpu().numpy(),source_indices=np.array([r['index'] for r in rows]))
    shutil.copy2(OUT/selected['config']['name']/'model_best.pt',OUT/'model_selected.pt')
    shutil.copy2(OUT/selected_static['config']['name']/'model_best.pt',OUT/'model_static.pt')
    unchanged=state_hash(net)==prov['weights_hash'];assert unchanged
    js(OUT/'VERIFICATION.json',dict(selected_full_forward=verification,weights_buffers_unchanged=unchanged,preflight=pre))
    result=dict(status='evaluated_bootstrap_pending',ziarno=SEED,provenance=prov,selected=selected,selected_static=selected_static,metrics=metrics_all,
                gate_passed=all(metrics_all['selected']['test'][k]>metrics_all['baseline']['test'][k] for k in ('AP50','AP50_95')),
                test_used_for_selection=False,weights_buffers_unchanged=unchanged,seconds=time.perf_counter()-start)
    # Save usable selected models and the report before optional resampling.
    js(OUT/'RESULT.json',result);report(result)
    js(OUT/'PROGRESS.json',dict(status='bootstrap',selected=selected['config']['name'],metrics=metrics_all,seconds=time.perf_counter()-start))
    for tag in ('selected','static'):
        result.setdefault('paired_film_CI95',{})[tag]=bootstrap(predictions[tag][sets['test']],base[sets['test']],T,gt,rows,sets['test'])
        js(OUT/'RESULT.json',result)
    result['seconds']=time.perf_counter()-start;result['status']='complete';js(OUT/'RESULT.json',result);report(result)
    zapisz_wynik(__file__,'search_k1000',result)
    js(OUT/'PROGRESS.json',dict(status='complete',selected=selected['config']['name'],metrics=metrics_all,seconds=result['seconds']))
    print('COMPLETE',json.dumps(metrics_all),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        js(OUT/'FAILURE.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()))
        traceback.print_exc();raise
