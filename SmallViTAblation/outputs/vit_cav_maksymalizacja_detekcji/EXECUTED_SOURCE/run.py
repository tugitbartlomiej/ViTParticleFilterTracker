"""Czy wspólny alpha poprawia AP50? Brak testowej przewagi obala korzyść detekcji."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[2]/'outputs/vit_cav_maksymalizacja_detekcji/mplconfig')
import argparse,ast,csv,hashlib,json,platform,shutil,time,traceback
import numpy as np,torch,cv2,timm,scipy
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from dane import *
from model_snapshot import MalyViT,dekoduj
from metryki_source import iou,ap50 as source_ap50
TCAV=ROOT/'outputs/vit_tcav_tozsamosc_narzedzia'
OLD=ROOT/'experiments/maly_vit_tozsamosc_czasteczki_pary_400'
GRID=np.array([-.5,-.25,-.1,-.05,-.02,-.01,-.005,0,.005,.01,.02,.05,.1,.25,.5])
THRESHOLDS=np.round(np.arange(.5,.96,.05),2)
MAIN=['full768','J128','historical_LDA'];RSEEDS=[202610051,202610052,202610053]
BOOT=2000;START=time.time();GELU_MIN=-.1699712075
def progress(stage,**kw):
    js(OUT/'PROGRESS.json',dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kw));print(stage,kw,flush=True)
def register_research_family():
    d=dict(rodzina=HERE.name,klasa='badawcza',pytanie='Czy jeden alpha na stałym CAV poprawia rzeczywiste AP50?',role=['BIEG','CAV','LOCK','VAL','TEST'],wymagane=['tag','ziarno'],kontrole=['random768','randomJ128'],uwagi='Source-compatible101point AP50; oracleGTROI; val selection, frozen test; scoped registry only')
    if (HERE/'WYNIKI.json').exists():assert read(HERE/'WYNIKI.json')['rodzina']==HERE.name
    else:js(HERE/'WYNIKI.json',d)
    js(OUT/'REGISTRATION.json',dict(**d,registered_by='register_research_family',shared_registry_write=False,plan_sha256=sha(HERE/'PLAN.md')))
def input_image(pair):
    im=cv2.imread(str(PAIRS/pair['original']));assert im is not None
    return cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
def init_net():
    torch.set_num_threads(4);threadpool_limits(4);cv2.setNumThreads(1);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu'
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>700*1024**2:
        torch.cuda.set_per_process_memory_fraction(.12);device='cuda'
    net=MalyViT().to(device).eval();ck=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000;net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False)
    return net,device
def forward(net,device,pair):
    x=torch.from_numpy(input_image(pair)).permute(2,0,1)[None].float().to(device)/255
    cache={};b=net.bloki[-1]
    h1=b.mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach().clone()))
    h2=b.register_forward_hook(lambda m,a,o:cache.update(q=o.detach().clone()))
    with torch.no_grad():o,_=net(x)
    h1.remove();h2.remove();return x,o,cache['z'],cache['q']
def weights_hash(net):return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
def lift(w):return w/np.linalg.norm(w)
def shifted(net,q,w,v,alphas):
    # Last block output q already includes fc2 bias and residual; perturb only the linear fc2 response.
    b=net.bloki[-1];vv=torch.tensor(v,dtype=q.dtype,device=q.device)
    u=b.ls2(F.linear(vv,b.mlp.fc2.weight,None))
    dl=torch.tensor(lift(w),dtype=q.dtype,device=q.device)[None,:,None]*u[None,None,:]
    aa=torch.tensor(alphas,dtype=q.dtype,device=q.device)[:,None,None]
    with torch.no_grad():o=net.glowa(net.norm(q+aa*dl)).reshape(len(alphas),20,30,5)
    assert torch.isfinite(o).all();return o
def bbox(row):
    x,y,w,h=row['bbox_source_480'];return np.array([x,y,x+w,y+h],np.float32)
def predictions(o,g):
    boxes,scores=dekoduj(o.detach().cpu());boxes=boxes.numpy();scores=scores.numpy();ov=iou(boxes,g)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    return dict(boxes=boxes,scores=scores,ious=ov)
def AP_all(scores,ious):
    # Source greedy matching with one GT/image. First qualifying local prediction is the unique TP.
    hit=ious[:,:,None]>=THRESHOLDS[None,None,:]
    tp=hit&(np.cumsum(hit,axis=1)==1)
    order=np.argsort(-scores.reshape(-1),kind='stable');ctp=np.cumsum(tp.reshape(-1,len(THRESHOLDS))[order],axis=0)
    rec=ctp/max(len(scores),1);prec=ctp/np.arange(1,len(order)+1)[:,None]
    prec=np.maximum.accumulate(prec[::-1],axis=0)[::-1];out=[]
    for k in range(len(THRESHOLDS)):
        ix=np.searchsorted(rec[:,k],np.linspace(0,1,101),side='left');safe=np.minimum(ix,len(order)-1)
        out.append(float(np.where(ix<len(order),prec[safe,k],0).sum()/101))
    return np.array(out)
def metrics(pred):
    aps=AP_all(pred['scores'],pred['ious']);out=dict(AP50=float(aps[0]),AP75=float(aps[5]),AP50_95=float(aps.mean()),AP_by_IoU=aps,mean_top1_IoU=float(pred['ious'][:,0].mean()),mean_confidence=float(pred['scores'][:,0].mean()))
    for t in [.5,.75]:
        hits=((pred['scores']>=.5)&(pred['ious']>=t)).any(1);n=(pred['scores']>=.5).sum();p=float(hits.sum()/max(n,1));r=float(hits.mean())
        out[f'precision{t}']=p;out[f'recall{t}']=r;out[f'F1{t}']=2*p*r/max(p+r,1e-12)
    return out
def source_equivalence(pred,gt):
    actual=source_ap50([(b,s) for b,s in zip(pred['boxes'],pred['scores'])],[g[None] for g in gt]);err=abs(actual-metrics(pred)['AP50']);assert err<1e-12;return err
def synthetic_metric_check():
    perfect=dict(boxes=np.array([[[0,0,2,2]]],np.float32),scores=np.array([[.8]]),ious=np.array([[1.]]))
    assert metrics(perfect)['AP50']==1.
    assert AP_all(np.array([[.9,.8]]),np.array([[0.,1.]]))[0]==.5
    assert AP_all(np.array([[.8]]),np.array([[0.]]))[0]==0.
    # Stable ties and missing match test match original implementation exactly.
    boxes=np.array([[[0,0,2,2],[5,5,6,6]],[[0,0,1,1],[0,0,2,2]]],np.float32);g=np.array([[0,0,2,2]]*2,np.float32);s=np.array([[.5,.5],[.5,.5]])
    assert source_equivalence(dict(boxes=boxes,scores=s,ious=np.array([iou(b,gg) for b,gg in zip(boxes,g)])),g)<1e-12
def source_inputs():
    rows,pairs,w,J,ids=load();meta=read(OLD/'POMIAR.json');assert sha(CKPT)==meta['checkpoint_sha256'];assert sha(HERE/'model_snapshot.py')==meta['model_source_sha256']
    assert sha(PAIRS/'manifest.json')==meta['manifest_sha256'];assert sha(PAIRS/'TOKEN_MASKI.npz')==meta['token_masks_sha256']
    cache=dict(np.load(OLD/'TOKENY_PARY.npz'));assert sha(OLD/'TOKENY_PARY.npz')==meta['npz_sha256']
    norms=[]
    for i in ids['train']:
        a,b=cache['offsets'][i:i+2];norms.append(np.linalg.norm(cache['states'][a:b,0]))
    T=float(np.median(norms));directions={k:np.load(TCAV/f'CAV_{k}.npz')['unit'].astype(np.float32) for k in MAIN}
    for k,v in directions.items():assert len(v)==768 and abs(np.linalg.norm(v)-1)<1e-6
    assert (directions['J128'][np.setdiff1d(np.arange(768),J)]==0).all()
    for support,axes in [('768',np.arange(768)),('J128',J)]:
        for seed in RSEEDS:
            v=np.zeros(768,np.float32);v[axes]=np.random.default_rng(seed).normal(size=len(axes));v/=np.linalg.norm(v);directions[f'random{support}_s{seed}']=v
    np.savez_compressed(OUT/'CAV_FIXED.npz',**directions,channels0_J128=J,grid_r=GRID,alphas=GRID*T)
    with (OUT/'DIRECTIONS_768.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['channel0','channel1','in_J128']+list(directions))
        for j in range(768):writer.writerow([j,j+1,int(j in J)]+[float(v[j]) for v in directions.values()])
    with (OUT/'DIRECTIONS_J128.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['channel0','channel1','unit_CAV_J128'])
        for j in J:writer.writerow([int(j),int(j+1),float(directions['J128'][j])])
    hashes={str(p):sha(p) for p in [CKPT,SPLIT,PAIRS/'manifest.json',PAIRS/'PARY.json',PAIRS/'TOKEN_MASKI.npz',OLD/'TOKENY_PARY.npz']+[TCAV/f'CAV_{k}.npz' for k in MAIN]}
    js(OUT/'CALIBRATION.json',dict(train_median_ROI_Frobenius=T,train_ROI_norms=norms,grid_r=GRID,grid_alpha=GRID*T,max_delta_Frobenius=.5*T,per_image_fitting=False,per_image_cap=False,train_only=True,source_hashes=hashes))
    js(OUT/'SPLITS_GT.json',dict(indices=ids,records=[dict(source_index=i,frame_id=r['frame_id'],film=r['video_id'],split=r['tcav_split'],GT_xyxy=bbox(r),original=str(PAIRS/pairs[i]['original'])) for i,r in enumerate(rows)],split_sha256=sha(SPLIT)))
    return rows,pairs,w,J,ids,cache,T,directions,hashes
def smoke(net,device,rows,pairs,w,ids,T,directions):
    diagnostics=[]
    for i in ids['val'][:3]:
        x,o,z,q=forward(net,device,pairs[i])
        for name in MAIN+[f'random768_s{RSEEDS[0]}',f'randomJ128_s{RSEEDS[0]}']:
            v=directions[name];d=torch.tensor(lift(w[i]),device=device)[None,:,None]*torch.tensor(v,device=device)[None,None,:]
            assert abs(float(d.norm())-1)<1e-5
            for alpha in [-.5*T,0,.5*T]:
                h=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,out:out+alpha*d)
                with torch.no_grad():full,_=net(x)
                h.remove();fast=shifted(net,q,w[i],v,[alpha]);maxerr=float((fast-full).abs().max())
                torch.testing.assert_close(fast,full,rtol=5e-5,atol=1e-5)
                a=predictions(full,bbox(rows[i]));b=predictions(fast,bbox(rows[i]));boxerr=float(np.max(abs(a['boxes']-b['boxes'])));scoreerr=float(np.max(abs(a['scores']-b['scores'])))
                assert boxerr<.01 and scoreerr<1e-5
                diagnostics.append(dict(source_index=int(i),method=name,alpha=alpha,max_map_error=maxerr,max_decoded_bbox_error=boxerr,max_score_error=scoreerr))
        progress('smoke_processing',source_index=int(i),processed=len(diagnostics),device=device)
    js(OUT/'SMOKE.json',dict(status='passed',tests=diagnostics,device=device,metric_source101_equivalence=True,maximum_map_error=max(d['max_map_error'] for d in diagnostics),maximum_bbox_error=max(d['max_decoded_bbox_error'] for d in diagnostics),gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
def validation(net,device,rows,pairs,w,ids,cache,T,directions,hashes):
    predictions_all={k:dict(boxes=[],scores=[],ious=[]) for k in directions};baseline={k:[] for k in ['boxes','scores','ious']};feasibility={k:[] for k in directions};qs=[]
    for count,i in enumerate(ids['val']):
        x,o,z,q=forward(net,device,pairs[i]);a,b=cache['offsets'][i:i+2];tids=cache['token_ids'][a:b].astype(int)
        np.testing.assert_allclose(z[0,tids].cpu().numpy(),cache['states'][a:b,0],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(o.cpu().numpy()[0],cache['outputs'][i,0],rtol=2e-5,atol=3e-5)
        p=predictions(o,bbox(rows[i]));qs.append(q.cpu().numpy()[0])
        for key in baseline:baseline[key].append(p[key][0])
        zn=z[0,tids].cpu().numpy();ll=lift(w[i])[tids]
        for name,v in directions.items():
            p=predictions(shifted(net,q,w[i],v,GRID*T),bbox(rows[i]))
            for key in p:predictions_all[name][key].append(p[key])
            invalid=[float(((zn+alpha*ll[:,None]*v[None,:])<GELU_MIN-1e-6).mean()) for alpha in GRID*T];feasibility[name].append(invalid)
        progress('validation_processing',processed=count+1,total=len(ids['val']),source_index=int(i),device=device)
    baseline={k:np.array(v) for k,v in baseline.items()};base=metrics(baseline);assert source_equivalence(baseline,[bbox(rows[i]) for i in ids['val']])<1e-12
    np.savez_compressed(OUT/'VALIDATION_BASELINE.npz',source_indices=ids['val'],**baseline,last_block_q=np.array(qs))
    curves={};chosen={}
    for name,p in predictions_all.items():
        p={k:np.array(v) for k,v in p.items()};np.savez_compressed(OUT/f'VAL_PRED_{name}.npz',source_indices=ids['val'],grid_r=GRID,alpha=GRID*T,**p)
        vals=[metrics({k:v[:,j] for k,v in p.items()}) for j in range(len(GRID))]
        zero=int(np.flatnonzero(GRID==0)[0]);assert abs(vals[zero]['AP50']-base['AP50'])<1e-12
        j=max(range(len(GRID)),key=lambda j:(vals[j]['AP50'],-abs(GRID[j]),-GRID[j]))
        assert vals[j]['AP50']>=base['AP50']-1e-12
        curves[name]=[dict(r=float(GRID[j]),alpha=float(GRID[j]*T),metrics=vals[j],mean_ROI_fraction_below_GELU_floor=float(np.mean(feasibility[name],axis=0)[j])) for j in range(len(GRID))]
        chosen[name]=dict(r=float(GRID[j]),alpha=float(GRID[j]*T),grid_index=j,validation_metrics=vals[j],grid_budget=15)
    best={}
    for support in ['768','J128']:
        names=[f'random{support}_s{s}' for s in RSEEDS];best[support]=max(names,key=lambda k:(chosen[k]['validation_metrics']['AP50'],-RSEEDS.index(int(k.split('_s')[-1]))))
    js(OUT/'VALIDATION_CURVES.json',dict(primary='source101_AP50',baseline=base,curves=curves,chosen=chosen))
    lock=dict(status='locked_before_test',lock_unix=time.time(),primary='source101_AP50_single_annotated_class',selection='max validation AP50; ties min abs r then smaller r',chosen=chosen,best_random_by_val=best,grid_r=GRID,T_train_only=T,direction_source_hashes=hashes,validation_curve_sha256=sha(OUT/'VALIDATION_CURVES.json'),test_accessed=False,refinement=False)
    js(OUT/'LOCK.json',lock);progress('locked_before_test',chosen_r={k:v['r'] for k,v in chosen.items()},best_random=best,lock_sha256=sha(OUT/'LOCK.json'))
    return lock,baseline
def test_once(net,device,rows,w,ids,cache,directions,lock):
    locksha=sha(OUT/'LOCK.json');assert lock['test_accessed'] is False
    # No reading/evaluating test activations until the immutable val settings are persisted.
    states=dict(np.load(TCAV/'HELDOUT_STATES.npz'));assert np.array_equal(states['test_indices'],ids['test'])
    q=states['last_block_output_q'];z=states['original_GELU'];preds={k:dict(boxes=[],scores=[],ious=[]) for k in ['baseline']+list(directions)};maps={k:[] for k in preds};records=[]
    for count,i in enumerate(ids['test']):
        qq=torch.from_numpy(q[count:count+1]).to(device);gt=bbox(rows[i]);base=shifted(net,qq,w[i],directions['full768'],[0])
        np.testing.assert_allclose(base[0].cpu().numpy(),cache['outputs'][i,0],rtol=2e-5,atol=3e-5)
        p=predictions(base,gt)
        for key in p:preds['baseline'][key].append(p[key][0])
        maps['baseline'].append(base[0].cpu().numpy());tids=np.flatnonzero(w[i]);norm=float(np.linalg.norm(z[count,tids]))
        for name,v in directions.items():
            alpha=lock['chosen'][name]['alpha'];oo=shifted(net,qq,w[i],v,[alpha]);pp=predictions(oo,gt)
            for key in pp:preds[name][key].append(pp[key][0])
            maps[name].append(oo[0].cpu().numpy());changed=z[count,tids]+alpha*lift(w[i])[tids,None]*v[None,:]
            records.append(dict(source_index=int(i),film=rows[i]['video_id'],method=name,alpha=alpha,r=lock['chosen'][name]['r'],delta_Frobenius=abs(alpha),original_ROI_norm=norm,relative_norm=abs(alpha)/norm,ROI_fraction_below_GELU_floor=float((changed<GELU_MIN-1e-6).mean()),GT_xyxy=gt))
        progress('test_locked_processing',processed=count+1,total=len(ids['test']),source_index=int(i),lock_sha256=locksha)
    preds={name:{k:np.array(v) for k,v in pred.items()} for name,pred in preds.items()}
    gt=[bbox(rows[i]) for i in ids['test']];assert source_equivalence(preds['baseline'],gt)<1e-12
    for name in preds:np.savez_compressed(OUT/f'TEST_PRED_{name}.npz',source_indices=ids['test'],GT_xyxy=gt,**preds[name],output_maps=np.array(maps[name]),alpha=0 if name=='baseline' else lock['chosen'][name]['alpha'])
    js(OUT/'TEST_INTERVENTIONS.json',records);assert sha(OUT/'LOCK.json')==locksha
    js(OUT/'TEST_EVALUATION.json',dict(evaluated_once=True,test_pairs=85,test_films=24,configuration_count=len(preds),lock_sha256=locksha,started_after_lock=True,test_grid_search=False,refit=False))
    return preds,records
def grouped_bootstrap(rows,ids,preds,lock):
    test=ids['test'];films=sorted({rows[i]['video_id'] for i in test});groups=[np.array([j for j,i in enumerate(test) if rows[i]['video_id']==f]) for f in films]
    rng=np.random.default_rng(SEED+321);draws=rng.integers(0,len(groups),(BOOT,len(groups)));draws_ix=[np.concatenate([groups[k] for k in draw]) for draw in draws]
    point={k:metrics(p) for k,p in preds.items()};samples={};keys=['AP50','AP75','AP50_95','top1_IoU_film','recall50_film']
    for name,p in preds.items():
        mean_iou=np.array([p['ious'][g,0].mean() for g in groups]);hit=((p['scores']>=.5)&(p['ious']>=.5)).any(1);filmhit=np.array([hit[g].mean() for g in groups]);values=[]
        for b,ix in enumerate(draws_ix):
            aps=AP_all(p['scores'][ix],p['ious'][ix]);values.append([aps[0],aps[5],aps.mean(),mean_iou[draws[b]].mean(),filmhit[draws[b]].mean()])
        samples[name]=np.array(values);point[name]['top1_IoU_film']=float(mean_iou.mean());point[name]['recall50_film']=float(filmhit.mean())
        progress('film_bootstrap',method=name,resamples=BOOT)
    out={};base=samples['baseline']
    for name in preds:
        support='J128' if 'J128' in name else '768';best=lock['best_random_by_val'][support]
        out[name]=dict(metrics=point[name],CI_columns=keys,film_bootstrap_CI95=np.quantile(samples[name],[.025,.975],axis=0),delta_vs_baseline={k:point[name][k]-point['baseline'][k] for k in keys},paired_delta_baseline_CI95=np.quantile(samples[name]-base,[.025,.975],axis=0),best_control_chosen_val=best,delta_vs_best_control={k:point[name][k]-point[best][k] for k in keys},paired_delta_best_control_CI95=np.quantile(samples[name]-samples[best],[.025,.975],axis=0),probability_boot_delta_AP50_positive=float((samples[name][:,0]>base[:,0]).mean()))
    np.savez_compressed(OUT/'BOOTSTRAP_SAMPLES.npz',films=np.array(films),film_draws=draws,columns=np.array(keys),**samples)
    return out
def plots_report(rows,pairs,ids,lock,valbaseline,preds,summary,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    curves=read(OUT/'VALIDATION_CURVES.json');fig,axs=plt.subplots(1,2,figsize=(13,5))
    for name,c in curves['curves'].items():
        ax=axs[1] if 'J128' in name else axs[0];ax.plot([r['r'] for r in c],[r['metrics']['AP50']*100 for r in c],marker='.' if name.startswith('random') else 'o',ls='--' if name.startswith('random') else '-',label=name)
        chosen=lock['chosen'][name];ax.scatter([chosen['r']],[100*chosen['validation_metrics']['AP50']],s=50)
    for ax in axs:ax.axhline(100*curves['baseline']['AP50'],color='black',lw=1);ax.set_xlabel('r; alpha = r * train median ROI norm');ax.set_ylabel('validation source101 AP50 [%]');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Shared scalar search on validation only; fixed CAV and detector');fig.tight_layout();fig.savefig(OUT/'VAL_AP50_CURVES.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));names=list(preds);pts=[100*summary[k]['metrics']['AP50'] for k in names];bounds=np.array([summary[k]['film_bootstrap_CI95'] for k in names])[:,:,0]*100
    ax.errorbar(np.arange(len(names)),pts,yerr=[np.maximum(0,np.array(pts)-bounds[:,0]),np.maximum(0,bounds[:,1]-np.array(pts))],fmt='o',capsize=4);ax.set_xticks(np.arange(len(names)),names,rotation=30,ha='right');ax.set_ylabel('test source101 AP50 [%] + film CI95');fig.tight_layout();fig.savefig(OUT/'TEST_AP50.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(4,4,figsize=(16,10));chosen_imgs=ids['test'][:4]
    for rowidx,i in enumerate(chosen_imgs):
        im=input_image(pairs[i]);g=bbox(rows[i])
        for col,name in enumerate(['baseline']+MAIN):
            ax=axs[rowidx,col];ax.imshow(im);b=preds[name]['boxes'][rowidx,0];ax.add_patch(Rectangle((g[0],g[1]),g[2]-g[0],g[3]-g[1],fill=False,edgecolor='yellow',lw=1.5));ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='lime',lw=1.5));ax.set_xlim(0,480);ax.set_ylim(320,0);ax.axis('off');ax.set_title(f'{name}\nIoU {preds[name]["ious"][rowidx,0]:.4f}; score {preds[name]["scores"][rowidx,0]:.4f}',fontsize=9)
    fig.suptitle('First four heldout images, no improvement-based example selection; GT yellow, top1 green');fig.tight_layout();fig.savefig(OUT/'BEFORE_AFTER_EXAMPLES.jpg',dpi=120);plt.close(fig)
    baseline=summary['baseline']['metrics'];lines=['# Wspólny skalar CAV a rzeczywista detekcja','', 'Primary objective: źródłowe AP50 jednej anotowanej klasy, interpolacja 101-punktowa. To diagnostyka oracle ROI, nie wdrażalny detektor. Skalar wybrano po75 obrazach/23filmach walidacji, zapisano LOCK, następnie jednokrotnie oceniono85 obrazów/24filmów testu. Wagi i kierunki pozostały zamrożone.','', f'Baseline test AP50={baseline["AP50"]:.8f}, AP75={baseline["AP75"]:.8f}, AP50:.95={baseline["AP50_95"]:.8f}.','', '| Kierunek | r val | alpha | val AP50 | test AP50 | delta AP50 | CI95 delta AP50 |','|---|---:|---:|---:|---:|---:|---|']
    for name in lock['chosen']:
        c=lock['chosen'][name];s=summary[name];lo,hi=s['paired_delta_baseline_CI95'][:,0];lines.append(f'|{name}|{c["r"]:.4g}|{c["alpha"]:.6g}|{c["validation_metrics"]["AP50"]:.6f}|{s["metrics"]["AP50"]:.6f}|{s["delta_vs_baseline"]["AP50"]:.6f}|[{lo:.6f}, {hi:.6f}]|')
    lines+=['','## Wynik i podłogi']
    for name in MAIN:
        s=summary[name];lo,hi=s['paired_delta_baseline_CI95'][:,0];b=s['best_control_chosen_val'];blo,bhi=s['paired_delta_best_control_CI95'][:,0]
        lines.append(f'{name}: testowa zmiana AP50={s["delta_vs_baseline"]["AP50"]:+.8f}, CI[{lo:+.8f},{hi:+.8f}]. Względem najlepszego po val dopasowanego random ({b}): {s["delta_vs_best_control"]["AP50"]:+.8f}, CI[{blo:+.8f},{bhi:+.8f}]. '+('Przedział wyklucza zero po stronie poprawy baseline.' if lo>0 else 'Nie wykazano dodatniej przewagi nad baseline z CI wykluczającym zero.'))
    lines+=['','## Aparat i ograniczenia','Wszystkie9 kierunków przeszły tę samą15-punktową siatkę. Alpha0 zapobiega pogorszeniu wybranej walidacji, lecz nie gwarantuje poprawy testu. Bestrandom wybierano tylko po val spośród3seedów, nie po teście. Film-bootstrap2000 na zapisanych predykcjach utrzymuje parowanie zmian; CI warunkowe na kierunki i wybór, bez ponownej selekcji.','', 'Norma lift=1, wspólny alpha=r*trainmedianROI. Żadnego fittingu alpha do obrazu, clippingu, poprawiania wag ani kierunków. Fastforward obejmuje fc2, residual, norm i densehead, zgodność z hookiem GELU i bbox sprawdzono na3 walidacyjnych obrazach.','', 'Punkty poniżej dolnej granicy GELU zapisano jako diagnostykę; interwencje nie muszą być realizowalne przez obraz. Oracle GT ROI i tylko jedna anotowana ramka mogą zawyżać użyteczność i zaniżać precision dla innych obecnych narzędzi.','', 'To najlepszy punkt w zadeklarowanej siatce, nie globalne maksimum. Oryginały były w puli treningowej ViT, J128 historycznie wybrano z tej puli i test już wcześniej wykorzystano. Brak dowodu patient-disjoint, detector-training-disjoint i niezależnej generalizacji. LaMa może konfudować semantykę CAV. Nie mylimy AP50 z confidence ani BA sondy.','', 'Pliki: CALIBRATION, SPLITS_GT, CAV_FIXED, DIRECTIONS_768/J128.csv, SMOKE, VALIDATION_CURVES, VAL_PRED_*, LOCK, TEST_PRED_* wraz z mapami i GT, TEST_INTERVENTIONS, BOOTSTRAP_SAMPLES, RESULT, AUDIT, PROVENANCE, HASHES.']
    txt='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(txt,encoding='utf-8');(HERE/'WYNIKI_MAKSYMALIZACJA.md').write_text(txt,encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);register_research_family();synthetic_metric_check()
    if not args.smoke and (OUT/'LOCK.json').exists():raise RuntimeError('Existing lock; refusing repeat test or overwrite')
    js(OUT/('SMOKE_PID.json' if args.smoke else 'PID.json'),dict(pid=os.getpid(),started=time.time(),python=sys.executable,command=sys.argv))
    rows,pairs,w,J,ids,cache,T,directions,hashes=source_inputs();net,device=init_net();wh=weights_hash(net)
    progress('source_verified',train_scale=T,device=device,frozen_weights=wh,primary='source101_AP50')
    smoke(net,device,rows,pairs,w,ids,T,directions)
    if args.smoke:progress('smoke_passed',device=device);return
    lock,valbase=validation(net,device,rows,pairs,w,ids,cache,T,directions,hashes)
    preds,records=test_once(net,device,rows,w,ids,cache,directions,lock);assert weights_hash(net)==wh
    progress('analysis_started');summary=grouped_bootstrap(rows,ids,preds,lock)
    result=dict(status='complete',tag='scalar_grid',ziarno=SEED,primary='source-compatible101-pointAP50_single_annotated_class',validation_pairs=75,validation_films=23,test_pairs=85,test_films=24,grid_r=GRID,train_scale=T,chosen=lock['chosen'],best_control_chosen_validation=lock['best_random_by_val'],summary=summary,bootstrap_films=2000,lock_sha256=sha(OUT/'LOCK.json'),oracleROI=True,perimage_fitting=False,weights_unchanged=True,global_maximum_claim=False)
    js(OUT/'RESULT.json',result);plots_report(rows,pairs,ids,lock,valbase,preds,summary,records)
    for p,h in hashes.items():assert sha(p)==h
    snapshot=OUT/'EXECUTED_SOURCE';snapshot.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,snapshot/p.name)
    for n in ['PLAN.md','README.md','WYNIKI.json']:shutil.copy2(HERE/n,snapshot/n)
    provenance=dict(source_hashes=hashes,code_hashes={p.name:sha(p) for p in HERE.glob('*.py')},checkpoint_updates=1000,weights_hash_before=wh,weights_hash_after=weights_hash(net),torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),device=device,metric_source_sha256=sha(HERE/'metryki_source.py'),previous_TCAV_metric_difference='previous TCAV integral AP, this source101 AP; do not equate baseline values',heldout_states_sha256=sha(TCAV/'HELDOUT_STATES.npz'),gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,normalization_chain='raw unit CAV; lift=w/||w||; shared alpha=r*train median ROI norm; q+alpha*ls2(W_fc2*v)*lift; norm; original densehead',inputs_readonly=True)
    js(OUT/'PROVENANCE.json',provenance)
    audit=dict(status='passed',smoke=read(OUT/'SMOKE.json'),source_metric_exact_equivalence=True,weights_unchanged=wh==weights_hash(net),input_hashes_unchanged=True,all_val_chosen_AP_ge_baseline=True,test_only_after_lock=True,no_perimage_alpha=True,configuration_count=10,validation_grid_evaluations_per_direction=15,own_directories_only=True)
    js(OUT/'AUDIT.json',audit)
    hs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','stdout.log','stderr.log','BIEG_scalar_grid.json']};js(OUT/'HASHES.json',hs)
    js(OUT/'BIEG_scalar_grid.json',dict(tag='scalar_grid',ziarno=SEED,status='complete',result_sha256=sha(OUT/'RESULT.json'),audit_sha256=sha(OUT/'AUDIT.json'),duration_seconds=time.time()-START))
    progress('complete',duration_seconds=time.time()-START,result=str(OUT/'RESULT.json'))
if __name__=='__main__':
    try:main()
    except Exception:progress('failed',traceback=traceback.format_exc());raise
