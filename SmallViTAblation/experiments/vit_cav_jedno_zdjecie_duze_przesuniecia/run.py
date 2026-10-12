"""Czy szeroki signed alpha poprawia właściwą ramkę? Wzrost samego score nie wystarcza."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[2]/'outputs/vit_cav_jedno_zdjecie_duze_przesuniecia/mplconfig')
import argparse,ast,csv,hashlib,shutil,time,traceback,platform
import cv2,numpy as np,torch,timm,scipy
from torch.nn import functional as F
from scipy.stats import pearsonr,spearmanr
from dane import *
from model_snapshot import MalyViT,dekoduj
from metryki_source import iou,ap50 as source_AP50
COARSE=np.array([-100,-30,-10,-3,-1,-.5,-.25,-.1,-.05,-.01,0,.01,.05,.1,.25,.5,1,3,10,30,100],float)
TCAV=ROOT/'outputs/vit_tcav_tozsamosc_narzedzia';SOURCE_CAV=TCAV/'CAV_full768.npz'
K=29;START=time.time();FLOOR=-.1699712075;THRESHOLDS=np.round(np.arange(.5,.96,.05),2)
def progress(stage,**kw):
    js(OUT/'PROGRESS.json',dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kw));print(stage,kw,flush=True)
def register_research_family():
    decl=dict(rodzina=HERE.name,klasa='badawcza',pytanie='Czy jeden fixed768CAV poprawia q i singleimage AP przy dużym signed alpha?',role=['BIEG','CAV','CURVE','LATENT','AUDIT'],wymagane=['tag','ziarno'],kontrole=['baseline','negative_arm'],uwagi='Oracle source image29; primary matchedconfidence*IoU; singleimage AP additional; scoped registry')
    if (HERE/'WYNIKI.json').exists():assert read(HERE/'WYNIKI.json')['rodzina']==HERE.name
    else:js(HERE/'WYNIKI.json',decl)
    js(OUT/'REGISTRATION.json',dict(**decl,registered_by='register_research_family',shared_registry_write=False,plan_sha256=sha(HERE/'PLAN.md')))
def state_hash(net):return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
def init_net():
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu'
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>700*1024**2:
        torch.cuda.set_per_process_memory_fraction(.12);device='cuda'
    net=MalyViT().to(device).eval();ck=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000;net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False);return net,device
def baseline(net,device,pair):
    im=cv2.imread(str(PAIRS/pair['original']));assert im is not None
    rgb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(rgb).permute(2,0,1)[None].float().to(device)/255;cache={};b=net.bloki[-1]
    h1=b.mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach().clone()))
    h2=b.register_forward_hook(lambda m,a,o:cache.update(q=o.detach().clone()))
    with torch.no_grad():o,_=net(x)
    h1.remove();h2.remove();return x,rgb,o,cache['z'],cache['q']
def fast(net,q,unit_q,r,N):
    alpha=torch.tensor(r,dtype=q.dtype,device=q.device)[:,None,None]*N
    with torch.no_grad():out=net.glowa(net.norm(q+alpha*unit_q)).reshape(len(r),20,30,5)
    return out
def full_hook(net,x,D,alpha):
    h=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:o+alpha*D)
    try:
        with torch.no_grad():o,_=net(x)
    finally:h.remove()
    return o
def APs(scores,overlaps):
    hit=overlaps[:,None]>=THRESHOLDS[None,:];tp=hit&(np.cumsum(hit,axis=0)==1);order=np.argsort(-scores,kind='stable')
    ctp=np.cumsum(tp[order],axis=0);precision=ctp/np.arange(1,len(scores)+1)[:,None];precision=np.maximum.accumulate(precision[::-1],axis=0)[::-1];aps=[]
    for j in range(len(THRESHOLDS)):
        ix=np.searchsorted(ctp[:,j],np.linspace(0,1,101));safe=np.minimum(ix,len(scores)-1);aps.append(float(np.where(ix<len(scores),precision[safe,j],0).sum()/101))
    return np.array(aps)
def describe(o,g,center,r,N,J,v,zn,weights,stage):
    boxes,scores=dekoduj(o.cpu());boxes=boxes[0].numpy();scores=scores[0].numpy()
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):return None
    overlaps=iou(boxes,g);qs=scores*overlaps;match=int(np.argmax(qs));ap=APs(scores,overlaps)
    # Verify every point's source AP50, not only an optimized shortcut.
    source=source_AP50([(boxes,scores)],[g[None]]);assert abs(source-ap[0])<1e-12
    cy,cx=divmod(center,30);logit=float(o.reshape(600,5)[center,0]);conf=float(torch.sigmoid(o.reshape(600,5)[center,0]))
    ds=o.reshape(600,5)[center,1:].exp().cpu().numpy()*16;px=(cx+.5)*16;py=(cy+.5)*16;fb=np.array([px-ds[0],py-ds[1],px+ds[2],py+ds[3]])
    alpha=float(r*N);tids=np.flatnonzero(weights);lift=weights[tids]/np.linalg.norm(weights);change=alpha*lift[:,None]*v[None,:]
    moved=zn[tids]+change;other=np.setdiff1d(np.arange(768),J)
    nJ=float(abs(alpha)*np.linalg.norm(v[J]));nother=float(abs(alpha)*np.linalg.norm(v[other]))
    assert abs(nJ*nJ+nother*nother-alpha*alpha)<max(1e-5,2e-6*alpha*alpha)
    return dict(r=float(r),alpha=alpha,stage=stage,valid=True,fixed_GT_center_logit=logit,fixed_GT_center_confidence=conf,fixed_GT_center_bbox=fb,fixed_GT_center_IoU=float(iou(fb[None],g)[0]),confidence_top1=float(scores[0]),IoU_top1=float(overlaps[0]),q_top1=float(qs[0]),matched_rank=match+1,matched_confidence=float(scores[match]),matched_IoU=float(overlaps[match]),q_star=float(qs[match]),AP50=float(ap[0]),AP75=float(ap[5]),AP50_95=float(ap.mean()),AP_by_IoU=ap,bbox_top1=boxes[0],bbox_matched=boxes[match],all_boxes=boxes,all_scores=scores,all_IoU=overlaps,delta_Frobenius=abs(alpha),delta_norm_J128=nJ,delta_norm_other640=nother,relative_ROI_norm=abs(float(r)),ROI_fraction_below_GELU_floor=float((moved<FLOOR-1e-6).mean()),ROI_min_activation=float(moved.min()),ROI_max_activation=float(moved.max()))
def best(records,sign=None,metric='q_star'):
    valid=[r for r in records if r['valid'] and (sign is None or (r['r']>=0 if sign>0 else r['r']<=0))]
    return max(valid,key=lambda r:(r[metric],r['q_star'],-abs(r['r']),-r['r']))
def check_hook(net,x,q,unitq,D,r,N,g,label):
    full=full_hook(net,x,D,float(r*N));cached=fast(net,q,unitq,[r],N);err=float((full-cached).abs().max())
    torch.testing.assert_close(full,cached,rtol=1e-4,atol=5e-6)
    fb,fs=dekoduj(full.cpu());cb,cs=dekoduj(cached.cpu());boxerr=float((fb-cb).abs().max());scoreerr=float((fs-cs).abs().max());assert boxerr<.02 and scoreerr<1e-5
    return dict(label=label,r=float(r),alpha=float(r*N),max_map_error=err,max_decoded_bbox_error=boxerr,max_score_error=scoreerr)
def correlations(records):
    coarse=[next(r for r in records if r['r']==x) for x in COARSE];out={}
    for name,points in [('signed_all',coarse),('positive_with_zero',[r for r in coarse if r['r']>=0]),('negative_with_zero',[r for r in coarse if r['r']<=0])]:
        points=[r for r in points if r['valid']];a=np.array([r['alpha'] for r in points]);out[name]=dict(n=len(points),r_grid=[r['r'] for r in points],coefficients={})
        for metric in ['confidence_top1','IoU_top1','q_star','fixed_GT_center_confidence','matched_confidence','matched_IoU']:
            values=np.array([r[metric] for r in points]);constant=np.ptp(values)==0
            out[name]['coefficients'][metric]=dict(Pearson=None if constant else float(pearsonr(a,values).statistic),Spearman=None if constant else float(spearmanr(a,values).statistic),constant=constant)
    js(OUT/'COARSE_CORRELATIONS.json',dict(primary_grid='prespecified coarse21 only; no dense/refinement',groups=out,interpretation='deterministic same-image response, grid dependent; not independent sample population correlation; no inferential p-values'))
    return out
def plots(rgb,g,records,selected,corr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    valid=sorted([r for r in records if r['valid']],key=lambda r:r['alpha']);a=np.array([r['alpha'] for r in valid]);fig,axs=plt.subplots(3,1,figsize=(12,11),sharex=True)
    for key in ['confidence_top1','IoU_top1','q_star']:axs[0].plot(a,[r[key] for r in valid],label=key)
    for key in ['matched_confidence','matched_IoU','fixed_GT_center_confidence']:axs[1].plot(a,[r[key] for r in valid],label=key)
    for key in ['AP50','AP75','AP50_95']:axs[2].plot(a,[r[key] for r in valid],label='single-image '+key)
    for ax in axs:ax.set_xscale('symlog',linthresh=.01);ax.set_ylim(-.03,1.03);ax.axvline(0,color='gray',ls='--');ax.axvline(selected['best_overall']['alpha'],color='black',ls=':',label='best q');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[2].set_xlabel('signed alpha; symlog axis; fixed full768 CAV, oracle single image');fig.tight_layout();fig.savefig(OUT/'ALPHA_RESPONSE.png',dpi=150);plt.close(fig)
    names=['baseline','best_overall','best_positive','best_negative'];fig,axs=plt.subplots(1,4,figsize=(17,4.8))
    for ax,name in zip(axs,names):
        rr=selected[name];ax.imshow(rgb);b=rr['bbox_matched'];b1=rr['bbox_top1'];ax.add_patch(Rectangle((g[0],g[1]),g[2]-g[0],g[3]-g[1],fill=False,edgecolor='yellow',lw=2));ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='lime',lw=2));ax.add_patch(Rectangle((b1[0],b1[1]),b1[2]-b1[0],b1[3]-b1[1],fill=False,edgecolor='cyan',lw=1,ls=':'));ax.set_xlim(0,480);ax.set_ylim(320,0);ax.axis('off');ax.set_title(f'{name}; r={rr["r"]:.5g}\nmatched score={rr["matched_confidence"]:.4f}; IoU={rr["matched_IoU"]:.4f}\nq={rr["q_star"]:.4f}; AP50:.95={rr["AP50_95"]:.4f}',fontsize=9)
    fig.suptitle('Same original photograph; altered detector activations only. GT yellow, q-matched green, top1 cyan.');fig.tight_layout();fig.savefig(OUT/'BASE_BEST_POSITIVE_NEGATIVE.jpg',dpi=140);plt.close(fig)
def report(result):
    ss=result['selected'];lines=['# Jeden fixed full768 CAV: szeroki signed alpha','', 'Badano wskazany oryginalny obraz29, TR_0154_S1_P03/00001926, na oryginalnym zamrożonym k1000. Jeden CAV ma768 stałych współczynników. Alpha przesuwa obrazowe600x768 aktywacje w jednym ustalonym kierunku na oryginalnym ROI. J128 służy tylko do oznaczania i pomiaru części zmiany; nie optymalizowano128 ani768 współczynników.','', 'Primary: q_star=maximum(confidence_j * IoU(box_j,GT)) po10 decodedboxes. To oracle optymalizacja tego samego obrazu z GT, nie generalizacja. Obraz nie został zmodyfikowany ani generowany przez SD.','', '| Wariant | r | alpha | confidence matched | IoU matched | q_star | AP50 | AP75 | AP50:.95 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['baseline','best_overall','best_positive','best_negative','best_AP50','best_AP75','best_AP50_95']:
        r=ss[name];lines.append(f'|{name}|{r["r"]:.6g}|{r["alpha"]:.6g}|{r["matched_confidence"]:.6f}|{r["matched_IoU"]:.6f}|{r["q_star"]:.6f}|{r["AP50"]:.6f}|{r["AP75"]:.6f}|{r["AP50_95"]:.6f}|')
    delta=ss['best_overall']['q_star']-ss['baseline']['q_star'];lines+=['',f'Najlepsza zmiana q względem naturalnego baseline: {delta:+.8f}. Zakres zbadany r=[{result["searched_r_range"][0]}, {result["searched_r_range"][1]}]; liczba punktów {result["curve_points"]}. Najlepszy punkt w zbadanej siatce z lokalnym refinement, nie dowód globalnego maksimum.','', '## AP i korelacje','AP50/AP75/meanAP(.50:.95) liczone źródłową101punktową metryką na jednymGT. To single-image AP diagnostic, nie populacyjne mAP. Wiele punktów ma identyczne0/1 lub1/rankAP; APwinner wybierany osobno, remisy rozstrzyga q. Primary/refinement nie zmieniono na AP po obejrzeniu wyniku.','', 'Pearson/Spearman tylko na zadeklarowanych21 coarsepoints, osobno signed21 oraz11positive/11negative z zerem. Wszystkie koeficjenty w COARSE_CORRELATIONS.json. To zależne próbki jednej deterministycznej funkcji, zależne od grid, saturacji, rank changes i nieliniowości, nie korelacja populacyjna ani wniosek o monotoniczności.','', '## Aparat i zastrzeżenia','Fastforward zachowuje oryginalnyresidualq, fc2, norm i glowa. AUDIT zawiera pełny forward z rzeczywistymhookGELU dla małych/dużych±punktów i wybranych najlepszych wyników. Modelstate i źródłowe hashe niezmienne.','', 'Duże perturbacje nie są obcinane. Wartości poniżej minimalnegoGELU są zapisane, a sztuczne najlepsze aktywacje nie muszą być realizowalne przez obraz. Wybrany obraz należy do historycznej puli treningowej i wcześniejszej selekcji. Oryginalny knownROI i użycieGT do wyborualpha sprawiają, że to eksperymentoracle jednej klatki. Nie dowodzi to poprawy detektora na zbiorze ani możliwości generowania obrazu z tym latentem.','', f'Best q: fraction belowGELUfloor={ss["best_overall"]["ROI_fraction_below_GELU_floor"]:.6f}, delta normJ128={ss["best_overall"]["delta_norm_J128"]:.6f}, other640={ss["best_overall"]["delta_norm_other640"]:.6f}.','', 'CURVE.csv/json: wszystkie punkty i10boxes. RAW_MAPS: surowe logits5kanałów. BASE_BEST_LATENTS: naturalbaseline oraz artificialbest600x768, jednostkowy lift, maskweights i indeksyJ128. CAV_COEFFICIENTS.csv: wszystkie768 współczynników z flagąJ128. AUDIT/PROVENANCE/HASHES i executedsource umożliwiają odtworzenie.']
    txt='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(txt,encoding='utf-8');(HERE/'WYNIKI_SZEROKI_ALPHA.md').write_text(txt,encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True);register_research_family()
    if not args.smoke and (OUT/'RESULT.json').exists():raise RuntimeError('Existing result; refusing overwrite')
    js(OUT/('SMOKE_PID.json' if args.smoke else 'PID.json'),dict(pid=os.getpid(),started=time.time(),python=sys.executable,command=sys.argv))
    rows,pairs,weights,J,ids=load();row=rows[K];selection=read(SOURCE/'SELECTION.json');assert selection['source_index']==K and row['frame_id']==selection['record']['frame_id']
    assert row['frame_id']=='cataract_lmm/tracking/TR_0154_S1_P03/00001926';assert sha(PAIRS/pairs[K]['original'])==selection['record']['sha256']
    meta=read(ROOT/'experiments/maly_vit_tozsamosc_czasteczki_pary_400/POMIAR.json');assert sha(CKPT)==meta['checkpoint_sha256'];assert sha(HERE/'model_snapshot.py')==meta['model_source_sha256']
    v=np.load(SOURCE_CAV)['unit'].astype(np.float32);assert v.shape==(768,) and abs(np.linalg.norm(v)-1)<1e-6
    hashes={str(p):sha(p) for p in [CKPT,SOURCE_CAV,SOURCE/'LATENT_PELNY.npz',SOURCE/'SELECTION.json',PAIRS/pairs[K]['original'],PAIRS/'TOKEN_MASKI.npz',PAIRS/'manifest.json',PAIRS/'PARY.json']}
    net,device=init_net();wh=state_hash(net);x,rgb,original,z,q=baseline(net,device,pairs[K]);zn=z[0].cpu().numpy();old=np.load(SOURCE/'LATENT_PELNY.npz')['before'];err=float(abs(old-zn).max());np.testing.assert_allclose(old,zn,rtol=2e-5,atol=3e-5)
    w=weights[K];roi=np.flatnonzero(w);N=float(np.linalg.norm(zn[roi]));lift=w/np.linalg.norm(w);D=torch.tensor(lift,device=device)[None,:,None]*torch.tensor(v,device=device)[None,None,:];assert abs(float(D.norm())-1)<1e-6
    b=net.bloki[-1];unit_q=b.ls2(F.linear(D,b.mlp.fc2.weight,None));xx,yy,bw,bh=row['bbox_source_480'];g=np.array([xx,yy,xx+bw,yy+bh],np.float32);cx=int(np.clip(round((g[0]+g[2])/32-.5),0,29));cy=int(np.clip(round((g[1]+g[3])/32-.5),0,19));center=cy*30+cx
    checks=[];progress('source_verified',source_index=K,frame_id=row['frame_id'],base_ROI_norm=N,device=device,CAV='fixed full768',original_before_max_error=err)
    for r in [-100,-.01,0,.01,100]:checks.append(check_hook(net,x,q,unit_q,D,r,N,g,'prespecified_smoke'));progress('smoke_hook_checked',r=r)
    probe=describe(original,g,center,0,N,J,v,zn,w,'smoke_baseline');assert probe is not None and probe['valid']
    js(OUT/'SMOKE.json',dict(status='passed',checks=checks,device=device,baseline_metrics=probe,curve_description_path_checked=True))
    if args.smoke:progress('smoke_passed');return
    records=[];maps=[];done=set()
    def evaluate(grid,stage):
        fresh=[float(r) for r in grid if float(r) not in done]
        for start in range(0,len(fresh),16):
            chunk=fresh[start:start+16];oo=fast(net,q,unit_q,chunk,N)
            for j,r in enumerate(chunk):
                done.add(r);m=oo[j:j+1]
                if torch.isfinite(m).all():desc=describe(m,g,center,r,N,J,v,zn,w,stage)
                else:desc=None
                if desc is None:desc=dict(r=r,alpha=r*N,stage=stage,valid=False,reason='nonfinite model map or decoded box')
                records.append(desc);maps.append(m[0].cpu().numpy())
            progress('curve_processing',stage_name=stage,processed=len(records),stage_target=len(fresh),stage_processed=min(start+16,len(fresh)))
    evaluate(np.unique(np.r_[COARSE,-np.logspace(-4,2,200),0,np.logspace(-4,2,200)]),'coarse_and_dense401')
    extension=[]
    for sign in [-1,1]:
        winner=best(records,sign)
        if abs(winner['r'])==100:extension.extend(sign*np.logspace(2,3,101)[1:])
    if extension:evaluate(extension,'endpoint_extension_to1000')
    refinement=[]
    for sign in [-1,1]:
        winner=best(records,sign);branch=sorted(set(r['r'] for r in records if r['valid'] and (r['r']>=0 if sign>0 else r['r']<=0)));j=branch.index(winner['r']);left=branch[max(0,j-1)];right=branch[min(len(branch)-1,j+1)];refinement.extend(np.linspace(left,right,201))
    evaluate(refinement,'local_q_refinement')
    selected=dict(baseline=next(r for r in records if r['r']==0),best_overall=best(records),best_positive=best(records,1),best_negative=best(records,-1))
    for metric in ['AP50','AP75','AP50_95']:selected['best_'+metric]=best(records,metric=metric)
    for name,r in selected.items():checks.append(check_hook(net,x,q,unit_q,D,r['r'],N,g,name));progress('selected_fullhook_verified',selection=name,r=r['r'],q=r['q_star'])
    assert wh==state_hash(net)
    for p,h in hashes.items():assert sha(p)==h
    corr=correlations(records);js(OUT/'CURVE.json',dict(primary='q_star matched confidence*IoU',source_index=K,frame_id=row['frame_id'],GT_xyxy=g,coarse_r=COARSE,records=records))
    scalar_keys=[k for k,val in selected['baseline'].items() if not isinstance(val,np.ndarray)]
    with (OUT/'CURVE.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=scalar_keys,extrasaction='ignore');writer.writeheader()
        for rr in sorted(records,key=lambda r:r['r']):writer.writerow(rr)
    with (OUT/'CAV_COEFFICIENTS.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.writer(fp);writer.writerow(['channel0','channel1','unit_full768_coefficient','in_J128_display_only'])
        for j in range(768):writer.writerow([j,j+1,float(v[j]),int(j in J)])
    np.savez_compressed(OUT/'RAW_MAPS.npz',r=np.array([r['r'] for r in records]),alpha=np.array([r['alpha'] for r in records]),maps=np.array(maps),valid=np.array([r['valid'] for r in records]))
    latents={name:(z+r['alpha']*D)[0].cpu().numpy() for name,r in selected.items()};np.savez_compressed(OUT/'BASE_BEST_LATENTS.npz',natural_baseline=zn,**{'artificial_'+k:v for k,v in latents.items() if k!='baseline'},unit_full768=v,unit_ROI_lift_D=D[0].cpu().numpy(),original_last_block_q=q[0].cpu().numpy(),mask_weights=w,ROI_indices=roi,J128_indices_display_only=J,GT_xyxy=g,source_index=K,base_ROI_norm=N)
    ties={metric:sum(r['valid'] and r[metric]==selected['best_'+metric][metric] for r in records) for metric in ['AP50','AP75','AP50_95']};range_= [min(done),max(done)]
    result=dict(status='complete',primary='q_star = max decoded confidence*IoU',additional_AP='single-image source101 AP diagnostic; not population mAP',source_index=K,frame_id=row['frame_id'],GT_xyxy=g,base_ROI_norm=N,selected=selected,AP_tie_counts=ties,curve_points=len(records),finite_points=sum(r['valid'] for r in records),searched_r_range=range_,endpoint_extension=bool(extension),local_refinement=True,CAV_fixed=True,number_optimized_scalars=1,J128_role='display partition only, not optimized or support restricted',oracle_same_image=True,population_generalization=False,weights_unchanged=True,grid_correlations=corr,seed=SEED)
    js(OUT/'RESULT.json',result);plots(rgb,g,records,selected,corr);report(result)
    audit=dict(status='passed',original_before_max_error=err,sourceimage_hash_matches=True,checkpoint_k1000=True,weights_before=wh,weights_after=state_hash(net),CAV_source_hash_unchanged=True,input_hashes_unchanged=True,norm_D=float(D.norm()),zero_outside_ROI=bool((D[0,np.setdiff1d(np.arange(600),roi)]==0).all()),fullhook_checks=checks,max_map_error=max(c['max_map_error'] for c in checks),max_bbox_error=max(c['max_decoded_bbox_error'] for c in checks),all_points_source_AP50_equivalent=True,source101_interpolation=True,no_clipping=True,only_one_image=True,no_SD=True)
    js(OUT/'AUDIT.json',audit);js(OUT/'PROVENANCE.json',dict(source_hashes=hashes,code_hashes={p.name:sha(p) for p in HERE.glob('*.py')},torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),device=device,gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,weights_unchanged=wh==state_hash(net),normalization='raw unit full768 CAV; D=original ROI normalized weights/||weights|| * v; alpha=r*base original ROI norm; q+ls2(W_fc2*(alphaD)), norm, densehead',source_latent_used='before only; never prior artificial after',metric_source_sha256=sha(HERE/'metryki_source.py')))
    ss=OUT/'EXECUTED_SOURCE';ss.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,ss/p.name)
    for n in ['PLAN.md','README.md','WYNIKI.json']:shutil.copy2(HERE/n,ss/n)
    hs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','stdout.log','stderr.log','BIEG_single_image.json']};js(OUT/'HASHES.json',hs)
    js(OUT/'BIEG_single_image.json',dict(tag='single_image',ziarno=SEED,status='complete',result_sha256=sha(OUT/'RESULT.json'),audit_sha256=sha(OUT/'AUDIT.json'),duration_seconds=time.time()-START))
    progress('complete',duration_seconds=time.time()-START,best_r=selected['best_overall']['r'],best_q=selected['best_overall']['q_star'],baseline_q=selected['baseline']['q_star'])
if __name__=='__main__':
    try:main()
    except Exception:progress('failed',traceback=traceback.format_exc());raise
