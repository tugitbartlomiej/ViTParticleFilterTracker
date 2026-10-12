"""Full-state per-checkpoint transition operators across SAME-image pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from koopman_przyrostowy import IncrementKoopman,self_test

HERE=Path(__file__).resolve().parent
OUT=HERE/'wyniki'/'macierze_wiele_obrazow'
ROOT=HERE/'wyniki'/'run_1810662'
PROBE=HERE.parent/'koopman_fastgan_petla/wyniki/local_1810448/run/probe'
VARIANTS=['original','rotation90','length08','black','yellow','foreground_bright']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def status(stage,**kw):write(OUT/'STATUS.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))

def prepare():
    OUT.mkdir(exist_ok=True)
    manifest={r['path']:r for r in json.loads((ROOT/'REMOTE_MANIFEST.json').read_text())}
    assert sha(PROBE/'manifest.json')==manifest['probe/manifest.json']['sha256']
    allrows=json.loads((PROBE/'manifest.json').read_text())
    rows=[r for r in allrows if r['variant'] in VARIANTS]
    assert len(rows)==72 and sum(r['role']=='fit' for r in rows)==48
    for r in rows:
        for key in ('image','mask'):
            assert sha(PROBE/r[key])==r[key+'_sha256']==manifest['probe/'+r[key]]['sha256']
    weights=[]
    receipt_path=OUT/'WEIGHTS_VERIFIED.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for p in sorted((ROOT/'pretrain').glob('model_*.pth')):
        step=int(p.stem.split('_')[-1])
        expected=manifest['pretrain/'+p.name]['sha256']
        stat=p.stat();stamp=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=expected)
        if receipt.get(p.name)!=stamp:assert sha(p)==expected
        receipt[p.name]=stamp
        weights.append(dict(step=step,path=str(p),sha256=manifest['pretrain/'+p.name]['sha256']))
    assert len(weights)>=2 and all(b['step']-a['step']==5 for a,b in zip(weights,weights[1:]))
    cfg=dict(rows=rows,weights=weights,variants=VARIANTS,relative_ridge=1e-3,shape=[576,384],
        same_image_pairs=(len(weights)-1)*len(rows),image_fit=48,image_holdout=24,
        probe_manifest_sha256=sha(PROBE/'manifest.json'),pca=False,
        purpose='Retrospective K_t identification; separate image holdout, not future-time forecast',
        operator='K_t = I + Xc_t.T C_t, affine bias; full cross-token and cross-channel state')
    config=OUT/'CONFIG.json'
    if config.exists():assert json.loads(config.read_text())==cfg,'Frozen configuration mismatch'
    else:write(config,cfg)
    write(receipt_path,receipt)
    write(OUT/'TESTS.json',self_test())
    return cfg

def extract(cfg,smoke=False):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from rfdetr import RFDETRNano
    from czastki import DifferentiableReader
    from deterministyka import diagnostic_kernels
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for bounded local extraction')
    rows=cfg['rows'];images=[];masks=[]
    for r in rows:
        a=np.array(Image.open(PROBE/r['image']).convert('RGB'),copy=True)
        images.append(F.interpolate(torch.from_numpy(a).permute(2,0,1)[None].float()/255,(384,384),mode='bilinear',align_corners=False))
        a=np.array(Image.open(PROBE/r['mask']).convert('L'),copy=True)
        masks.append(F.interpolate(torch.from_numpy(a)[None,None].float()/255,(24,24),mode='area').flatten().numpy())
    np.save(OUT/'MASK_WEIGHTS.npy',np.array(masks))
    wrapper=RFDETRNano(num_classes=1,resolution=384,pretrain_weights=None,device='cuda',amp=False)
    net=wrapper.model.model.to(device='cuda',dtype=torch.float32).eval()
    for p in net.parameters():p.requires_grad_(False)
    items=cfg['weights'][:1] if smoke else cfg['weights']
    with diagnostic_kernels(),DifferentiableReader(net,wrapper.means,wrapper.stds) as reader,torch.no_grad():
        for item in items:
            target=OUT/f"TOKENS_{item['step']:07d}.npy"
            proof=target.with_suffix('.json')
            if not smoke and target.exists() and proof.exists():
                record=json.loads(proof.read_text())
                assert record['checkpoint_sha256']==item['sha256'] and record['sha256']==sha(target)
                continue
            status('extracting',step=item['step'],image=0,total_images=len(rows))
            payload=torch.load(item['path'],map_location='cpu',weights_only=False)
            net.load_state_dict(payload['model'],strict=True);del payload
            data=np.empty((len(rows),576,384),dtype=np.float32)
            for i,img in enumerate(images[:2] if smoke else images):
                feature=reader.features(img.cuda()).permute(0,2,3,1).reshape(576,384)
                data[i]=feature.cpu().numpy()
                if i==0:
                    repeat=reader.features(img.cuda()).permute(0,2,3,1).reshape(576,384).cpu().numpy()
                    err=float(np.linalg.norm(repeat-data[i])/max(np.linalg.norm(data[i]),1e-12))
                    assert err<=1e-5,('repeatability',err)
                    ref=np.load(ROOT/'pretrain'/f"CZASTKI_{item['step']:07d}.npz")['tokens']
                    parity=float(np.linalg.norm(ref-data[i])/max(np.linalg.norm(ref),1e-12))
                    assert parity<=1e-4,('Eden/local feature mismatch',parity)
                if i%12==0:status('extracting',step=item['step'],image=i,total_images=len(rows))
            if smoke:
                write(OUT/'GPU_SMOKE.json',dict(repeatability=err,eden_parity=parity,images=2,
                    feature_shape=list(data[:2].shape),max_memory_bytes=torch.cuda.max_memory_allocated(),passed=True))
                print('GPU smoke passed',flush=True);return
            assert np.isfinite(data).all()
            tmp=target.with_suffix('.tmp.npy');np.save(tmp,data);tmp.replace(target)
            write(proof,dict(sha256=sha(target),checkpoint_sha256=item['sha256'],repeatability=err,eden_parity=parity))
    del net,wrapper
    torch.cuda.empty_cache()

def metric_rows(pred,actual,initial,mask):
    n=len(pred);err=pred-actual;delta=pred-initial;truth=actual-initial
    norm=np.linalg.norm(delta,axis=1);tn=np.linalg.norm(truth,axis=1)
    cos=np.divide((delta*truth).sum(1),norm*tn,out=np.full(n,np.nan),where=norm*tn>0)
    mse=np.mean(err**2,axis=1)
    pertoken=np.mean(err.reshape(n,576,384)**2,axis=2)
    fg=(pertoken*mask).sum(1)/mask.sum(1)
    return [dict(mse=float(mse[i]),rmse=float(np.sqrt(mse[i])),foreground_mse=float(fg[i]),
        cosine=float(cos[i]) if np.isfinite(cos[i]) else None,
        increment_norm_ratio=float(norm[i]/tn[i]) if tn[i]>0 else None) for i in range(n)]

def analyze(cfg):
    fit=np.array([r['role']=='fit' for r in cfg['rows']]);mask=np.load(OUT/'MASK_WEIGHTS.npy')
    records=[];transitions=[];previous=None
    for a,b in zip(cfg['weights'],cfg['weights'][1:]):
        status('fitting_transition',from_step=a['step'],to_step=b['step'])
        x=np.load(OUT/f"TOKENS_{a['step']:07d}.npy").reshape(72,-1).astype(np.float64)
        y=np.load(OUT/f"TOKENS_{b['step']:07d}.npy").reshape(72,-1).astype(np.float64)
        model=IncrementKoopman().fit(x[fit],y[fit],cfg['relative_ridge'])
        pred=model.predict(x)
        # Exact full operator factors: Xc recovered from referenced full states.
        op=OUT/f"OPERATOR_{a['step']:07d}_{b['step']:07d}.npz"
        np.savez(op,mu_x=model.mx,mean_delta=model.md,coeff=model.coeff,ridge=model.ridge,
                 fit_indices=np.flatnonzero(fit),source_step=a['step'],target_step=b['step'])
        candidates={'fitted_K_t':pred,'persistence':x}
        if previous is not None:
            candidates['previous_K']=previous.predict(x)
            older=np.load(OUT/f"TOKENS_{a['step']-5:07d}.npy").reshape(72,-1)
            candidates['last_increment']=2*x-older
        summary=dict(from_step=a['step'],to_step=b['step'],ridge=model.ridge,
                     interpretation='K_t fitted using fit-image data at BOTH endpoints')
        for name,value in candidates.items():
            ms=metric_rows(value,y,x,mask)
            for i,r in enumerate(ms):
                records.append(dict(from_step=a['step'],to_step=b['step'],image_index=i,
                    group_id=cfg['rows'][i]['group_id'],variant=cfg['rows'][i]['variant'],role=cfg['rows'][i]['role'],method=name,**r))
            summary[name]={role:float(np.sqrt(np.mean([ms[i]['mse'] for i in range(72) if fit[i]==isfit]))) for role,isfit in [('fit',True),('holdout',False)]}
        if previous is not None:
            # ||X1.T C1 - X0.T C0||_F without materializing D x D.
            def norm2(m):return float(np.sum((m.xc@m.xc.T)*(m.coeff@m.coeff.T)))
            cross=float(np.sum((model.xc@previous.xc.T)*(model.coeff@previous.coeff.T)))
            summary['delta_K_frobenius']=float(np.sqrt(max(norm2(model)+norm2(previous)-2*cross,0)))
        transitions.append(summary);previous=model
        write(OUT/'TRANSITIONS.json',transitions)
    keys=list(records[0])
    with (OUT/'ERRORS_PER_IMAGE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    totals={};matched_totals={}
    for method in sorted(set(r['method'] for r in records)):
        totals[method]={}
        matched_totals[method]={}
        for role in ['fit','holdout']:
            rs=[r for r in records if r['method']==method and r['role']==role]
            totals[method][role]=dict(rmse=float(np.sqrt(np.mean([r['mse'] for r in rs]))),pairs=len(rs))
            common=[r for r in rs if r['from_step']>=cfg['weights'][1]['step']]
            matched_totals[method][role]=dict(rmse=float(np.sqrt(np.mean([r['mse'] for r in common]))),pairs=len(common))
    report=dict(state='completed',images=72,checkpoint_count=len(cfg['weights']),
        fit_groups=8,holdout_groups=4,same_image_pairs=cfg['same_image_pairs'],totals=totals,matched_totals=matched_totals,
        holdout_transitions_beating_persistence=sum(t['fitted_K_t']['holdout']<t['persistence']['holdout'] for t in transitions),
        transitions=len(transitions),pca=False,deep_koopman=False,
        limitations=['72 correlated variants of 12 sources, not 72 independent videos',
            'per-transition fit sees both checkpoint endpoints on training images',
            'previous_K predicts new time; fitted_K_t only generalizes across held-out images',
            'affine fit errors do not prove exact linearity or identify unique true operator',
            'no semantic identity attribution or detector performance intervention'])
    write(OUT/'REPORT.json',report)
    lines=['# Macierz osobno dla każdego przejścia między checkpointami','',
        f"{len(cfg['weights'])} checkpointów; {cfg['same_image_pairs']} par tego samego obrazu; pełne 576×384 bez PCA.",
        '48 wariantów z 8 grup do dopasowania, 24 warianty z 4 osobnych grup do oceny. K_t widzi oba końce danego odcinka na obrazach fit. Test holdout sprawdza przeniesienie mapy na inne obrazy, nie prognozę nieznanego czasu. previous_K jest prognozą kolejnego odcinka z poprzedniej mapy.','',
        '| Metoda | RMSE fit | RMSE holdout |','|---|---:|---:|']
    for name,v in matched_totals.items():lines.append(f"| {name} | {v['fit']['rmse']:.7f} | {v['holdout']['rmse']:.7f} |")
    lines+=['','Tabela używa wspólnych odcinków od drugiego przejścia; poprzednia macierz nie istnieje dla pierwszego. Pełne wyniki wszystkich odcinków są osobno w REPORT.json.',f"K_t lepszy od braku zmian na holdout w {report['holdout_transitions_beating_persistence']}/{len(transitions)} odcinkach.",
        'Pełna tabela: ERRORS_PER_IMAGE.csv. Macierze: OPERATOR_*.npz; K=I+Xc.T@coeff, Xc rekonstruuje się z odpowiedniego TOKENS i fit_indices, b=mean_delta−mu_x@(K−I).',
        'To analiza afiniczna. Nie uruchamiano deep Koopman ani FastGAN. Mały błąd nie dowodzi liniowości całej dynamiki ani domknięcia tożsamości.']
    (OUT/'RAPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    status('completed');print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    cfg=prepare()
    if not a.analyze_only:extract(cfg,a.smoke)
    if not a.smoke:analyze(cfg)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status('failed',error=repr(e))
        raise
