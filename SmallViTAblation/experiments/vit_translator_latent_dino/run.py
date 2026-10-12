"""Paired film-disjoint pilot. All 600x768 values; CPU ridge, GPU frozen encoders."""
import os,sys,time,json,hashlib,traceback
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import torch,cv2
from PIL import Image
from torchvision import transforms
from translator import Translator,save,unit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'outputs'/HERE.name
sys.path.insert(0,str(ROOT/'experiments/vit_kierunek_nadzorowany_detekcja128'))
from model_snapshot import MalyViT
START=time.time()
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def progress(stage,**kwargs):
    obj=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kwargs);write('PROGRESS.json',obj);print(json.dumps(obj),flush=True)
def metrics(pred,target):
    sim=pred@target.T;ranks=1+(sim>np.diag(sim)[:,None]).sum(1)
    return dict(mean_cosine=float(np.sum(pred*target,1).mean()),normalized_MSE=float(np.mean((pred-target)**2)),retrieval_top1=float(np.mean(ranks==1)),retrieval_top5=float(np.mean(ranks<=5)),median_rank=float(np.median(ranks)),gallery_size=len(target))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'RUN.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:run()
    except Exception:
        write('FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:lock.unlink(missing_ok=True)
def run():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    pairs_root=ROOT/'experiments/maly_vit_test_orientacji_narzedzia_400'
    pairs=json.loads((pairs_root/'PARY.json').read_text(encoding='utf-8-sig'))
    split_path=ROOT/'outputs/vit_cav_pelny_latent_pary400/SPLITS.json'
    records=[r for r in json.loads(split_path.read_text(encoding='utf-8-sig'))['records'] if r['kind']=='original']
    assert len(pairs)==len(records)==400
    ids={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','val','test']}
    films={s:{records[i]['film'] for i in ix} for s,ix in ids.items()}
    assert not(films['train']&films['val'] or films['train']&films['test'] or films['val']&films['test'])
    paths=[pairs_root/p['original'] for p in pairs]
    for i,p in enumerate(paths):
        assert pairs[i]['frame_id']==records[i]['frame_id'] and sha(p)==records[i]['sha256']
    ckpt=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
    hub=Path('C:/Users/bartl/.cache/torch/hub/facebookresearch_dinov2_main')
    weights=Path('C:/Users/bartl/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
    config=dict(vit_checkpoint=str(ckpt),vit_checkpoint_sha256=sha(ckpt),dino_model='dinov2_vits14',dino_weights_sha256=sha(weights),dino_output='L2 normalized CLS, 384 dimensions',preprocess='RGB Resize(shorter edge 256, bicubic), CenterCrop224, ImageNet normalize',latent='postGELU final MLP, full ordered 600x768, no pooling or PCA',train=len(ids['train']),val=len(ids['val']),test=len(ids['test']),historical_development_data=True,independent_generalization=False,alphas=[.0001,.001,.01,.1,1.,10.])
    write('CONFIG.json',config);write('MANIFEST.json',dict(records=records,paths=list(map(str,paths)),split_sha256=sha(split_path)))
    if not (OUT/'vit.npy').exists() or not (OUT/'dino.npy').exists():
        progress('loading_frozen_encoders')
        net=MalyViT().cuda().eval();net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['net']);net.requires_grad_(False);net.zapisuj_latent(True)
        dino=torch.hub.load(str(hub),'dinov2_vits14',source='local',pretrained=False)
        dino.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));dino=dino.cuda().eval().requires_grad_(False)
        prep=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        z=np.lib.format.open_memmap(OUT/'vit.partial.npy',mode='w+',dtype='float32',shape=(400,600,768))
        y=np.empty((400,384),np.float32)
        with torch.inference_mode():
            for lo in range(0,400,4):
                subset=paths[lo:lo+4];v=[];d=[]
                for p in subset:
                    a=cv2.imread(str(p));assert a is not None
                    a=cv2.resize(cv2.cvtColor(a,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
                    v.append(torch.from_numpy(a).permute(2,0,1).float()/255)
                    with Image.open(p) as im:d.append(prep(im.convert('RGB')))
                _,latent=net(torch.stack(v).cuda());assert latent.shape[1:]==(600,768)
                z[lo:lo+len(subset)]=latent.cpu().numpy()
                features=dino.forward_features(torch.stack(d).cuda())['x_norm_clstoken']
                y[lo:lo+len(subset)]=torch.nn.functional.normalize(features,dim=1).cpu().numpy()
                if lo%40==0:progress('paired_features',completed=lo+len(subset),total=400)
        z.flush();del z;os.replace(OUT/'vit.partial.npy',OUT/'vit.npy');np.save(OUT/'dino.npy',y)
        del net,dino;torch.cuda.empty_cache()
    z=np.load(OUT/'vit.npy',mmap_mode='r');y=np.load(OUT/'dino.npy');assert np.isfinite(y).all() and np.isfinite(z).all()
    tr,va,te=[ids[s] for s in ['train','val','test']]
    x=z[tr].reshape(len(tr),-1);trials=[]
    for alpha in config['alphas']:
        progress('ridge_selection',alpha=alpha)
        path=OUT/'candidate.npz';save(path,x,y[tr],(600,768),alpha)
        model=Translator(path);m=metrics(model.predict(z[va]),y[va]);trials.append(dict(alpha=alpha,**m))
    best=max(trials,key=lambda r:(r['retrieval_top1'],r['mean_cosine']))
    save(OUT/'TRANSLATOR.npz',x,y[tr],(600,768),best['alpha']);del x
    model=Translator(OUT/'TRANSLATOR.npz')
    test_pred=model.predict(z[te]);val_pred=model.predict(z[va])
    baseline=np.broadcast_to(unit(y[tr].mean(0)),y[te].shape)
    # Control: nearest real training latent and its DINO feature, same complete representation.
    flat=z.reshape(400,-1);anchor=np.array(flat[tr]);anchor=unit(anchor)
    nn=unit(np.array(flat[te]))@anchor.T;nn_pred=y[tr[nn.argmax(1)]]
    results=dict(selection=best,validation_trials=trials,test=metrics(test_pred,y[te]),mean_baseline_test=metrics(baseline,y[te]),nearest_vit_baseline_test=metrics(nn_pred,y[te]))
    target_path=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
    target=np.load(target_path);pred=model.predict(target['after'])[0];np.save(OUT/'DINO_CEL_SZTUCZNY.npy',pred)
    before=model.predict(target['before'])[0];np.save(OUT/'DINO_CEL_PRZED.npy',before)
    similarities=y@pred;order=np.argsort(-similarities)[:32]
    write('KANDYDACI_CEL_SZTUCZNY.json',dict(scope='400 historical pilot originals, NOT full training index',target_sha256=sha(target_path),translation_before_after_cosine=float(pred@before),candidates=[dict(index=int(i),path=str(paths[i]),frame_id=records[i]['frame_id'],cosine=float(similarities[i])) for i in order]))
    np.savez(OUT/'TEST_PREDICTIONS.npz',indices=te,predicted=test_pred,actual=y[te])
    # Reload independent artifact and verify parity.
    reloaded=Translator(OUT/'TRANSLATOR.npz');np.testing.assert_allclose(reloaded.predict(z[te[:2]]),test_pred[:2],atol=1e-6)
    results['verification']=dict(reload_passed=True,full_input_shape=[600,768],output_dim=384,finite=True,test_not_used_for_selection=True)
    write('RESULT.json',results)
    report='# Translator ViT do DINO — pilot\n\nPełny latent 600 × 768 jest bezpośrednio mapowany przez liniową regresję ridge w postaci dualnej na 384 cechy DINOv2 ViT-S/14. Nie ma SD, PCA ani uśredniania tokenów. ViT i DINO zamrożone.\n\n240 obrazów treningowych, 75 walidacyjnych, 85 testowych; całe filmy rozłączne. Są to historyczne dane rozwojowe, więc nie jest to niezależny dowód generalizacji. Hiperparametr dobrano wyłącznie na walidacji.\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n\nRetrieval odnosi się do galerii zdjęć z odpowiedniej odłożonej części pilota, nie do całych 35 tysięcy. Cosine nie jest procentem zgodności wartości. Sztuczny cel może leżeć poza rozkładem treningowym; jego translacja i ranking są hipotezą, a nie dowodem osiągalności ani korzyści dla uczenia detektora. Indeks FAISS musi używać dokładnie tych samych wag, CLS i preprocessingu.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf-8');progress('complete',test=results['test'])
if __name__=='__main__':main()
