"""Intermediate checkpoint: one real image, inference only, no optimizer step."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json, hashlib, time, copy
from pathlib import Path
import numpy as np
import torch
import pykoopman
from model import ExpandedNNDMD, rollout
from state import AuxHistory, Normalizer, expanded_input

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'maly_vit_pykoopman_30_obrazow_128'
EXPECTED='85099613250068ba97637a3cd933b89f4f79b381083c7818019950d8b4fff5a5'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

start=time.perf_counter()
checkpoint_path=HERE/'LOCAL_INFERENCE_SNAPSHOT_20261002_1831.pt'
assert sha(checkpoint_path)==EXPECTED
checkpoint=torch.load(checkpoint_path,map_location='cpu')
config=json.loads((HERE/'CONFIG.json').read_text(encoding='utf-8-sig'))
manifest=json.loads((HERE/'MANIFEST_1500.json').read_text(encoding='utf-8-sig'))
oldmanifest=json.loads((OLD/'MANIFEST.json').read_text(encoding='utf-8-sig'))
metadata=json.loads((OLD/'DANE_768.json').read_text(encoding='utf-8-sig'))
lookup={r['frame_id']:r['index'] for r in oldmanifest['records']}
record=next(r for r in manifest['records'] if r['frame_id'] in lookup)
index=record['index']; oldindex=lookup[record['frame_id']]
assert index==23 and oldindex==29
images=np.load(OLD/'OBRAZY.npy',mmap_mode='r')
assert hashlib.sha256(images[oldindex].tobytes()).hexdigest()==record['input_rgb_sha256']
oldhash={r['step']:r['sha256'] for r in metadata['checkpoints']}
checked=[]
for r in manifest['checkpoints']:
    if 88<=r['index']<=100:
        assert oldhash[r['optimizer_step']]==r['sha256']
        checked.append(r)
aux=AuxHistory.load(HERE/'AUX_HISTORY.npz')
norm=Normalizer.load(HERE/'NORMALIZER_FINAL.npz'); assert norm.fit_last==90
mc=checkpoint['model_config']
device='cuda' if torch.cuda.is_available() else 'cpu'
model=ExpandedNNDMD(**{k:mc[k] for k in ['q','native_dim','learned_dim','hidden','seed','lr_network','lr_generator']},device=device)
loaded=model.module.load_state_dict(checkpoint['state_dict'],strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
for name,value in model.module.state_dict().items():assert torch.equal(value.cpu(),checkpoint['state_dict'][name])
for p in model.module.parameters():p.requires_grad_(False)
model.module.eval()
details={k:checkpoint.get(k) for k in ['phase','epoch','last_completed_origin','optimizer_steps','total_optimizer_steps','normalization_fit_last']}
details.update({'strict_load':True,'manifest_index':index,'oldcache_index':oldindex,'frame_id':record['frame_id'],'device':device,'torch':torch.__version__,'pykoopman':pykoopman.__version__,'normalizer_fit_last':norm.fit_last})
dump('CHECKPOINT_READY.json',details)
print('STRICT_LOAD_READY',json.dumps(details),flush=True)
latent=np.load(OLD/'LATENT_768.npy',mmap_mode='r')
past=np.array(latent[88:91,oldindex:oldindex+1],copy=True)
assert past.shape==(3,1,600,768) and np.isfinite(past).all()
prediction,auxmean,auxstd,proxy=rollout(model,past,90,aux,norm,n=10,micro_tokens=1024)
print('REAL_ROLLOUT_COMPLETE',prediction.shape,flush=True)
# Future truth is read only after autonomous prediction has completed.
truth=np.array(latent[91:101,oldindex:oldindex+1],copy=True)
assert np.isfinite(truth).all()
J=np.array(config['channels_j128']); assert len(J)==128
baselines={'model':prediction,'persistence':np.broadcast_to(past[-1],prediction.shape),'historical_mean':np.broadcast_to(past.mean(axis=0),prediction.shape),'continuation':past[-1][None]+np.arange(1,11)[:,None,None,None]*(past[-1]-past[-2])[None]}
rows=[]
for h in range(10):
    row={'horizon_checkpoints':h+1,'horizon_source_updates':10*(h+1)}
    for key,forecast in baselines.items():
        row[key]={}
        for block,ix in [('full768',np.arange(768)),('J128',J)]:
            a=forecast[h][...,ix].astype(np.float64); b=truth[h][...,ix].astype(np.float64); z=past[-1][...,ix].astype(np.float64)
            mse=float(np.mean((a-b)**2))
            row[key][block]={'mse':mse,'rmse':float(np.sqrt(mse)),'sign_all':float(np.mean(np.sign(a-z)==np.sign(b-z)))}
    for block in ['full768','J128']:
        ratio=row['model'][block]['mse']/row['persistence'][block]['mse']
        row['model'][block].update({'mse_skill_vs_persistence':1-ratio,'rmse_ratio_vs_persistence':float(np.sqrt(ratio))})
    rows.append(row)
dump('METRICS_FIRST_RESULT.json',{'checkpoint':details,'horizons':rows})
print('H10_FIRST_RESULT',json.dumps(rows[-1]),flush=True)
np.save(HERE/'PREDICTION_FULL768.npy',prediction)
np.savez(HERE/'AUX_PREDICTION_DIAGNOSTICS.npz',mean=auxmean,std=auxstd)
# Poison all future true auxiliary values: inference must remain identical.
poison=copy.deepcopy(aux)
poison.values[91:]=np.nan;poison.targets[91:]=np.nan;poison.age[91:]=np.nan
again,*_=rollout(model,past,90,poison,norm,n=10,micro_tokens=1024)
assert np.array_equal(prediction,again)
with torch.no_grad():
    A=model.operator(); assert bool(torch.isfinite(A).all())
    identity_distance=float(torch.linalg.norm(A-torch.eye(768,device=device))/np.sqrt(768))
    inp=expanded_input(torch.tensor(past.reshape(3,600,768),device=device),90,aux,norm,90)
    phi=model.encode(inp); reconstruction=(model.decode(phi)[:,:768]*norm.native_scale).cpu().numpy().reshape(1,600,768)
    manual=[]
    for h in range(10):
        phi=phi@A.T;manual.append((model.decode(phi)[:,:768]*norm.native_scale).cpu().numpy().reshape(1,600,768))
    manual=np.stack(manual);assert np.array_equal(manual,prediction)
for name,value in model.module.state_dict().items():assert torch.equal(value.cpu(),checkpoint['state_dict'][name])
np.save(HERE/'RECONSTRUCTION_H0.npy',reconstruction)
np.save(HERE/'A_DISCRETE.npy',A.cpu().numpy())
aggregate={}
for key,forecast in baselines.items():
    aggregate[key]={}
    for block,ix in [('full768',np.arange(768)),('J128',J)]:
        mse=float(np.mean((forecast[...,ix].astype(np.float64)-truth[...,ix].astype(np.float64))**2))
        aggregate[key][block]={'mse':mse,'rmse':float(np.sqrt(mse))}
reconstruction_metrics={block:float(np.sqrt(np.mean((reconstruction[...,ix].astype(np.float64)-past[-1][...,ix].astype(np.float64))**2))) for block,ix in [('full768',np.arange(768)),('J128',J)]}
result={'status':'complete_intermediate_inference_only','checkpoint':details,'checkpoint_sha256':EXPECTED,'image_count':1,'tokens':600,'native_channels':768,'horizons':rows,'aggregate':aggregate,'reconstruction_h0_rmse':reconstruction_metrics,'A_finite':True,'A_minus_I_relative_frobenius':identity_distance,'future_aux_poison_identical':True,'manual_encode_once_rollout_identical':True,'weights_unchanged':True,'optimizer_updates_in_this_check':0,'source_checkpoint_metadata_sha_matches':checked,'sourcecache_sha256_declared_not_recomputed':metadata['latent_sha256'],'input_history_sha256':hashlib.sha256(past.tobytes()).hexdigest(),'truth_sha256':hashlib.sha256(truth.tobytes()).hexdigest(),'prediction_sha256':sha(HERE/'PREDICTION_FULL768.npy'),'runtime_difference':'local Torch2.1.2+cu121 versus Eden Torch2.9.1+cu128; source cache extracted previously using source model and identical checkpoint metadata','seconds':time.perf_counter()-start}
dump('METRICS.json',result)
summary=f"""# Lokalny test POŚREDNIEGO checkpointu — jeden obraz\n\nSnapshot {EXPECTED}, phase {details['phase']}, epoch {details['epoch']}, ostatni ukończony origin {details['last_completed_origin']}, total optimizer steps {details['total_optimizer_steps']}. To nieukończony refit, diagnostyka bez strojenia.\n\nObraz exact1500 index {index}, frame {record['frame_id']}, oldcache index {oldindex}. Wszystkie 600 cząsteczek i 768 współrzędnych. Wejście cp88..90, samodzielna prognoza cp91..100. Jeden checkpoint odpowiada 10 aktualizacjom źródłowego ViT. J128 odczytano po pełnym decoderze.\n\n| H aktualizacji | RMSE full768 model | persistence | RMSE J128 model | persistence | znak J128 |\n|---:|---:|---:|---:|---:|---:|\n"""
for r in rows:summary+=f"| {r['horizon_source_updates']} | {r['model']['full768']['rmse']:.8f} | {r['persistence']['full768']['rmse']:.8f} | {r['model']['J128']['rmse']:.8f} | {r['persistence']['J128']['rmse']:.8f} | {100*r['model']['J128']['sign_all']:.3f}% |\n"
summary+=f"\nRMSE to pierwiastek z MSE agregowanego równomiernie po wszystkich cząsteczkach i kanałach, w surowych jednostkach latentu. Historyczna średnia i kontynuacja ostatniego przyrostu są zapisane w METRICS.json. Znak uwzględnia wszystkie współrzędne, sign(0)=0.\n\nStrict load PASS, wszystkie tensory zgodne bitowo ze snapshotem i niezmienione po inferencji. Przyszłe auxiliary zatrute NaN nie zmieniły prognozy. Niezależny encode-once rollout phi_h=phi_(h-1) A^T jest identyczny. A finite; ||A−I||F/||I||F={identity_distance:.8g}. Żadnych aktualizacji optymalizatora ani przyszłych wejść.\n\nRuntime lokalny {torch.__version__}, PyKoopman {pykoopman.__version__}, {device}; Eden używa Torch2.9.1+cu128. Różnice kerneli FP32 mogą powodować drobne różnice numeryczne. Prawdziwe cele pochodzą z wcześniej zweryfikowanego lokalnego cache, identyczne ID obrazu/piksele i hashe źródłowych checkpointów cp88..100 potwierdzono. SHA całego cache pochodzi z jego metadanych; nie przeliczano ponownie 6GB. Wynik dotyczy wyłącznie jednego obrazu, nie wszystkich1500 ani finalnego checkpointu.\n"
(HERE/'RAPORT_POSREDNI.md').write_text(summary,encoding='utf-8')
print('COMPLETE',json.dumps({'seconds':result['seconds'],'A_minus_I':identity_distance,'reconstruction':reconstruction_metrics}),flush=True)
