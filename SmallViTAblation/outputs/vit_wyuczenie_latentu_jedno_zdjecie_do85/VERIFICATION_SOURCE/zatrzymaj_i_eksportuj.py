"""Po jawnym STOP eksportuj ostatni trwały model bez update i potwierdź fresh forward przed przekazaniem do SD."""
import os,sys,json,time,csv,shutil,subprocess
from datetime import datetime,timezone
import torch
import aparat as a
from aparat import HERE,ROOT,SOURCE,OUT,CONFIG,ARCH,load,sha,export,js
from model_snapshot import MalyViT,cele
def main():
    start=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=json.loads((OUT/'RUN_CONFIG.json').read_text(encoding='utf-8'));device=env['device']
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.22)
    saved=torch.load(OUT/'MODEL_best_live.pt',map_location='cpu',weights_only=False)
    assert all(float(v['step'])==saved['step'] for v in saved['optimizer']['state'].values())
    for tag in ('best','final'):shutil.copy2(OUT/'MODEL_best_live.pt',OUT/f'MODEL_{tag}.pt')
    step=saved['step'];m=saved['metrics']
    meta,rgb,image,before,target,J,K,g=load(device);targets=tuple(t.to(device) for t in cele([g[None]],20,30))
    original=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'];a._REFERENCE=original
    net=MalyViT().to(device).eval();net.requires_grad_(False)
    for tag in ('before','best','final'):
        ck=torch.load(OUT/f'MODEL_{tag}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['net'],strict=True)
        measured=export(net,image,target,before,J,K,g,targets,tag,original,ck['trainable_parameters'],rgb)
        for key in ['agreement128','MSE128','MSE640','MSE768','cosine128','confidence','iou','GTloss']:assert abs(measured[key]-ck['metrics'][key])<2e-5
    records=[]
    for line in (OUT/'HISTORY.jsonl').read_text(encoding='utf-8').splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:pass
    kept=[r for r in records if r['step']<=step]
    if kept[-1]['step']!=step:kept.append(dict(step=step,seconds=records[-1]['seconds'],lr=saved['optimizer']['param_groups'][0]['lr'],**m))
    with (OUT/'HISTORY.csv').open('w',newline='',encoding='utf-8') as f:
        fields=sorted(set().union(*(r.keys() for r in kept)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
    js(OUT/'HISTORY.json',kept);js(OUT/'LR_EVENTS.json',[])
    result=dict(status='stopped_by_user_pending_fresh_reload',ziarno=CONFIG['seed'],baseline=kept[0],best=m,final=m,best_step=step,final_step=step,
                new_updates=step-2000,stop_reason='explicit_human_stop_before85',elapsed_seconds=kept[-1]['seconds'],config=CONFIG,optimizer='Adam',resumed_optimizer=True,
                resumed_from_step=2000,uses_source_GT=True,single_image_overfit=True,success=False,candidate_agreement85=False,
                cuda_peak_allocated_bytes=0,lr_events=[],training_worker_pid=env['pid'],training_worker_stopped=True,
                stopped_utc=datetime.now(timezone.utc).isoformat(),last_logged_step=records[-1]['step'],
                stop_protocol='No runtime stop flag; durable model+Adam verified before stopping only owned PID. Recover final complete atomic best_live after stop. No new updates.')
    js(OUT/'RESULT.json',result);js(OUT/'STOP_REQUESTED.json',dict(reason='explicit user STOP; no further training',worker_pid=env['pid'],saved_step=step))
    js(OUT/'PROGRESS.json',dict(status='stopped_by_user_pending_fresh_reload',step=step,best=m))
    del net;torch.cuda.empty_cache() if device=='cuda' else None
    subprocess.run([sys.executable,'-B','-u',str(HERE/'sprawdz.py')],cwd=str(ROOT),check=True)
    result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'));assert result['fresh_reload_verified']
    result['status']='stopped_by_user';result['export_and_verification_seconds']=time.perf_counter()-start;js(OUT/'RESULT.json',result);a.zapisz_wynik(__file__,'training',result)
    ready=dict(status='ready_for_sd',training_stopped_by_user=True,training_worker_pid=env['pid'],canonical_best_checkpoint=str((OUT/'MODEL_best.pt').resolve()),
               checkpoint_sha256=sha(OUT/'MODEL_best.pt'),final_checkpoint=str((OUT/'MODEL_final.pt').resolve()),step=step,metrics=result['best'],architecture=ARCH,
               torch_version=env['torch'],timm_version=env['timm'],model_source_path=str((HERE/'model_snapshot.py').resolve()),model_source_sha256=sha(HERE/'model_snapshot.py'),
               fresh_reload_verified=True,plain_forward_without_hooks_verified=True,activation_replacement=False,optimizer_saved=True,trainable_parameters=saved['trainable_parameters'],
               RGB_pipeline=dict(source_image=str((SOURCE/'ZDJECIE_ZRODLOWE.jpg').resolve()),prepared_rgb=str((SOURCE/'WEJSCIE_VIT.png').resolve()),
                 decode='cv2 BGR then BGR2RGB',resize=[480,320],interpolation='cv2.INTER_AREA',input_range='float32 RGB /255',layout='BCHW',
                 normalization_inside_model=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),latent_hook='bloki[-1].mlp.act GELU read-only',latent_shape=[600,768],grid_shape=[20,30]),
               full_original_target_path=str((SOURCE/'LATENT_PELNY.npz').resolve()),full_original_target_sha256=sha(SOURCE/'LATENT_PELNY.npz'),target_key='after',target_unchanged=True,
               channels0=J.cpu().tolist(),natural_latent_path=str((OUT/'LATENT_NATURAL_best.npz').resolve()),natural_latent_key='natural',
               primary_agreement_formula='100*max(0,1-sqrt(MSE128/mean(target128**2)))',single_image_GT_fit=True,generalization_tested=False,
               next_task_note='Separate SD from random noise using trained ViT as guide; no SD run in this family.')
    js(OUT/'READY_FOR_SD.json',ready)
    stopnote='\n\n## Jawne zatrzymanie przez użytkownika\n\nUżytkownik zakończył dążenie do85. Nie było runtime stop flag; trwały checkpoint i Adam potwierdzono przed zatrzymaniem wyłącznie własnego PID. Odzyskano ostatni kompletny atomowy best_live. Best i final oznaczają ten sam zabezpieczony krok. Nie wykonano późniejszych aktualizacji. GPU peak nie został odzyskany z zakończonego procesu; wartość0 w metadanych eksportu nie jest pomiarem treningu. READY_FOR_SD.json zawiera kanoniczny checkpoint, SHA, architekturę/wersje, pipeline RGB, niezmieniony target/J128 i świeżo zweryfikowane metryki. Generacja SD należy do kolejnego odrębnego zadania.\n'
    for p in (OUT/'RAPORT.md',HERE/'WYNIKI_TRAIN.md'):
        text=p.read_text(encoding='utf-8');p.write_text('Stan końcowy: zatrzymano na jawne polecenie użytkownika.\n\n'+text+stopnote,encoding='utf-8')
    js(OUT/'PROGRESS.json',dict(status='stopped_by_user',training_pid=env['pid'],step=step,ready_for_sd=True,best=result['best']))
    js(OUT/'HASHES.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','TRAIN.stdout.log','TRAIN.stderr.log','WRITER.lock']})
    print('READY_FOR_SD',json.dumps(dict(step=step,metrics=result['best'],checkpoint=ready['canonical_best_checkpoint'],sha256=ready['checkpoint_sha256'])),flush=True)
if __name__=='__main__':main()
