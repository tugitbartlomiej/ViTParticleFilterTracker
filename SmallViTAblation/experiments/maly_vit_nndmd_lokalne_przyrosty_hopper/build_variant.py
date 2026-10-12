"""One-time creation of the explicitly approved separate residual variant."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'maly_vit_nndmd_1500_expandedstate_eden'
for name in ['model.py','state.py','train.py','gpu_smoke.py','test_expanded.py','test_driver.py','test_distributed.py']:
    assert not (HERE/name).exists(),f'Refusing overwrite: {name}'
    (HERE/name).write_text((OLD/name).read_text(encoding='utf-8'),encoding='utf-8')
cfg=json.loads((OLD/'CONFIG.json').read_text(encoding='utf-8-sig'))
cfg.update(train_first=50,lr_generator_candidates=[1e-4,1e-3],lr_generator=1e-4,forecast_mode='anchored_residual',target_world_size=11)
(HERE/'CONFIG.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')

p=HERE/'model.py';s=p.read_text()
s=s.replace('native_rec=torch.sum((out[:,:d]-z0)**2)/(n_total*d)','decoded_origin=out[:,:d]  # BOTH branches stay in graph; no detach.\n    native_rec=torch.sum((decoded_origin-z0)**2)/(n_total*d)')
s=s.replace('err=torch.sum((out[:,:d]-truth)**2)','anchored=z0+(out[:,:d]-decoded_origin)\n        err=torch.sum((anchored-truth)**2)')
s=s.replace('def rollout(model,past_native,origin,aux,norm,n=10,micro_tokens=1024):','def rollout(model,past_native,origin,aux,norm,n=10,micro_tokens=1024,residual=True):')
s=s.replace('phi=model.encode(expanded_input(raw,origin,aux,norm,allowed=origin))\n            for h in range(n):','phi=model.encode(expanded_input(raw,origin,aux,norm,allowed=origin))\n            decoded_origin=model.decode(phi)[:,:d]\n            for h in range(n):')
s=s.replace('output[h,first:last]=(decoded[:,:d]*norm.native_scale).cpu().numpy()','native=raw[-1]+(decoded[:,:d]-decoded_origin)*norm.native_scale if residual else decoded[:,:d]*norm.native_scale\n                output[h,first:last]=native.cpu().numpy()')
s=s.replace('model.optimizer.step();model.optimizer_steps+=1','diagnostics,before=step_diagnostics_before(model)\n    model.optimizer.step();model.optimizer_steps+=1\n    diagnostics.update(step_diagnostics_after(model,before))')
s=s.replace('stats.update(dict(zip(keys,scalars.cpu().tolist())),','stats.update(diagnostics,**dict(zip(keys,scalars.cpu().tolist())),')
s=s.replace('stats.update(optimizer_step=model.optimizer_steps,','stats.update(diagnostics,optimizer_step=model.optimizer_steps,')
# Capture pre-clipping block norms explicitly before clip; diagnostics afterclip separate.
s=s.replace('grad=torch.nn.utils.clip_grad_norm_(model.module.parameters(),clip_norm)','preclip=gradient_block_norms(model)\n    grad=torch.nn.utils.clip_grad_norm_(model.module.parameters(),clip_norm)')
s=s.replace('diagnostics,before=step_diagnostics_before(model)','diagnostics,before=step_diagnostics_before(model)\n    diagnostics.update(preclip)')
s+='''\n\ndef gradient_block_norms(model):
    generator=list(model.module._koopman_propagator.parameters())
    network=list(model.module._encoder.parameters())+list(model.module._decoder.parameters())
    def norm(params):return float(torch.sqrt(sum(torch.sum(p.grad.detach()**2) for p in params if p.grad is not None)).cpu())
    return {'gradient_G_norm_before_clip':norm(generator),'gradient_network_norm_before_clip':norm(network)}

def step_diagnostics_before(model):
    with torch.no_grad():
        a=model.operator().detach().clone()
        network=[p.detach().clone() for p in list(model.module._encoder.parameters())+list(model.module._decoder.parameters())]
        g=model.module._koopman_propagator.get_K().detach().clone()
    return {'lr_generator':model.lr_generator},(a,g,network)

def step_diagnostics_after(model,before):
    a0,g0,n0=before
    with torch.no_grad():
        a=model.operator().detach();g=model.module._koopman_propagator.get_K().detach()
        net=list(model.module._encoder.parameters())+list(model.module._decoder.parameters())
        dn=torch.sqrt(sum(torch.sum((p-old)**2) for p,old in zip(net,n0)))
        nn=torch.sqrt(sum(torch.sum(old**2) for old in n0))
        return {'A_minus_I_relative_frobenius':float((torch.linalg.norm(a-torch.eye(model.learned_dim,device=model.device))/model.learned_dim**.5).cpu()),
          'delta_A_relative_frobenius':float((torch.linalg.norm(a-a0)/torch.clamp(torch.linalg.norm(a0),min=1e-30)).cpu()),
          'delta_G_frobenius':float(torch.linalg.norm(g-g0).cpu()),
          'delta_network_relative_norm':float((dn/torch.clamp(nn,min=1e-30)).cpu())}
'''
p.write_text(s,encoding='utf-8')

p=HERE/'train.py';s=p.read_text()
s=s.replace('return range(last-horizon+1)','return range(50,last-horizon+1)')
s=s.replace('assert world in (1,2,4)','assert world in (1,2,3,4,8,11)')
s=s.replace("reader.get_history_batch(local_ids,0)","reader.get_history_batch(local_ids,50)")
s=s.replace("reader.get_history_batch(ids,0)","reader.get_history_batch(ids,50)")
s=s.replace('optimizer_step(reference,full,0,aux,norm,80','optimizer_step(reference,full,50,aux,norm,80')
s=s.replace('distributed_optimizer_step(parallel,raw,0,aux,norm,80','distributed_optimizer_step(parallel,raw,50,aux,norm,80')
s=s.replace("'actual_dataset_GPU_preflight',1,0,case","'actual_dataset_GPU_preflight',1,50,case")
s=s.replace("case+1,'preflight',0","case+1,'preflight',50")
s=s.replace("reader.ensure_window(0,10)","reader.ensure_window(50,10)")
s=s.replace("reader.get_state(ids,0)","reader.get_state(ids,50)")
s=s.replace("native_calibration_checkpoint=0","native_calibration_checkpoint=50")
s=s.replace("def make_model():return ExpandedNNDMD", "def make_model(lr=None):return ExpandedNNDMD")
s=s.replace("lr_generator=cfg['lr_generator'],device=", "lr_generator=cfg['lr_generator'] if lr is None else lr,device=")
s=s.replace('seconds*3337','seconds*987').replace('seconds*3807','seconds*1457')
# Original resume blocks are replaced by the candidate-aware protocol below.
start=s.index('        resume_path=os.environ.get(')
stop=s.index('        def train_epoch(',start)
s=s[:start]+"        resume=None\n        def restore(model,saved):\n            model.module.load_state_dict(saved['state_dict']);model.optimizer.load_state_dict(saved['optimizer']);model.optimizer_steps=saved['optimizer_steps']\n"+s[stop:]
start=s.index("        if resume and resume['phase']=='inner_train':restore(model)")
stop=s.index("        collective_call(lambda:save_checkpoint(HERE/'FROZEN_MODEL.pt'",start)
s=s[:start]+'''        del model;torch.cuda.empty_cache()
        candidate_results=[]
        def operator_diagnostics(model):
            a=model.operator().detach().cpu().numpy().astype(np.float64)
            return dict(spectral_radius=float(np.max(np.abs(np.linalg.eigvals(a)))),
                operator_norm=float(np.linalg.norm(a,2)),power10_norm=float(np.linalg.norm(np.linalg.matrix_power(a,10),2)),
                A_minus_I_relative_frobenius=float(np.linalg.norm(a-np.eye(768))/np.sqrt(768)),
                note='identity proximity penalty does not guarantee spectral stability; anchored offset is outside learned768')
        for candidate,lr in enumerate(cfg['lr_generator_candidates']):
            tag='LRG_%d'%candidate;phase='inner_'+tag;val_rows=[]
            for file in sorted(HERE.glob(tag+'_VALIDATION_EPOCH_*.json')):val_rows.append(json.loads(file.read_text()))
            model=make_model(lr)
            path=HERE/(phase+'_latest.pt');resume=None
            if path.exists():
                resume=torch.load(path,map_location='cpu',weights_only=False)
                assert resume['config']==cfg and resume['phase']==phase and resume['normalization_fit_last']==80
                assert resume['model_config']['lr_generator']==lr
                restore(model,resume);total_steps=max(total_steps,resume['total_optimizer_steps'])
            start_epoch=resume['epoch'] if resume else 1
            for epoch in range(start_epoch,cfg['max_epochs']+1):
                if early_stop(val_rows,cfg['early_stopping_patience'],cfg['early_stopping_min_delta_raw_rmse']):break
                if any(v['epoch']==epoch for v in val_rows):continue
                duration=train_epoch(model,norm,80,phase,epoch)
                collective_call(lambda:reader.ensure_window(80,10),'parallel validation extraction')
                val=broadcast_object(collective_call(lambda:evaluate(model,reader,80,aux,norm,cfg,progress,window_prepared=True),'rank0 validation',True))
                diag=broadcast_object(collective_call(lambda:operator_diagnostics(model),'validation operator diagnostics',True))
                val.update(epoch=epoch,training_epoch_seconds=duration,lr_generator=lr,operator_diagnostics=diag);val_rows.append(val)
                collective_call(lambda:atomic_json(HERE/(tag+'_VALIDATION_EPOCH_%02d.json'%epoch),val),'validation save',True)
                collective_call(lambda:save_checkpoint(HERE/(tag+'_INNER_EPOCH_%02d.pt'%epoch),model,epoch,norm,cfg,total_steps),'epoch checkpoint',True)
            chosen_epoch=select_epoch(val_rows);best=next(v for v in val_rows if v['epoch']==chosen_epoch)
            candidate_results.append(dict(lr_generator=lr,selected_epoch=chosen_epoch,full768_rmse=best['full768']['aggregate_rmse_state'],
                J128_rmse=best['J128']['aggregate_rmse_state'],persistence_rmse=best['full768']['aggregate_rmse_persistence'],tag=tag))
            del model;torch.cuda.empty_cache()
        selected=min(candidate_results,key=lambda r:(r['full768_rmse'],r['lr_generator'],r['selected_epoch']))
        chosen=selected['selected_epoch'];selected_lr=selected['lr_generator']
        collective_call(lambda:atomic_json(HERE/'SELECTION.json',dict(selected_epoch=chosen,selected_lr_generator=selected_lr,
            objective='minimum full768 aggregate raw validation RMSE h1..10 at origin80; ties lower lr then earlier epoch',
            candidates=candidate_results,test_opened=False)),'selection save',True)
        norm=Normalizer.fit(scale,aux,90)
        collective_call(lambda:norm.save(HERE/'NORMALIZER_FINAL.npz'),'final normalizer save',True);model=make_model(selected_lr)
        resume=None;refit_start=1;path=HERE/'final_refit_latest.pt'
        if path.exists():
            resume=torch.load(path,map_location='cpu',weights_only=False)
            assert resume['config']==cfg and resume['phase']=='final_refit' and resume['normalization_fit_last']==90
            assert resume['model_config']['lr_generator']==selected_lr
            restore(model,resume);refit_start=resume['epoch'];total_steps=max(total_steps,resume['total_optimizer_steps'])
        for epoch in range(refit_start,chosen+1):train_epoch(model,norm,90,'final_refit',epoch)
''' +s[stop:]
s=s.replace("fit_last=90,test_origin=90,test_last=100,selected_epoch=chosen,total_optimizer_steps=total_steps,", "fit_first=50,fit_last=90,test_origin=90,test_last=100,selected_epoch=chosen,selected_lr_generator=selected_lr,total_optimizer_steps=total_steps,")
s=s.replace("forecast='origin-only encoder, learned A^h, nonlinear decoder; no future signals'", "forecast='z0+s*(D_native(A^h phi0)-D_native(phi0)); origin memory offset; no future signals'")
s=s.replace("'Fit obejmuje 90 przejść między checkpointami, czyli 900 aktualizacji źródłowego ViT.","'Fit obejmuje 40 unikalnych przejść cp50..90, czyli 400 aktualizacji źródłowego ViT;31 sliding windows H10.")
s=s.replace("'Wybrana epoka: %d. Całkowite kroki Adam: %d. Final refit: %d kroków.'%(chosen,total_steps,model.optimizer_steps),", "'Wybrana epoka: %d. lrG: %.8g. Całkowite kroki Adam: %d. Final refit: %d kroków.'%(chosen,selected_lr,total_steps,model.optimizer_steps),")
s=s.replace("'Encoder użyto jeden raz.","'Prognoza jest anchored residual: z0+s*(D(A^h phi0)-D(phi0)); stały offset poza learned768. H0 dokładne przez kotwicę, nie przez doskonałą rekonstrukcję. Encoder użyto jeden raz.")
s=s.replace("'actual_training_steps':0,'local_GPU_training_cancelled':True", "'actual_training_steps':0,'train_first':50,'forecast_mode':'anchored_residual','local_GPU_training_cancelled':True")
p.write_text(s,encoding='utf-8')

p=HERE/'gpu_smoke.py';s=p.read_text().replace('assert world in (2,4)','assert world in (2,3,4,8,11)');p.write_text(s,encoding='utf-8')
p=HERE/'test_driver.py';s=p.read_text().replace('len(train.training_origins(80)),71','len(train.training_origins(80)),21').replace('len(train.training_origins(90)),81','len(train.training_origins(90)),31').replace('47*71,3337','47*21,987').replace('47*81,3807','47*31,1457');p.write_text(s,encoding='utf-8')
p=HERE/'test_expanded.py';s=p.read_text().replace('phi=model.encode(expanded_input(torch.tensor(native.reshape(3,10,3)),80,aux,norm,80));a=model.operator()','phi=model.encode(expanded_input(torch.tensor(native.reshape(3,10,3)),80,aux,norm,80));a=model.operator();base=model.decode(phi)[:,:3]')
s=s.replace('decoded=model.decode(phi)[:,:3]*norm.native_scale','decoded=torch.tensor(native[-1].reshape(10,3))+(model.decode(phi)[:,:3]-base)*norm.native_scale');p.write_text(s,encoding='utf-8')
p=HERE/'test_distributed.py';s=p.read_text().replace('for world in (2,4):','for world in (4,11):').replace('timedelta(seconds=120)','timedelta(seconds=300)');s=s.replace('self.assertEqual(len(rows),8)','self.assertEqual(len(rows),14)').replace('list(range(1,9))','list(range(1,15))').replace('[32,28]*4','[32,28]*7').replace("freeze['total_optimizer_steps'],8","freeze['total_optimizer_steps'],14").replace("['actual_training_steps'],8","['actual_training_steps'],14")
# Two candidates each stop after3 epochs in fixture +one best final epoch=14steps.
p.write_text(s,encoding='utf-8')
