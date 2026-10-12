"""Czy fresh plain forward osiąga agreement128>=85? Cosine/confidence/IoU nie są warunkami sukcesu."""
import os,json,time,shutil
import numpy as np,torch
from model_snapshot import MalyViT,cele
from dane import HERE,ROOT,SOURCE,OUT,load,sha
from aparat import js,measure,audit,successful,zapisz_wynik,PREVIOUS,RESUME
def main():
    start=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=json.loads((OUT/'RUN_CONFIG.json').read_text(encoding='utf-8'));device=env['device']
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.22)
    meta,rgb,image,before,target,J,K,g=load(device)
    targets=tuple(t.to(device) for t in cele([g[None]],20,30))
    original=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'];results={}
    for tag in ('before','best','final'):
        net=MalyViT().to(device).eval();saved=torch.load(OUT/f'MODEL_{tag}.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(saved['net'],strict=True);net.requires_grad_(False)
        assert not any(m._forward_hooks or m._forward_pre_hooks for m in net.modules())
        with torch.no_grad():plain,_=net(image)
        cache={}
        def read(m,i,o):cache['z']=o.detach()
        h=net.bloki[-1].mlp.act.register_forward_hook(read)
        with torch.no_grad():observed,_=net(image)
        h.remove();torch.testing.assert_close(plain,observed,rtol=0,atol=0)
        stored=np.load(OUT/f'LATENT_NATURAL_{tag}.npz')
        assert np.array_equal(stored['target'],target.cpu().numpy())
        assert np.array_equal(stored['channels0'],J.cpu().numpy())
        torch.testing.assert_close(cache['z'][0].cpu(),torch.from_numpy(stored['natural']),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(plain.cpu(),torch.from_numpy(stored['detector_map']),rtol=2e-5,atol=2e-5)
        audit_result=audit(net,original,saved['trainable_parameters'])
        assert audit_result['frozen_unchanged'] and audit_result['BN_buffers_unchanged']
        assert saved['changed_parameters']==audit_result['changed_parameters']
        optimizer_steps=[float(v['step']) for v in saved['optimizer']['state'].values()]
        assert all(s==saved['step'] for s in optimizer_steps)
        m=measure(cache['z'],plain,target,J,K,g,targets)
        results[tag]=dict(metrics=m,step=saved['step'],audit=audit_result,optimizer_steps=optimizer_steps,plain_forward_no_hooks=True,read_only_hook_no_effect=True,
             latent_max_abs_error=float(np.max(abs(cache['z'][0].cpu().numpy()-stored['natural']))),
             map_max_abs_error=float(np.max(abs(plain.cpu().numpy()-stored['detector_map']))),checkpoint_sha256=sha(OUT/f'MODEL_{tag}.pt'))
        del net,cache,plain,observed
    baseline=json.loads((OUT/'BASELINE_CHECK.json').read_text(encoding='utf-8'))
    assert results['before']['step']==2000
    for n,h in baseline['source_hashes'].items():assert sha(SOURCE/n)==h
    assert sha(RESUME)==baseline['resume_checkpoint_sha256']
    assert sha(PREVIOUS/'LATENT_NATURAL_final.npz')==baseline['previous_final_latent_sha256']
    # Recheck original0 with no replacement; historical1000 numbers are already fresh-verified.
    net=MalyViT().to(device).eval();net.load_state_dict(original);net.requires_grad_(False)
    cache={}
    with torch.no_grad():plain0,_=net(image)
    def read0(m,i,o):cache['z']=o.detach()
    h=net.bloki[-1].mlp.act.register_forward_hook(read0)
    with torch.no_grad():observed0,_=net(image)
    h.remove();torch.testing.assert_close(plain0,observed0,rtol=0,atol=0)
    torch.testing.assert_close(cache['z'][0],before,rtol=2e-5,atol=2e-5)
    reference0=measure(cache['z'],plain0,target,J,K,g,targets)
    previous=json.loads((PREVIOUS/'RESULT.json').read_text(encoding='utf-8'))
    first=json.loads((ROOT/'outputs/vit_wyuczenie_latentu_jedno_zdjecie/RESULT.json').read_text(encoding='utf-8'))
    assert first['fresh_reload_verified'] and previous['fresh_reload_verified']
    for key in ['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss']:
        assert abs(reference0[key]-first['baseline'][key])<2e-5
        assert abs(results['before']['metrics'][key]-previous['final'][key])<2e-5 if 'final' in previous else abs(results['before']['metrics'][key]-previous['comparison']['trained2000'][key])<2e-5
    result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'))
    result['status']='complete';result['fresh_reload_verified']=True;result['best']=results['best']['metrics'];result['final']=results['final']['metrics']
    result['success']=successful(result['best']);result['primary_stop_criterion']='agreement128>=85 only; fresh plain forward'
    result['verification_seconds']=time.perf_counter()-start
    result['comparison']={'original0':reference0,'trained1000':first['best'],'trained2000':results['before']['metrics'],'final':result['final']}
    if result['candidate_agreement85']:assert result['success'],'Candidate did not reach85 after fresh reload'
    js(OUT/'FRESH_RELOAD.json',dict(pid=os.getpid(),results=results,original0_verified=reference0,source_files_unchanged=True,resume_checkpoint_unchanged=True,primary_success=result['success'],seconds=result['verification_seconds']))
    js(OUT/'RESULT.json',result);zapisz_wynik(__file__,'training',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,4,figsize=(20,4.5))
    for ax,(label,d) in zip(axes,result['comparison'].items()):
        ax.imshow(rgb)
        for box,color,legend in [(g,'cyan','GT'),(d['bbox_xyxy'],'lime','Prediction')]:
            ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=2,label=legend))
        ax.set_title(f"{label}\nAgreement {d['agreement128']:.3f}%\nConf {d['confidence']:.6f}; IoU {d['iou']:.6f}");ax.axis('off');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'POROWNANIE_0_1000_2000_FINAL.png',dpi=160);plt.close(fig)
    fields=['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss']
    lines=['# Naturalny latent do85 procent','',f"Primary agreement>=85: {'OSIĄGNIĘTE' if result['success'] else 'NIEOSIĄGNIĘTE'}.",
           '', '| Pomiar | Oryginał0 | Wyuczone1000 | Wyuczone2000 | Final | Best |','|---|---:|---:|---:|---:|---:|']
    for k in fields:lines.append('| '+k+' | '+' | '.join(f"{m[k]:.9g}" for m in [*result['comparison'].values(),result['best']])+' |')
    lines+=['','## Pytanie i aparat','',
            'Czy dalsze uczenie tych samych norm2/fc1 od kroku2000 osiąga naturalny target J128 na85 procent? Agreement =100 max(0,1-sqrt(MSE128/mean(target128²))). Cosine/confidence/IoU są osobno, bez warunku sukcesu. Nie normalizujemy wektorów, nie skalujemy ani nie przycinamy targetu.',
            '',f"Adam przywrócony z MODEL_final.pt rodziny2000. Best krok{result['best_step']}, final{result['final_step']}; nowych aktualizacji{result['new_updates']}. Stop: {result['stop_reason']}. Czas trenera wraz z eksportem {result['elapsed_seconds']:.3f} s; weryfikacja {result['verification_seconds']:.3f} s; GPU peak {result['cuda_peak_allocated_bytes']} bajtów.",
            '', 'Tylko norm2 weight/bias i fc1 weight/bias ostatniego bloku. Loss bez zmian: MSE128/energia128 +10 MSE640/energia640 +0.02 GT CenterNet/GIoU. Float32, eval BN, bez autocast/TF32, pełny forward. Początkowy LR3e-5; każde zdarzenie zmiany zapisano w LR_EVENTS.',
            '',json.dumps(result['config'],ensure_ascii=False),'',json.dumps(result['lr_events'],ensure_ascii=False),
            '', '## Podłoga i audyt','',f"Przed pierwszym update odtworzono naturalny latent2000: max błąd {baseline['latent_max_abs_error']:.9g}; Adam cztery step2000. Energia128 {env['target_energy128']:.9g}, energia640 {env['target_energy640']:.9g}. Ten sam RGB480x320 INTER_AREA i ImageNet wewnątrz modelu.",
            '', 'Nowy proces utworzył modele before/best/final i wykonał plain forward bez hooków. Oddzielny hook jest wyłącznie odczytujący; jego wpływ na detektor wynosi bitowo zero. Oryginał0 również świeżo sprawdzono. Audyt każdego state_dict potwierdza identyczne zamrożone parametry i BN, tylko cztery dozwolone zmiany. Sprawdzono optimizer steps, źródłowe SHA oraz nienaruszony checkpoint2000.',
            '', '## Zakres wyniku','',
            'Jeden uprzednio korzystnie wybrany obraz i GT. Target pochodzi z optymalizacji GT. Trening widzi target/GT, inference tylko piksele i wagi. To overfit, bez dowodu generalizacji lub poprawy na nowych obrazach. Żadnych lookup, biasfield, nowych warstw, zmian architektury, downstream lub trenowalnych parametrów. Nieosiągnięcie progu w budżecie/plateau nie dowodzi niemożliwości.',
            '', '## Artefakty','',
            'MODEL_before/best/final.pt z Adamem i metadanymi; best_live/latest oraz chunk checkpoints; naturalne latenty600x768 i pełne CSV J128; HISTORY.jsonl/json/csv; audyty, LR_EVENTS, FRESH_RELOAD i overlay. EXECUTED_SOURCE: kod z chwili startu. VERIFICATION_SOURCE: końcowa weryfikacja. Deklaracja rodziny lokalna, bez zapisu poza dwoma dozwolonymi katalogami.']
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI_TRAIN.md').write_text(text,encoding='utf-8')
    snapshots=OUT/'VERIFICATION_SOURCE';snapshots.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,snapshots/p.name)
    js(OUT/'HASHES.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','TRAIN.stdout.log','TRAIN.stderr.log','VERIFY.stdout.log','VERIFY.stderr.log','WRITER.lock']})
    js(OUT/'PROGRESS.json',dict(status='complete',training_pid=env['pid'],verification_pid=os.getpid(),step=result['final_step'],success=result['success'],best=result['best'],elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(status='complete',success=result['success'],best_step=result['best_step'],best=result['best']),indent=2),flush=True)
if __name__=='__main__':main()
