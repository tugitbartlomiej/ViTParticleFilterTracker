"""Czy świeży plain forward odtwarza kontynuację? Rozbieżności wag/latentów lub optimizer step obalają audyt."""
import os,json,time,shutil
import numpy as np,torch
from model_snapshot import MalyViT
from dane import HERE,SOURCE,OUT,load,sha
from trenuj import js,measure,audit,successful,cele,zapisz_wynik,PREVIOUS,RESUME
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
        torch.testing.assert_close(cache['z'][0].cpu(),torch.from_numpy(stored['natural']),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(plain.cpu(),torch.from_numpy(stored['detector_map']),rtol=2e-5,atol=2e-5)
        a=audit(net,original,saved['trainable_parameters']);assert a['frozen_unchanged'] and a['BN_buffers_unchanged']
        assert saved['changed_parameters']==a['changed_parameters']
        optimizer_steps=[float(v['step']) for v in saved['optimizer']['state'].values()]
        assert all(s==saved['step'] for s in optimizer_steps)
        m=measure(cache['z'],plain,target,J,K,g,targets)
        results[tag]=dict(metrics=m,step=saved['step'],audit=a,optimizer_steps=optimizer_steps,plain_forward_no_hooks=True,read_only_hook_no_effect=True,
             latent_max_abs_error=float(np.max(abs(cache['z'][0].cpu().numpy()-stored['natural']))),
             map_max_abs_error=float(np.max(abs(plain.cpu().numpy()-stored['detector_map']))),checkpoint_sha256=sha(OUT/f'MODEL_{tag}.pt'))
        del net,cache,plain,observed
    assert results['before']['step']==1000 and results['final']['step']==2000
    baseline=json.loads((OUT/'BASELINE_CHECK.json').read_text(encoding='utf-8'))
    for n,h in baseline['source_hashes'].items():assert sha(SOURCE/n)==h
    assert sha(RESUME)==baseline['resume_checkpoint_sha256']
    assert sha(PREVIOUS/'LATENT_NATURAL_final.npz')==baseline['previous_final_latent_sha256']
    previous=json.loads((PREVIOUS/'RESULT.json').read_text(encoding='utf-8'))
    # Independently verify the original0 forward too, against the preceding family's baseline.
    net=MalyViT().to(device).eval();net.load_state_dict(original);net.requires_grad_(False)
    cache={}
    with torch.no_grad():plain0,_=net(image)
    def read0(m,i,o):cache['z']=o.detach()
    h=net.bloki[-1].mlp.act.register_forward_hook(read0)
    with torch.no_grad():observed0,_=net(image)
    h.remove();torch.testing.assert_close(plain0,observed0,rtol=0,atol=0)
    torch.testing.assert_close(cache['z'][0],before,rtol=2e-5,atol=2e-5)
    reference0=measure(cache['z'],plain0,target,J,K,g,targets)
    for key in ['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss']:
        assert abs(reference0[key]-previous['baseline'][key])<2e-5
    result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'));result['status']='complete';result['fresh_reload_verified']=True
    result['best']=results['best']['metrics'];result['success']=successful(result['best']);result['verification_seconds']=time.perf_counter()-start
    result['restart_from_original']=False;result['continued_from_step']=1000
    result['comparison']={'original0':reference0,'trained1000':results['before']['metrics'],'trained2000':results['final']['metrics']}
    js(OUT/'FRESH_RELOAD.json',dict(pid=os.getpid(),results=results,original0_verified=reference0,source_files_unchanged=True,resume_checkpoint_unchanged=True,optimizer_final_step=2000,seconds=result['verification_seconds']))
    js(OUT/'RESULT.json',result);zapisz_wynik(__file__,'training',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,d,title in zip(axes,[reference0,results['before']['metrics'],results['final']['metrics']],['Original natural step0','Trained natural step1000','Trained natural step2000']):
        ax.imshow(rgb)
        for box,color,label in [(g,'cyan','GT'),(d['bbox_xyxy'],'lime','Prediction')]:
            ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=2,label=label))
        ax.set_title(f"{title}\nConf {d['confidence']:.6f}; IoU {d['iou']:.6f}");ax.axis('off');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'POROWNANIE_0_1000_2000.png',dpi=160);plt.close(fig)
    fields=['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss']
    lines=['# Kontynuacja do 2000 aktualizacji','',f"Łączne kryterium: {'OSIĄGNIĘTE' if result['success'] else 'NIEOSIĄGNIĘTE'}.",'',
           '| Pomiar | Oryginał0 | Wyuczone1000 | Wyuczone2000 | Best kontynuacji |','|---|---:|---:|---:|---:|']
    for k in fields:lines.append('| '+k+' | '+' | '.join(f"{m[k]:.9g}" for m in [reference0,results['before']['metrics'],results['final']['metrics'],results['best']['metrics']])+' |')
    lines+=['','## Pytanie i aparat','',
            'Czy dalsze1000 aktualizacji tych samych wag i tego samego celu poprawi naturalne dopasowanie GELU? Adam przywrócony z MODEL_T1_norm2_fc1_final.pt rodziny vit_wyuczenie_latentu_jedno_zdjecie; wszystkie cztery stany step1000 na wejściu, step2000 na końcu. LR stały3e-5, bez warmup i bez nowego cosine. Żadnego restartu od oryginału.',
            '', 'Uczone tylko norm2 affine i fc1 weight/bias ostatniego bloku. Loss = MSE128/energia target128 + 10 MSE640/energia target640 + 0.02 GT CenterNet/GIoU. Float32, eval BN, bez autocast/TF32, frozen downstream.',
            '',json.dumps(result['config'],ensure_ascii=False),
            '',f"Czas kontynuacji z przygotowaniem i eksportem: {result['elapsed_seconds']:.3f} s. Weryfikacja: {result['verification_seconds']:.3f} s. Szczytowa alokacja GPU: {result['cuda_peak_allocated_bytes']} bajtów."]
    for r in result['runs']:
        lines+=['',f"{r['steps']} nowych aktualizacji; łącznie {r['total_steps']}; best krok {r['best_step']}; stop {r['stop_reason']}; {r['trainable_count']} uczonych parametrów; pętla {r['seconds']:.3f} s."]
    lines+=['','## Podłoga i audyt','',
            f"Przed pierwszą aktualizacją odtworzono naturalny checkpoint1000 i jego latent; max błąd {baseline['latent_max_abs_error']:.9g}. Ten sam RGB480x320 INTER_AREA i normalizacja ImageNet wewnątrz modelu. Stała energia target128 {env['target_energy128']:.9g}, target640 {env['target_energy640']:.9g}.",
            '', 'Świeży proces utworzył modele i wykonał plain forward bez hooków. Oddzielny hook wyłącznie odczytujący nie zmienił mapy ani o jeden bit. Oryginał0 również zweryfikowano ponownie. Pełne audyty porównują każdy tensor state_dict do oryginału0: zmieniły się tylko dozwolone cztery parametry, bufory BN identyczne. Checkpoint wznowienia, pierwotny checkpoint, NPZ i obrazy zachowały SHA256. Wszystkie optimizer steps i zmienione parametry zapisano i sprawdzono.',
            '', 'Writer JSON ma unikatowe PID/czas, ograniczone retry OSError i niekrytyczny progress. Best jest zapisywany przed progress. Nie było zatrzymującego błędu w kontynuacji. T2 i earlystop nie były stosowane.',
            '', '## Zastrzeżenia','',
            'Jeden uprzednio korzystnie wybrany obraz i jego GT. Target również pochodzi z optymalizacji GT. Trening widzi target i GT, inference tylko piksele i wyuczone wagi. To dopasowanie wewnątrz próby; nie dowodzi generalizacji ani poprawy na nowych zdjęciach. Nie zmieniono architektury, parametrówki, objective ani targetu. Bez lookup, biasfield, aktywacji zastępujących forward, rescale i clipping celu. Nieosiągnięcie90 procent jest wynikiem w zadanym budżecie, nie dowodem niemożliwości.',
            '', '## Artefakty','',
            'MODEL_before/best/final.pt (before=krok1000) z Adamem, architekturą/wersjami i changed_parameters; LATENT_NATURAL_*.npz600x768; VEKTORY_J128_*.csv600x128; HISTORY; AUDIT i FRESH_RELOAD; OVERLAY i POROWNANIE_0_1000_2000.png. RUN_CONFIG zawiera konfigurację, PID, hashes, optimizer i źródło. EXECUTED_SOURCE jest migawką z chwili startu, VERIFICATION_SOURCE końcową weryfikacją. Deklaracja rodziny jest lokalna w granicy wyłącznej własności.']
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI_TRAIN.md').write_text(text,encoding='utf-8')
    snapshots=OUT/'VERIFICATION_SOURCE';snapshots.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,snapshots/p.name)
    js(OUT/'HASHES.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','TRAIN.stdout.log','TRAIN.stderr.log']})
    js(OUT/'PROGRESS.json',dict(status='complete',training_pid=env['pid'],verification_pid=os.getpid(),step=2000,success=result['success'],best=result['best'],elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(status='complete',success=result['success'],comparison={t:{k:m[k] for k in fields} for t,m in result['comparison'].items()}),indent=2),flush=True)
if __name__=='__main__':main()
