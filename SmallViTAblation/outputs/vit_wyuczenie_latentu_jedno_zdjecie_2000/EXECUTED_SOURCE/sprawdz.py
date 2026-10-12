"""Czy świeże modele bez podmiany odtwarzają naturalny wynik? Rozbieżność lub zmiana zamrożonych wag obala audyt."""
import os,sys,time,json,shutil
import numpy as np,torch
from model_snapshot import MalyViT
from dane import HERE,SOURCE,OUT,load,sha
from trenuj import js,measure,audit,successful,cele,zapisz_wynik
def main():
    start=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Separate process, same backend for tight numerical comparison; small memory cap.
    env=json.loads((OUT/'RUN_CONFIG.json').read_text(encoding='utf-8'));device=env['device']
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.22)
    meta,rgb,image,before,target,J,K,g=load(device)
    targets=tuple(t.to(device) for t in cele([g[None]],20,30))
    original=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'];results={}
    # Add explicit changed-parameter metadata to every checkpoint without changing weights.
    for p in OUT.glob('MODEL_*.pt'):
        saved=torch.load(p,map_location='cpu',weights_only=False)
        changed=[n for n,v in saved['net'].items() if not torch.equal(v,original[n])]
        assert set(changed)<=set(saved['trainable_parameters'])
        if saved.get('changed_parameters')!=changed:
            saved['changed_parameters']=changed
            tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');torch.save(saved,tmp)
            for attempt in range(12):
                try:os.replace(tmp,p);break
                except OSError:
                    if attempt==11:raise
                    time.sleep(min(.02*2**attempt,.25))
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
        m=measure(cache['z'],plain,target,J,K,g,targets)
        results[tag]=dict(metrics=m,audit=a,plain_forward_no_hooks=True,read_only_hook_no_effect=True,
             latent_max_abs_error=float(np.max(abs(cache['z'][0].cpu().numpy()-stored['natural']))),
             map_max_abs_error=float(np.max(abs(plain.cpu().numpy()-stored['detector_map']))),checkpoint_sha256=sha(OUT/f'MODEL_{tag}.pt'))
        del net,cache,plain,observed
    baseline=json.loads((OUT/'BASELINE_CHECK.json').read_text(encoding='utf-8'))
    for n,h in baseline['source_hashes'].items():assert sha(SOURCE/n)==h
    result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'));result['status']='complete';result['fresh_reload_verified']=True
    result['best']=results['best']['metrics'];result['success']=successful(result['best']);result['verification_seconds']=time.perf_counter()-start
    result['failed_attempts']=['STARTUP_ERROR.json','LOCK_FAILURE.json'];result['restart_from_original']=True
    js(OUT/'FRESH_RELOAD.json',dict(pid=os.getpid(),results=results,source_files_unchanged=True,seconds=result['verification_seconds']))
    js(OUT/'RESULT.json',result)
    zapisz_wynik(__file__,'training',result)
    # Source/GT overlays, including explicitly labelled historical artificial reference.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,d,title in zip(axes,[results['before']['metrics'],meta['intervention'],result['best']],['Original natural','Historical artificial target (reference)','Trained natural best']):
        ax.imshow(rgb)
        for box,color,label in [(g,'cyan','GT'),(d['bbox_xyxy'],'lime','Prediction')]:
            ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=2,label=label))
        ax.set_title(f"{title}\nConf {d['confidence']:.6f}; IoU {d['iou']:.6f}");ax.axis('off');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'POROWNANIE_PRZED_CEL_NATURAL.png',dpi=160);plt.close(fig)
    fields=['MSE128','MSE640','MSE768','agreement128','cosine128','confidence','iou','GTloss']
    lines=['# Naturalne wyuczenie latentu jednego obrazu','',
           'Pytanie: czy uczenie oryginalnych wag przed GELU osiągnie zapisany sztuczny target after bez podmiany aktywacji?',
           '',f"Wynik kryterium łącznego: {'OSIĄGNIĘTY' if result['success'] else 'NIEOSIĄGNIĘTY'}.",
           '', '| Pomiar | Oryginalny naturalny | Najlepszy naturalny | Końcowy naturalny |','|---|---:|---:|---:|']
    for k in fields:lines.append('| '+k+' | '+' | '.join(f"{results[t]['metrics'][k]:.9g}" for t in ['before','best','final'])+' |')
    lines+=['','Agreement128 = 100 max(0, 1-sqrt(MSE128/mean(target128²))). MSE640 mierzy zmianę wobec niezmienionych kanałów before. Cosine jest osobno i nie zastępuje agreement.',
            '',f"Czas treningu wraz z przygotowaniem i eksportem: {result['elapsed_seconds']:.3f} s. Weryfikacja i odczyt: {result['verification_seconds']:.3f} s.",
            '', '## Aparat i biegi', '',json.dumps(result['config'],ensure_ascii=False), '']
    for r in result['runs']:
        lines.append(f"{r['stage']}: {r['steps']} aktualizacji, best krok {r['best_step']}, stop {r['stop_reason']}, {r['trainable_count']} uczonych parametrów, czas {r['seconds']:.3f} s. Końcowy przyrost najlepszego agreement przez 250 kroków: {r['tail_gain_pp']:.6f} pp.")
    if len(result['runs'])==1:
        lines+=['','T2 nie został uruchomiony: końcowa poprawa T1 nie spełniła zadeklarowanej definicji plateau. Limit T1 został wyczerpany.']
    lines+=['','Pierwszy start przerwał błąd typu w przygotowaniu GT. Następny start wykonał 96 aktualizacji i został przerwany przez Windows PermissionError podczas atomowego zapisu progress, nie przez OOM. Nie było zapisanych wag po aktualizacjach, więc zakończony bieg rozpoczął od oryginału. Zachowano STARTUP_ERROR i LOCK_FAILURE. Writer ma unikatowe pliki PID/czas, ponawianie OSError i niekrytyczny progress; każde poprawione best jest zapisywane przed progress.']
    lines+=['','## Podłoga i audyt','',f"Oryginalny checkpoint i before odtworzone: max błąd GELU {baseline['latent_max_abs_error']:.9g}, rtol/atol 2e-5. Ten sam RGB480x320 INTER_AREA, ImageNet wewnątrz modelu. Target128 energia {env['target_energy128']:.9g}; target640 energia {env['target_energy640']:.9g}.",
            '', 'Niezależny proces utworzył nowe modele, załadował before/best/final, wykonał forward bez hooków i sprawdził detekcje. Odczytujący hook nie zmienił mapy ani o jeden bit. Audyt obejmuje każdy tensor state_dict; wszystkie nieuczone parametry i bufory BN pozostały identyczne. Źródłowe checkpoint, NPZ i obrazy zachowały SHA256. FRESH_RELOAD.json zawiera pełny audyt i błędy odczytu.',
            '', '## Zastrzeżenia i zakres wyniku','',
            'To optymalizacja jednego uprzednio korzystnie wybranego obrazu oraz jego GT. Target był wcześniej optymalizowany z użyciem GT. Uczenie ma bezpośredni dostęp do targetu, a inference używa tylko obrazu i nowych wag. Wynik nie dowodzi generalizacji, poprawy na nowych zdjęciach ani możliwości uzyskania targetu przy oryginalnych zamrożonych wagach. T2, jeśli uruchomiony, ma większą pojemność i może zmienić wejście residual; oceniamy rzeczywistą głowę. Nie dodano lookup, biasfield ani nowych warstw; nie zmieniono lub przycięto targetu.',
            '', 'Przy nieosiągnięciu kryterium zapisany najlepszy residual jest wynikiem negatywnym w zadanym budżecie; nie dowodzi niemożliwości dopasowania innym treningiem.',
            '', '## Artefakty','', 'MODEL_before/best/final.pt, oryginalne checkpointy etapów z Adamem, LATENT_NATURAL_*.npz (600x768), VEKTORY_J128_*.csv (600x128), HISTORY.csv/json, OVERLAY_*.png i POROWNANIE_PRZED_CEL_NATURAL.png. RUN_CONFIG.json identyfikuje architekturę, wersje, źródło, kod, PID i stałe energie. Deklaracja rodziny pozostaje lokalna w dozwolonym zakresie.']
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI_TRAIN.md').write_text(text,encoding='utf-8')
    snapshots=OUT/'EXECUTED_SOURCE';snapshots.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,snapshots/p.name)
    js(OUT/'HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','TRAIN.stdout.log','TRAIN.stderr.log']})
    js(OUT/'PROGRESS.json',dict(status='complete',training_pid=env['pid'],verification_pid=os.getpid(),success=result['success'],best=result['best'],elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(status='complete',success=result['success'],best=result['best']),indent=2),flush=True)
if __name__=='__main__':main()
