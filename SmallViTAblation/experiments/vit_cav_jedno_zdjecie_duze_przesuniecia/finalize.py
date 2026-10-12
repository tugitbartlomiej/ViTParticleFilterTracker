"""Czy zapisany latent jest dokładnie baseline+alphaD? Rozjazd obala interwencję."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import shutil,time
from dane import *
def main():
    r=read(OUT/'RESULT.json');a=read(OUT/'AUDIT.json');curve=read(OUT/'CURVE.json')['records'];f=np.load(OUT/'BASE_BEST_LATENTS.npz');maps=np.load(OUT/'RAW_MAPS.npz');z=f['natural_baseline'];D=f['unit_ROI_lift_D'];J=f['J128_indices_display_only'];outside=np.flatnonzero(f['mask_weights']==0)
    assert len(curve)==len(maps['maps'])==817 and r['finite_points']==817
    threshold=np.round(np.arange(.5,.96,.05),2);aperr=0
    for i,c in enumerate(curve):
        assert c['r']==maps['r'][i] and c['alpha']==maps['alpha'][i]
        order=np.argsort(-np.array(c['all_scores']),kind='stable');ov=np.array(c['all_IoU'])[order]
        # One GT AP must be reciprocal of first qualifying prediction rank, or zero.
        expected=np.array([1/(np.flatnonzero(ov>=t)[0]+1) if np.any(ov>=t) else 0 for t in threshold]);aperr=max(aperr,float(abs(expected-np.array(c['AP_by_IoU'])).max()))
    assert aperr<1e-12
    latent_errors={}
    for name,chosen in r['selected'].items():
        if name=='baseline':continue
        zz=f['artificial_'+name];delta=zz-z;alpha=chosen['alpha'];err=float(abs(delta-alpha*D).max());assert err<2e-6
        assert np.array_equal(zz[outside],z[outside]);latent_errors[name]=err
        actualnorm=float(np.linalg.norm(delta));assert abs(actualnorm-abs(alpha))<max(2e-6,2e-6*abs(alpha))
    ss=r['selected'];intro=('\nPrzeszukano817 punktów r od -100 do100 i lokalnie doprecyzowano najlepszy q. '
        'Najlepszy dodatni punkt r=0.62183158, alpha=18.13375569: q wzrasta z0.74341142 do0.74373132 (zaledwie+0.00031990), '
        'confidence rośnie z0.86786950 do0.87667030, lecz IoU maleje z0.85659355 do0.84835923. '
        'AP50 iAP75 pozostają1.0; single-image meanAP(.50:.95) spada z0.8 do0.7. Nie wykazano poprawyAP ani lepszej geometrii w punkcie wybranym wedługq.\n\n'
        'Najlepsze meanAP w całej siatce nadal wynosi0.8, czyli tyle co baseline. Wśród474 remisów wybór przezq daje r=0.44487828. '
        'Duże r=+100 daje q=0.28291574, a r=-100 daje q=0.03051965: oba wyraźnie pogarszają detekcję. '
        'Najlepsza strona ujemna leży praktycznie przy zerze (r=-0.00010719), a przyrostq około jednej milionowej nie stanowi przekonującej korzyści wobec błędów numerycznych FP32.\n')
    lines=['\n## Korelacje coarsegrid (bez pvalue)\n','| Zakres alpha | Metryka | Pearson | Spearman |','|---|---|---:|---:|']
    for group,stats in r['grid_correlations'].items():
        for metric in ['confidence_top1','IoU_top1','q_star']:
            c=stats['coefficients'][metric];lines.append(f'|{group}|{metric}|{c["Pearson"]:.6f}|{c["Spearman"]:.6f}|')
    lines+=['','Na dodatniej stronie większe alpha w szerokim zakresie zwykle pogarsza IoU (Spearman około-0.991); confidence początkowo wzrasta, potem spada. Wniosek zależy od21 zadeklarowanychpunktów i nie jest prawem populacyjnym.']
    for p in [OUT/'RAPORT.md',HERE/'WYNIKI_SZEROKI_ALPHA.md']:
        txt=p.read_text(encoding='utf-8');first,rest=txt.split('\n',1);p.write_text(first+'\n'+intro+rest+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    a.update(saved_latent_equals_baseline_plus_alphaD=True,latent_max_errors=latent_errors,outside_ROI_bitwise_unchanged=True,all817_AP50_AP75_AP50_95_singleGT_rank_identity_verified=True,maximum_AP_rank_identity_error=aperr,additional_model_forward_during_final_audit=0);js(OUT/'AUDIT.json',a)
    prov=read(OUT/'PROVENANCE.json');prov['finalizer_sha256']=sha(HERE/'finalize.py');js(OUT/'PROVENANCE.json',prov)
    for name,h in prov['code_hashes'].items():assert sha(OUT/'EXECUTED_SOURCE'/name)==h
    shutil.copy2(HERE/'finalize.py',OUT/'EXECUTED_SOURCE/finalize.py')
    hs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','BIEG_single_image.json','stdout.log','stderr.log','finalize_stdout.log','finalize_stderr.log']};js(OUT/'HASHES.json',hs)
    bie=read(OUT/'BIEG_single_image.json');bie.update(audit_sha256=sha(OUT/'AUDIT.json'),hashes_sha256=sha(OUT/'HASHES.json'));js(OUT/'BIEG_single_image.json',bie)
    progress=read(OUT/'PROGRESS.json');progress.update(audited=True,finalized_unix=time.time(),hashes_sha256=sha(OUT/'HASHES.json'));js(OUT/'PROGRESS.json',progress)
    print('FINAL AUDIT PASSED:817 finite points, APs and latent contract; no extra model forward',flush=True)
if __name__=='__main__':main()
