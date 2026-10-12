"""Czy val lock i zapisane predykcje odtwarzają wynik? Rozjazd AP obala selekcję."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import csv,shutil,time
from dane import *
from run import TCAV,MAIN,GRID,source_equivalence,metrics,weights_hash
def main():
    rows,pairs,w,J,ids=load();result=read(OUT/'RESULT.json');lock=read(OUT/'LOCK.json');cal=read(OUT/'CALIBRATION.json');curves=read(OUT/'VALIDATION_CURVES.json');audit=read(OUT/'AUDIT.json');lockhash=sha(OUT/'LOCK.json')
    assert result['status']=='complete' and audit['status']=='passed';worst=0;checked=0
    for name,chosen in lock['chosen'].items():
        p=np.load(OUT/f'VAL_PRED_{name}.npz');aps=[]
        for j in range(len(GRID)):
            pred={k:p[k][:,j] for k in ['boxes','scores','ious']};err=source_equivalence(pred,[bbox(rows[i]) for i in ids['val']]);worst=max(err,worst);aps.append(metrics(pred)['AP50']);checked+=1
        expected=max(range(len(GRID)),key=lambda j:(aps[j],-abs(GRID[j]),-GRID[j]));assert expected==chosen['grid_index'];assert aps[expected]>=curves['baseline']['AP50']
        p=np.load(OUT/f'TEST_PRED_{name}.npz');pred={k:p[k] for k in ['boxes','scores','ious']};worst=max(worst,source_equivalence(pred,[bbox(rows[i]) for i in ids['test']]));checked+=1
        assert abs(metrics(pred)['AP50']-result['summary'][name]['metrics']['AP50'])<1e-12
    assert sha(OUT/'LOCK.json')==lockhash
    original=np.load(TCAV/'CAV_J128.npz');assert np.array_equal(original['axes0'],J)
    with (OUT/'J128_SEPARATOR_COEFFICIENTS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['channel0','channel1','unit_raw_CAV','raw_separator_coefficient','standardized_separator_coefficient','train_mean','train_SD'])
        for n,j in enumerate(J):writer.writerow([int(j),int(j+1),float(original['unit'][j]),float(original['raw_coef'][j]),float(original['raw_coef'][j]*original['sd'][n]),float(original['mean'][n]),float(original['sd'][n])])
    js(OUT/'GT_LABELS.json',dict(class_id=0,class_name='annotated_tool',one_GT_per_image=True,source_category='cystotome',records=[dict(source_index=i,frame_id=r['frame_id'],film=r['video_id'],split=r['tcav_split'],category=r['source_category'],class_id=0,bbox_source_xywh=r['bbox_source_480']) for i,r in enumerate(rows)]))
    records=read(OUT/'TEST_INTERVENTIONS.json');feasibility={k:dict(mean_fraction_below_GELU_floor=float(np.mean([r['ROI_fraction_below_GELU_floor'] for r in records if r['method']==k])),maximum_relative_norm=float(max(r['relative_norm'] for r in records if r['method']==k))) for k in lock['chosen']}
    summary='\nNa zablokowanym teście AP50: baseline 45.7589 procent; full768 46.6715 procent (+0.9127 punktu), historyczna LDA 46.6652 procent (+0.9064 punktu), J128 43.3084 procent (-2.4504 punktu). Wszystkie CI zmian względem baseline i najlepszego po val random obejmują zero. Nie wykazano pewnej przewagi jakości detekcji.\n\nKażdy główny kierunek wybrał r=+0.5, alpha=14.61536598, czyli górną granicę zadeklarowanej siatki. Nie określono optimum poza tym zakresem. Przy tym alpha średni udział współrzędnych ROI poniżej teoretycznej dolnej granicy GELU wynosi full768 17.42 procent, LDA 18.72 procent, J128 4.18 procent. Nie przycinano współrzędnych, aby zachować stały kierunek. To interwencje diagnostyczne w aktywacjach, bez dowodu realizowalności przez obraz.\n'
    for path in [OUT/'RAPORT.md',HERE/'WYNIKI_MAKSYMALIZACJA.md']:
        txt=path.read_text(encoding='utf-8');first,rest=txt.split('\n',1);path.write_text(first+'\n'+summary+rest,encoding='utf-8')
    result['selected_at_grid_boundary']={k:abs(v['r'])==.5 for k,v in lock['chosen'].items()};result['feasibility_diagnostic']=feasibility;js(OUT/'RESULT.json',result)
    audit.update(all_validation_curve_points_source_AP50_verified=True,source_AP50_audit_prediction_sets=checked,max_source_AP50_error=worst,lock_hash_unchanged=True,chosen_alpha_verified_against_saved_val_curves=True,additional_model_forwards=0);js(OUT/'AUDIT.json',audit)
    prov=read(OUT/'PROVENANCE.json');prov['finalizer_sha256']=sha(HERE/'finalize.py');js(OUT/'PROVENANCE.json',prov)
    for name,h in prov['code_hashes'].items():assert sha(OUT/'EXECUTED_SOURCE'/name)==h
    shutil.copy2(HERE/'finalize.py',OUT/'EXECUTED_SOURCE/finalize.py')
    hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['HASHES.json','PROGRESS.json','BIEG_scalar_grid.json','stdout.log','stderr.log','finalize_stdout.log','finalize_stderr.log']};js(OUT/'HASHES.json',hashes)
    bie=read(OUT/'BIEG_scalar_grid.json');bie.update(result_sha256=sha(OUT/'RESULT.json'),audit_sha256=sha(OUT/'AUDIT.json'),hashes_sha256=sha(OUT/'HASHES.json'));js(OUT/'BIEG_scalar_grid.json',bie)
    p=read(OUT/'PROGRESS.json');p.update(audited=True,finalized_unix=time.time(),hashes_sha256=sha(OUT/'HASHES.json'));js(OUT/'PROGRESS.json',p)
    print('FINAL AUDIT PASSED',checked,'saved prediction sets, max sourceAP error',worst,'no extra model forward',flush=True)
def bbox(row):
    x,y,w,h=row['bbox_source_480'];return np.array([x,y,x+w,y+h],np.float32)
if __name__=='__main__':main()
