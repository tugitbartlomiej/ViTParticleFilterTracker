"""Selectivity and sensitivity checks; inspect strongest intervention case."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'outputs/vit_semantyka_ruchu_20261008/pilot_v1'
spec=importlib.util.spec_from_file_location('experiment',Path(__file__).with_name('run.py'));exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
motion=pd.read_csv(out/'MOTION.csv');motion=motion[motion.role=='holdout'].copy()
motion['excess']=motion.rollback_effect-motion.random_norm_effect
probes=pd.read_csv(out/'PROBES.csv');probes=probes[probes.role=='holdout']
matrix=probes.pivot_table(index='basis',columns='perturbation',values='capture')
overlap=pd.read_csv(out/'OVERLAP.csv').pivot_table(index='basis',columns='other',values='overlap')
summary={}
for f in exp.FACTORS:
    data=motion[motion.factor==f]
    own=float(matrix.loc[f,f]);other=float(matrix.loc[f].drop(f).max())
    summary[f]=dict(own_capture=own,strongest_other_capture=other,selectivity_difference=own-other,mean_effect_minus_norm_control=float(data.excess.mean()),median_effect_minus_norm_control=float(data.excess.median()),mean_excluding_step500=float(data[data.step!=500].excess.mean()),mean_excluding_image11=float(data[data.image!=11].excess.mean()))
# Recompute largest case in float32 (do not use float16 activation archive for intervention).
module=exp.load_model_module();torch.set_num_threads(4)
model=module.MalyViT().cuda().eval();records=json.loads((exp.SOURCE/'dane/images.json').read_text(encoding='utf-8'));images=np.load(exp.SOURCE/'dane/obrazy.npy')
x=torch.tensor(images[11:12].transpose(0,3,1,2).copy(),dtype=torch.float32,device='cuda')/255
capture={};h=model.bloki[-1].mlp.act.register_forward_hook(lambda m,i,o:capture.update(z=o.clone()))
with torch.inference_mode():
    model.load_state_dict(torch.load(exp.SOURCE/'wagi/k0450.pt',map_location='cpu',weights_only=False)['net']);model(x);previous=capture['z']
    model.load_state_dict(torch.load(exp.SOURCE/'wagi/k0500.pt',map_location='cpu',weights_only=False)['net']);original,_=model(x);current=capture['z'];h.remove()
    state=np.load(out/'STATES_BASES.npz');q=torch.tensor(state['bases'][9,1],device='cuda')
    edit=((current-previous)@q)@q.T
    h=model.bloki[-1].mlp.act.register_forward_hook(lambda m,i,o:current-edit)
    changed,_=model(x);h.remove()
    _,box=exp.mask_box(records[11]);truth=torch.tensor(box[None],device='cuda')
    case={}
    for label,pred in [('original',original),('rollback_color',changed)]:
        b,s=module.dekoduj(pred,k=10)
        case[label]=dict(boxes=b[0].cpu().tolist(),scores=s[0].cpu().tolist(),ious=exp.iou(b[0],truth.expand(10,-1)).cpu().tolist())
    case['activation_edit_relative_norm']=float(edit.norm()/current.norm())
result=dict(selectivity=summary,overlap=overlap.to_dict(),dominant_case_step500_image11=case)
(out/'INTERPRETATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
lines=['','## Co wynika z danych','', 'Sondy nie dają obecnie czystego podziału na semantyczne moduły. Średnie nakładanie się podprzestrzeni (suma kwadratów cosinusów kątów głównych / 8) wynosi 0.54 dla koloru i pozycji, 0.55 dla koloru i proporcji oraz 0.68 dla pozycji i proporcji. Nazwa sondy opisuje sposób jej wyznaczenia, nie udowodnione znaczenie każdego kierunku.','']
for f,s in summary.items():
    lines.append(f'- {f}: własna odpowiedź przechwycona w {s["own_capture"]:.3f}, najsilniejsza obca w {s["strongest_other_capture"]:.3f}. Efekt minus kontrola normy: średnia {s["mean_effect_minus_norm_control"]:.5f}, mediana {s["median_effect_minus_norm_control"]:.5f}; bez kroku 500 średnia {s["mean_excluding_step500"]:.5f}; bez obrazu 11 {s["mean_excluding_image11"]:.5f}.')
lines+=['','Sonda pozycji jest częściowo selektywna; sondy koloru i proporcji nie przechwytują swojej perturbacji mocniej niż każdej obcej. Korelacje udziału ruchu z przyrostem IoU są bliskie zeru. Nie ma podstaw do wniosku, że poprawa detekcji wynika przede wszystkim z jednego z trzech czynników.','',f'Największy efekt ingerencji pochodzi z pojedynczego obrazu 11 na kroku 500: IoU najwyżej punktowanej ramki zmienia się z {case["original"]["ious"][0]:.4f} na {case["rollback_color"]["ious"][0]:.4f}. Względna norma edycji aktywacji: {case["activation_edit_relative_norm"]:.5f}. Ranking ramek jest nieciągły, dlatego nie należy utożsamiać tego skoku z globalną poprawą reprezentacji. Pełne ramki i pewności zapisano w INTERPRETATION.json.','', 'Weryfikacja QA.json: zgodność rzutu z niezależnym obliczeniem z archiwum, ortonormalność baz, brak zmian koloru poza maską, dokładność cyfrowej translacji, identyczny wynik zerowej interwencji.','', 'Dalszy eksperyment powinien użyć bardziej selektywnych sond (wyodrębnienie wspólnej odpowiedzi i model kontrastujący czynniki), izolowanej deformacji narzędzia z kontrolą artefaktów, kilku przebiegów treningu i nowych obrazów. Dopiero potem można przypisywać składowym ruchu znaczenie semantyczne.']
with (out/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
