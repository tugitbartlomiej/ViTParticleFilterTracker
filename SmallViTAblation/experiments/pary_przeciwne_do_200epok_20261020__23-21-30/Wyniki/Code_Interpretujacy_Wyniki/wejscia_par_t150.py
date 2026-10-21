"""Input similarity (W1 folded with LayerNorm gain, centered) of all same-token >=150 deg pairs at each epoch 60-200."""
from pathlib import Path
import csv,sys
import numpy as np
import torch

EXP=Path(__file__).resolve().parents[2]
CK=EXP.parent/'vit_kontynuacja_do_200epok_20261020__19-06-42/Wyniki/Dane/checkpoints'
sys.stdout.reconfigure(encoding='utf-8')
pairs=[r for r in csv.DictReader(open(EXP/'Wyniki/Dane/pary_epoki_60_200/CANDIDATE_PAIRS_60_200.csv',encoding='utf-8')) if int(r['count_ge_150'])>0]
rows=[]
for step in sorted({int(r['step']) for r in pairs}):
    st=torch.load(CK/f'k{step:06d}.pt',map_location='cpu',weights_only=False)['net'];vals=[]
    for r in (p for p in pairs if int(p['step'])==step):
        b=int(r['block'])-1;a,c=int(r['neuron_a']),int(r['neuron_b'])
        W1=st[f'bloki.{b}.mlp.fc1.weight'].double().numpy()*st[f'bloki.{b}.norm2.weight'].double().numpy()
        W1=W1-W1.mean(1,keepdims=True);cin=float(W1[a]@W1[c]/np.linalg.norm(W1[a])/np.linalg.norm(W1[c]))
        vals.append(cin);rows.append(dict(step=step,epoch=step*32/35000,block=b+1,a=a,b=c,w2_angle=float(r['weight_angle']),w1_cos=cin))
    v=np.array(vals)
    print(f"epoka {step*32/35000:5.0f}: par {len(v):2d} | cos W1 mediana {np.median(v):+.2f}, min {v.min():+.2f}, maks {v.max():+.2f} | >0,9: {(v>.9).sum()} | >0,5: {(v>.5).sum()}")
with open(EXP/'Wyniki/Dane/pary_epoki_60_200/WEJSCIA_PAR_T150.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
