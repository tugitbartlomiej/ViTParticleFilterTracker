"""Fate of the same-token >=150 deg pairs present at epoch 110: input cosine and W2 angle at 110 vs 200, survivors vs lost."""
from pathlib import Path
import csv,sys
import numpy as np
import torch

EXP=Path(__file__).resolve().parents[2]
CK=EXP.parent/'vit_kontynuacja_do_200epok_20261020__19-06-42/Wyniki/Dane/checkpoints'
sys.stdout.reconfigure(encoding='utf-8')
pairs=[r for r in csv.DictReader(open(EXP/'Wyniki/Dane/pary_epoki_60_200/CANDIDATE_PAIRS_60_200.csv',encoding='utf-8')) if int(r['count_ge_150'])>0]
ident=lambda s:{(int(r['block']),int(r['neuron_a']),int(r['neuron_b'])) for r in pairs if int(r['step'])==s}
A,Z=ident(120313),ident(218750)

def geometry(step,keys):
    st=torch.load(CK/f'k{step:06d}.pt',map_location='cpu',weights_only=False)['net'];out={}
    for b,a,c in keys:
        W1=st[f'bloki.{b-1}.mlp.fc1.weight'].double().numpy()*st[f'bloki.{b-1}.norm2.weight'].double().numpy();W1=W1-W1.mean(1,keepdims=True)
        V=st[f'bloki.{b-1}.mlp.fc2.weight'].double().numpy().T;bh=st[f'bloki.{b-1}.mlp.fc1.bias'].double().numpy()+st[f'bloki.{b-1}.mlp.fc1.weight'].double().numpy()@st[f'bloki.{b-1}.norm2.bias'].double().numpy()
        cos=lambda x,y:float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
        out[(b,a,c)]=(cos(W1[a],W1[c]),float(np.degrees(np.arccos(np.clip(cos(V[a],V[c]),-1,1)))),float(abs(bh[a]-bh[c])))
    return out
g110,g200=geometry(120313,A),geometry(218750,A)
print('pary w epoce 110:',len(A),'| przetrwały do 200:',len(A&Z),'| zniknęły:',len(A-Z),'| nowe w 200:',len(Z-A))
for name,keys in [('przetrwały',A&Z),('zniknęły',A-Z)]:
    d=np.array([[*g110[k],*g200[k]] for k in keys])
    print(f"{name:10s} cos W1: {np.median(d[:,0]):+.2f} -> {np.median(d[:,3]):+.2f} | kąt W2: {np.median(d[:,1]):5.1f}° -> {np.median(d[:,4]):5.1f}° | "
          f"różnica biasów |Δb̂|: {np.median(d[:,2]):.2f} -> {np.median(d[:,5]):.2f} | ile ma kąt W2 > 30° w ep.200: {(d[:,4]>30).sum()}/{len(d)}")
for k in sorted(A-Z):
    print('  zniknęła',k,'cos W1 %.2f->%.2f, kąt W2 %.1f°->%.1f°, |Δb̂| %.2f->%.2f'%(g110[k][0],g200[k][0],g110[k][1],g200[k][1],g110[k][2],g200[k][2]))
