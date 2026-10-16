"""Post-hoc initialization control: is selected cancellation present before training?"""
from pomiar import ROOT,OUT,PREV,LD,decode,jacobian,pair_sets,npcpu
import numpy as np
import pandas as pd
import torch,copy,json

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=np.load(PREV/'pomiar.npz');ids=data['ids'][:,:8]
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    net=LD.modul_modelu().MalyViT().to(dev).eval().requires_grad_(False)
    rows=[];rng=np.random.default_rng(151017)
    for step in [0,1000,33000,65640]:
        i=int(np.flatnonzero(data['steps']==step)[0])
        net.load_state_dict(torch.load(ROOT/f'assets/MalyViT_35000_od_zera_s47/wagi/k{step:05d}.pt',map_location='cpu',weights_only=True)['net'])
        pre=torch.tensor(data['x_pre'][i,:,:8].reshape(24,192),device=dev)
        post=torch.tensor(data['x_post'][i,:,:8].reshape(24,192),device=dev).double()
        z=net.bloki[-1].mlp.act(net.bloki[-1].mlp.fc1(net.bloki[-1].norm2(pre)))
        V=net.bloki[-1].mlp.fc2.weight;U=(z[:,:,None]*V.T[None]).double()
        head=torch.nn.Sequential(copy.deepcopy(net.norm),copy.deepcopy(net.glowa)).double()
        y,J=jacobian(head,post,torch.tensor(ids.reshape(-1),device=dev))
        A=npcpu((J[:,1:]@V.double())*z.double()[:,None,:]).transpose(0,2,1)
        for t in range(24):
            methods,_=pair_sets(A[t],npcpu(U[t]),rng)
            for method in ['functional_cancel','functional_reinforce','matched_cancel','matched_reinforce']:
                pp=torch.tensor(methods[method],device=dev)
                with torch.no_grad():
                    ds=decode(head(post[t][None,None]-U[t,pp]),torch.tensor(int(ids.reshape(-1)[t]),device=dev))-y[t]
                    dj=decode(head(post[t][None]-U[t,pp].sum(1)),torch.tensor(int(ids.reshape(-1)[t]),device=dev))-y[t]
                den=ds[:,:,1:].norm(dim=-1).sum(1)
                q=dj[:,1:].norm(dim=-1)/den
                for p in range(4):
                    rows.append(dict(step=step,image=t//8,token_slot=t%8,method=method,pair_slot=p,
                        q=float(q[p]),joint_px=float(dj[p,1:].norm()/2)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'kontrola_startu.csv',index=False)
    table=df.groupby(['step','method'])[['q','joint_px']].median()
    table.to_csv(OUT/'kontrola_startu_podsumowanie.csv');print(table.to_string())

if __name__=='__main__':main()
