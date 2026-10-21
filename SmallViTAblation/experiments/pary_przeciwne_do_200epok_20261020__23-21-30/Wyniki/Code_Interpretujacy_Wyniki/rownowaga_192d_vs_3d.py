"""Collective balance of all 768 MLP forces on a token: full 192D versus the app's 3D projection P3.

R = ||sum_c u_c|| / sum_c ||u_c||  (1 = all forces aligned, 0 = perfect balance), u_c = z_c * W2[:,c] (fc2 bias excluded).
P3 = the app's projection (lasery_dane.py): 3 principal directions of the normalized fc2 columns of the epoch-200 checkpoint.
Usage: python rownowaga_192d_vs_3d.py <folder with frame .npz files and rows.json>
"""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys
import numpy as np
import torch

EXP=Path(__file__).resolve().parents[2]
CK=EXP.parent/'vit_kontynuacja_do_200epok_20261020__19-06-42'
OUT=EXP/'Wyniki/Dane/rownowaga_192d_vs_3d';OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('model',CK/'Kod_eksperymentu/model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def p3_app(V):
    # Same recipe as app/lasery_dane.py: normalize columns, center across columns, top-3 left singular vectors.
    Vn=V/np.maximum(np.linalg.norm(V,axis=0,keepdims=True),1e-12)
    U,_,_=np.linalg.svd(Vn-Vn.mean(1,keepdims=True),full_matrices=False)
    return U[:,:3].T

def balance(z,V,P):
    # z: (T,768) GELU outputs, V: (192,768) fc2 weight, P: (3,192). Returns per-token measures.
    s=np.abs(z)*np.linalg.norm(V,axis=0);PV=P@V;s3=np.abs(z)*np.linalg.norm(PV,axis=0)
    F=np.linalg.norm(z@V.T,axis=1);F3=np.linalg.norm(z@PV.T,axis=1)
    return dict(R192=F/s.sum(1),R3=F3/s3.sum(1),Rrand192=np.sqrt((s**2).sum(1))/s.sum(1),Rrand3=np.sqrt((s3**2).sum(1))/s3.sum(1),
                visible_net=F3/F,visible_total=s3.sum(1)/s.sum(1))

def main():
    sys.stdout.reconfigure(encoding='utf-8');src=Path(sys.argv[1])
    rows=json.loads((src/'rows.json').read_text())
    imgs=[]
    for r in rows:
        p=src/Path(r['image']).name;assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256']
        with np.load(p) as d:imgs.append(d['rgb'].copy())
    x=torch.as_tensor(np.stack(imgs)).permute(0,3,1,2).float()/255
    st200=torch.load(CK/'Wyniki/Dane/checkpoints/k218750.pt',map_location='cpu',weights_only=False)['net']
    P={b:p3_app(st200[f'bloki.{b}.mlp.fc2.weight'].double().numpy()) for b in range(6)}
    net=M.MalyViT().eval().requires_grad_(False);cache={}
    for b,blk in enumerate(net.bloki):blk.mlp.act.register_forward_hook(lambda m,a,o,b=b:cache.__setitem__(b,o.detach()))
    ids=np.arange(600);cx=(ids%30+.5)*16;cy=(ids//30+.5)*16
    out=[];tok_rows=[]
    for name,epoch in [('k065640.pt',60),('k218750.pt',200)]:
        st=torch.load(CK/'Wyniki/Dane/checkpoints'/name,map_location='cpu',weights_only=False)['net'];net.load_state_dict(st,strict=True)
        with torch.inference_mode():raw=net(x)[0].reshape(len(rows),600,5).double().numpy()
        for b in range(6):
            V=st[f'bloki.{b}.mlp.fc2.weight'].double().numpy();z=cache[b].double().numpy()
            m=balance(z.reshape(-1,768),V,P[b]);zc=(z-z.mean(1,keepdims=True)).reshape(-1,768);mc=balance(zc,V,P[b])
            out.append(dict(epoch=epoch,block=b+1,**{f'median_{k}':float(np.median(v)) for k,v in m.items()},
                frac_tokens_3d_looks_more_balanced=float((m['R3']<m['R192']).mean()),
                median_R192_centered=float(np.median(mc['R192'])),median_R3_centered=float(np.median(mc['R3'])),
                P3_energy_of_fc2=float((np.linalg.norm(P[b]@V)**2)/(np.linalg.norm(V)**2))))
            for j,r in enumerate(rows):
                g=np.array(r['gt'][0]);gc=((g[0]+g[2])/2,(g[1]+g[3])/2)
                d2=(cx-gc[0])**2+(cy-gc[1])**2
                for kind,t in [('zwycięzca',int(np.argmax(raw[j,:,0]))),('środek GT',int(np.argmin(d2))),('tło',int(np.argmax(d2)))]:
                    k=j*600+t;tok_rows.append(dict(epoch=epoch,block=b+1,frame=r['id'],token_kind=kind,token=t,
                        **{key:float(v[k]) for key,v in m.items()}))
    for fn,rr in [('PODSUMOWANIE_BLOKI.csv',out),('TOKENY.csv',tok_rows)]:
        with (OUT/fn).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    (OUT/'PROVENANCE.json').write_text(json.dumps(dict(frames=[r['id'] for r in rows],frame_sha256=[r['image_sha256'] for r in rows],
        checkpoints=['k065640.pt','k218750.pt'],projection='app P3 (lasery_dane.py recipe) from epoch-200 fc2 of each block',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print('epoka blok | mediana R 192D | R 3D | R losowe 192D | R losowe 3D | widoczna część wypadkowej | widoczna część sumy sił | tokeny z pozorną większą równowagą | R 192D/3D (z−z̄)')
    for r in out:
        print(f"{r['epoch']:4d} b{r['block']} | {r['median_R192']:.3f} | {r['median_R3']:.3f} | {r['median_Rrand192']:.3f} | {r['median_Rrand3']:.3f} | "
              f"{r['median_visible_net']*100:5.1f}% | {r['median_visible_total']*100:5.1f}% | {r['frac_tokens_3d_looks_more_balanced']*100:5.1f}% | {r['median_R192_centered']:.3f}/{r['median_R3_centered']:.3f}")
    print('\nblok 6, epoka 200, wybrane tokeny:')
    for r in tok_rows:
        if r['block']==6 and r['epoch']==200:
            print(f"{r['frame']:22s} {r['token_kind']:10s} R192={r['R192']:.3f} R3={r['R3']:.3f} widać {r['visible_net']*100:5.1f}% wypadkowej")

if __name__=='__main__':main()
