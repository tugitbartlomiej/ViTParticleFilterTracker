"""Does the block-6 laser kick increase or decrease the tool/background separation? Compares centroid distance before
and after x -> x + F (F = sum of beam forces; fc2 bias is common to all colloids and does not change separations)."""
from pathlib import Path
import csv,importlib.util,json,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dyn',HERE/'dynamika_koloidow.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

def main():
    sys.stdout.reconfigure(encoding='utf-8');frames,hist=Path(sys.argv[1]),Path(sys.argv[2]);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=importlib.util.spec_from_file_location('model',D.CONT/'Kod_eksperymentu/model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    rows=json.loads((frames/'rows.json').read_text());imgs=[]
    for r in rows:
        with np.load(frames/Path(r['image']).name) as d:imgs.append(d['rgb'].copy())
    x_in=torch.as_tensor(np.stack(imgs),device=dev).permute(0,3,1,2).float()/255
    ids=torch.arange(600,device=dev);cx=(ids%30+.5)*16;cy=(ids//30+.5)*16
    tool=torch.zeros(len(rows),600,dtype=torch.bool,device=dev)
    for f,r in enumerate(rows):
        for x1,y1,x2,y2 in r['gt']:tool[f]|=(cx>=x1)&(cx<=x2)&(cy>=y1)&(cy<=y2)
    net=M.MalyViT().to(dev).eval().requires_grad_(False);cache={};blk=net.bloki[D.K]
    blk.norm2.register_forward_hook(lambda m,a,o:cache.__setitem__('x',a[0].detach()))
    blk.mlp.act.register_forward_hook(lambda m,a,o:cache.__setitem__('z',o.detach()))
    out=[]
    for step,path,h in D.checkpoints(hist):
        assert D.sha(path)==h
        st=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(st.get('net',st),strict=True)
        with torch.inference_mode():net(x_in)
        X=cache['x'].double();F=cache['z'].double()@blk.mlp.fc2.weight.double().T;acc=[]
        for f in range(len(rows)):
            m=tool[f];x=X[f]-X[f].mean(0);R=x.pow(2).sum(1).mean().sqrt();Ff=F[f]
            d0=x[m].mean(0)-x[~m].mean(0);d1=d0+Ff[m].mean(0)-Ff[~m].mean(0);u=d0/d0.norm()
            common=Ff.mean(0)
            acc.append([float(d0.norm()/R),float(d1.norm()/R),float((Ff[m].mean(0)@u)/R),float((Ff[~m].mean(0)@u)/R),float(common.norm()/R),
                        float((Ff-common).norm(dim=1).mean()/R)])
        a=np.mean(acc,0);out.append(dict(epoch=step*32/35000,sep_before=a[0],sep_after=a[1],tool_push_along_sep=a[2],bg_push_along_sep=a[3],
            common_push=a[4],individual_push=a[5]))
        print(f"{step*32/35000:6.1f} sep {a[0]:.3f} -> {a[1]:.3f} ({(a[1]/a[0]-1)*100:+.1f}%) | push along tool-bg axis: tool {a[2]:+.3f}, bg {a[3]:+.3f} | "
              f"wspólne pchnięcie {a[4]:.3f}, indywidualne {a[5]:.3f} (jednostki R)")
    with open(D.OUT/'KOPNIECIE_A_ROZDZIAL.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0]));w.writeheader();w.writerows(out)

if __name__=='__main__':main()
