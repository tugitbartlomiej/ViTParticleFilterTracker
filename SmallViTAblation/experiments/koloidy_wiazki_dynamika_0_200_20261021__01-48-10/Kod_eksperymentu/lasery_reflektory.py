"""Per-laser view of block 6 over training (26 checkpoints, 12 frames): power, direction, duty cycle, tool/background
selectivity, aiming at the tool-background axis, contribution to the separation (sorting vs braking), rotation, elite turnover.
Usage: python lasery_reflektory.py <frames folder> <history weights folder>
"""
from pathlib import Path
import csv,importlib.util,json,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dyn',HERE/'dynamika_koloidow.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
ELITE=50

def gini(x):
    x=np.sort(np.abs(x));n=len(x);return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum())) if x.sum()>0 else 0.

def main():
    sys.stdout.reconfigure(encoding='utf-8');frames,hist=Path(sys.argv[1]),Path(sys.argv[2]);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=importlib.util.spec_from_file_location('model',D.CONT/'Kod_eksperymentu/model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    rows=json.loads((frames/'rows.json').read_text());imgs=[]
    for r in rows:
        p=frames/Path(r['image']).name;assert D.sha(p)==r['image_sha256']
        with np.load(p) as d:imgs.append(d['rgb'].copy())
    x_in=torch.as_tensor(np.stack(imgs),device=dev).permute(0,3,1,2).float()/255
    ids=torch.arange(600,device=dev);cx=(ids%30+.5)*16;cy=(ids//30+.5)*16
    tool=torch.zeros(len(rows),600,dtype=torch.bool,device=dev)
    for f,r in enumerate(rows):
        for x1,y1,x2,y2 in r['gt']:tool[f]|=(cx>=x1)&(cx<=x2)&(cy>=y1)&(cy<=y2)
    net=M.MalyViT().to(dev).eval().requires_grad_(False);cache={};blk=net.bloki[D.K]
    blk.norm2.register_forward_hook(lambda m,a,o:cache.__setitem__('x',a[0].detach()))
    blk.mlp.act.register_forward_hook(lambda m,a,o:cache.__setitem__('z',o.detach()))
    keys=['power','duty_pos','active_frac','mean_abs_z','s_tool','s_bg','selectivity','aim','sep_contrib','rotation_deg']
    arr={k:[] for k in keys};dirs=[];summary=[];prev=None;prev_elite=None;steps=[]
    for step,path,h in D.checkpoints(hist):
        assert D.sha(path)==h
        st=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(st.get('net',st),strict=True)
        with torch.inference_mode():net(x_in)
        X=cache['x'].double();Z=cache['z'].double();V=blk.mlp.fc2.weight.double();pw=V.norm(dim=0);Vh=V/pw.clamp_min(1e-12)
        S=Z.abs()*pw;act=S>.05*S.max(-1,keepdim=True).values
        st_t=S[tool].mean(0);st_b=S[~tool].mean(0)
        aim=[];contrib=[]
        for f in range(len(rows)):
            m=tool[f];x=X[f]-X[f].mean(0);R=x.pow(2).sum(1).mean().sqrt();d=x[m].mean(0)-x[~m].mean(0);u=d/d.norm()
            aim.append(Vh.T@u)
            # Laser c moves the tool centroid by mean_tool(z_c) v_c and the background centroid by mean_bg(z_c) v_c;
            # its effect on the separation along u is (mean_tool z - mean_bg z) (v_c.u) / R. Positive = sorts, negative = brakes.
            contrib.append((Z[f][m].mean(0)-Z[f][~m].mean(0))*(V.T@u)/R)
        rot=torch.rad2deg(torch.arccos((Vh*prev).sum(0).clamp(-1,1))) if prev is not None else torch.full((768,),float('nan'),device=dev,dtype=torch.float64)
        vals=dict(power=pw,duty_pos=(Z>0).double().mean((0,1)),active_frac=act.double().mean((0,1)),mean_abs_z=Z.abs().mean((0,1)),
            s_tool=st_t,s_bg=st_b,selectivity=(st_t-st_b)/(st_t+st_b).clamp_min(1e-30),aim=torch.stack(aim).mean(0),
            sep_contrib=torch.stack(contrib).mean(0),rotation_deg=rot)
        for k in keys:arr[k].append(vals[k].float().cpu().numpy())
        dirs.append(Vh.T.float().cpu().numpy());prev=Vh;steps.append(step)
        use=(S.mean((0,1))).cpu().numpy();elite=set(np.argsort(-use)[:ELITE].tolist())
        sc=vals['sep_contrib'].cpu().numpy();brk=np.sort(sc[sc<0]);srt=sc[sc>0]
        e=step*32/35000;de=None
        r=dict(epoch=e,lasers_never_active=int((vals['active_frac']==0).sum()),lasers_rare_lt_0_1pct=int((vals['active_frac']<1e-3).sum()),
            power_gini=gini(vals['power'].cpu().numpy()),use_gini=gini(use),elite_kept=len(elite&prev_elite)/ELITE if prev_elite else None,
            spearman_power_vs_use=float(np.corrcoef(np.argsort(np.argsort(vals['power'].cpu().numpy())),np.argsort(np.argsort(use)))[0,1]),
            tool_selective=int((vals['selectivity']>.5).sum()),bg_selective=int((vals['selectivity']<-.5).sum()),
            median_abs_aim=float(vals['aim'].abs().median()),aim_q99=float(vals['aim'].abs().quantile(.99)),
            sort_total=float(srt.sum()),brake_total=float(brk.sum()),net_sep_change=float(sc.sum()),
            brakers_for_half=int(np.searchsorted(np.cumsum(brk)/brk.sum(),.5)+1) if len(brk) else 0,
            sorters_for_half=int(np.searchsorted(np.cumsum(np.sort(srt)[::-1])/srt.sum(),.5)+1) if len(srt) else 0,
            mean_duty_pos=float(vals['duty_pos'].mean()),rotation_mean=float(rot.nanmean()) if prev_elite else None,
            spearman_rotation_vs_use=float(np.corrcoef(np.argsort(np.argsort(rot.cpu().numpy())),np.argsort(np.argsort(use)))[0,1]) if prev_elite else None)
        prev_elite=elite;summary.append(r)
        print(f"{e:6.1f} martwe {r['lasers_never_active']:3d} | elita zachowana {r['elite_kept'] if r['elite_kept'] is not None else float('nan'):.2f} | "
              f"selektywne narzędzie {r['tool_selective']:3d} tło {r['bg_selective']:3d} | sortuje {r['sort_total']:+.3f} hamuje {r['brake_total']:+.3f} "
              f"(połowę hamowania daje {r['brakers_for_half']} laserów) | gini mocy {r['power_gini']:.2f} użycia {r['use_gini']:.2f}",flush=True)
    np.savez_compressed(D.OUT/'LASERY_26_CHECKPOINTOW.npz',steps=np.array(steps),directions=np.stack(dirs),**{k:np.stack(v) for k,v in arr.items()})
    with open(D.OUT/'LASERY_PODSUMOWANIE.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

if __name__=='__main__':main()
