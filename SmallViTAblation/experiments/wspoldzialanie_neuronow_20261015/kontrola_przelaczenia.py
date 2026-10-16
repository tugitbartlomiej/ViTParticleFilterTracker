"""Independent full-forward replay of the largest late group displacement (post hoc)."""
from pathlib import Path
from pomiar import ROOT,OUT,LD,npcpu,iou,decode
import numpy as np
import torch,json,copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    groups=json.loads((OUT/'grupy_neuronow.json').read_text())
    selection=next(x for x in groups if x['step']==55000 and x['image']==2 and x['method']=='reinforce' and x['size']==32)
    pairs=np.array(selection['neurons']);hist=np.load(ROOT/'app/cache_uczenia/lasery/2cca2743e1c74b14/dane.npz')
    rgb=hist['rgb'];gt=hist['gt'];image=torch.tensor(rgb,device='cuda').permute(2,0,1)[None].float()/255
    M=LD.modul_modelu();net=M.MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(ROOT/'assets/MalyViT_35000_od_zera_s47/wagi/k55000.pt',map_location='cpu',weights_only=True)['net'])
    pam={};net.norm.register_forward_pre_hook(lambda m,a:pam.__setitem__('post',a[0].detach()))
    head=torch.nn.Sequential(copy.deepcopy(net.norm),copy.deepcopy(net.glowa)).double()
    entries=[];maps=[]
    for label,neurons in [('base',[]),('half_a',pairs[:,0]),('half_b',pairs[:,1]),('joint',pairs.flatten())]:
        def ablate(m,a,o):
            q=o.clone();q[:,:,neurons]=0;return q
        h=net.bloki[-1].mlp.act.register_forward_hook(ablate) if len(neurons) else None
        with torch.no_grad():raw,_=net(image)
        if h is not None:h.remove()
        raw=raw.reshape(600,5);idx=raw[:,0].argsort(descending=True)[:5]
        with torch.no_grad():
            official,conf=M.dekoduj(raw.reshape(1,20,30,5),k=1)
            double_raw=head(pam['post'].double()).reshape(600,5)
            double_idx=double_raw[:,0].argmax()
        assert int(idx[0])==int(double_idx)
        box=npcpu(official[0,0]);maps.append(npcpu(raw[:,0]))
        entries.append(dict(variant=label,neurons=list(map(int,neurons)),winner=int(idx[0]),
            row=int(idx[0])//30,column=int(idx[0])%30,box=box.tolist(),iou=float(iou(box,gt)),
            confidence=float(conf[0,0]),logit=float(raw[idx[0],0]),gap=float(raw[idx[0],0]-raw[idx[1],0]),
            top5_ids=npcpu(idx).tolist(),top5_logits=npcpu(raw[idx,0]).tolist(),
            double_winner=int(double_idx),double_float_max_logit_error=float((double_raw[:,0]-raw[:,0]).abs().max())))
    winner_ids=sorted(set(e['winner'] for e in entries))
    info=dict(step=55000,image=2,gt=gt.tolist(),selected_post_hoc=True,variants=entries,
              winner_logits={str(i):[float(m[i]) for m in maps] for i in winner_ids})
    (OUT/'przelaczenie_weryfikacja.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    titles=['Bez wyłączenia','Wyłączona połowa A','Wyłączona połowa B','Wyłączone A i B']
    for ax,e,title in zip(axes,entries,titles):
        ax.imshow(rgb)
        for box,color in [(gt,'#50ee60'),(e['box'],'#ff7050')]:
            ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,ec=color,lw=1.4))
        ax.scatter([(e['column']+.5)*16],[(e['row']+.5)*16],c='#ff7050',s=12)
        ax.set_title(f'{title}\nToken {e["winner"]}; IoU {e["iou"]:.4f}',fontsize=10);ax.axis('off')
    fig.suptitle('Przełączenie zwycięskiej detekcji: obraz 3, krok 55000; zielona = GT\nPrzypadek wybrany po analizie jako największy skok w późnych checkpointach',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'przelaczenie_ramki.png',dpi=180);plt.close(fig)
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
