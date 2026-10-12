"""Czy wspólny regulator działa bez etykiety obrazu? Inferencja nie dopasowuje parametrów."""
import argparse
from pathlib import Path
import cv2,numpy as np,torch
from dane_bazowe import CKPT,js,sha,OUT
from model_snapshot import MalyViT,dekoduj
from regulator import Regulator,predict_tail

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True)
    p.add_argument('--model',type=Path,default=OUT/'model_selected.pt');p.add_argument('--output',type=Path,default=OUT/'INFERENCE.json');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(a.model,map_location='cpu',weights_only=False)
    assert sha(CKPT)==saved['provenance']['checkpoint_sha']
    net=MalyViT().cuda().eval();net.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True)['net']);net.requires_grad_(False)
    stats={k:v.cuda() for k,v in saved['stats'].items()};reg=Regulator(saved['config'],stats).eval();reg.load_state_dict(saved['state'])
    J=saved['channels'].cuda();im=cv2.imread(str(a.image));assert im is not None
    native_height,native_width=im.shape[:2]
    im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(im).permute(2,0,1)[None].float().cuda()/255
    cache={};b=net.bloki[-1]
    h1=b.mlp.act.register_forward_hook(lambda m,args,o:cache.update(z=o[:,:,J].detach()))
    h2=b.register_forward_hook(lambda m,args,o:cache.update(q=o.detach()))
    with torch.no_grad():
        original,_=net(x);h1.remove();h2.remove();edited=predict_tail(net,J,cache['z'],cache['q'],reg)
        records={}
        scale=torch.tensor([native_width/480,native_height/320,native_width/480,native_height/320],device='cuda')
        for name,o in [('baseline',original),('regulated',edited)]:
            boxes,scores=dekoduj(o)
            records[name]=dict(boxes_xyxy_native=(boxes[0]*scale).cpu().tolist(),scores=scores[0].cpu().tolist())
    js(a.output,dict(image=a.image,model=a.model,native_size=[native_width,native_height],config=saved['config'],
                     labels_used=False,optimization_on_image=False,detections=records))
    print(a.output)

if __name__=='__main__':main()
