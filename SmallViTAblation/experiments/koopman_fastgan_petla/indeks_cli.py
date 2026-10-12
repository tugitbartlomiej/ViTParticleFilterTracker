"""CLI izolujące pamięć niezależnej wyszukiwarki."""
import argparse
from wyszukiwanie import build_index,query_index
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('pool','data-root','hub','dino-weights','index','image','exclude'):ap.add_argument('--'+k)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--topk',type=int,default=20)
    a=ap.parse_args()
    if a.index:query_index(a.index,a.image,a.exclude or '[]',a.out,a.topk,device=a.device)
    else:build_index(a.pool,a.data_root,a.out,torch_hub_dir=a.hub,weights=a.dino_weights,device=a.device)
