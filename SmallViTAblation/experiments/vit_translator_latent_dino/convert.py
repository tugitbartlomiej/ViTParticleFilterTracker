"""Translate a saved full latent; optionally query a compatible FAISS index."""
import argparse,json
from pathlib import Path
import numpy as np
from translator import Translator
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--latent',type=Path,required=True)
    p.add_argument('--key',default='after');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--index',type=Path);p.add_argument('--index-config',type=Path);p.add_argument('--top-k',type=int,default=32)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if not a.out.resolve().is_relative_to(root):raise ValueError('Output must stay under SmallViTAblation')
    loaded=np.load(a.latent,allow_pickle=False)
    z=loaded[a.key] if isinstance(loaded,np.lib.npyio.NpzFile) else loaded
    feature=Translator(a.model).predict(z)
    a.out.parent.mkdir(parents=True,exist_ok=True);np.save(a.out,feature)
    if a.index:
        if not a.index_config:raise ValueError('Specify index-config to verify compatible DINO features')
        expected=json.loads((a.model.parent/'CONFIG.json').read_text(encoding='utf-8'))
        got=json.loads(a.index_config.read_text(encoding='utf-8-sig'))
        for field in ['dino_model','dino_weights_sha256','dino_output','preprocess']:
            if got.get(field)!=expected[field]:raise ValueError(f'Index representation mismatch: {field}')
        import faiss
        index=faiss.read_index(str(a.index));assert index.d==feature.shape[1]
        scores,ids=index.search(np.ascontiguousarray(feature),min(a.top_k,index.ntotal))
        a.out.with_suffix('.retrieval.json').write_text(json.dumps(dict(row_indices=ids.tolist(),cosine=scores.tolist(),index=str(a.index)),indent=2),encoding='utf-8')
if __name__=='__main__':main()
