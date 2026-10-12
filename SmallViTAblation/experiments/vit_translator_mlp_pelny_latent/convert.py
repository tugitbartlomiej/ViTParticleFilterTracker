"""Convert a saved full latent with a trained checkpoint-specific MLP."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from model import load
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--latent',type=Path,required=True)
    p.add_argument('--key',default='after');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    if not a.out.resolve().is_relative_to(root):raise ValueError('Output must stay under SmallViTAblation')
    torch.set_num_threads(4);m,ck=load(a.model)
    source=np.load(a.latent,allow_pickle=False);z=source[a.key] if isinstance(source,np.lib.npyio.NpzFile) else source
    if z.shape==(600,768):z=z[None]
    with torch.inference_mode():feature=m(torch.tensor(z,dtype=torch.float32))[0].numpy()
    assert np.isfinite(feature).all();np.testing.assert_allclose(np.linalg.norm(feature,axis=1),1.,atol=1e-5)
    a.out.parent.mkdir(parents=True,exist_ok=True);np.save(a.out,feature)
    a.out.with_suffix('.json').write_text(json.dumps(dict(model=str(a.model),latent=str(a.latent),key=a.key,shape=list(feature.shape),vit_checkpoint_sha256=ck['config']['vit_checkpoint_sha256'],dino_weights_sha256=ck['config']['dino_weights_sha256'],preprocess=ck['config']['preprocess']),indent=2),encoding='utf-8')
if __name__=='__main__':main()
