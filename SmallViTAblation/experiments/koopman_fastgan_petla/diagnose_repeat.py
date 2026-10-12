"""Read-only szczegóły różnicy powtórzeń; nie zmienia tolerancji akceptacji."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from verify_states import equivalent
root=Path(sys.argv[1]);torch.set_num_threads(2)
a=torch.load(root/'control'/'FINAL.ckpt',map_location='cpu',weights_only=False)
for name in ['repeat','continuous']:
    b=torch.load(root/name/'FINAL.ckpt',map_location='cpu',weights_only=False)
    errors=sorted([(float((v.double()-b['state_dict'][k].double()).abs().max()),k) for k,v in a['state_dict'].items() if v.numel()],reverse=True)
    print(name,json.dumps(errors[:10]))
    for k in ['experiment_rng','lr_schedulers','global_step','epoch']:
        print(k,equivalent(a[k],b[k]))
    for f in sorted((root/'control').glob('LATENT_*.npz')):
        other=root/name/f.name
        if other.exists():
            x=np.load(f)['features'];y=np.load(other)['features'];print(f.name,float(np.max(np.abs(x-y))))
    print('samples_equal',json.loads((root/'control'/'samples_epoch001.json').read_text())==json.loads((root/name/'samples_epoch001.json').read_text()))
