"""Read-only opis stanu pętli pełnego checkpointu eksperymentu."""
import sys,json
import torch
for name in sys.argv[1:]:
    c=torch.load(name,map_location='cpu',weights_only=False)
    print(json.dumps(dict(path=name,epoch=c.get('epoch'),step=c.get('global_step'),loops=c.get('loops')),indent=2,default=str))
