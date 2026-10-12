"""Czy cel przekracza konieczne granice aktywacji zamrożonego ViT?
Brak przekroczeń nie dowodzi osiągalności. Przekroczenie ścisłej górnej granicy
wyklucza dokładny cel dla tej warstwy niezależnie od użytego obrazu.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_latent768_pelny_zbior'
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
state=torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net']
W=state['bloki.5.mlp.fc1.weight'].double().numpy()
b=state['bloki.5.mlp.fc1.bias'].double().numpy()
gamma=state['bloki.5.norm2.weight'].double().numpy()
beta=state['bloki.5.norm2.bias'].double().numpy()
assert W.shape==(768,192) and gamma.shape==(192,)
# LayerNorm standardized vector q has sum(q)=0 and ||q|| <= sqrt(D).
# h_j=(W_j*gamma) dot q + W_j dot beta + b_j.
# Centering coefficients exploits sum(q)=0; Cauchy-Schwarz gives this bound.
# GELU(h)=h*Phi(h) <= max(0,h), so no monotonicity assumption is needed.
a=W*gamma[None]
c=W@beta+b
upper_h=c+np.sqrt(192)*np.linalg.norm(a-a.mean(1,keepdims=True),axis=1)
# Loose numeric safety margin avoids using tiny float32 roundoff as evidence.
upper_z=np.maximum(0,upper_h)+.01
with np.load(SOURCE/'LATENT_PELNY.npz') as f:
    target=f['after'].astype(np.float64);original=f['before'].astype(np.float64);J=f['channels0']
assert (original<=upper_z[None]).all(), 'Source activation violates bound; check model/layer'
excess=np.maximum(target-upper_z[None],0)
other=np.ones(768,bool);other[J]=False
violated=np.flatnonzero((excess>0).any(0))
result=dict(exact_target_excluded=bool(excess.max()>0),layer='bloki.5.norm2 -> mlp.fc1 -> GELU',dimension=192,latent_dimension=768,formula='z_j <= max(0, W_j beta + b_j + sqrt(192) ||W_j gamma - mean(W_j gamma)||_2) + 0.01 safety',target_max=float(target.max()),largest_channel_upper=float(upper_z.max()),violating_coordinates=int((excess>0).sum()),violating_channels0=violated.tolist(),max_excess=float(excess.max()),necessary_mse768_lower_bound=float((excess**2).mean()),necessary_mse128_lower_bound=float((excess[:,J]**2).mean()),necessary_mse640_lower_bound=float((excess[:,other]**2).mean()),original_satisfies_bounds=True,target_sha256=hashlib.sha256((SOURCE/'LATENT_PELNY.npz').read_bytes()).hexdigest(),limitation='Necessary coordinatewise bound only; absence of violations would not prove a natural image exists. Not a bound on detector accuracy.')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'DIAGNOSTYKA_OSIAGALNOSCI.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'GRANICE_LATENTU.npz',upper768=upper_z,target=target,excess=excess,channels0=J)
print(json.dumps(result,indent=2),flush=True)
