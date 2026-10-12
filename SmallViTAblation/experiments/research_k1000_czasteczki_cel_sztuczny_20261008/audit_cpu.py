"""Jednorazowy probe badawczy CPU; bez forwardu obrazu i bez zmiany modeli."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / Path(__file__).parent.name / 'evidence'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
ckpath = ROOT / 'assets/MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt'
ck = torch.load(ckpath, map_location='cpu', weights_only=True)
state = ck['net']
name = 'bloki.5.mlp.fc2.weight'
W = state[name].numpy().astype(np.float64)
u, s, vt = np.linalg.svd(W, full_matrices=False)
tol = np.finfo(np.float64).eps * max(W.shape) * s[0]
rank = int((s > tol).sum())
V = vt[:rank].T
rng = np.random.default_rng(20261008)
a = rng.normal(size=W.shape[1])
row = V @ (V.T @ a)
null = a - row
g = W.T @ rng.normal(size=W.shape[0])
adam_direction = g / (np.abs(g) + 1e-8)
adam_null = adam_direction - V @ (V.T @ adam_direction)
lin_dim = state['bloki.5.mlp.fc1.weight'].shape[1]
W1=state['bloki.5.mlp.fc1.weight'].numpy().astype(np.float64)
b1=state['bloki.5.mlp.fc1.bias'].numpy().astype(np.float64)
gamma=state['bloki.5.norm2.weight'].numpy().astype(np.float64)
beta=state['bloki.5.norm2.bias'].numpy().astype(np.float64)
wg=W1*gamma[None,:]
wg_centered=wg-wg.mean(axis=1,keepdims=True)
center=W1@beta+b1
radius=np.sqrt(lin_dim)*np.linalg.norm(wg_centered,axis=1)
pre_low,pre_high=center-radius,center+radius
gelu=lambda x:x*ndtr(x)
post_upper=np.maximum(gelu(pre_low),gelu(pre_high))
post_lower=np.minimum(gelu(pre_low),gelu(pre_high))
critical=-0.7517915246935645
post_lower=np.where((pre_low<=critical)&(pre_high>=critical),gelu(critical),post_lower)
np.savez(OUT/'NATURAL_GELU_BOUNDS.npz',pre_low=pre_low,pre_high=pre_high,post_lower=post_lower,post_upper=post_upper)
historical={}
hp=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
if hp.exists():
    with np.load(hp,allow_pickle=False) as arrays:
        historical['keys']=arrays.files
        historical['sha256']=hashlib.sha256(hp.read_bytes()).hexdigest()
        historical['arrays']={}
        for key in ('before','after'):
            arr=arrays[key]
            if arr.shape[-1:]!=(768,) or arr.dtype.kind not in 'fc':
                continue
            flat=arr.reshape(-1,768)
            historical['arrays'][key]=dict(shape=list(arr.shape),maximum=float(flat.max()),
                upper_violations=int((flat>post_upper+1e-5).sum()),
                lower_violations=int((flat<post_lower-1e-5).sum()),
                max_upper_excess=float(np.max(flat-post_upper)))
        dd=(arrays['after']-arrays['before']).astype(np.float64).reshape(-1,768)
        visible=dd@V@V.T
        invisible=dd-visible
        historical['delta_null_energy_fraction']=float(np.sum(invisible**2)/np.sum(dd**2))
        aa=arrays['after'].astype(np.float64).reshape(-1,768)
        box_residual=np.maximum(aa-post_upper,0)+np.minimum(aa-post_lower,0)
        historical['minimum_possible_natural_preimage_rmse_lower_bound']=float(np.sqrt(np.mean(box_residual**2)))
snapshot = 1500 * 600 * 768 * 4
paths = [
    'AGENTS.md',
    'experiments/vit_k1000_1500_pelne_latenty_90_20261008/PLAN.md',
    'experiments/vit_k1000_1500_pelne_latenty_90_20261008/pelne_latenty.py',
    'experiments/vit_k1000_1500_pelne_latenty_90_20261008/model_snapshot.py',
    'experiments/vit_k1000_1500_pelne_latenty_90_20261008/test_pelne_latenty.py',
    'experiments/maly_vit_ruch_czastek_gesta_historia/WYNIKI.md',
    'experiments/maly_vit_odpowiedz_latentu_na_paczke/WYNIKI.md',
    'outputs/vit_dmdc_paczki_20261008/full_v2/REPORT.md',
    'outputs/vit_k8000_podmiana_8z256_dino_m0011_eden_20261008/results/latent_comparison/compatible_k8000_target/DIRECTION_TO_COMPATIBLE_K8000_TARGET.md',
]
evidence=[]
for rel in paths:
    p=ROOT / rel
    if p.exists():
        blob=p.read_bytes()
        evidence.append(dict(path=rel,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
        # Immutable copies confined to the authorized research output directory.
        (OUT / rel.replace('/', '__')).write_bytes(blob)
result=dict(
    device='cpu', seed=20261008, no_image_forward=True, no_training=True,
    checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
    checkpoint_keys=list(ck.keys()), krok=ck.get('krok'), fc2_shape=list(W.shape),
    fc1_input_dim=int(lin_dim), fc2_numerical_rank=rank, rank_tolerance=float(tol),
    singular_max=float(s[0]), singular_min=float(s[-1]), condition=float(s[0]/s[-1]),
    nullity=W.shape[1]-rank,
    null_test_relative_residual=float(np.linalg.norm(W@null)/(np.linalg.norm(W)*np.linalg.norm(null))),
    row_projection_preserves_fc2_relative_error=float(np.linalg.norm(W@a-W@row)/np.linalg.norm(W@a)),
    synthetic_adam_first_step_null_energy_fraction=float(np.dot(adam_null,adam_null)/np.dot(adam_direction,adam_direction)),
    synthetic_adam_warning='Synthetic gradient in row(W), not measured oracle update. Coordinatewise Adam preconditioning may leave row(W).',
    head_shapes={k:list(v.shape) for k,v in state.items() if k.startswith('glowa.')},
    layernorm_keys=[k for k in state if k.startswith('norm.')],
    natural_gelu_upper_bound_min=float(post_upper.min()),natural_gelu_upper_bound_max=float(post_upper.max()),
    bound_derivation='fc1(LN2(h)): |preactivation-(W1 beta+b1)| <= sqrt(192)*||center(W1 gamma)||. Exact necessary upper bound in frozen eval; not a sufficiency condition.',
    historical_latent=historical,
    bytes_full_snapshot=snapshot, bytes_501_states=snapshot*501,
    bytes_endpoints_3_and_milestones_3=snapshot*6,
    bytes_panel_256_501_4fields=256*501*768*4*4,
    evidence=evidence,
)
(OUT/'CPU_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))
