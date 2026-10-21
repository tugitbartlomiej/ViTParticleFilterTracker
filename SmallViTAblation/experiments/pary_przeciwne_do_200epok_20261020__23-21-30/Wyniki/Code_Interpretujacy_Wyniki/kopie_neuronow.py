"""Count near-duplicate MLP neuron pairs (same input pattern, same output direction) per block over epochs 60-200."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
CK = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_kontynuacja_do_200epok_20261020__19-06-42\Wyniki\Dane\checkpoints')
II, JJ = np.triu_indices(768, 1)


def cos_rows(M):
    u = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-30)
    return (u @ u.T)[II, JJ]


print('epoka | kopie na blok b1..b6 (cos W1 po LN > 0.9 i kąt W2 < 30°) | razem | przeciwne wyjścia przy cos W1 > 0.9 | para 370-391 kopią?')
for f in sorted(CK.glob('k*.pt')):
    st = torch.load(f, map_location='cpu', weights_only=False)['net']
    per_block, anti, total = [], 0, 0
    for b in range(6):
        W1 = st[f'bloki.{b}.mlp.fc1.weight'].double().numpy()
        g = st[f'bloki.{b}.norm2.weight'].double().numpy()
        V = st[f'bloki.{b}.mlp.fc2.weight'].double().numpy().T
        eff = W1 * g
        eff -= eff.mean(1, keepdims=True)  # LayerNorm output has zero mean
        cin, cout = cos_rows(eff), cos_rows(V)
        copies = (cin > .9) & (cout > np.cos(np.radians(30)))
        per_block.append(int(copies.sum()))
        anti += int(((cin > .9) & (cout < -.5)).sum())
        if b == 5:
            k = np.flatnonzero((II == 370) & (JJ == 391))[0]
            pair = f'cos W1={cin[k]:.2f}, kąt W2={np.degrees(np.arccos(cout[k])):.0f}°'
    print(f"{int(f.stem[1:])*32/35000:5.0f} | {' '.join(f'{c:3d}' for c in per_block)} | {sum(per_block):4d} | {anti:3d} | {pair}")
print('pary w bloku:', len(II))
