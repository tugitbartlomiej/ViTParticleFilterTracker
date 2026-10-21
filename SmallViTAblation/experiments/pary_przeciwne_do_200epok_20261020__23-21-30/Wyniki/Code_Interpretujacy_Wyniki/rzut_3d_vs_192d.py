"""How a 3D projection distorts angles between MLP output directions (W2 columns) compared with the full 192D."""
import sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
CK = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_kontynuacja_do_200epok_20261020__19-06-42\Wyniki\Dane\checkpoints'
II, JJ = np.triu_indices(768, 1)


def angles(V):
    n = np.linalg.norm(V, axis=1)
    ok = (n[II] > 1e-12) & (n[JJ] > 1e-12)
    u = V / np.maximum(n[:, None], 1e-30)
    a = np.degrees(np.arccos(np.clip((u @ u.T)[II, JJ], -1, 1)))
    return a, ok


def signed_max(a):
    # A beam pair can show alpha or 180-alpha depending on the GELU signs.
    return np.maximum(a, 180 - a)


rng = np.random.default_rng(3047)
for name, epoch in [('k065640.pt', 60), ('k218750.pt', 200)]:
    st = torch.load(f'{CK}\\{name}', map_location='cpu', weights_only=False)
    print(f'=== epoka {epoch}')
    for b in range(6):
        V = st['net'][f'bloki.{b}.mlp.fc2.weight'].double().numpy().T
        a192, ok = angles(V)
        s192 = signed_max(a192)[ok]
        # Best 3D view: top-3 singular directions of the 768 output vectors.
        _, S, Vt = np.linalg.svd(V, full_matrices=False)
        energy = (S[:3] ** 2).sum() / (S ** 2).sum()
        a_pca, okp = angles(V @ Vt[:3].T)
        s_pca = signed_max(a_pca)[ok & okp]
        rand = []
        for _ in range(20):
            Q, _ = np.linalg.qr(rng.standard_normal((192, 3)))
            ar, okr = angles(V @ Q)
            rand.append((signed_max(ar)[ok & okr] >= 150).sum())
        diff = np.abs(a_pca[ok & okp] - a192[ok & okp])
        print(f'blok {b+1}: 192D pary mogace dac >=150°: {(s192>=150).sum():6d} (maks {s192.max():.1f}°) | '
              f'3D PCA (energia {energy*100:4.1f}%): {(s_pca>=150).sum():6d} | 3D losowy rzut: {int(np.mean(rand)):6d} | '
              f'kąt zmieniony o >30°: {(diff>30).mean()*100:4.1f}% par')
print('pary w bloku:', len(II), '| losowe kierunki w 3D: P(kąt>=150° lub <=30°) =', round(1 - np.cos(np.radians(30)), 4))
