"""Test hipotezy „768 wiązek lasera trzyma 600 koloidów” na 806 checkpointach biegu 35 000 (60 epok).
Latent tokenu to z_i = GELU(W u_i + b): u_i ∈ R^192 = wejście MLP ostatniego bloku po LayerNorm („kulka”, położenie tokenu
wyliczone przez całą sieć przed tą warstwą), wiersze W ∈ R^{768×192} = 768 kierunków odczytu („wiązki”, wspólne dla wszystkich
tokenów i zdjęć). Ruch z między checkpointami A → B rozkładamy dokładnie:
  wiązki ruszyły, kulka stoi:   Δz_W = GELU(W_B u_A + b_B) − GELU(W_A u_A + b_A)
  kulka ruszyła, wiązki stoją:  Δz_u = GELU(W_A u_B + b_A) − GELU(W_A u_A + b_A)
  reszta (oba naraz):           Δz_I = Δz − Δz_W − Δz_u
Dla opóźnień 1, 10 i 40 checkpointów w ośmiu etapach (wczesny co 10 kroków, potem siatka co 100 kroków w epokach 1–5 … 50–60):
energie części w 768D (bezwzględne i bez modu zdjęcia), udział wiązek, korelacje przesunięć (6 sąsiadów, pary losowe w zdjęciu,
pary między zdjęciami) osobno dla Δz, Δz_W i Δz_u; ruch kulek w ich własnej przestrzeni 192D; obrót wiązek (cos w_k między
checkpointami, ‖ΔW‖/‖W‖). Plus cała droga od końca 1. epoki do końca 60. i krzywe od kroku 1094 (obrót wiązek, odejście kulek).
Forward w fp32 (bez autocast, bez TF32), z liczone z u i W ręcznie, zgodność z hakiem sieci sprawdzana.
Zapis: WIAZKI.json, wiazki.png; asset: u192_20zdjec.npy (806, 20, 600, 192) f16, W_fc1.npy (806, 768, 192) f16, b_fc1.npy, META_WIAZKI.json.
Uruchomienie (katalog kodu biegu 35 000): python wiazki_koloidy.py --wagi ... --manifest ... --zipy ... --out ... [--n 50]"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dane as D  # noqa: E402
import model as M  # noqa: E402
from trening35000 import wczytaj_z_zipow, wiersze_35000  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--wagi', required=True); ap.add_argument('--manifest', required=True); ap.add_argument('--zipy', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n_train', type=int, default=3000); ap.add_argument('--n_val', type=int, default=1000); ap.add_argument('--ziarno', type=int, default=2026)
ap.add_argument('--n', type=int, default=50, help='zdjęć na część (TRAIN i VALID), te same co eksport ruch60')
ap.add_argument('--n_zapis', type=int, default=10, help='zdjęć na część z zapisem u (asset)')
ap.add_argument('--paczka', type=int, default=25)
ap.add_argument('--max_krok', type=int, default=65640)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
TOK, EP = 600, 1094
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
log = lambda *s: print(*s, f'[{time.time() - t0:.0f} s]', flush=True)

# ---------------------------------------------------------------- checkpointy i zdjęcia (jak pomiar_eden.py / ruch60_rp.py)
pliki = {int(m[1]): p for p in Path(a.wagi).iterdir() if (m := re.fullmatch(r'k([0-9]+)\.pt', p.name))}
kmax = min(max(pliki), a.max_krok)
kroki = np.array(sorted(k for k in pliki if k <= kmax and ((k <= 1000 and k % 10 == 0) or k % 100 == 0 or k % 1094 == 0 or k == kmax)))
K = len(kroki)
rng = np.random.default_rng(a.ziarno)
wt = wiersze_35000(a.manifest)
idx_tr = sorted(rng.choice(len(wt), a.n_train, replace=False).tolist())
wal_w, wal_X = D.walidacja()
idx_val = sorted(rng.choice(len(wal_w), a.n_val, replace=False).tolist())
n = a.n
X = np.concatenate([wczytaj_z_zipow([wt[i] for i in idx_tr[:n]], a.zipy), wal_X[idx_val[:n]]]); del wal_X
N = len(X)
split = np.array([0] * n + [1] * n)
zapis = np.concatenate([np.arange(a.n_zapis), n + np.arange(a.n_zapis)])
log(f'checkpointów {K}, zdjęć {N} ({n} TRAIN + {n} VALID), zapis u dla {len(zapis)}')

# ---------------------------------------------------------------- forward: u (wejście fc1 ostatniego bloku) i W, b na każdym checkpoincie
net = M.MalyViT().to(dev).eval()
net.zapisuj_latent(True)
fc1 = net.bloki[-1].mlp.fc1
cap = {}
fc1.register_forward_hook(lambda m, i, o: cap.__setitem__('u', i[0]))
Xg = torch.from_numpy(X).to(dev)
U = np.empty((K, N, TOK, 192), np.float16)                                       # ~18,6 GB w RAM przy 100 zdjęciach
Wk = np.empty((K, 768, 192), np.float32); Bk = np.empty((K, 768), np.float32)
zgodnosc = []
for t, k in enumerate(kroki):
    st = torch.load(pliki[int(k)], map_location='cpu', weights_only=False)
    net.load_state_dict(st['net'], strict=True)
    W = fc1.weight.detach().float(); b = fc1.bias.detach().float()
    Wk[t] = W.cpu().numpy(); Bk[t] = b.cpu().numpy()
    with torch.no_grad():
        for s in range(0, N, a.paczka):
            x = Xg[s:s + a.paczka].permute(0, 3, 1, 2).float() / 255.
            _, lat = net(x)
            u = cap['u'].float()
            U[t, s:s + len(u)] = u.half().cpu().numpy()
            if s == 0 and t % 100 == 0:
                z = F.gelu(u @ W.T + b)
                zgodnosc.append(float(((z - lat.float()).norm() / lat.float().norm()).item()))
    if t % 100 == 0 or t == K - 1:
        log(f'krok {k} ({t + 1}/{K}); zgodność z hakiem (błąd względny) {zgodnosc[-1] if zgodnosc else "-"}')
    del st
np.save(OUT / 'u192_20zdjec.npy', U[:, zapis])
np.save(OUT / 'W_fc1.npy', Wk.astype(np.float16)); np.save(OUT / 'b_fc1.npy', Bk)
log('zapisano asset u192/W/b')

# ---------------------------------------------------------------- rozkład ruchu
g = torch.Generator(device='cpu').manual_seed(0)


def na(t):
    return torch.from_numpy(U[t].astype(np.float32)).to(dev), torch.from_numpy(Wk[t]).to(dev), torch.from_numpy(Bk[t]).to(dev)


def z_(u, W, b):
    return F.gelu(u @ W.T + b)                                                   # (N, 600, 768)


def kosinusy(d, nb, perm, perm_img):
    """d (N, 600, 768): średni kosinus przesunięć 6 sąsiadów, par losowych w zdjęciu, par między zdjęciami."""
    dn = d / (d.norm(dim=-1, keepdim=True) + 1e-12)
    ii = torch.arange(N, device=dev)[:, None, None]
    sas = (dn[:, :, None, :] * dn[ii, nb]).sum(-1).mean().item()
    los = (dn * dn[:, perm]).sum(-1).mean().item()
    mie = (dn * dn[perm_img][:, perm]).sum(-1).mean().item()
    return dict(sasiedzi=sas, pary_w_zdjeciu=los, miedzy_zdjeciami=mie)


def rozloz(tA, tB, z_kor=True):
    uA, WA, bA = na(tA); uB, WB, bB = na(tB)
    zA = z_(uA, WA, bA); dF = z_(uB, WB, bB) - zA; dW = z_(uA, WB, bB) - zA; dU = z_(uB, WA, bA) - zA; dI = dF - dW - dU
    E = lambda d: float((d ** 2).sum(-1).mean().item())
    rel = lambda d: d - d.mean(1, keepdim=True)
    w = dict(E=E(dF), E_wiazki=E(dW), E_kulki=E(dU), E_reszta=E(dI), E_wzgl=E(rel(dF)), E_wiazki_wzgl=E(rel(dW)), E_kulki_wzgl=E(rel(dU)),
             E_kulki_192=float(((uB - uA) ** 2).sum(-1).mean().item()), E_kulki_192_wzgl=float((rel(uB - uA) ** 2).sum(-1).mean().item()),
             cos_wiazek=float(F.cosine_similarity(WA, WB, dim=1).mean().item()), dW_do_W=float(((WB - WA).norm() / WA.norm()).item()),
             cos_dz_W_dz_u=float((F.cosine_similarity(rel(dW), rel(dU), dim=-1)).mean().item()))
    w['udzial_wiazek'] = w['E_wiazki'] / (w['E_wiazki'] + w['E_kulki'] + 1e-12)
    w['udzial_wiazek_wzgl'] = w['E_wiazki_wzgl'] / (w['E_wiazki_wzgl'] + w['E_kulki_wzgl'] + 1e-12)
    if z_kor:
        Ar = rel(zA)
        nb = torch.stack([torch.topk(torch.cdist(Ar[j], Ar[j]) + torch.eye(TOK, device=dev) * 1e9, 6, largest=False).indices for j in range(N)])
        perm = torch.randperm(TOK, generator=g).to(dev); perm_img = ((torch.arange(N) + 1 + torch.randint(0, N - 1, (1,), generator=g)) % N).to(dev)
        for nazwa, d in (('pelne', rel(dF)), ('wiazki', rel(dW)), ('kulki', rel(dU))):
            w[f'kor_{nazwa}'] = kosinusy(d, nb, perm, perm_img)
        w['kor_pelne_bezwzgl'] = kosinusy(dF, nb, perm, perm_img)
    return w


ETAPY = {'wczesny_100_1000': [i for i, k in enumerate(kroki) if 100 <= k <= 1000 and k % 10 == 0]}
for e1, e2 in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)):
    ETAPY[f'{e1}-{e2}'] = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
LAGI = (1, 10, 40)
etapy = {}
for nazwa, idx in ETAPY.items():
    etapy[nazwa] = dict(krok_co=int(kroki[idx[1]] - kroki[idx[0]]), lagi={})
    for L in LAGI:
        if L >= len(idx) - 2:
            continue
        starty = np.unique(np.linspace(0, len(idx) - 1 - L, min(30, len(idx) - L)).astype(int))
        ws = [rozloz(idx[s], idx[s + L], z_kor=(j % 3 == 0)) for j, s in enumerate(starty)]
        sr = {}
        for key in ws[0]:
            if isinstance(ws[0][key], dict):
                sr[key] = {kk: float(np.mean([w[key][kk] for w in ws if key in w])) for kk in ws[0][key]}
            else:
                sr[key] = float(np.mean([w[key] for w in ws]))
        sr['n_par'] = len(ws); sr['krokow'] = int(L * etapy[nazwa]['krok_co'])
        etapy[nazwa]['lagi'][L] = sr
        log(f'{nazwa:>16s} L={L:2d} ({sr["krokow"]:5d} kroków): udział wiązek {sr["udzial_wiazek"]:.2f} (wzgl {sr["udzial_wiazek_wzgl"]:.2f}), reszta {sr["E_reszta"] / sr["E"]:.2f}, '
            f'kor. sąsiadów pełne/wiązki/kulki {sr["kor_pelne"]["sasiedzi"]:.2f}/{sr["kor_wiazki"]["sasiedzi"]:.2f}/{sr["kor_kulki"]["sasiedzi"]:.2f}, '
            f'między zdjęciami {sr["kor_pelne"]["miedzy_zdjeciami"]:.2f}/{sr["kor_wiazki"]["miedzy_zdjeciami"]:.2f}/{sr["kor_kulki"]["miedzy_zdjeciami"]:.2f}, cos wiązek {sr["cos_wiazek"]:.4f}')

# cała droga i krzywe od końca 1. epoki
t1 = int(np.argmin(abs(kroki - EP)))
cala = rozloz(t1, K - 1)
log(f'cała droga epoka 1 → 60: udział wiązek {cala["udzial_wiazek"]:.2f} (wzgl {cala["udzial_wiazek_wzgl"]:.2f}), reszta {cala["E_reszta"] / cala["E"]:.2f}, cos wiązek {cala["cos_wiazek"]:.3f}')
krzywe = dict(krok=[], cos_wiazek=[], E_wiazki=[], E_kulki=[], E=[], E_kulki_192_wzgl=[])
uA, WA, bA = na(t1)
for t in range(t1, K, 5):
    uB, WB, bB = na(t)
    zA = z_(uA, WA, bA)
    krzywe['krok'].append(int(kroki[t])); krzywe['cos_wiazek'].append(float(F.cosine_similarity(WA, WB, dim=1).mean().item()))
    krzywe['E'].append(float(((z_(uB, WB, bB) - zA) ** 2).sum(-1).mean().item()))
    krzywe['E_wiazki'].append(float(((z_(uA, WB, bB) - zA) ** 2).sum(-1).mean().item()))
    krzywe['E_kulki'].append(float(((z_(uB, WA, bA) - zA) ** 2).sum(-1).mean().item()))
    d = uB - uA; d = d - d.mean(1, keepdim=True); krzywe['E_kulki_192_wzgl'].append(float((d ** 2).sum(-1).mean().item()))

wyn = dict(opis=__doc__.split('\n')[0], kroki=[int(k) for k in kroki], zdjec=N, zgodnosc_z_hakiem=zgodnosc, etapy=etapy, cala_droga_e1_e60=cala, krzywe_od_e1=krzywe,
           zapis=dict(u192='(K, 20, 600, 192) f16: zdjęcia 0–9 TRAIN i 50–59 VALID eksportu ruch60 (te same zdjęcia)', W_fc1='(K, 768, 192) f16', b_fc1='(K, 768) f32'))
json.dump(wyn, open(OUT / 'WIAZKI.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(dict(kroki=[int(k) for k in kroki], zdjecia_zapis=[int(v) for v in zapis], split_zapis=[int(split[v]) for v in zapis],
               frame_id=[(wt[idx_tr[v]]['frame_id'] if v < n else wal_w[idx_val[v - n]]['frame_id']) for v in zapis],
               u='wejście fc1 ostatniego bloku (po norm2), fp32 forward, zapis f16', pochodzenie=dict(skrypt='wiazki_koloidy.py', zadanie=os.environ.get('SLURM_JOB_ID'))),
          open(OUT / 'META_WIAZKI.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:  # noqa: BLE001
    log(f'bez rysunku: {e}'); sys.exit(0)
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5), dpi=100)
nm = list(etapy); x = np.arange(len(nm))
a = ax[0, 0]
for L, c in zip(LAGI, ('#ffffff', '#ffd166', '#ff4433')):
    a.plot(x, [etapy[e]['lagi'].get(L, {}).get('udzial_wiazek_wzgl', np.nan) for e in nm], 'o-', color=c, label=f'opóźnienie {L} checkpointów')
a.axhline(0.5, color='#8a8f99', ls='--', lw=1); a.set_xticks(x); a.set_xticklabels(nm, rotation=25, ha='right'); a.set_ylim(0, 1)
a.set_ylabel('udział wiązek w energii ruchu (bez modu zdjęcia)'); a.legend(fontsize=8); a.grid(True, alpha=0.4)
a.set_title('1. kto rusza tokenem: wiązki (W) czy kulki (u)', fontsize=10)
a = ax[0, 1]
w_ = 0.27
for j, (cz, c) in enumerate((('kor_pelne', '#ffffff'), ('kor_wiazki', '#4a8cff'), ('kor_kulki', '#ffd166'))):
    a.bar(x + j * w_, [etapy[e]['lagi'][1][cz]['sasiedzi'] for e in nm], w_, color=c, label=cz.replace('kor_', ''))
a.set_xticks(x + w_); a.set_xticklabels(nm, rotation=25, ha='right'); a.set_ylabel('⟨cos⟩ przesunięć 6 sąsiadów'); a.legend(fontsize=8); a.grid(True, axis='y', alpha=0.4)
a.set_title('2. skąd korelacja sąsiadów (opóźnienie 1)', fontsize=10)
a = ax[1, 0]
for j, (cz, c) in enumerate((('kor_pelne', '#ffffff'), ('kor_wiazki', '#4a8cff'), ('kor_kulki', '#ffd166'))):
    a.bar(x + j * w_, [etapy[e]['lagi'][1][cz]['miedzy_zdjeciami'] for e in nm], w_, color=c, label=cz.replace('kor_', ''))
a.set_xticks(x + w_); a.set_xticklabels(nm, rotation=25, ha='right'); a.set_ylabel('⟨cos⟩ par tokenów z różnych zdjęć'); a.legend(fontsize=8); a.grid(True, axis='y', alpha=0.4)
a.set_title('3. skąd korelacja między zdjęciami', fontsize=10)
a = ax[1, 1]
kk = np.array(krzywe['krok']) / EP
a.plot(kk, krzywe['E'], color='#ffffff', label='|Δz|² od epoki 1 (pełne)')
a.plot(kk, krzywe['E_wiazki'], color='#4a8cff', label='tylko wiązki')
a.plot(kk, krzywe['E_kulki'], color='#ffd166', label='tylko kulki')
a2 = a.twinx(); a2.plot(kk, krzywe['cos_wiazek'], ':', color='#ff4433', label='cos wiązek względem epoki 1'); a2.set_ylabel('cos w_k', color='#ff4433')
a.set_xlabel('epoka'); a.set_ylabel('⟨|Δz|²⟩ 768D'); a.legend(loc='upper left', fontsize=8); a2.legend(loc='lower right', fontsize=8); a.grid(True, alpha=0.4)
a.set_title(f'4. od epoki 1: udział wiązek na całej drodze {cala["udzial_wiazek"]:.2f}', fontsize=10)
fig.suptitle(f'„768 wiązek trzyma 600 koloidów”: rozkład ruchu tokenów na obrót wiązek W i ruch kulek u, {N} zdjęć, {K} checkpointów', fontsize=12)
fig.tight_layout(); fig.savefig(OUT / 'wiazki.png')
log('zapisano WIAZKI.json, wiazki.png')
