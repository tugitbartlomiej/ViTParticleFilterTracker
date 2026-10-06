"""Granica wartości latentu osiągalna w trakcie uczenia przepisem małego ViT (AdamW), niezależnie od danych.

Przepis (experiments/maly_vit_od_zera/trening.py): AdamW, lr 5e-4, rozgrzewka liniowa 300 kroków, potem stały lr,
betas (0,9; 0,999), weight decay 0,05 tylko dla macierzy (ndim >= 2); biasy i LayerNorm bez decay.
Start: nn.Linear domyślnie U(-1/sqrt(fan_in), 1/sqrt(fan_in)) dla wag i biasów, LayerNorm g = 1, b = 0.

Krok Adama na jeden parametr: |m_hat / sqrt(v_hat)| <= B_t, gdzie z nierówności Cauchy'ego-Schwarza
B_t = (1-b1)/sqrt(1-b2) * sqrt(sum_{k<t} (b1^2/b2)^k) * sqrt(1-b2^t)/(1-b1^t)   (niezależnie od gradientów i clip).
Z decay: R_t = (1 - lr_t*wd) R_{t-1} + lr_t B_t, bez decay: R_t = R_{t-1} + lr_t B_t.
Neuron j ostatniego bloku: a_j <= sum_k |W_jk||b_k| + |c_j| + sqrt(d) ||W_j * g|| <= d R_W (R_b + 1 + R_g) + R_c.
"""
import json, math, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / HERE.name
HIST = ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi'
LR, WD, ROZG, B1, B2, D = 5e-4, 0.05, 300, 0.9, 0.999, 192
gelu = lambda a: 0.5 * a * (1 + math.erf(a / math.sqrt(2)))

def koperta(T):
    R0 = 1 / math.sqrt(D)
    Rw, Rc, Rg, Rb = R0, R0, 0.0, 0.0                      # Rg: odchylenie g od 1
    s, out = 0.0, []
    for t in range(1, T + 1):
        lr = LR * min(1.0, t / ROZG)
        s = s * (B1 * B1 / B2) + 1.0
        Bt = (1 - B1) / math.sqrt(1 - B2) * math.sqrt(s) * math.sqrt(1 - B2 ** t) / (1 - B1 ** t)
        Rw = (1 - lr * WD) * Rw + lr * Bt
        Rc, Rg, Rb = Rc + lr * Bt, Rg + lr * Bt, Rb + lr * Bt
        a = D * Rw * (Rb + 1 + Rg) + Rc
        out.append((t, Bt, Rw, Rg, a, a))                   # GELU(a) ~ a dla dużych a
    return out

def granica_wag(sd):
    blok = max(int(k.split('.')[1]) for k in sd if k.startswith('bloki.'))
    g, b = sd[f'bloki.{blok}.norm2.weight'].double(), sd[f'bloki.{blok}.norm2.bias'].double()
    W, c = sd[f'bloki.{blok}.mlp.fc1.weight'].double(), sd[f'bloki.{blok}.mlp.fc1.bias'].double()
    Wg = W * g; r = math.sqrt(D) * (Wg - Wg.mean(1, keepdim=True)).norm(dim=1)
    a = W @ b + c + r
    return 0.5 * a * (1 + torch.erf(a / math.sqrt(2))), W.abs().max().item(), (g - 1).abs().max().item()

if __name__ == '__main__':
    T = 1000
    kop = koperta(T)
    trajektoria = []
    for p in sorted(HIST.glob('k*.pt')):
        sd = torch.load(p, map_location='cpu', weights_only=True)
        h, wmax, gdev = granica_wag(sd['net'])
        trajektoria.append(dict(krok=int(sd['krok']), h_max_maks=float(h.max()), h_max_mediana=float(h.median()),
                                maks_waga_fc1=wmax, maks_odchylenie_g=gdev))
    # Granica po nieskończenie wielu krokach: macierze z decay mają punkt stały R* = B_inf / wd,
    # ale g, b LayerNorm i bias fc1 nie mają decay, więc rosną bez ograniczenia (lr * B * t).
    Binf = (1 - B1) / math.sqrt(1 - B2) / math.sqrt(1 - B1 * B1 / B2)
    wynik = dict(
        przepis=dict(lr=LR, wd=WD, rozgrzewka=ROZG, betas=[B1, B2], decay_tylko_macierze=True),
        krok_adama_maks_B_inf=Binf, punkt_staly_wag_z_decay=Binf / WD,
        nieskonczone_uczenie='brak skonczonej granicy: g, b LayerNorm i bias fc1 bez weight decay rosna liniowo z liczba krokow',
        hipotetycznie_wszystko_z_decay_granica_a=D * (Binf / WD) * (Binf / WD + 1 + Binf / WD) + Binf / WD,
        koperta={str(t): dict(B=B, R_W=Rw, R_g=Rg, a_max=a) for t, B, Rw, Rg, a, _ in kop if t in (1, 10, 50, 100, 300, 500, 1000)},
        trajektoria_biegu_ziarno_47=trajektoria[::10] + [trajektoria[-1]],
        minimum_zawsze=-0.16997)
    (OUT / 'GRANICE_UCZENIA.json').write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = [k[0] for k in kop]; ax.semilogy(t, [k[4] for k in kop], label='twarda granica z AdamW (dowolne dane)')
    ax.semilogy([r['krok'] for r in trajektoria], [r['h_max_maks'] for r in trajektoria], label='granica z wag prawdziwego biegu (maks. kanału)')
    ax.semilogy([r['krok'] for r in trajektoria], [r['h_max_mediana'] for r in trajektoria], label='to samo, mediana kanałów')
    ax.axhline(51.35, color='r', ls='--', label='sztuczny latent J128 (51,4)')
    ax.axhline(4.43, color='k', ls=':', label='naturalnie na 400 zdjęciach k1000 (4,4)')
    ax.set_xlabel('krok uczenia'); ax.set_ylabel('największa możliwa wartość latentu')
    ax.set_title('Ile może wynosić wartość latentu po t krokach uczenia'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'granice_uczenia.png', dpi=110)
    print(json.dumps({k: wynik[k] for k in ('krok_adama_maks_B_inf', 'punkt_staly_wag_z_decay', 'hipotetycznie_wszystko_z_decay_granica_a', 'koperta')}, indent=1, ensure_ascii=False))
    for r in trajektoria[::20] + [trajektoria[-1]]: print(r)
