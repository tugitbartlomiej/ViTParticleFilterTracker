"""Jak często obcięcie normy gradientu do 1 było naprawdę aktywne i czy może tłumaczyć brak „temperatury”.

W każdym biegu MałegoViT po L.backward() jest torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) (trening.py:80 biegu s47;
biegi z pokrętłami wołają ten sam Uczen.kroki, także z SGD). Skrypt odtwarza warunki kroku uczenia: klatki 35 000 TRAIN
z archiwów (SHA-256 z manifestem), rozciągnięcie do 480 × 320, odbicie 0,5, jasność/kontrast ±20 %, paczka biegu, autocast bf16,
strata M.strata, BatchNorm łodygi w trybie train (statystyki paczki, running stats bez zmian). Dla wybranych checkpointów liczy
całkowitą normę gradientu PRZED obcięciem na wielu losowych paczkach z puli 3072 klatek.

A. Bieg s47 (assets/MalyViT_35000_od_zera_s47, AdamW 5e-4): kroki 0–1000 i epoki 1, 5, 10, 11, 12, 20, 40, 60.
B. Biegi z pokrętłami (Eden → assets/.../05_biegi_pokretla_eden/wyniki, stany pelny/stan_eNN.pt): AdamW lr 1,25e-4 / 5e-4 / 2e-3
   w krokach 12 034 i 13 128 (okno pomiaru τ_1e) oraz SGD z pędem 0,9 (lr 0,03 / 0,1 / 0,3 przy paczce 32; paczka 16 / 64 / 128
   przy lr 0,1) w dwóch ostatnich epokach.
C. Historia uczenia ze stanu optymalizatora: dla AdamW Σ v̂ (średnia wykładnicza kwadratów OBCIĘTEGO gradientu, β₂ = 0,999,
   ok. 1000 ostatnich kroków) = średni kwadrat normy obciętego gradientu, ≤ 1, a ≈ 1 znaczy obcięcie prawie w każdym kroku;
   do tego norma kroku AdamW lr·‖m̂/(√v̂+ε)‖. Dla SGD norma bufora pędu i norma kroku lr·‖bufor‖.
Uruchom (z katalogu obciecie_gradientu): F:/Python/3.13/python.exe -X utf8 pomiar_normy_gradientu.py  → NORMA_GRADIENTU.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
MANIFEST = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl'
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
PELNY_60 = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/eden/pelny/pelny'
POK = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki'
sys.path.insert(0, str(KOD))
import model as M                                                                   # noqa: E402

W_, H_ = 480, 320
PULA, ZIARNO = 3072, 2026
DEV = 'cuda'
KROKI_S47 = [0, 10, 50, 100, 300, 500, 1000, 1094, 5470, 10940, 12034, 13128, 21880, 43760, 65640]
ADAMW = [('dlugi_lr_1.25e-4_rp', 1.25e-4), ('dlugi_lr_5e-4_rp', 5e-4), ('dlugi_lr_2e-3_rp', 2e-3)]
SGD = [('sgd_lr_0.03_rp', 0.03, 32), ('sgd_lr_0.1_rp', 0.1, 32), ('sgd_lr_0.3_rp', 0.3, 32),
       ('sgd_lr_0.1_b16_rp', 0.1, 16), ('sgd_lr_0.1_b64_rp', 0.1, 64), ('sgd_lr_0.1_b128_rp', 0.1, 128)]
GRUPY = dict(lodyga=('lodyga.',), **{f'blok{i}': (f'bloki.{i}.',) for i in range(6)}, glowa=('glowa.', 'norm.'))
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
f4 = lambda x: float(np.round(float(x), 4))

# ---------------- pula klatek i cele ----------------
wiersze = [json.loads(l) for l in open(MANIFEST, encoding='utf-8')]
pula = [wiersze[i] for i in np.sort(np.random.default_rng(ZIARNO).choice(len(wiersze), PULA, replace=False))]


def wczytaj(pula):
    X = np.empty((len(pula), H_, W_, 3), np.uint8)
    grupy = defaultdict(list)
    for i, r in enumerate(pula):
        grupy[r['archive']].append(i)

    def archiwum(nazwa):
        with zipfile.ZipFile(nazwa) as z:
            for i in grupy[nazwa]:
                raw = z.read(pula[i]['member'])
                if hashlib.sha256(raw).hexdigest() != pula[i]['image_sha256']:
                    raise ValueError(f'SHA-256 niezgodne: {pula[i]["id"]}')
                im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if im is None or im.shape[1::-1] != (pula[i]['width'], pula[i]['height']):
                    raise ValueError(f'zła klatka {pula[i]["id"]}')
                X[i] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (W_, H_), interpolation=cv2.INTER_AREA)
        return len(grupy[nazwa])

    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(archiwum, sorted(grupy)))
    log(f'klatki: {n} z {len(grupy)} archiwów')
    return X


def ramki(r):
    sx, sy = W_ / r['width'], H_ / r['height']
    return np.array([[g['bbox_xywh'][0] * sx, g['bbox_xywh'][1] * sy, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * sx,
                      (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * sy] for g in r['gt']], np.float32).reshape(-1, 4)


X = wczytaj(pula)
R = [ramki(r) for r in pula]
hh, ww = H_ // M.KROK, W_ // M.KROK
CEL0 = M.cele(R, hh, ww)
CEL1 = M.cele([np.stack([W_ - b[:, 2], b[:, 1], W_ - b[:, 0], b[:, 3]], 1) if len(b) else b for b in R], hh, ww)
log('cele policzone')


def plan(B, n):
    r = np.random.default_rng(ZIARNO + B)
    return [(np.sort(r.choice(PULA, B, replace=False)), r.random(B) < .5, r.uniform(.8, 1.2, (B, 2)).astype(np.float32)) for _ in range(n)]


def paczka(idx, odb, a):
    x = torch.from_numpy(X[idx]).to(DEV).permute(0, 3, 1, 2).float() / 255.
    x = torch.where(torch.from_numpy(odb).to(DEV)[:, None, None, None], x.flip(3), x)
    a = torch.from_numpy(a).to(DEV)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * a[:, 0, None, None, None] + m * a[:, 1, None, None, None]).clamp(0, 1)
    it, o_ = torch.from_numpy(idx), torch.from_numpy(odb)
    heat = torch.where(o_[:, None, None], CEL1[0][it], CEL0[0][it]).to(DEV)
    ltrb = torch.where(o_[:, None, None, None], CEL1[1][it], CEL0[1][it]).to(DEV)
    poz = torch.where(o_[:, None, None], CEL1[2][it], CEL0[2][it]).to(DEV)
    return x, heat, ltrb, poz


# ---------------- gradient przed obcięciem ----------------
net = M.MalyViT().to(DEV)
NAZWY = [n for n, _ in net.named_parameters()]
GRUPA = [next(g for g, pre in GRUPY.items() if n.startswith(pre)) for n in NAZWY]


def gradient(x, heat, ltrb, poz, mikro):
    """Gradient straty paczki jak w Uczen.kroki; mikro > 1 tylko gdy cała paczka nie mieści się w pamięci GPU
    (wtedy BatchNorm widzi statystyki części paczki, a strata jest średnią strat części)."""
    net.zero_grad(set_to_none=True)
    Ls = 0.0
    for s in range(mikro):
        sl = slice(s * len(x) // mikro, (s + 1) * len(x) // mikro)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x[sl])
        L, _ = M.strata(o, heat[sl], ltrb[sl], poz[sl])
        (L / mikro).backward()
        Ls += float(L) / mikro
    return Ls


def normy(sd, B, n_paczek):
    net.load_state_dict(sd)
    for mod in net.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.momentum = 0.0                                                      # statystyki paczki, running stats bez zmian
    net.train()
    mikro = 1
    while True:
        try:
            wyn = dict(norma=[], strata=[], udzial=defaultdict(list))
            for idx, odb, a in plan(B, n_paczek):
                L = gradient(*paczka(idx, odb, a), mikro)
                sq = torch.stack([p.grad.detach().float().pow(2).sum() if p.grad is not None else torch.zeros((), device=DEV)
                                  for p in net.parameters()]).cpu().numpy().astype(np.float64)
                tot = sq.sum()
                wyn['norma'].append(float(np.sqrt(tot))); wyn['strata'].append(L)
                for g in GRUPY:
                    wyn['udzial'][g].append(float(sq[[k for k, gg in enumerate(GRUPA) if gg == g]].sum() / tot))
            wyn['mikro'] = mikro
            return wyn
        except torch.OutOfMemoryError:
            net.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            mikro *= 2
            log(f'brak pamięci GPU przy paczce {B}: dzielę na {mikro} części')
            if mikro > 8:
                raise


def statystyki(w):
    n = np.asarray(w['norma']); c = np.minimum(1.0, 1.0 / n)
    return dict(paczek=int(len(n)), mikro=int(w['mikro']), ulamek_obcietych=f4((n > 1).mean()),
                norma=dict(p10=f4(np.percentile(n, 10)), med=f4(np.median(n)), p90=f4(np.percentile(n, 90)), srednia=f4(n.mean()), max=f4(n.max())),
                wspolczynnik_obciecia=dict(srednia=f4(c.mean()), cv=f4(c.std() / c.mean()), p10=f4(np.percentile(c, 10)),
                                           med=f4(np.median(c)), p90=f4(np.percentile(c, 90))),
                srednia_kwadratu_obcietej_normy=f4((np.minimum(n, 1.0) ** 2).mean()),
                strata_med=f4(np.median(w['strata'])),
                udzial_grup_w_normie_kw={g: f4(np.median(v)) for g, v in w['udzial'].items()},
                normy=[round(v, 4) for v in w['norma']])


def stan_optymalizatora(st, lr):
    grupa = st['opt']['param_groups'][0]
    stan = st['opt']['state']
    if 'betas' in grupa:
        b1, b2 = grupa['betas']; eps = grupa['eps']
        sv = sm = su = 0.0; n_el = 0
        for s in stan.values():
            k = float(s['step']); v = s['exp_avg_sq'].double() / (1 - b2 ** k); m = s['exp_avg'].double() / (1 - b1 ** k)
            sv += float(v.sum()); sm += float((m * m).sum()); su += float(((m / (v.sqrt() + eps)) ** 2).sum()); n_el += v.numel()
        return dict(opt='adamw', krok=int(st['krok']), rms_obcietej_normy_z_historii=f4(np.sqrt(sv)), norma_m=f4(np.sqrt(sm)),
                    norma_kroku_lr_razy_m_przez_sqrt_v=float(f'{lr * np.sqrt(su):.4e}'),
                    krok_na_wspolrzedna_rms=float(f'{lr * np.sqrt(su / n_el):.4e}'))
    nb = float(np.sqrt(sum(float(s['momentum_buffer'].double().pow(2).sum()) for s in stan.values() if s.get('momentum_buffer') is not None)))
    return dict(opt='sgd', krok=int(st['krok']), norma_bufora_pedu=f4(nb), norma_kroku_lr_razy_bufor=f4(lr * nb))


wynik = dict(opis=(__doc__ or '').split('\n\n')[0], ustawienia=dict(pula_klatek=PULA, ziarno=ZIARNO, autocast='bf16', obciecie_w_treningu=1.0,
                                                                     batchnorm='train, momentum 0', manifest=str(MANIFEST.relative_to(ROOT))))

# ---------------- A. bieg s47 ----------------
A = {}
for k in KROKI_S47:
    sd = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
    A[str(k)] = dict(epoka=round(k / 1094, 3), **statystyki(normy(sd, 32, 64)))
    log('A', k, {kk: A[str(k)][kk] for kk in ('ulamek_obcietych', 'norma', 'wspolczynnik_obciecia', 'strata_med')})
wynik['A_s47'] = A

# stan AdamW biegu 60-epokowego (pelny/stan_e59, e60): czy to ten sam bieg co wagi s47
C60 = {}
for e in (59, 60):
    p = PELNY_60 / f'stan_e{e:02d}.pt'
    if p.exists():
        st = torch.load(p, map_location='cpu', weights_only=False)
        k = int(st['krok'])
        ref = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net'] if (WAGI / f'k{k:05d}.pt').exists() else None
        zgodne = ref is not None and all(torch.equal(st['net'][n].cpu(), ref[n].cpu()) for n in ref)
        C60[f'e{e}'] = dict(wagi_identyczne_z_s47=bool(zgodne), **stan_optymalizatora(st, 5e-4))
        log('C60', e, C60[f'e{e}'])
wynik['C_s47_stan_adamw'] = C60

# ---------------- B. biegi z pokrętłami ----------------
B_ = {}
for bieg, lr in ADAMW:
    B_[bieg] = dict(opt='adamw', lr=lr, paczka=32)
    for e in (11, 12):
        st = torch.load(POK / bieg / 'pelny' / f'stan_e{e:02d}.pt', map_location='cpu', weights_only=False)
        B_[bieg][f'e{e}'] = dict(krok=int(st['krok']), **statystyki(normy(st['net'], 32, 64)), stan_optymalizatora=stan_optymalizatora(st, lr))
        log('B', bieg, e, {kk: B_[bieg][f'e{e}'][kk] for kk in ('ulamek_obcietych', 'norma', 'wspolczynnik_obciecia', 'stan_optymalizatora')})
for bieg, lr, Bp in SGD:
    B_[bieg] = dict(opt='sgd', lr=lr, paczka=Bp, ped=0.9)
    for p in sorted((POK / bieg / 'pelny').glob('stan_e*.pt')):
        st = torch.load(p, map_location='cpu', weights_only=False)
        B_[bieg][p.stem] = dict(krok=int(st['krok']), **statystyki(normy(st['net'], Bp, 48)), stan_optymalizatora=stan_optymalizatora(st, lr))
        log('B', bieg, p.stem, {kk: B_[bieg][p.stem][kk] for kk in ('ulamek_obcietych', 'norma', 'wspolczynnik_obciecia', 'stan_optymalizatora')})
wynik['B_pokretla'] = B_


# ---------------- wnioski liczbowe ----------------
def nachylenie(xs, ys):
    return f4(np.polyfit(np.log(xs), np.log(ys), 1)[0])


lr_a = [lr for _, lr in ADAMW]
med_a = [np.mean([B_[b][f'e{e}']['norma']['med'] for e in (11, 12)]) for b, _ in ADAMW]
c_a = [np.mean([B_[b][f'e{e}']['wspolczynnik_obciecia']['srednia'] for e in (11, 12)]) for b, _ in ADAMW]
krok_a = [np.mean([B_[b][f'e{e}']['stan_optymalizatora']['norma_kroku_lr_razy_m_przez_sqrt_v'] for e in (11, 12)]) for b, _ in ADAMW]
sgd32 = [(b, lr) for b, lr, Bp in SGD if Bp == 32]
ost = lambda b: sorted(k for k in B_[b] if k.startswith('stan_e'))
med_s = [np.mean([B_[b][k]['norma']['med'] for k in ost(b)]) for b, _ in sgd32]
c_s = [np.mean([B_[b][k]['wspolczynnik_obciecia']['srednia'] for k in ost(b)]) for b, _ in sgd32]
krok_s = [np.mean([B_[b][k]['stan_optymalizatora']['norma_kroku_lr_razy_bufor'] for k in ost(b)]) for b, _ in sgd32]
wynik['wnioski_liczbowe'] = dict(
    adamw=dict(lr=lr_a, norma_med=[f4(v) for v in med_a], wsp_obciecia_sredni=[f4(v) for v in c_a], norma_kroku=[float(f'{v:.4e}') for v in krok_a],
               wykladnik_normy_gradientu_wzgledem_lr=nachylenie(lr_a, med_a), wykladnik_normy_kroku_wzgledem_lr=nachylenie(lr_a, krok_a),
               uwaga='AdamW jest niezmienniczy na stałą skalę gradientu: wspólny współczynnik obcięcia się znosi; znaczenie ma tylko jego zmienność (cv).'),
    sgd_paczka32=dict(lr=[lr for _, lr in sgd32], norma_med=[f4(v) for v in med_s], wsp_obciecia_sredni=[f4(v) for v in c_s],
                      norma_kroku=[f4(v) for v in krok_s], wykladnik_normy_gradientu_wzgledem_lr=nachylenie([lr for _, lr in sgd32], med_s),
                      wykladnik_efektywnego_lr=nachylenie([lr for _, lr in sgd32], [lr * c for (_, lr), c in zip(sgd32, c_s)]),
                      wykladnik_normy_kroku_wzgledem_lr=nachylenie([lr for _, lr in sgd32], krok_s)))
log('wnioski', wynik['wnioski_liczbowe'])
(HERE / 'NORMA_GRADIENTU.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
log('zapisano NORMA_GRADIENTU.json')
