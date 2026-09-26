#!/usr/bin/env python3
"""Lokalna część obrotu k pętli (PLAN.md): prognoza latentu, FastGAN, wyszukanie klatek, selekcje trzech pętli.

Dla pętli „koopman”:
 1. Historia opisu 72 sond w 5 kierunkach (skalar tożsamości w_lda·(narzędzie − tło) i 4 kierunki S_4 średniej
    narzędzia, te same co w pokaz_fastgan.py): bieg 1810662 kroki 5..245 (tokeny lokalne) i LATENT_*.npz
    wszystkich wcześniejszych obrotów tej pętli (zapis co 5 aktualizacji; LATENT = czastki.descriptor dla 192 sond,
    72 z nich to sondy tej analizy, zgodność sprawdzona do 1e-6).
 2. Prognoza: wspólny operator na 8 ostatnich przyrostach (dryf Koopmana z pamięcią, grzbiet λ = 0,1 na
    cechach standaryzowanych, jak w ../langevin_latent_detektora), osobno dla skalara i dla S_4, dopasowany do
    całej historii; rollout średniej po sondach o H = 33 kroki (165 aktualizacji = jeden obrót).
 3. FastGAN: obecny kod 3 kotwic odczytany bieżącym detektorem, cel = obecny + przyrost; inwersja kotwicy
    (w obrocie 1 pełna jak w pokaz_fastgan_inwersja.py, dalej start z z* poprzedniego obrotu plus 2 losowe
    restarty) i krok Gaussa–Newtona do celu oraz do obecnego (kontrola, jak w pokaz_fastgan_gn.py).
 4. Wyszukanie DINOv2 (indeks równoległej sesji, tylko odczyt) dla obrazu celu i, kontrolnie, obrazu obecnego;
    nowe klatki na przemian z list trzech kotwic aż do 134, bez pretreningu, kotwic i klatek już dodanych.
Pętla „losowe”: 134 losowe nieużyte klatki (ziarno 1000 + k). Pętla „bez”: 1500 klatek bazy.
Zapis: wyniki/obroty/kXX/ (raport, obrazy, zapytania) i wyniki/selekcje/{petla}_kXX.json.

    <venv rfdetr>/python obrot_lokalny.py --k 1
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministyczne kernele czytnika
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
KT = EXP / 'koopman_tozsamosc_kierunki'
PETLA = EXP / 'koopman_fastgan_petla'
sys.path.insert(0, str(PETLA))
PROBE_MANIFEST = PETLA / 'wyniki/local_1810448/run/probe/manifest.json'
FASTGAN = EXP / 'fastgan_narzedzie/wyniki/eden_20260923_50k/results_50k/checkpoint_050000.pth'
VENDOR = EXP / 'fastgan_narzedzie/vendor/deepfake_gi_fastGAN'
TOKENY = KT / 'wyniki/tokeny_pelne'
OPS = KT / 'wyniki/OPERATORY_K73_S48.npz'
WAGI_START = EXP / 'koopman_gesta_historia/wyniki/run_1810662/pretrain/model_0000245.pth'
INDEKS = PETLA / 'wyniki/local_1810448/run/index'
WYKL = KT / 'wyniki/dobor_klatek/wykluczone_pretrain_1500_plus_kotwice.json'
BAZA_SEL = KT / 'wyniki/dobor_klatek/trening5ep/selekcja_bez_dodania.json'
W = HERE / 'wyniki'
AKTUALIZACJI = 165                 # na obrót (epoch_size 2640 / 16)
CO_ILE = 5                         # zapis latentu co 5 aktualizacji
H = AKTUALIZACJI // CO_ILE         # 33 kroki prognozy
L_PAM = 8
LAM = 0.1
NOWYCH = 134
TOP_K = 90
SEED = 42


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def zj(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding='utf-8')


def projekcja():
    ops = np.load(OPS)
    P = np.zeros((1536, 5)); P[:384, 0] = ops['w_lda']; P[384:768, 0] = -ops['w_lda']; P[:384, 1:] = ops['Q32'][:, :4]
    return P


def sondy():
    """Wiersze 72 sond, mapowanie na 192 sondy manifestu i kotwice jak w pokaz_fastgan.py."""
    rows = jz(TOKENY / 'CONFIG.json')['rows']
    m192 = jz(PROBE_MANIFEST)
    idx = {r['image_sha256']: i for i, r in enumerate(m192)}
    mapa = np.array([idx[r['image_sha256']] for r in rows])
    orig = [i for i, r in enumerate(rows) if r['variant'] == 'original']
    fit_o = [i for i in orig if rows[i]['role'] == 'fit']; hold_o = [i for i in orig if rows[i]['role'] == 'holdout']
    kotwice = [fit_o[0], fit_o[len(fit_o) // 2], hold_o[0]]
    return rows, mapa, kotwice


def historia_bazowa(P):
    f = W / 'historia_bazowa.npz'
    if f.exists():
        return np.load(f)['Y']
    from czastki import descriptor_numpy
    maska = np.load(TOKENY / 'MASK_WEIGHTS.npy')
    Y = []
    for s in range(5, 246, 5):
        Z = np.load(TOKENY / f'TOKENS_{s:07d}.npy', mmap_mode='r')
        D = descriptor_numpy(np.asarray(Z, np.float64).reshape(72, 24, 24, 384).transpose(0, 3, 1, 2), maska.reshape(72, 1, 24, 24))
        Y.append(D @ P)
    Y = np.stack(Y)
    np.savez(f, Y=Y, steps=np.arange(5, 246, 5))
    return Y


def latent_obrotu(petla, j, P, mapa):
    """Opisy sond z obrotu j pętli (kroki 5..165); krok 0 = koniec poprzedniego obrotu, więc pomijany."""
    d = W / 'lancuchy' / petla / f'k{j:02d}' / 'latent'
    Y = []
    for s in range(CO_ILE, AKTUALIZACJI + 1, CO_ILE):
        F = np.load(d / f'LATENT_{s:07d}.npz')['features'].astype(np.float64)
        Y.append(F[mapa] @ P)
    return np.stack(Y)


def historia(k, P, mapa, petla='koopman'):
    czesci = [historia_bazowa(P)] + [latent_obrotu(petla, j, P, mapa) for j in range(1, k)]
    return np.concatenate(czesci)


def dopasuj_i_prognozuj(Y):
    """Wspólny operator na L_PAM ostatnich przyrostach, osobno skalar i S_4; rollout średniej o H kroków."""
    V = np.diff(Y, axis=0)                                            # (T-1, 72, 5)
    delta, diag = np.zeros(5), {}
    for nazwa, dims in (('skalar', [0]), ('S4', [1, 2, 3, 4])):
        X, y = [], []
        for t in range(L_PAM, V.shape[0]):
            X.append(np.concatenate([V[t - l][:, dims] for l in range(1, L_PAM + 1)], 1)); y.append(V[t][:, dims])
        X, y = np.concatenate(X), np.concatenate(y)
        mx, sx, my = X.mean(0), X.std(0) + 1e-12, y.mean(0)
        Zs = (X - mx) / sx
        Wt = np.linalg.solve(Zs.T @ Zs + LAM * len(y) * np.eye(Zs.shape[1]), Zs.T @ (y - my))
        res = y - (my + Zs @ Wt)
        hist = [V[t].mean(0)[dims] for t in range(V.shape[0] - L_PAM, V.shape[0])]   # najstarsza pierwsza
        kroki = []
        for _ in range(H):
            x = np.concatenate([hist[-l] for l in range(1, L_PAM + 1)])
            v = my + ((x - mx) / sx) @ Wt
            kroki.append(v); hist.append(v)
        delta[dims] = np.sum(kroki, 0)
        diag[nazwa] = dict(przejsc=int(len(y) // 72), umiejetnosc_1krok_w_probie=float(1 - np.mean(res ** 2) / np.mean(y ** 2)),
                           przyrost_prognozowany=np.sum(kroki, 0).tolist())
    return delta, diag


def fastgan(k, wagi, delta, skala, rows, kotwice, out, device='cuda'):
    import torch
    from czastki import DifferentiableReader
    import generowanie as gen
    from rfdetr import RFDETRNano
    P = projekcja()
    np.savez(out / 'directions.npz', mean=np.zeros(1536), scale=np.ones(1536), projection=P, target_scale=skala)
    projection = gen.DescriptorProjection.from_npz(out / 'directions.npz')
    idx_man = {r['image']: i for i, r in enumerate(jz(PROBE_MANIFEST))}
    generator, gmeta = gen.load_fastgan(str(FASTGAN), str(VENDOR), device)
    wrapper = RFDETRNano(num_classes=1, resolution=384, pretrain_weights=str(wagi), trust_checkpoint=True, amp=False)
    net = wrapper.model.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    skala_t = torch.tensor(skala, dtype=torch.float32, device=device)
    rap = {}
    with gen._generation_numerics(device), DifferentiableReader(net, wrapper.means, wrapper.stds) as reader:
        for i in kotwice:
            t1 = time.time()
            kd = out / f'kotwica_{i:02d}'; kd.mkdir(parents=True, exist_ok=True)
            anchor_image, anchor_mask, _ = gen._load_anchor((str(PROBE_MANIFEST), idx_man[rows[i]['image']]), device)
            with torch.no_grad():
                cur = projection.project_torch(reader.descriptors(anchor_image, anchor_mask))[0].cpu().numpy().astype(np.float64)
            cel = cur + delta
            # --- inwersja: z -> obecny kod
            poprz = W / 'obroty' / f'k{k - 1:02d}' / f'kotwica_{i:02d}' / 'inwersja_best.npz'
            starty = []
            if poprz.exists():
                p = np.load(poprz)
                starty.append((torch.tensor(p['z'], dtype=torch.float32, device=device), int(p['noise_seed']), 'z*_poprzedni'))
                restarty = 2
            else:
                restarty = 6
            kand = []
            for r in range(restarty):
                rng = torch.Generator(device='cpu').manual_seed(42 + r + 100 * k)
                z0 = torch.randn((1, 256), generator=rng).to(device)
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, cur, z0, steps=100, lr=.05,
                                          noise_seed=1042 + r)
                kand.append((float(res['best']['total']), res, 1042 + r, f'restart_{r}'))
            kand.sort(key=lambda x: x[0])
            doszlif = [(z, s, n) for z, s, n in starty] + [(c[1]['z'].detach().clone(), c[2], c[3]) for c in kand[:1 if starty else 2]]
            zwyc = None
            for z0, szum, nazwa in doszlif:
                res = gen.optimize_latent(generator, reader, anchor_image, anchor_mask, projection, cur, z0, steps=200, lr=.01,
                                          noise_seed=szum)
                b = res['best']
                if zwyc is None or b['total'] < zwyc[0]['total']:
                    zwyc = (b, res, szum, nazwa)
            b, res, szum, nazwa = zwyc
            z_star = res['z'].detach().clone()
            np.savez_compressed(kd / 'inwersja_best.npz', z=z_star.cpu().numpy(), code=res['code'].cpu().numpy(), noise_seed=szum)
            gen._save_image(kd / 'inwersja_best.png', res['rgba'][:, :3])
            inw = dict(start=nazwa, blad_skalowany=float(b['feature'] ** .5), anchor_l1=float(b['anchor_l1']),
                       iou=float(gen.shape_iou(res['rgba'][:, 3:4], anchor_mask)))

            # --- krok Gaussa-Newtona od z* do celu i do obecnego (kontrola)
            def kod(z):
                rgba = gen.fixed_generator_output(generator, z, seed=szum)
                return projection.project_torch(reader.descriptors(rgba[:, :3], anchor_mask))[0], rgba

            def krok_gn(z, cel_t, lam0=1e-2):
                zq = z.detach().clone().requires_grad_(True)
                c, _ = kod(zq)
                r = (cel_t - c) / skala_t
                J = torch.stack([torch.autograd.grad(-(c[j] / skala_t[j]), zq, retain_graph=True)[0][0] for j in range(5)])
                r0 = float(r.norm() / np.sqrt(5)); JJt = J @ J.T; sl = float(torch.trace(JJt) / 5); lam = lam0
                for _ in range(8):
                    dz = -J.T @ torch.linalg.solve(JJt + lam * sl * torch.eye(5, device=J.device), r.detach())
                    with torch.no_grad():
                        c2, _ = kod(z + dz[None]); r2 = float(((cel_t - c2) / skala_t).norm() / np.sqrt(5))
                    if r2 < r0:
                        return (z + dz[None]).detach(), r0, r2
                    lam *= 4
                return z.detach(), r0, r0

            ramiona = {}
            with gen._frozen_eval(generator, reader.net):
                for arm, c_arm in (('target', cel), ('current', cur)):
                    cel_t = torch.tensor(c_arm, dtype=torch.float32, device=device)
                    z = z_star.clone(); it = 0
                    for it in range(12):
                        z, r0, r2 = krok_gn(z, cel_t)
                        if r2 >= r0 or r2 < .02:
                            break
                    with torch.no_grad():
                        c, rgba = kod(z)
                    gen._save_image(kd / f'{arm}.png', rgba[:, :3]); gen._save_image(kd / f'{arm}_mask.png', rgba[:, 3:4])
                    ramiona[arm] = dict(reszta=float(((cel_t - c) / skala_t).norm() / np.sqrt(5)), iteracje=it + 1,
                                        kod=c.cpu().numpy().tolist(), rgb=rgba[:, :3].detach().cpu())
            l1 = float((ramiona['target'].pop('rgb') - ramiona['current'].pop('rgb')).abs().mean())
            rap[str(i)] = dict(obraz=rows[i]['image'], rola=rows[i]['role'], obecny=cur.tolist(), cel=cel.tolist(),
                               inwersja=inw, ramiona=ramiona, l1_rgb_cel_vs_obecny=l1, sekundy=time.time() - t1)
            print(f'  kotwica {i}: inwersja {inw["blad_skalowany"]:.3f} ({nazwa}), reszta cel {ramiona["target"]["reszta"]:.3f}, '
                  f'L1 cel-obecny {l1:.4f} ({time.time() - t1:.0f} s)', flush=True)
    del net, wrapper, generator
    torch.cuda.empty_cache()
    return rap


def wyszukaj(out, kotwice, wykluczone):
    from wyszukiwanie import query_index
    listy, pokrycie = {}, {}
    for i in kotwice:
        kd = out / f'kotwica_{i:02d}'
        wyn = {}
        for arm in ('target', 'current'):
            f = kd / f'zapytanie_{arm}.json'
            if not f.exists():
                query_index(INDEKS, kd / f'{arm}.png', sorted(wykluczone), out=f, top_k=TOP_K, device='cuda')
            wyn[arm] = [c['frame_id'] for c in jz(f)['candidates']]
        listy[i] = [(fid, c['score'], c.get('video_id')) for fid, c in
                    ((c['frame_id'], c) for c in jz(kd / 'zapytanie_target.json')['candidates'])]
        pokrycie[str(i)] = dict(top45_wspolne=len(set(wyn['target'][:45]) & set(wyn['current'][:45])),
                                top90_wspolne=len(set(wyn['target']) & set(wyn['current'])))
    nowe, widziane, poz = [], set(wykluczone), {i: 0 for i in kotwice}
    while len(nowe) < NOWYCH:
        ruch = False
        for i in kotwice:
            while poz[i] < len(listy[i]) and listy[i][poz[i]][0] in widziane:
                poz[i] += 1
            if poz[i] < len(listy[i]) and len(nowe) < NOWYCH:
                fid, sc, vid = listy[i][poz[i]]
                nowe.append(dict(frame_id=fid, score=float(sc), video=vid, kotwica=f'{i:02d}'))
                widziane.add(fid); ruch = True
        if not ruch:
            raise RuntimeError('za mało kandydatów; zwiększ TOP_K')
    return nowe, pokrycie


def pula_nieuzyta():
    from wyszukiwanie import load_index
    _, _, rows, _ = load_index(INDEKS)
    return [r['frame_id'] for r in rows]


def selekcja(petla, k, dodane, baza):
    ids = list(baza['frame_ids']) + [d['frame_id'] for d in dodane]
    assert len(ids) == len(set(ids)), 'duplikaty w selekcji'
    return dict(run_id=f'petla_{petla}_k{k:02d}', baza=baza['baza'], dodane=len(dodane), frame_ids=ids, dodane_szczegoly=dodane)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, required=True)
    a = ap.parse_args(argv)
    k = a.k
    t0 = time.time()
    out = W / 'obroty' / f'k{k:02d}'; out.mkdir(parents=True, exist_ok=True)
    P = projekcja(); rows, mapa, kotwice = sondy()
    baza = jz(BAZA_SEL)
    wykl_baza = set(jz(WYKL))
    poprzednie = {p: (jz(W / 'selekcje' / f'{p}_k{k - 1:02d}.json')['dodane_szczegoly'] if k > 1 else []) for p in ('koopman', 'losowe')}
    # --- pętla koopman: prognoza
    Y = historia(k, P, mapa)
    delta, diag = dopasuj_i_prognozuj(Y)
    skala = Y[-1].std(0)
    wagi = WAGI_START if k == 1 else W / 'lancuchy' / 'koopman' / f'k{k - 1:02d}' / 'model_0000165.pth'
    assert Path(wagi).is_file(), wagi
    rap = dict(k=k, wagi=str(wagi), punktow_historii=int(len(Y)), horyzont_krokow=H, pamiec=L_PAM, lam=LAM,
               przyrost=delta.tolist(), przyrost_w_odchyleniach=(delta / skala).tolist(), skala=skala.tolist(), diagnostyka=diag)
    if k > 1:
        pr = jz(W / 'obroty' / f'k{k - 1:02d}' / 'RAPORT.json')
        rz = Y[-1].mean(0) - Y[-1 - H].mean(0)
        rap['sprawdzenie_poprzedniej_prognozy'] = dict(
            przewidziany=pr['przyrost'], zmierzony=rz.tolist(),
            blad_w_odchyleniach=((np.array(pr['przyrost']) - rz) / skala).tolist(),
            blad_wzgl_braku_zmian=float(np.linalg.norm(np.array(pr['przyrost']) - rz) / max(np.linalg.norm(rz), 1e-12)))
    print(f'obrót {k}: historia {len(Y)} punktów, przyrost w odchyleniach {np.round(delta / skala, 3).tolist()}', flush=True)
    rap['kotwice'] = fastgan(k, wagi, delta, skala, rows, kotwice, out)
    zj(out / 'RAPORT.json', rap)
    # --- wyszukanie i selekcje
    wykl_k = wykl_baza | {d['frame_id'] for d in poprzednie['koopman']}
    nowe, pokrycie = wyszukaj(out, kotwice, wykl_k)
    for d in nowe:
        d['obrot'] = k
    rap['pokrycie_cel_vs_obecny'] = pokrycie
    zj(W / 'selekcje' / f'koopman_k{k:02d}.json', selekcja('koopman', k, poprzednie['koopman'] + nowe, baza))
    pula = [f for f in pula_nieuzyta() if f not in (wykl_baza | {d['frame_id'] for d in poprzednie['losowe']})]
    rng = np.random.default_rng(1000 + k)
    los = [dict(frame_id=pula[j], obrot=k) for j in rng.choice(len(pula), NOWYCH, replace=False)]
    zj(W / 'selekcje' / f'losowe_k{k:02d}.json', selekcja('losowe', k, poprzednie['losowe'] + los, baza))
    zj(W / 'selekcje' / f'bez_k{k:02d}.json', selekcja('bez', k, [], baza))
    rap['sekundy'] = time.time() - t0
    zj(out / 'RAPORT.json', rap)
    print(f'obrót {k}: selekcje zapisane (koopman +{len(nowe)}, losowe +{len(los)}), pokrycie {pokrycie}, {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
