"""Czy progi laserów MLP (768 neuronów ukrytych, GELU) leżą MIĘDZY tokenami narzędzia a tokenami tła, tj. czy lasery są wybiórcze klasowo,
i jaką część skoku Fishera w MLP b robią lasery wybiórcze. Pomiar na istniejących checkpointach 3047 (46) i 10 klatkach TEST.

Numeracja bloków 1-based (b = 1..6; blok b = net.bloki[b-1]): stan przed MLP b = X[2b-1], po MLP b = X[2b] (zgodnie z lodyga.py: Fisher m4 = X[8]).
Jasność lasera z = wyjście GELU (po aktywacji; hak na mlp.act), „zapalony” = z > PROG_Z (jak w pomiar_3047.py). Blok timm bez LayerScale
(ls2 = Identity), więc wkład MLP do tokenu j to Σ_c z_jc v_c + fc2.bias, v_c = fc2.weight[:, c].
Wynik: Wyniki/Dane/lasery_klasy.npz, lasery_klasy_tabele.json. Użycie: python lasery_klasy.py <katalog_wyjsciowy>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BAZA = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
REPL = BAZA / 'experiments/plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41'
sys.path.insert(0, str(REPL / 'Kod_eksperymentu'))
import wspolne as WS  # noqa: E402
from wspolne import dev  # noqa: E402

WY = Path(sys.argv[1]); WY.mkdir(parents=True, exist_ok=True)
PROG_Z = 0.05
KLASY = ['obie', 'tylko_narzedzie', 'tylko_tlo', 'ciemne', 'mieszane']
KROKI_SUROWE = (0, 1000, 10000, 65640, 437500)
NB = 6


def fisher(X, a_idx, b_idx):
    """‖μ_T − μ_B‖² / średnia (ważona liczebnością) suma wariancji wewnątrzklasowych, jak w lodyga.py (na stanach wycentrowanych po tokenach)."""
    X = X - X.mean(0, keepdims=True)
    a, b = X[a_idx], X[b_idx]; d = a.mean(0) - b.mean(0)
    return float(d @ d / ((a.var(0).sum() * len(a) + b.var(0).sum() * len(b)) / (len(a) + len(b))))


def pT_pB(Zb, a_idx, b_idx):
    on = Zb > PROG_Z
    return on[a_idx].double().mean(0), on[b_idx].double().mean(0)


def klasy(pT, pB):
    """Liczby laserów w pięciu klasach (kolejność KLASY)."""
    obie = (pT >= .5) & (pB >= .5); tn = (pT >= .5) & (pB <= .1); tt = (pB >= .5) & (pT <= .1); ci = (pT <= .1) & (pB <= .1)
    mie = ~(obie | tn | tt | ci)
    return np.array([int(m.sum()) for m in (obie, tn, tt, ci, mie)]), tn, tt


KL = json.loads((REPL / 'Wyniki/Dane/klatki.json').read_text(encoding='utf-8'))
OBR = []
for k in KL:
    rgb = WS.obraz(k['film'], k['klatka'])[0]; u = WS.etykieta_cystotomu(k['film'], k['klatka'])['udzial']
    OBR.append((rgb, np.where(u >= WS.PROG_NARZ)[0], np.where(u == 0)[0]))
rng_los = np.random.default_rng(0)                                  # kontrola Fishera: losowy podzbiór laserów tej samej liczności co wybiórcze
rng = np.random.default_rng(0)                                      # jedno przetasowanie na klatkę, stały seed 0
ZERO = []
for _, na, tl in OBR:
    p = rng.permutation(np.concatenate([na, tl])); ZERO.append((p[:len(na)], p[len(na):]))
print('klatki', len(OBR), 'narzędzie/tło/pośrednie:', [(len(a), len(b), 600 - len(a) - len(b)) for _, a, b in OBR], flush=True)

net = WS.siec()
kroki = WS.checkpointy(); nK, nF = len(kroki), len(OBR)
print('checkpointy', nK, kroki, 'dev', dev, flush=True)
f4 = lambda *s: np.full(s, np.nan)
R = dict(lit_narz=f4(nK, nF, NB), lit_tlo=f4(nK, nF, NB), klasy=f4(nK, nF, NB, 5), klasy_zero=f4(nK, nF, NB, 5),
         absd_med=f4(nK, nF, NB), absd_frac=f4(nK, nF, NB), absd_med_zero=f4(nK, nF, NB), absd_frac_zero=f4(nK, nF, NB),
         n_sel=f4(nK, nF, NB), n_sel_zero=f4(nK, nF, NB), fisher=f4(nK, nF, NB, 4), fisher_zero=f4(nK, nF, NB, 2), fisher_los=f4(nK, nF, NB),
         cos_v=f4(nK, nF, NB, 3), blad_rekonstrukcji=f4(nK, nF, NB), pT4=f4(len(KROKI_SUROWE), nF, 768), pB4=f4(len(KROKI_SUROWE), nF, 768),
         pT4_zero=f4(len(KROKI_SUROWE), nF, 768), pB4_zero=f4(len(KROKI_SUROWE), nF, 768))
t0 = time.time()
for ik, krok in enumerate(kroki):
    WS.wczytaj(net, krok)
    for jf, (rgb, na, tl) in enumerate(OBR):
        P = WS.przejscie(net, rgb); X = P['X'].double(); Z = P['Z'].double()
        for b in range(1, NB + 1):
            blk = net.bloki[b - 1]; V = blk.mlp.fc2.weight.double().T                  # (768, 192): wiersz c = v_c
            bias = blk.mlp.fc2.bias.double(); Zb = Z[b - 1]; Xp, Xo = X[2 * b - 1], X[2 * b]
            on = Zb > PROG_Z
            R['lit_narz'][ik, jf, b - 1] = float(on[na].double().mean(1).median()); R['lit_tlo'][ik, jf, b - 1] = float(on[tl].double().mean(1).median())
            pT, pB = pT_pB(Zb, na, tl); pT0, pB0 = pT_pB(Zb, *ZERO[jf])
            kl, tn, tt = klasy(pT.cpu().numpy(), pB.cpu().numpy()); R['klasy'][ik, jf, b - 1] = kl
            R['klasy_zero'][ik, jf, b - 1] = klasy(pT0.cpu().numpy(), pB0.cpu().numpy())[0]
            d = (pT - pB).abs(); d0 = (pT0 - pB0).abs(); sel = d > .5; sel0 = d0 > .5
            R['absd_med'][ik, jf, b - 1] = float(d.median()); R['absd_frac'][ik, jf, b - 1] = float(sel.double().mean())
            R['absd_med_zero'][ik, jf, b - 1] = float(d0.median()); R['absd_frac_zero'][ik, jf, b - 1] = float(sel0.double().mean())
            R['n_sel'][ik, jf, b - 1] = int(sel.sum()); R['n_sel_zero'][ik, jf, b - 1] = int(sel0.sum())
            # dekompozycja wkładu MLP
            dall = Zb @ V + bias; dsel = Zb[:, sel] @ V[sel]; dnon = Zb[:, ~sel] @ V[~sel] + bias
            R['blad_rekonstrukcji'][ik, jf, b - 1] = float((Xo - Xp - dall).norm() / (Xo - Xp).norm())
            Xp_, Xo_ = Xp.cpu().numpy(), Xo.cpu().numpy()
            R['fisher'][ik, jf, b - 1] = [fisher(Xp_, na, tl), fisher(Xo_, na, tl), fisher(Xp_ + dsel.cpu().numpy(), na, tl), fisher(Xp_ + dnon.cpu().numpy(), na, tl)]
            dsel0 = Zb[:, sel0] @ V[sel0]; dnon0 = Zb[:, ~sel0] @ V[~sel0] + bias        # zero: lasery „wybiórcze” wg przetasowanych etykiet
            R['fisher_zero'][ik, jf, b - 1] = [fisher(Xp_ + dsel0.cpu().numpy(), na, tl), fisher(Xp_ + dnon0.cpu().numpy(), na, tl)]
            los = torch.from_numpy(rng_los.permutation(768)[:int(sel.sum())]).to(Zb.device)       # losowy podzbiór o liczności |sel| (kontrola dodatkowa)
            R['fisher_los'][ik, jf, b - 1] = fisher(Xp_ + (Zb[:, los] @ V[los]).cpu().numpy(), na, tl)
            # cos(v_c, μ_T − μ_B po MLP b)
            u = torch.from_numpy(Xo_[na].mean(0) - Xo_[tl].mean(0)).to(Zb.device); u = u / u.norm()
            cs = (V / V.norm(dim=1, keepdim=True)) @ u
            for q, m in enumerate((torch.from_numpy(tn).to(cs.device), torch.from_numpy(tt).to(cs.device), ~sel)):
                if int(m.sum()):
                    R['cos_v'][ik, jf, b - 1, q] = float(cs[m].median())
            if b == 4 and krok in KROKI_SUROWE:
                s = KROKI_SUROWE.index(krok)
                R['pT4'][s, jf], R['pB4'][s, jf] = pT.cpu().numpy(), pB.cpu().numpy(); R['pT4_zero'][s, jf], R['pB4_zero'][s, jf] = pT0.cpu().numpy(), pB0.cpu().numpy()
    f = np.nanmedian(R['fisher'][ik], 0)
    print(f'krok {krok:6d} [{time.time() - t0:5.0f}s]  lit narz/tlo (blok 1..6) '
          + ' '.join(f'{np.median(R["lit_narz"][ik, :, i]):.2f}/{np.median(R["lit_tlo"][ik, :, i]):.2f}' for i in range(NB))
          + f'  Fisher blok4 przed/po/sel/niesel {f[3, 0]:.2f}/{f[3, 1]:.2f}/{f[3, 2]:.2f}/{f[3, 3]:.2f}  błąd rek. max {np.nanmax(R["blad_rekonstrukcji"][ik]):.2e}', flush=True)

np.savez_compressed(WY / 'lasery_klasy.npz', kroki=np.array(kroki), kroki_surowe=np.array(KROKI_SUROWE), klasy_nazwy=np.array(KLASY),
                    n_narz=np.array([len(a) for _, a, _ in OBR]), n_tlo=np.array([len(b) for _, _, b in OBR]), prog_z=PROG_Z, **R)
med = lambda a: np.nanmedian(a, 0)
tab = {}
for ik, krok in enumerate(kroki):
    tab[str(krok)] = dict(
        lit_narz=med(R['lit_narz'][ik]).tolist(), lit_tlo=med(R['lit_tlo'][ik]).tolist(),
        klasy={KLASY[q]: med(R['klasy'][ik][..., q]).tolist() for q in range(5)}, klasy_zero={KLASY[q]: med(R['klasy_zero'][ik][..., q]).tolist() for q in range(5)},
        absd_mediana=med(R['absd_med'][ik]).tolist(), absd_odsetek_gt_05=med(R['absd_frac'][ik]).tolist(),
        absd_mediana_zero=med(R['absd_med_zero'][ik]).tolist(), absd_odsetek_gt_05_zero=med(R['absd_frac_zero'][ik]).tolist(),
        n_wybiorczych=med(R['n_sel'][ik]).tolist(), n_wybiorczych_zero=med(R['n_sel_zero'][ik]).tolist(),
        fisher={n: med(R['fisher'][ik][..., q]).tolist() for q, n in enumerate(['przed', 'po', 'przed+tylko_wybiorcze', 'przed+tylko_niewybiorcze'])},
        fisher_zero={n: med(R['fisher_zero'][ik][..., q]).tolist() for q, n in enumerate(['przed+tylko_wybiorcze_zero', 'przed+tylko_niewybiorcze_zero'])},
        fisher_los_losowy_podzbior=med(R['fisher_los'][ik]).tolist(),
        cos_v={n: med(R['cos_v'][ik][..., q]).tolist() for q, n in enumerate(['tylko_narzedzie', 'tylko_tlo', 'niewybiorcze'])},
        blad_rekonstrukcji_max=np.nanmax(R['blad_rekonstrukcji'][ik], 0).tolist())
json.dump(dict(opis='mediany po 10 klatkach; listy po blokach 1..6', klasy_nazwy=KLASY, tabele=tab), open(WY / 'lasery_klasy_tabele.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('blad rekonstrukcji: max', float(np.nanmax(R['blad_rekonstrukcji'])), 'mediana', float(np.nanmedian(R['blad_rekonstrukcji'])))
print('zapisano', WY / 'lasery_klasy.npz', time.time() - t0, 's')
