#!/usr/bin/env python3
"""Analiza pętli uczenia (PLAN.md): tempo uczenia i wynik trzech pętli po obrotach, jakość prognozy Koopmana,
pokrycie wyboru z sąsiadami samej kotwicy, efekt lokalny przy kotwicach.

Wejścia: wyniki/STAN.json, wyniki/lancuchy/{petla}/kXX/{REPORT.json, model_0000165.pth}, wyniki/obroty/kXX/RAPORT.json,
../koopman_tozsamosc_kierunki/wyniki/dobor_klatek/efekt_lokalny/{PODOBIENSTWO_VAL.npz, PREDYKCJE_start_0245.npz}.
Etapy:
  predykcje  wagi końca każdego obrotu każdej pętli: 10 procent val najbliższych kotwicom; ostatni obrót: całe val.
  analiza    -> wyniki/ANALIZA.json, wyniki/ANALIZA.png.

    <venv rfdetr>/python analiza.py --etap wszystko
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KT = HERE.parent / 'koopman_tozsamosc_kierunki'
sys.path.insert(0, str(KT))
import efekt_lokalny as E                       # noqa: E402
import efekt_lokalny_5ep as E5                  # noqa: E402

W = HERE / 'wyniki'
PRED = W / 'predykcje'
PETLE = ('koopman', 'losowe', 'bez')
PROGI_TEMPA = (0.890, 0.895, 0.900)


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def obroty():
    return sorted(int(d.name[1:]) for d in (W / 'lancuchy' / 'bez').glob('k*') if (d / 'REPORT.json').exists()) \
        if (W / 'lancuchy' / 'bez').exists() else []


def etap_predykcje(batch=32):
    import torch
    from PIL import Image
    from rfdetr import RFDETRNano
    _, rows = E.wczytaj_val()
    pz = E5.podzbiory(rows)
    PRED.mkdir(parents=True, exist_ok=True)
    ks = obroty()
    for p in PETLE:
        for k in ks:
            w = W / 'lancuchy' / p / f'k{k:02d}' / 'model_0000165.pth'
            cel = PRED / f'PRED_{p}_k{k:02d}.npz'
            if cel.exists() or not w.exists():
                continue
            idx = pz['wszystkie'] if k == max(ks) else pz['top10proc']
            t0 = time.time()
            model = RFDETRNano(pretrain_weights=str(w), num_classes=1, resolution=384, amp=False, device='cuda', trust_checkpoint=True)
            B, S, off = E5.przewiduj(model, rows, idx, batch, torch, Image)
            np.savez_compressed(cel, idx=idx, boxes=B, scores=S, offsets=off)
            print(f'{p} k{k}: {len(idx)} klatek, {time.time() - t0:.0f} s', flush=True)
            del model
            torch.cuda.empty_cache()


def etap_analiza():
    _, rows = E.wczytaj_val()
    N = len(rows)
    gt = [np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                    for a in r['annotations']], float).reshape(-1, 4) for r in rows]
    pz = E5.podzbiory(rows)
    ks = obroty()
    start, _ = E5.wczytaj_pred(E.OUT / 'PREDYKCJE_start_0245.npz', N)
    lok0 = {pn: E.ap_coco(start, gt, list(pz[pn]), [.5])[.5] for pn in ('top134', 'top10proc')}
    wyn = {}
    for p in PETLE:
        d = dict(mAP50=[], mAP50_95=[], F1=[], klatek=[], top134=[], top10proc=[])
        for k in ks:
            rep = jz(W / 'lancuchy' / p / f'k{k:02d}' / 'REPORT.json'); v = rep['validation'][0]
            d['mAP50'].append(float(v['val/mAP_50'])); d['mAP50_95'].append(float(v['val/mAP_50_95']))
            d['F1'].append(float(v['val/F1'])); d['klatek'].append(rep['klatek_train'])
            f = PRED / f'PRED_{p}_k{k:02d}.npz'
            if f.exists():
                pr, _ = E5.wczytaj_pred(f, N)
                d['top134'].append(E.ap_coco(pr, gt, list(pz['top134']), [.5])[.5])
                d['top10proc'].append(E.ap_coco(pr, gt, list(pz['top10proc']), [.5])[.5])
            else:
                d['top134'].append(None); d['top10proc'].append(None)
        d['tempo_obrot_do_progu'] = {str(t): next((k for k, m in zip(ks, d['mAP50']) if m >= t), None) for t in PROGI_TEMPA}
        d['pole_pod_krzywa_mAP50'] = float(np.mean(d['mAP50'])) if ks else None
        wyn[p] = d
    roznice = {}
    for p in ('koopman', 'losowe'):
        for ref in ('bez', 'losowe'):
            if p == ref:
                continue
            roznice[f'{p}-{ref}'] = dict(
                mAP50=[a - b for a, b in zip(wyn[p]['mAP50'], wyn[ref]['mAP50'])],
                top134=[None if a is None or b is None else a - b for a, b in zip(wyn[p]['top134'], wyn[ref]['top134'])],
                pole_pod_krzywa=wyn[p]['pole_pod_krzywa_mAP50'] - wyn[ref]['pole_pod_krzywa_mAP50'] if ks else None)
    prognozy, pokrycie = [], []
    for k in ks:
        f = W / 'obroty' / f'k{k:02d}' / 'RAPORT.json'
        if not f.exists():
            continue
        r = jz(f)
        pokrycie.append(dict(k=k, **{kk: v['top45_wspolne'] for kk, v in r.get('pokrycie_cel_vs_obecny', {}).items()}))
        prognozy.append(dict(k=k, przyrost_w_odchyleniach=r['przyrost_w_odchyleniach'],
                             sprawdzenie=r.get('sprawdzenie_poprzedniej_prognozy'),
                             l1_rgb_cel_vs_obecny={kk: v['l1_rgb_cel_vs_obecny'] for kk, v in r['kotwice'].items()}))
    kryt = None
    if len(ks) == 8 and wyn['koopman']['top134'][-1] is not None:
        kryt = dict(top134_koopman_minus_losowe=wyn['koopman']['top134'][-1] - wyn['losowe']['top134'][-1],
                    mAP50_koopman_minus_bez=wyn['koopman']['mAP50'][-1] - wyn['bez']['mAP50'][-1])
        kryt['spelnione_jedno_ziarno'] = bool(kryt['top134_koopman_minus_losowe'] > 0 and kryt['mAP50_koopman_minus_bez'] >= -0.002)
    raport = dict(obroty=ks, start_lokalnie=lok0, petle=wyn, roznice=roznice, prognozy=prognozy, pokrycie_top45=pokrycie,
                  kryterium=kryt, uwaga='Jedno ziarno (42): różnice poniżej około 0,01 lokalnie i 0,003 globalnie mieszczą się '
                                         'w rozrzucie między ziarnami z kampanii 5 epok.')
    (W / 'ANALIZA.json').write_text(json.dumps(raport, indent=1, ensure_ascii=False), encoding='utf-8')
    rysuj(raport)
    for p, d in wyn.items():
        print(f'{p:8s} mAP50 ' + ' '.join(f'{m:.4f}' for m in d['mAP50']) + f'  top134 ' +
              ' '.join('-' if x is None else f'{x:.3f}' for x in d['top134']) + f'  tempo {d["tempo_obrot_do_progu"]}')
    print('kryterium:', kryt)


def rysuj(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kol = dict(koopman='C0', losowe='C1', bez='k')
    ks = r['obroty']
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for p, d in r['petle'].items():
        ax[0, 0].plot(ks, d['mAP50'], 'o-', color=kol[p], label=p)
        y = [np.nan if v is None else v for v in d['top134']]
        ax[0, 1].plot([0] + ks, [r['start_lokalnie']['top134']] + y, 'o-', color=kol[p], label=p)
    ax[0, 0].set_title('mAP@0,5 całego val po obrotach (Eden)'); ax[0, 1].set_title('AP@0,5 na 134 klatkach val przy kotwicach')
    for a in ax[0]:
        a.set_xlabel('obrót (165 aktualizacji)'); a.grid(alpha=.3); a.legend(fontsize=8)
    pk = [p['k'] for p in r['prognozy'] if p['sprawdzenie']]
    if pk:
        ax[1, 0].plot(pk, [p['sprawdzenie']['blad_wzgl_braku_zmian'] for p in r['prognozy'] if p['sprawdzenie']], 'o-')
    ax[1, 0].axhline(1, color='k', lw=.8); ax[1, 0].set_title('Błąd prognozy średniego przyrostu latentu (1 = jak brak zmian)')
    ax[1, 0].set_xlabel('obrót sprawdzany'); ax[1, 0].grid(alpha=.3)
    for kk in ('0', '24', '48'):
        ax[1, 1].plot([p['k'] for p in r['pokrycie_top45']], [p.get(kk) for p in r['pokrycie_top45']], 'o-', label=f'kotwica {kk}')
    ax[1, 1].axhline(45 * .9, color='k', lw=.8, ls='--'); ax[1, 1].set_ylim(0, 46)
    ax[1, 1].set_title('Wspólne klatki top-45: obraz prognozy vs obraz obecny'); ax[1, 1].set_xlabel('obrót')
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=8)
    fig.suptitle('Pętla uczenia: prognoza latentu → FastGAN → wyszukanie → uczenie (ziarno 42)')
    fig.tight_layout(); fig.savefig(W / 'ANALIZA.png', dpi=120); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--etap', choices=['predykcje', 'analiza', 'wszystko'], default='wszystko')
    a = ap.parse_args(argv)
    if a.etap in ('predykcje', 'wszystko'):
        etap_predykcje()
    if a.etap in ('analiza', 'wszystko'):
        etap_analiza()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
