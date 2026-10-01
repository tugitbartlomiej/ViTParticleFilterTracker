#!/usr/bin/env python3
"""Analiza kampanii pętli 4.3 na Edenie (docs/26, krok 10): czy metoda uczy szybciej i lepiej niż klatki losowe i kolejne.

Wejście: kopia plików JSON z Edenu w `wyniki_eden43/` (`eden/pobierz43.sh`): petla43_s<ziarno>/<ramię>/OCENY.json
i k??/OBROT.json. Miary na ziarno i ramię (tylko oceny po rozgrzewce, obroty ≥ 3):
  pole  = średnie mAP@0,5 ze wszystkich ocen po rozgrzewce (pole pod krzywą uczenia: szybkość uczenia),
  koniec = mAP@0,5 ostatniej oceny,
  krok_do_progu = pierwszy krok, w którym mAP@0,5 ≥ próg (0,40 i 0,45), albo brak.
Porównania sparowane po ziarnach (ta sama rozgrzewka): odsetek ziaren, w których metoda wygrywa, średnia różnica,
test Wilcoxona (sparowany, dwustronny). Kryteria ustalone przed wynikiem (docs/26): K1 metoda > losowe w ≥ 70 % ziaren
i p < 0,05; K2 metoda > kolejne w ≥ 70 %; K3 metoda > sd_bez_prognozy w ≥ 70 %.
Do tego: trafność obu prognoz (Langevin z jądrem, Koopman/Langevin w. 3) po obrotach i pokrycie latentu przez SD.

    python analiza43.py   ->  wyniki_eden43/WYNIKI_EDEN43.json, WYNIKI_EDEN43.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
WE = HERE / 'wyniki_eden43'
RAMIONA = ('metoda', 'metoda_kl3', 'sd_bez_prognozy', 'losowe', 'kolejne', 'bez')
PROGI = (0.40, 0.45)
KONTRASTY = [('metoda', 'losowe'), ('metoda', 'kolejne'), ('metoda', 'sd_bez_prognozy'), ('metoda', 'bez'),
             ('metoda_kl3', 'losowe'), ('metoda_kl3', 'kolejne'), ('metoda', 'metoda_kl3'), ('sd_bez_prognozy', 'losowe')]


def wczytaj():
    dane = {}
    for d in sorted(WE.glob('petla43b_s*')):
        s = int(d.name.split('_s')[1])
        for r in RAMIONA:
            f = d / r / 'OCENY.json'
            if not (d / r / 'KONIEC.json').exists() or not f.exists():
                continue
            oc = [o for o in json.loads(f.read_text(encoding='utf-8')) if o['obrot'] >= 3]
            obr = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((d / r).glob('k??/OBROT.json'))]
            dane.setdefault(s, {})[r] = dict(kroki=np.array([o['krok'] for o in oc]), map=np.array([o['mAP50'] for o in oc]),
                                             obroty=obr)
    return dane


def miary(x):
    m = x['map']
    out = dict(pole=float(m.mean()), koniec=float(m[-1]), najlepsze=float(m.max()))
    for p in PROGI:
        i = np.where(m >= p)[0]
        out[f'krok_do_{p:.2f}'] = int(x['kroki'][i[0]]) if len(i) else None
    return out


def main() -> int:
    dane = wczytaj()
    ziarna = sorted(s for s in dane if all(r in dane[s] for r in RAMIONA))
    if not ziarna:
        print('brak pełnych ziaren'); return 1
    M = {s: {r: miary(dane[s][r]) for r in RAMIONA} for s in ziarna}
    wyn = dict(ziarna=ziarna, ramiona={}, kontrasty={}, kryteria={})
    for r in RAMIONA:
        wyn['ramiona'][r] = {k: (float(np.mean([M[s][r][k] for s in ziarna])) if not k.startswith('krok_do') else
                                 [M[s][r][k] for s in ziarna]) for k in M[ziarna[0]][r]}
        trg, trk, pt, pl = [], [], [], []
        for s in ziarna:
            for o in dane[s][r]['obroty']:
                tp = o.get('trafnosc_prognozy')
                if tp:
                    trg.append(tp['gle']['umiejetnosc']); trk.append(tp['kl3']['umiejetnosc'])
                for x in o.get('obrazy_sd') or []:
                    pt.append(x['pokrycie_tozsamosc']); pl.append(x['pokrycie_latent'])
        wyn['ramiona'][r].update(prognoza_langevin_jadro_mediana=float(np.median(trg)) if trg else None,
                                 prognoza_kl3_mediana=float(np.median(trk)) if trk else None,
                                 sd_pokrycie_tozsamosci_mediana=float(np.median(pt)) if pt else None,
                                 sd_pokrycie_latentu_mediana=float(np.median(pl)) if pl else None,
                                 sd_obrazow_ponad_90_obu=float(np.mean([(a >= 90) and (b >= 90) for a, b in zip(pt, pl)])) if pt else None)
    for a, b in KONTRASTY:
        for k in ('pole', 'koniec'):
            d = np.array([M[s][a][k] - M[s][b][k] for s in ziarna])
            p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else None
            wyn['kontrasty'][f'{a} - {b} ({k})'] = dict(srednia_roznica=float(d.mean()), wygrane=float((d > 0).mean()),
                                                         p_wilcoxon=p, roznice=d.round(4).tolist())
    kp = wyn['kontrasty']
    wyn['kryteria'] = dict(
        K1=kp['metoda - losowe (pole)']['wygrane'] >= .7 and (kp['metoda - losowe (pole)']['p_wilcoxon'] or 1) < .05,
        K2=kp['metoda - kolejne (pole)']['wygrane'] >= .7,
        K3=kp['metoda - sd_bez_prognozy (pole)']['wygrane'] >= .7)
    (WE / 'WYNIKI_EDEN43.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    # wykres: średnia krzywa mAP po ziarnach i różnice pola
    kol = dict(metoda='#c0392b', metoda_kl3='#e67e22', sd_bez_prognozy='#8e44ad', losowe='#2c7fb8', kolejne='#27ae60', bez='#7f8c8d')
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for r in RAMIONA:
        K = dane[ziarna[0]][r]['kroki']
        Y = np.stack([np.interp(K, dane[s][r]['kroki'], dane[s][r]['map']) for s in ziarna])
        ax[0].plot(K, Y.mean(0), color=kol[r], lw=2.2 if r == 'metoda' else 1.5, label=f'{r} (pole {Y.mean():.4f})')
        ax[0].fill_between(K, Y.mean(0) - Y.std(0) / np.sqrt(len(ziarna)), Y.mean(0) + Y.std(0) / np.sqrt(len(ziarna)),
                           color=kol[r], alpha=.12)
    ax[0].set_xlabel('krok uczenia'); ax[0].set_ylabel('mAP@0,5 walidacji (średnia ziaren ± błąd średniej)')
    ax[0].set_title(f'Krzywe uczenia, {len(ziarna)} ziaren'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    nazwy = [f'{a} − {b}' for a, b in KONTRASTY]
    for i, (a, b) in enumerate(KONTRASTY):
        d = np.array(kp[f'{a} - {b} (pole)']['roznice'])
        ax[1].scatter(np.full(len(d), i), d, color='0.4', s=14, zorder=3)
        ax[1].bar(i, d.mean(), color=kol[a], alpha=.6)
    ax[1].axhline(0, color='k', lw=1); ax[1].set_xticks(range(len(nazwy))); ax[1].set_xticklabels(nazwy, rotation=35, ha='right', fontsize=8)
    ax[1].set_ylabel('różnica pola pod krzywą mAP (punkty = ziarna)'); ax[1].set_title('Sparowane różnice'); ax[1].grid(alpha=.3, axis='y')
    fig.suptitle('Pętla 4.3: Stable Diffusion prowadzone latentem tożsamości, prognoza Langevina z jądrem pamięci')
    fig.tight_layout(); fig.savefig(WE / 'WYNIKI_EDEN43.png', dpi=125)
    for k, v in kp.items():
        if '(pole)' in k:
            print(f'{k:40s} różnica {v["srednia_roznica"]:+.4f}, wygrane {v["wygrane"]:.0%}, p {v["p_wilcoxon"]}')
    print('kryteria', wyn['kryteria'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
