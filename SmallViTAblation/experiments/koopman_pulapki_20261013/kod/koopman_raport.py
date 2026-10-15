"""Tabela z KOOPMAN_PULAPKI.json: prognozy (stosunek błędu do „stoi”) dla pułapek i drgań, wobec surogatu błądzenia;
DMDc (sterowanie składowymi przyrostu wag) wobec kontroli; obrót i czasy z widma. Uruchomienie: python koopman_raport.py <katalog>"""
import json
import sys
from pathlib import Path


def f(pr, hs=('1', '5', '20')):
    if not pr:
        return '   –   '
    return ' '.join(f'{pr[h]["do_stoi"]:5.2f}' if h in pr else '   – ' for h in hs)


def main():
    W = json.load(open(Path(sys.argv[1]) / 'KOOPMAN_PULAPKI.json', encoding='utf-8'))
    print('stosunek błędu prognozy do „pułapka stoi” dla h = 1, 5, 20 kroków siatki (<1 = model przewiduje lepiej)')
    for pn, P in W['przestrzenie'].items():
        for rz, R in P.items():
            for sp in ('TRAIN', 'VALID'):
                m = R[sp]['mediany']
                if not m:
                    continue
                print(f'\n== {pn} {rz} {sp} ({R[sp]["zdjec"]} zdjęć)')
                for co in ('pulapki', 'drgania'):
                    q = m.get(co) or {}
                    s = (m.get('surogat') or {}).get(co) or {}
                    A = (q.get('A_operator') or {}).get('prognozy')
                    B5 = ((q.get('B_dmd_r') or {}).get('5') or {}).get('prognozy')
                    print(f'  {co:8s} A {f(A)} | sur {f(s.get("A_operator"))} || B5 {f(B5)} | sur {f(s.get("B5"))}')
                q = m.get('pulapki') or {}
                if q.get('D_dmdc'):
                    for rr, v in q['D_dmdc'].items():
                        print(f'  DMDc {rr}: ' + ' | '.join(f'{nm} {f(x["prognozy"])}' for nm, x in v.items()))
                wA = (q.get('A_operator') or {}).get('widmo') or {}
                wB = (q.get('B_dmd') or {}).get('widmo') or {}
                sB = q.get('B_surogat') or {}
                print(f'  widmo A: τ_med {wA.get("tau_med")}, kąt/100 kroków {wA.get("kat_med_na_100")}, |λ|max {wA.get("mod_max")} | '
                      f'B: kąt {wB.get("kat_med_na_100")} (surogat {sB.get("kat_med_na_100")}), τ_med {wB.get("tau_med")} (surogat {sB.get("tau_med")})')
                if q.get('B_mody'):
                    print('  mody B (najwolniejsze): ' + '; '.join(f'|λ| {x["mod_lambda"]:.3f} kąt {x["kat"]:.1f}° gładkość {x["gladkosc_sasiedzi"]:.2f} narzędzie×{(x["wzbogacenie_narzedzie"] or 0):.2f}'
                                                                 for x in q['B_mody']))
    print('\nudział wariancji przyrostu wag w składowych:', W.get('wejscia_udzial_wariancji'))


if __name__ == '__main__':
    main()
