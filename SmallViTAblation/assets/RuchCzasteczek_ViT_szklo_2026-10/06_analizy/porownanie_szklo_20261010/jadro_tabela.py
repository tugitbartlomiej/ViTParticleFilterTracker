"""Tabela testu jądra (eden/wyniki/pokretla/JADRO_<bieg>_<stan>.json z kod/jadro_pola.py): dla każdego stanu — rodziny przesunięć
(losowe kierunki prekondycjonowane / izotropowe, prawdziwy krok AdamW na paczce, część szumu, kontrola numeryczna) kontra zmierzone
przyrosty: C1 heterogeniczności i powłoki (CR), korelacja z sąsiadami z innych zdjęć, cos sąsiedzi − losowe, rozkład energii kroku,
Spearman mapy ruchliwości. Uruchomienie: F:/Python/3.13/python.exe jadro_tabela.py → JADRO_TABELA.json"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda v, d=2: ('  -  ' if v is None else f'{v:.{d}f}')
out = {}
for p in sorted(glob.glob(os.path.join(HERE, 'eden', 'wyniki', 'pokretla', 'JADRO_*.json'))):
    j = json.load(open(p, encoding='utf-8'))
    naz = os.path.basename(p)[len('JADRO_'):-len('.json')]
    print(f"\n=== {naz}: krok {j['krok']}, {j['sond']} sond, norma kroku AdamW {j['norma_kroku_adamw']:.4f}, okno zmierzone {j['okno_zmierzone']}")
    print(f"{'rodzina':<12}{'K':>4}{'E768':>8}{'C1':>6}{'pow2':>6}{'pow3':>6}{'cross':>7}{'cos':>6}{'C1pop':>7}{'wsp':>5}{'zdj':>5}{'sąs':>5}{'wł':>5}{'Sp.surowa':>10}{'Sp.wzgl':>8}{'Sp.zdj':>7}")
    wiersze = {}
    rodz = dict(j['rodziny']); rodz['zmierzone'] = j.get('zmierzone', {})
    sp = j.get('mapa_ruchliwosci_spearman', {})
    for nazwa, r in rodz.items():
        hb = r.get('het_blok') or {}; hp = r.get('het_populacja') or {}; rz = r.get('rozklad') or {}; s = sp.get(nazwa, {})
        pow_ = hb.get('C_powloki') or [None, None, None]
        w = dict(K=r.get('K'), E=r.get('E_768', r.get('E_128')), C1=hb.get('C1'), pow2=pow_[1], pow3=pow_[2], cross=r.get('het_blok_cross'), cos=r.get('cos_sasiedzi'),
                 C1pop=hp.get('C1'), wsp=rz.get('wspolny'), zdj=rz.get('zdjecie'), sas=rz.get('sasiedztwo'), wl=rz.get('wlasny'),
                 sp_surowa=s.get('surowa'), sp_wzgl=s.get('wzgledna'), sp_zdj=s.get('zdjecia'), blad=r.get('blad'))
        wiersze[nazwa] = w
        print(f"{nazwa:<12}{str(w['K']):>4}{f(w['E'], 3):>8}{f(w['C1']):>6}{f(w['pow2']):>6}{f(w['pow3']):>6}{f(w['cross']):>7}{f(w['cos']):>6}{f(w['C1pop']):>7}"
              f"{f(w['wsp']):>5}{f(w['zdj']):>5}{f(w['sas']):>5}{f(w['wl']):>5}{f(w['sp_surowa']):>10}{f(w['sp_wzgl']):>8}{f(w['sp_zdj']):>7}" + (f"  BŁĄD {w['blad']}" if w['blad'] else ''))
    out[naz] = dict(krok=j['krok'], norma_kroku=j['norma_kroku_adamw'], okno=j['okno_zmierzone'], rodziny=wiersze)
json.dump(out, open(os.path.join(HERE, 'JADRO_TABELA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nzapisano JADRO_TABELA.json')
