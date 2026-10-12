"""Składa lokalną część assetu RuchCzasteczek_ViT_szklo_2026-10 z katalogów experiments/ i outputs/ (część z Edena pobiera
pobierz_z_edena.sh). Pliki tekstowe (json, md, py, txt, sh, slurm, jsonl, html, log, csv) są kopiowane naprawdę. Pliki binarne
(q16, npz, npy, pt, mp4, png, jpg, pdf, ...) są twardymi dowiązaniami NTFS: ta sama treść pod drugą ścieżką, bez zajmowania
miejsca; usunięcie oryginału w experiments/ albo outputs/ NIE usuwa danych z assets. Binarnych zapisów nikt nie edytuje w miejscu
(aplikacja zapisuje nowe pliki przez os.replace), więc dowiązanie zachowuje się jak kopia. Rejestr: KOPIA_LOKALNA.tsv.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 zloz_lokalnie.py"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.normpath(os.path.join(HERE, '..', '..'))          # SmallViTAblation
TEKST = {'.json', '.md', '.py', '.txt', '.sh', '.slurm', '.jsonl', '.html', '.log', '.csv', '.bat', '.yaml', '.yml', '.tsv'}
GUI = 'outputs/ruch_czasteczek_gui/'
E35 = 'experiments/maly_vit_od_zera_35000_21epok_20261009/'

PLAN = [  # (źródło względem SmallViTAblation, cel względem assetu)
    (E35 + 'wyniki/gesty', '01_bieg_35000_gesty_zapis/lokalnie/gesty'),
    (E35 + 'wyniki/pelny/dane', '01_bieg_35000_gesty_zapis/lokalnie/pelny_dane'),
    (E35 + 'wejscia', '01_bieg_35000_gesty_zapis/lokalnie/wejscia'),
    (GUI + '20261010T_natural_eden_pelny_3000tr_1000val', '02_osie_ruchu_tokenow/A_35000_4000zdjec_do_21_epoki/20261010T_natural_eden_pelny_3000tr_1000val'),
    (GUI + '20261009T162654_natural_78cd84bd6edd', '02_osie_ruchu_tokenow/B_35000_100zdjec_konce_epok_do_21/20261009T162654_natural_78cd84bd6edd'),
    (GUI + '20261009T221404_natural_8083acc06418', '02_osie_ruchu_tokenow/C_35000_20zdjec_konce_epok_do_60/20261009T221404_natural_8083acc06418'),
    (GUI + '20261009T172905_natural_5de07a0979e3', '02_osie_ruchu_tokenow/D_35000_5zdjec_konce_epok_do_21/20261009T172905_natural_5de07a0979e3'),
    (GUI + '20261010T_natural_szklo_klatka_35k', '02_osie_ruchu_tokenow/E_35000_100zdjec_uklad_klatki_gesty/20261010T_natural_szklo_klatka_35k'),
]
for n in ('20261009T080332_natural_0051d7971abb', '20261009T080332_natural_0051d7971abb_q16', '20261009T081936_natural_07e68c7cdb6f',
          '20261009T081936_natural_07e68c7cdb6f_q16', '20261009T094824_natural_41cd2ac1904b', '20261009T094824_natural_41cd2ac1904b_q16',
          '20261009T110313_natural_fa3c69aee473'):
    PLAN.append((GUI + n, '02_osie_ruchu_tokenow/F_K1000_1500zdjec_kroki_0-1000/' + n))
PLAN.append((GUI + '20261009T172911_natural_ef7da43f8027', '02_osie_ruchu_tokenow/G_K1000_3zdjecia_kroki_0-1000/20261009T172911_natural_ef7da43f8027'))
for n in sorted(os.listdir(os.path.join(S, GUI))):
    p = os.path.join(S, GUI, n)
    if n.startswith('20261010T_szklo_eden_'):
        PLAN.append((GUI + n, '04_szklo_koloidalne_BD/osie_aplikacji/' + n))
    elif n.startswith('20261009T') and not any(n == os.path.basename(a) for a, _ in PLAN):
        PLAN.append((GUI + n, '02_osie_ruchu_tokenow/Z_proby_i_nieudane/' + n))
PLAN += [
    ('outputs/porownanie_szklo_20261010', '06_analizy/porownanie_szklo_20261010'),
    ('outputs/animacje_ruchu_20261010', '06_analizy/animacje_ruchu_20261010'),
    ('outputs/app_ruch_czasteczek_20261009', '06_analizy/app_ruch_czasteczek_20261009'),
    ('experiments/maly_vit_szklo_pokretla_20261010', '07_kod/lokalnie/maly_vit_szklo_pokretla_20261010'),
    ('experiments/app_ruch_czasteczek_20261009', '07_kod/lokalnie/app_ruch_czasteczek_20261009'),
    ('literatura_szklo_tokeny_20261011', '08_literatura/literatura_szklo_tokeny_20261011'),
]
for f in ('README.md', 'STATUS.md', 'LITERATURA_FIZYKA_RUCHU.md', 'bieg_eden.slurm', 'pobierz_gesty.sh', 'pobierz_z_edena.sh', 'kod'):
    PLAN.append((E35 + f, '07_kod/lokalnie/maly_vit_od_zera_35000_21epok_20261009/' + f))
for f in ('ruch_zbioru.py', 'ruch_gpu.py', 'ruch_naturalny.py', 'zakladka_ruchu.py', 'historie_checkpointow.py', 'README.md'):
    PLAN.append(('app/' + f, '07_kod/lokalnie/app_format_osi/' + f))

rej, n_kop, n_dow, b_kop, b_dow = [], 0, 0, 0, 0


def jeden(src, dst):
    global n_kop, n_dow, b_kop, b_dow
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rozm = os.path.getsize(src)
    if os.path.exists(dst):
        if os.path.getsize(dst) == rozm:
            rej.append((src, dst, rozm, 'juz_jest')); return
        os.remove(dst)
    if os.path.splitext(src)[1].lower() in TEKST:
        shutil.copy2(src, dst); n_kop += 1; b_kop += rozm; sp = 'kopia'
    else:
        os.link(src, dst); n_dow += 1; b_dow += rozm; sp = 'dowiazanie'
    rej.append((src, dst, rozm, sp))


for a, b in PLAN:
    src, dst = os.path.join(S, a), os.path.join(HERE, b)
    if not os.path.exists(src):
        print('BRAK źródła:', a); continue
    if os.path.isfile(src):
        jeden(src, dst); continue
    for kat, podk, pliki in os.walk(src):
        podk[:] = [d for d in podk if d != '__pycache__']
        for f in pliki:
            if f.endswith('.pyc'):
                continue
            s = os.path.join(kat, f)
            jeden(s, os.path.join(dst, os.path.relpath(s, src)))
    print(f'{a} -> {b}', flush=True)

with open(os.path.join(HERE, 'KOPIA_LOKALNA.tsv'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('zrodlo\tcel\tbajty\tsposob\n')
    for s, d, r, sp in rej:
        f.write(f'{os.path.relpath(s, S)}\t{os.path.relpath(d, HERE)}\t{r}\t{sp}\n')
print(f'kopie: {n_kop} plików, {b_kop / 1e9:.2f} GB | dowiązania: {n_dow} plików, {b_dow / 1e9:.2f} GB | razem wpisów {len(rej)}')
