"""Generuje README.md assetu i jego podfolderów z manifestów osi, metadanych biegów i listingów katalogów, żeby liczby w opisach
nie były przepisywane ręcznie. Uruchomienie po złożeniu (zloz_lokalnie.py) i pobraniu (pobierz_z_edena.sh):
F:/Python/3.13/python.exe -X utf8 opisz.py"""
import ast
import glob
import json
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
S = os.path.normpath(os.path.join(H, '..', '..'))


def rozm(p):
    if os.path.isfile(p):
        return os.path.getsize(p), 1
    b = n = 0
    for k, _, fs in os.walk(p):
        for f in fs:
            b += os.path.getsize(os.path.join(k, f))
            n += 1
    return b, n


def gb(b):
    return f'{b / 1e9:.2f} GB' if b >= 1e8 else f'{b / 1e6:.1f} MB'


def zapisz(rel, txt):
    p = os.path.join(H, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(txt.rstrip() + '\n')


def man(d):
    p = os.path.join(d, 'manifest.json')
    return json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else None


def liczba(x):
    return f'{x:,}'.replace(',', ' ')


# ---------------------------------------------------------------- 02 osie ruchu tokenów
OPIS_GRUP = {
    'A_35000_4000zdjec_do_21_epoki': 'Model 35 000 od zera (asset `MalyViT_35000_od_zera_s47`). 3000 zdjęć TRAIN + 1000 VALID, 341 checkpointów do 21. epoki '
                                     '(co 10 kroków do 1000, potem co 100, plus końce epok). Pomiar na Edenie (`pomiar_eden.py`). Główny materiał analiz TRAIN/VALID.',
    'B_35000_100zdjec_konce_epok_do_21': 'Model 35 000. 100 losowych klatek uczących (`outputs/vit_35000_losowe100_20261009`), końce epok 0–21.',
    'C_35000_20zdjec_konce_epok_do_60': 'Model 35 000. 20 klatek, końce epok 0–60 (pełne 60 epok, wagi z assetu `MalyViT_35000_od_zera_s47/wagi`).',
    'D_35000_5zdjec_konce_epok_do_21': 'Model 35 000. 5 klatek, końce epok 0–21 (podgląd).',
    'E_35000_100zdjec_uklad_klatki_gesty': 'Oś pochodna, nie pomiar checkpointów: gęsty zapis 100 sond biegu 35 000 (folder 01) w układzie względem klatki '
                                           '(CR, odjęty wspólny ruch 6 sąsiadów), PCA32, kroki 300–22 970. Skrypt `eksport_osi_szkla.py`.',
    'F_K1000_1500zdjec_kroki_0-1000': 'Inny model: gęsta historia K1000 (`experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi`, odniesienie '
                                      '`MalyViT_K1000_SD_conditioning`). 1500 zdjęć `MANIFEST_1500`, kroki 0–1000. Pomiary różnią się co ile kroków '
                                      'i wielkością paczki; `_q16` to kopie 16-bitowe tych samych osi.',
    'G_K1000_3zdjecia_kroki_0-1000': 'Model K1000, 3 zdjęcia, kroki 0–1000 co 10 (podgląd).',
    'Z_proby_i_nieudane': 'Próby i przerwane pomiary z 2026-10-09: optymalizacja latentu (3 kroki), pomiary przerwane lub nieudane (status w tabeli). '
                          'Zachowane dla kompletności historii, bez wartości analitycznej.',
}


def opis_osi(d):
    m = man(d)
    b, n = rozm(d)
    if m is None:
        return dict(status='brak manifestu', b=b, n=n)
    st = m.get('steps') or []
    c = m.get('config') or {}
    return dict(status=m.get('status'), b=b, n=n, zdj=m.get('image_count'), pkt=m.get('point_count'), kroki=len(st),
                zakres=f'{st[0]}–{st[-1]}' if st else '–', fmt=m.get('frame_format') or 'f32',
                co=c.get('checkpoint_stride', c.get('save_every', '')), paczka=c.get('batch_size', ''))


w = ['# Osie ruchu tokenów (format aplikacji, zakładka „Ruch zbioru”)', '',
     'Każdy podfolder to jedna oś czasu w formacie aplikacji (schemat 2): `manifest.json` (kroki, SHA-256 klatek, zdjęcia, parametry), '
     'klatki `step_NNNN.q16` albo `.f32` (cząstka = token; [N, 4] = x, y, z w PCA3 z kroku 0 i norma przesunięcia w pełnych 768D), '
     '`metrics.npy` (pewność, IoU, długość kroku na zdjęcie i krok), `initial_pca.npz` (baza PCA), `images.json`, czasem `pca32_podzbior.npz` '
     '(32 składowe dla podzbioru zdjęć) i `OCENY_POMIAR.json`. Wczytanie: aplikacja `app/app.py`, zakładka „Ruch zbioru”, otwarcie osi z podfolderu; '
     'w Pythonie `app/ruch_zbioru.py` (kopia w `07_kod/lokalnie/app_format_osi`). Oryginały leżą w `outputs/ruch_czasteczek_gui/`.', '']
for g in sorted(os.listdir(os.path.join(H, '02_osie_ruchu_tokenow'))):
    gd = os.path.join(H, '02_osie_ruchu_tokenow', g)
    if not os.path.isdir(gd):
        continue
    w += [f'## {g}', '', OPIS_GRUP.get(g, ''), '',
          '| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for a in sorted(os.listdir(gd)):
        o = opis_osi(os.path.join(gd, a))
        pkt = liczba(o['pkt']) if o.get('pkt') is not None else '–'
        w.append(f"| `{a}` | {o['status']} | {o.get('zdj', '–')} | {pkt} | {o.get('kroki', '–')} | {o.get('zakres', '–')} | "
                 f"{o.get('co', '')} | {o.get('paczka', '')} | {o.get('fmt', '–')} | {gb(o['b'])} |")
    w.append('')
zapisz('02_osie_ruchu_tokenow/README.md', '\n'.join(w))

# ---------------------------------------------------------------- 03 pomiar 60 epok
p60 = os.path.join(H, '03_pomiar_naturalny_60_epok_eden', 'pomiar_pelny60_3000tr_1000val')
w = ['# Pomiar naturalny do 60. epoki, 3000 TRAIN + 1000 VALID (Eden)', '']
m = man(p60)
if m:
    st = m['steps']
    b, n = rozm(p60)
    w += [f"Model 35 000 od zera (wagi `assets/MalyViT_35000_od_zera_s47/wagi`), {m.get('image_count')} zdjęć × {m.get('tokens_per_image')} tokenów = "
          f"{liczba(m.get('point_count'))} cząstek, {len(st)} checkpointów, kroki {st[0]}–{st[-1]} (epoka = 1094 kroki, krok 65 640 = koniec 60. epoki). "
          f"Status `{m.get('status')}`, {n} plików, {gb(b)}.", '',
          'Klatki `step_*.q16` tej osi skasowano na Edenie dla miejsca (2026-10-10). Zostały: `manifest.json`, `metrics.npy` (pewność, IoU, '
          'długość kroku dla każdego zdjęcia i checkpointu), `pca32_podzbior.npz` (32 składowe PCA dla podzbioru zdjęć we wszystkich krokach; '
          'główny zapis ruchu cząsteczek do 60. epoki), `initial_pca.npz`, `images.json`, `OCENY_POMIAR.json`, `TRAIN_VAL_pelny60.json` i '
          '`train_val_pelny60.png` (porównanie TRAIN/VALID). Klatki można odtworzyć na Edenie: `pomiar_eden.slurm 3000 1000 pelny60 65640` '
          '(kod w `07_kod/eden/2026-10-10_szklo_koloidalne/kod`).', '',
          f"Podział: `{m.get('split')}`.", '', f"Norma: `{m.get('norm')}`. Precyzja: `{m.get('precision')}`.", '',
          'Sumy kontrolne policzone na Edenie: `SHA256SUMS_EDEN_pomiar60.txt` (zgodność sprawdzona po pobraniu, log w `../pobieranie_z_edena.log`).']
else:
    w.append('Pobieranie z Edena jeszcze nie zakończone albo nieudane, zob. `../pobieranie_z_edena.log`.')
for r60 in sorted(glob.glob(os.path.join(H, '03_pomiar_naturalny_60_epok_eden', 'ruch60_*'))):
    b, n = rozm(r60)
    w += ['', f'## `{os.path.basename(r60)}`: trajektorie tokenów po wszystkich 806 checkpointach (60 epok)', '',
          'Skrypt `ruch60_rp.py` (zadanie Eden `ruch60_rp`, 2026-10-11). Te same zdjęcia co początek podzbioru PCA32 pomiaru pelny60 '
          '(n TRAIN + n VALID). Pliki: `rp128.npy` (K, N, 600, 128) f16, losowy rzut 768 → 128 identyczny jak w biegach pokręteł (`R_rp128.npy`); '
          '`pelne768.npy` (K, N768, 600, 768) f16, pełny latent dla kilku pierwszych zdjęć; `srednia768.npy` (mod zdjęcia); `normy768.npy` '
          '(|z| i |z − z0| w 768D); `oceny.npy` (pewność, IoU); `META.json` (kroki, zdjęcia, SHA wag). '
          f'{n} plików, {gb(b)}.']
    mp = os.path.join(r60, 'META.json')
    if os.path.isfile(mp):
        mt = json.load(open(mp, encoding='utf-8'))
        w += ['', 'META: ' + ', '.join(f'{k} = {str(v)[:80]}' for k, v in mt.items() if not isinstance(v, (list, dict)))]
zapisz('03_pomiar_naturalny_60_epok_eden/README.md', '\n'.join(w))

# ---------------------------------------------------------------- 04 szkło BD
w = ['# Szkło koloidalne: symulacje dynamiki browna (kontrola dodatnia)', '',
     'Model odniesienia: miękkie kule polidyspersyjne, dynamika browna (BD), stała temperatura (`stala`) albo stygnięcie (`stygnace`), '
     'kilka gęstości upakowania φ. Kod `model_szkla_eden.py`, `szklo_eden.slurm` (kopie w `07_kod/eden/2026-10-10_szklo_koloidalne/kod`). '
     'Porównanie z tokenami: `06_analizy/porownanie_szklo_20261010/README.md`, sekcje 4 i 4b.', '',
     '## osie_aplikacji: te same symulacje w formacie aplikacji', '',
     'Przyrostek `_duze`: duży bieg N = 4096, 10⁷ kroków. Bez przyrostka: bieg szybki N = 2048.', '',
     '| oś | tryb | φ | cząstek | kroków zapisu | zakres | rozmiar |', '|---|---|---|---|---|---|---|']
od = os.path.join(H, '04_szklo_koloidalne_BD', 'osie_aplikacji')
for a in (sorted(os.listdir(od)) if os.path.isdir(od) else []):
    m = man(os.path.join(od, a)) or {}
    st = m.get('steps') or []
    r = re.search(r'_(stala|stygnace)_phi_([0-9.]+)', a)
    zakres = f'{st[0]}–{st[-1]}' if st else '–'
    w.append(f"| `{a}` | {r.group(1) if r else ''} | {r.group(2) if r else ''} | {m.get('point_count')} | {len(st)} | {zakres} | "
             f"{gb(rozm(os.path.join(od, a))[0])} |")
w += ['', '## eden_symulacje: surowe wyniki z Edena', '']
ed = os.path.join(H, '04_szklo_koloidalne_BD', 'eden_symulacje')
if os.path.isdir(ed):
    w += ['| katalog | plików | rozmiar |', '|---|---|---|']
    for k in ('wyniki/stala', 'wyniki/stygnace', 'wyniki/szybko_stala', 'wyniki/szybko_stygnace', 'wyniki/proba', 'logi'):
        p = os.path.join(ed, k)
        if os.path.isdir(p):
            b, n = rozm(p)
            w.append(f'| `{k}` | {n} | {gb(b)} |')
    w += ['', 'Sumy kontrolne z Edena: `eden_symulacje/SHA256SUMS_EDEN_szklo_bd.txt`.']
zapisz('04_szklo_koloidalne_BD/README.md', '\n'.join(w))

# ---------------------------------------------------------------- 05 biegi pokręteł
w = ['# Biegi pokręteł (Eden, `~/smallvit_runs/2026-10-10_szklo_pokretla`)', '',
     'Krótkie biegi uczenia od zera z obserwatorem tokenów: „temperatura” (lr), „gęstość” (liczba klatek), ziarna, optymalizator (SGD), '
     'architektura (CNN), warstwy i zadanie (klasyfikacja Imagenette). Każdy bieg `wyniki/<bieg>/`: `dane/` (metadata.json z pełnym przepisem, '
     'LOG.json strata, OCENY.json walidacja), `pelny/stan_eNN.pt` (dwa ostatnie stany: wagi, AdamW, generatory), `wagi/k*.pt` (wybrane kroki), '
     '`gesty/` (SONDY.json, BAZA_PCA32.npz, PACZKI.npz, porcje `pca32_NNN.npz`; przy rzucie `rp128` pod kluczem pca32 leży losowy rzut 128D; '
     '`pelne768_NNN.npz` z pełnym latentem 768D jednej sondy) oraz `ANALIZA*.json` (odciski szkła, heterogeniczność, testy substancji). '
     'Surowe porcje `gesty` większości biegów skasowano na Edenie po analizie. Pełne zostały dla `lr_5e-4_rp` i `dlugi_lr_5e-4_rp`. '
     'Opis eksperymentu: `07_kod/lokalnie/maly_vit_szklo_pokretla_20261010/README.md`. Wyniki: README porównania, sekcje 5d, 6, 6b, 6c, 6d.', '',
     '| bieg | zadanie | model | opt | lr | paczka | ziarno | epoki | kroki | rzut | sondy | gęsto od | porcje gesty | stany | ANALIZ | rozmiar |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
wd = os.path.join(H, '05_biegi_pokretla_eden', 'wyniki')
for bieg in (sorted(os.listdir(wd)) if os.path.isdir(wd) else []):
    bd = os.path.join(wd, bieg)
    if not os.path.isdir(bd):
        continue
    mp = os.path.join(bd, 'dane', 'metadata.json')
    md = json.load(open(mp, encoding='utf-8')) if os.path.isfile(mp) else {}
    pk = md.get('pokretla') or {}
    if isinstance(pk, str):
        pk = ast.literal_eval(pk)
    gs = glob.glob(os.path.join(bd, 'gesty', '**', 'pca32_*.npz'), recursive=True)
    stany = sorted(os.path.basename(p) for p in glob.glob(os.path.join(bd, 'pelny', 'stan_e*.pt')))
    an = glob.glob(os.path.join(bd, 'ANALIZA*.json'))
    rz = f"{pk.get('rzut')}{pk.get('rzut_dim', '')}" if pk.get('rzut') else 'pca32'
    go = pk.get('gesto_od')
    w.append(f"| `{bieg}` | {md.get('zadanie', 'detekcja')} | {pk.get('model', 'vit')} | {pk.get('opt', 'adamw')} | {pk.get('lr', md.get('lr'))} | "
             f"{pk.get('paczka', 32)} | {pk.get('ziarno', 47)} | {md.get('epoki', '')} | {md.get('krokow', '')} | {rz} | {pk.get('sondy_n', '')} | "
             f"{'–' if go in (None, 1000000000) else go} | {len(gs) if gs else '–'} | {', '.join(stany) or '–'} | {len(an)} | {gb(rozm(bd)[0])} |")
if os.path.isfile(os.path.join(wd, 'OCENA_KLASYFIKACJI.json')):
    w += ['', '`wyniki/OCENA_KLASYFIKACJI.json`: ocena biegów Imagenette ze stanów (top-1, top-5, mAP, pewność, ECE), README porównania sekcja 6c.']
w += ['', 'Logi Slurm: `logi/`. Sumy kontrolne z Edena: `SHA256SUMS_EDEN_pokretla.txt`.']
zapisz('05_biegi_pokretla_eden/README.md', '\n'.join(w))

# ---------------------------------------------------------------- 01 bieg 35000 gęsty zapis
w = ['# Bieg 35 000 od zera z gęstym zapisem tokenów (21 epok, ziarno 47)', '',
     'Ten sam przepis co asset `../MalyViT_35000_od_zera_s47` (tam wagi 60 epok, tu nie duplikowane), ale osobny bieg z obserwatorem gęstym '
     '(`obserwator_gesty.py`). Zapis tokenów: 100 sond (zdjęć) × 600 tokenów. Latent = wyjście GELU MLP ostatniego bloku (768 neuronów na token).', '',
     '- `lokalnie/gesty/gesty/`: `pca32_NNN.npz` (rzut na PCA32 z kroku 0, co 1 krok do 2188, potem co 10 do 22 970; 98,6 % wariancji statycznej, '
     'ale tylko 4–14 % energii ruchu, zob. README porównania 5b), `pelne768_NNN.npz` (pełny latent 768D 8 pierwszych sond, co 1 krok do 1094, '
     'potem co 50), `BAZA_PCA32.npz`, `PACZKI.npz` (które próbki były w każdej paczce), `SONDY.json`.',
     '- `lokalnie/gesty/dane/`, `lokalnie/pelny_dane/`: LOG.json, OCENY.json (mAP@0,5), metadata.json biegu gęstego i biegu pełnego.',
     '- `lokalnie/wejscia/`: `MANIFEST_35000.jsonl` i adnotacje walidacji.',
     '- `eden/gesty/wagi/`: 341 checkpointów biegu gęstego `k<krok>.pt` (do kroku 22 974); `eden/gesty/pelny/`: `stan_e20.pt`, `stan_e21.pt`.',
     '- `eden/pelny/pelny/`: `stan_e59.pt`, `stan_e60.pt`, pełne stany biegu 60 epok (wagi, AdamW, generatory) do wznowienia.',
     '- `eden/proba`, `eden/gesty_proba`: biegi próbne, 512 klatek, 2 epoki.', '',
     'Na Edenie surowy gęsty zapis (31 GB) skasowano 2026-10-10 po pobraniu. Ta kopia i oryginał w `experiments/` to jedyne pełne egzemplarze.', '',
     '| podfolder | plików | rozmiar |', '|---|---|---|']
for k in ('lokalnie/gesty/gesty', 'lokalnie/gesty/dane', 'lokalnie/pelny_dane', 'lokalnie/wejscia', 'eden/gesty', 'eden/pelny', 'eden/proba', 'eden/gesty_proba'):
    p = os.path.join(H, '01_bieg_35000_gesty_zapis', k)
    if os.path.isdir(p):
        b, n = rozm(p)
        w.append(f'| `{k}` | {n} | {gb(b)} |')
zapisz('01_bieg_35000_gesty_zapis/README.md', '\n'.join(w))

# ---------------------------------------------------------------- 06, 07, 08
zapisz('06_analizy/README.md', '\n'.join([
    '# Analizy i rysunki', '',
    '- `porownanie_szklo_20261010/`: całe badanie „ruch tokenów jak w szkle”. README.md z wynikami (sekcje 1–8), skrypty analiz, JSON-y wyników, '
    'rysunki, animacje oraz `eden/wyniki/` (ANALIZA_*.json wszystkich biegów pokręteł, MODEL_SZKLA_EDEN_*.json, TRAIN_VAL_pelny60.json, OCENA_KLASYFIKACJI.json).',
    '- `animacje_ruchu_20261010/`: filmy MP4 i arkusze klatek (rozgrzewka, drgania w klatce, temperatura narzędzie/tło, szkło BD).',
    '- `app_ruch_czasteczek_20261009/`: audyt i stan pomiaru naturalnego w aplikacji z 2026-10-09.']))
zapisz('07_kod/README.md', '\n'.join([
    '# Kod, którym powstały zapisy', '',
    '- `eden/`: kod w wersji z Edena (`kod/` trzech katalogów zadań), sumy `SHA256SUMS_EDEN_kod.txt`.',
    '- `lokalnie/maly_vit_szklo_pokretla_20261010/`: eksperyment pokręteł (slurm, kod, README).',
    '- `lokalnie/maly_vit_od_zera_35000_21epok_20261009/`: bieg 35 000 (README, STATUS, kod, slurm, skrypty pobierania, notatka literaturowa).',
    '- `lokalnie/app_ruch_czasteczek_20261009/`: pomiar naturalny i audyt w aplikacji.',
    '- `lokalnie/app_format_osi/`: moduły aplikacji czytające format osi (`ruch_zbioru.py`: klatki q16/f32 i manifest; '
    '`ruch_naturalny.py`: pomiar po checkpointach).']))
zapisz('08_literatura/README.md', '\n'.join([
    '# Literatura', '', 'Kopia zbioru `SmallViTAblation/literatura_szklo_tokeny_20261011` (szkła koloidalne, cage rattling, pułapki optyczne, '
    'efekt Morana, NTK, sieci jako szkło): `linki/` (dwie listy), `PDF_all/` (pobrane PDF-y), `nieudane/` (paywalle), README zbioru.']))

# ---------------------------------------------------------------- starsze eksperymenty (tylko indeks)
STARE = ['maly_vit_ruch_czastek_gesta_historia', 'koopman_gesta_historia', 'koopman_tozsamosc_kierunki', 'maly_vit_gesta_historia_paczka',
         'maly_vit_wektor_tozsamosci', 'maly_vit_petla_sila_tozsamosci', 'maly_vit_prognoza_tozsamosci_500_bez_pca',
         'maly_vit_tozsamosc_czasteczki_pary_400', 'maly_vit_sd_prowadzenie_tozsamoscia', 'maly_vit_tozsamosc90_bez_pca_langevin',
         'maly_vit_ruch_czastek_maska_1px', 'maly_vit_langevin_wspolny_ruch', 'maly_vit_langevin_kierunki_tozsamosci', 'maly_vit_tozsamosc_tla',
         'maly_vit_tozsamosc_odswiezana_a_map', 'maly_vit_prognoza_tozsamosci_z_historii', 'maly_vit_dynamika_pelnych_czasteczek_checkpointy',
         'maly_vit_liniowy_ruch_czastek', 'vit_k1000_taniec_czasteczek_20261009', 'vit_k1000_taniec_czasteczek_gpu_20261009',
         'research_k1000_czasteczki_cel_sztuczny_20261008', 'vit_semantyka_ruchu_20261008', 'vit_tcav_tozsamosc_narzedzia',
         'domykanie_tozsamosci_large', 'tozsamosc_czastki_bledy', 'maly_vit_tozsamosc_jeden_krok', 'wybieracz_tozsamosc_koopman']
w = ['# Wcześniejsze eksperymenty z ruchem cząsteczek (NIE skopiowane, tylko indeks)', '',
     'Badania sprzed 2026-10-09 (tożsamość cząstki, Koopman, Langevin, gęsta historia K1000). Nie należą do badania „szkło” i leżą na miejscu. '
     'Wyjątek: osie K1000 używane w aplikacji są w `02_osie_ruchu_tokenow/F_*` i `G_*`.', '',
     '| eksperyment | rozmiar | plików | pierwsza linia README |', '|---|---|---|---|']
for e in STARE:
    p = os.path.join(S, 'experiments', e)
    if not os.path.isdir(p):
        continue
    b, n = rozm(p)
    r = os.path.join(p, 'README.md')
    t = open(r, encoding='utf-8', errors='replace').readline().strip().lstrip('# ') if os.path.isfile(r) else ''
    w.append(f'| `experiments/{e}` | {gb(b)} | {n} | {t[:110]} |')
zapisz('STARSZE_EKSPERYMENTY_RUCHU.md', '\n'.join(w))

# ---------------------------------------------------------------- README główne
CZ = [('01_bieg_35000_gesty_zapis', 'Gęsty zapis tokenów biegu 35 000 (co 1 krok, PCA32 i pełne 768D), checkpointy biegu gęstego, stany do wznowienia.'),
      ('02_osie_ruchu_tokenow', 'Pomiary naturalne ruchu tokenów po checkpointach w formacie aplikacji (4000 zdjęć do 21. epoki, 20 zdjęć do 60., K1000).'),
      ('03_pomiar_naturalny_60_epok_eden', 'Pomiar 3000 TRAIN + 1000 VALID do 60. epoki: PCA32 podzbioru, metryki, porównanie TRAIN/VALID.'),
      ('04_szklo_koloidalne_BD', 'Kontrola dodatnia: symulacje szkła koloidalnego (dynamika browna), osie aplikacji i surowe wyniki.'),
      ('05_biegi_pokretla_eden', 'Biegi pokręteł: lr, gęstość, ziarna, SGD, CNN, warstwy, Imagenette, okno po 60. epoce; stany, zapisy tokenów, analizy, test jądra.'),
      ('06_analizy', 'Wszystkie analizy, JSON-y wyników, rysunki i animacje badania.'),
      ('07_kod', 'Kod z Edena i lokalny, którym powstały zapisy; moduły czytające format osi.'),
      ('08_literatura', 'Zbiór literatury (102 linki, 66 PDF).')]
w = ['# Ruch cząsteczek (tokenów) małego ViT a szkło koloidalne: wszystkie zapisy, 2026-10-09…11', '',
     'Jedno miejsce na zapisy ruchu tokenów i przebiegi uczenia z badania „ruch cząsteczek jak w szkle” '
     '(`06_analizy/porownanie_szklo_20261010/README.md`). Cząstka = token obrazu (600 na zdjęcie 480 × 320), położenie = 768 aktywacji GELU '
     'warstwy ukrytej MLP ostatniego bloku, czas = krok uczenia. Wagi modelu 35 000 (60 epok, 806 checkpointów) są w `../MalyViT_35000_od_zera_s47`.', '',
     '## Zawartość', '', '| folder | co zawiera | plików | rozmiar |', '|---|---|---|---|']
for k, o in CZ:
    p = os.path.join(H, k)
    b, n = rozm(p) if os.path.isdir(p) else (0, 0)
    w.append(f'| `{k}/` | {o} | {liczba(n)} | {gb(b)} |')
w += ['', 'Każdy folder ma własne README z tabelą zawartości wygenerowaną z manifestów i metadanych (`opisz.py`). '
      'Wcześniejsze eksperymenty z ruchem cząsteczek, których tu nie kopiowano: `STARSZE_EKSPERYMENTY_RUCHU.md`.', '',
      '## Jak powstała ta kopia', '',
      '- `zloz_lokalnie.py`: części z `experiments/` i `outputs/`. Pliki tekstowe skopiowane, pliki binarne jako twarde dowiązania NTFS '
      '(ta sama treść, zero dodatkowego miejsca; usunięcie oryginału nie usuwa danych stąd). Rejestr każdego pliku: `KOPIA_LOKALNA.tsv`.',
      '- `pobierz_z_edena.sh`: części, które były tylko na Edenie, strumieniem tar. Każda część sprawdzona sumami SHA-256 policzonymi na Edenie '
      '(`SHA256SUMS_EDEN_*.txt`, wynik w `pobieranie_z_edena.log`). Na Edenie niczego nie zmieniano.',
      '- Duże pliki binarne są wyłączone z gita (`.gitignore` tego folderu i główny).', '',
      '## Uwaga o dowiązaniach', '',
      'Plik binarny edytowany w miejscu zmieniłby się w obu lokalizacjach. Zapisy ruchu nie są edytowane (aplikacja i skrypty piszą nowe pliki), '
      'więc w praktyce to kopia. Niezależną kopię na innym dysku daje zwykłe skopiowanie tego folderu.']
zapisz('README.md', '\n'.join(w))
print('README wygenerowane')
