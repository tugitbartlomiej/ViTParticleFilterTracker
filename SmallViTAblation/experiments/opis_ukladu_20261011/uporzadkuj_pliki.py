"""Kopiuje istniejące raporty/metadane i indeksuje pliki. Nie uruchamia modeli."""
from pathlib import Path
import hashlib,json,shutil,zipfile,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'OpisUkładu'
OUT.mkdir(exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def link(p,label):return f'[{label}]({p.resolve().as_posix()})'

# Korekta syntezy: pakiet opisuje tylko wcześniej istniejące wyniki.
p=OUT/'01_OPIS_UKLADU.md';s=p.read_text(encoding='utf-8')
s=s.replace('Nowy eksport 60 epok jest odczytem istniejących checkpointów.',
            'Pakiet końcowy zawiera istniejące pliki i odnośniki; rozpoczęte odtwarzanie aktywacji zatrzymano na polecenie użytkownika.')
s=s.replace('806 checkpointów; odczyty kontrolne 768D i nowy pełny eksport 8 sond',
            '806 checkpointów; zachowane pełne odczyty 768D ośmiu sond dla 75 wybranych stanów')
s=s.replace('Osiem sond pełnego eksportu','Osiem sond pełnego pomiaru kontrolnego')
a=s.index('Tablica pełnych aktywacji zachowuje');b=s.index('### 8.2.',a)
s=s[:a]+'''Tablica pełnych aktywacji zachowuje każdą z 768 współrzędnych każdego tokenu i każdego wybranego obrazu. Dla biegu 60 epok istnieją 75 plików kontrolnych, każdy z naturalną tablicą `fp32` o kształcie `(8,600,768)`. Osiem obrazów oznacza 4800 cząsteczek. Kolejne pliki wyznaczają stany tych samych tokenów; trzeba sortować je po liczbowym numerze kroku. W tych samych plikach są osobno odczyty bf16 i kontrola zamrożonego BN.

Kompletna robocza tablica 806 stanów ośmiu sond została usunięta przez historyczny skrypt po zbudowaniu macierzy Grama. Nie ma jej w końcowym pakiecie. Pliki kontrolne pozwalają liczyć na pełnych współrzędnych wybranych stanów, w tym końców wszystkich 60 epok. Brakujących kroków nie należy przedstawiać jako pomierzonych ani wypełniać interpolacją bez zmiany definicji analizy.

''' + s[b:]
p.write_text(s,encoding='utf-8')

reports=[];families=[];copies=[];targets=[]
for area in ['experiments','outputs']:
    for family in sorted((ROOT/area).iterdir()):
        if not family.is_dir() or family.name=='opis_ukladu_20261011' or family.name=='surowe_trajektorie_60epok_768d_20261011':continue
        chosen=[]
        # Limit głębokości: opisy rodzin i raporty w typowych katalogach wynikowych.
        frontier=[] if '--reuse' in sys.argv else [family]
        if '--reuse' in sys.argv:
            chosen=[ROOT/p.relative_to(OUT/'zrodla') for p in (OUT/'zrodla'/area/family.name).rglob('*.md')]
        for depth in range(4):
            nxt=[]
            for d in frontier:
                for f in d.iterdir():
                    if f.is_file() and f.suffix.lower()=='.md' and any(v in f.name.upper() for v in ['README','RAPORT','WYNIK','STATUS','PODSUMOWANIE','OCENA','TEORIA']):
                        chosen.append(f)
                    elif f.is_dir() and depth<3 and (f.name in ['wyniki','eden','pelny','gesty','dane','provenance','kod','analiza_co1','analiza_co10','analiza_60_120'] or f.name.startswith('run')):
                        nxt.append(f)
            frontier=nxt
        chosen=sorted(set(chosen))
        families.append(dict(area=area,family=family.name,path=str(family),reports=[str(p) for p in chosen]))
        targets.append(dict(path=str(family),shortcut=f'rodziny/{area}/{family.name}.lnk'))
        for src in chosen:
            dst=OUT/'zrodla'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            record=dict(source=str(src),copy=str(dst),sha256=sha(src),bytes=src.stat().st_size)
            copies.append(record);reports.append((src,dst))

catalog=['# Katalog zachowanych eksperymentów i raportów','',
         'Przegląd nazw rodzin i istniejących raportów. Synteza kluczowych pomiarów jest w 01_OPIS_UKLADU.md. '
         'Scalony plik zachowuje również raporty historyczne, których werdykty mogą być skorygowane w późniejszych audytach. '
         'Wyszukiwanie raportów obejmuje rodzinę i trzy poziomy podkatalogów; brak raportu w tym indeksie nie oznacza braku danych. '
         'Nie przeprowadzono nowych eksperymentów ani ponownego obliczania metryk.','']
for f in families:
    catalog.extend([f'## {f["area"]}/{f["family"]}','',link(Path(f['path']),'Oryginalny folder')+'; skrót w `linki/rodziny/`.', ''])
    if not f['reports']:catalog.extend(['Brak rozpoznanego raportu Markdown w sprawdzonym zakresie.',''])
    for path in f['reports']:
        src=Path(path);dst=OUT/'zrodla'/src.relative_to(ROOT)
        catalog.append('- '+link(dst,str(src.relative_to(ROOT)))+' — '+link(src,'oryginał'))
    catalog.append('')
(OUT/'03_KATALOG_EKSPERYMENTOW.md').write_text('\n'.join(catalog),encoding='utf-8')
merged=['# Zachowane raporty w jednym pliku','',
        'Każda część to kopia treści istniejącego raportu. Nie usuwa się sprzecznych historycznych interpretacji: '
        'ich korekty opisuje 01_OPIS_UKLADU.md. Linki względne wewnątrz treści odnoszą się do oryginalnego katalogu raportu.','']
for src,dst in reports:
    merged.extend(['---','',f'## Źródło: {src.relative_to(ROOT).as_posix()}','',link(src,'Otwórz oryginał'),''])
    merged.append(src.read_text(encoding='utf-8-sig',errors='replace'))
(OUT/'04_RAPORTY_SCALONE.md').write_text('\n'.join(merged),encoding='utf-8')

important={
'DANE_60_EPOK_768D_KONTROLE':'outputs/audyt_szkla_768d_20261010/precision_controls',
'WAGI_60_EPOK_806_CHECKPOINTOW':'assets/MalyViT_35000_od_zera_s47/wagi',
'LOGI_TRENINGU_60_EPOK':'assets/MalyViT_35000_od_zera_s47/dane',
'MODEL_60_EPOK_CALY_FOLDER':'assets/MalyViT_35000_od_zera_s47',
'KOD_TRENINGU_35K':'experiments/maly_vit_od_zera_35000_21epok_20261009/kod',
'DANE_GESTE_21_EPOK_768D_I_PACZKI':'experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty',
'K1000_LATENT_GESTY_768D':'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_GESTY.npy',
'K1000_DANE_I_OBRAZY':'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane',
'POMIAR_60_EPOK_4000_ZDJEC_METADANE':'outputs/porownanie_szklo_20261010/eden/wyniki/pomiar60',
'ZBIORCZE_ARCHIWUM_RUCHU':'assets/RuchCzasteczek_ViT_szklo_2026-10',
'AUDYT_768D_WYNIKI_POCHODNE':'outputs/audyt_szkla_768d_20261010',
'APP_KOD':'app',
}
for name,rel in important.items():targets.append(dict(path=str(ROOT/rel),shortcut=name+'.lnk'))
dense=ROOT/important['DANE_GESTE_21_EPOK_768D_I_PACZKI']
for f in sorted(dense.glob('pelne768_*.npz')):targets.append(dict(path=str(f),shortcut='pelne768_21epok/'+f.name+'.lnk'))
controls=ROOT/important['DANE_60_EPOK_768D_KONTROLE']
for f in sorted(controls.glob('k*.npz')):targets.append(dict(path=str(f),shortcut='pelne768_60epok/'+f.name+'.lnk'))

metadata=[dense/'SONDY.json',dense/'BAZA_PCA32.npz',dense/'PACZKI.npz',dense/'.zdalne.sha256',
 ROOT/'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl',
 ROOT/'assets/MalyViT_35000_od_zera_s47/README.md',
 ROOT/'assets/MalyViT_35000_od_zera_s47/SHA256SUMS_EDEN.txt',
 ROOT/'assets/MalyViT_35000_od_zera_s47/SHA256SUMS_LOKALNE.txt']
for dirname in [ROOT/'assets/MalyViT_35000_od_zera_s47/dane',ROOT/important['POMIAR_60_EPOK_4000_ZDJEC_METADANE'],ROOT/important['AUDYT_768D_WYNIKI_POCHODNE']]:
    if not dirname.exists():continue
    metadata.extend(f for f in dirname.iterdir() if f.is_file() and f.suffix.lower() in ['.json','.jsonl','.md'] and f.stat().st_size<8_000_000)
for src in sorted(set(metadata)):
    if not src.exists():continue
    dst=OUT/'metadane'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    copies.append(dict(source=str(src),copy=str(dst),sha256=sha(src),bytes=src.stat().st_size))

def header(f):
    v=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,v)
    return dict(shape=list(shape),dtype=str(dtype),fortran_order=fort)
inventory=[]
paths=[ROOT/important['K1000_LATENT_GESTY_768D']]
paths+=list(dense.glob('*.npz'))+list(controls.glob('*.npz'))
paths+=list((ROOT/important['WAGI_60_EPOK_806_CHECKPOINTOW']).glob('k*.pt'))
for p in sorted(paths):
    rec=dict(path=str(p),bytes=p.stat().st_size)
    if p.suffix=='.npy':
        with p.open('rb') as f:rec['array']=header(f)
    elif p.suffix=='.npz':
        with zipfile.ZipFile(p) as z:
            rec['arrays']={}
            for n in z.namelist():
                if n.endswith('.npy'):
                    with z.open(n) as f:rec['arrays'][n[:-4]]=header(f)
    inventory.append(rec)
write('INWENTARZ_DANYCH.json',dict(date='2026-10-11',method='Ścieżki, rozmiary i nagłówki; bez forwardu i obliczania metryk',files=inventory))
write('MANIFEST_KOPII.json',dict(date='2026-10-11',files=copies))
write('CELE_SKROTOW.json',targets)
write('RODZINY.json',families)
print(json.dumps(dict(families=len(families),reports=len(reports),copies=len(copies),shortcuts=len(targets),indexed_files=len(inventory)),ensure_ascii=False))
