import json,csv,collections,re,math
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent
d=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'));files=d['files']
copies=json.loads((OUT/'verified_copies.json').read_text(encoding='utf-8'))
existing={f['path'] for f in files}
for h in copies:
 if h.get('size') and h['path'] not in existing:
  files.append(dict(path=h['path'],size=h['size'],blocks=h['blocks'],dev=h['dev'],ino=h['ino'],nlink=h['nlink'],realpath=h['realpath'],mtime=None))
verified={h['path']:h for h in copies if h.get('verified_local')}
checked={h['path']:h for h in copies}
deploy=json.loads((OUT/'deploy_verified.json').read_text(encoding='utf-8'))
deploy_ok={x['archive'] for x in deploy if x['stable'] and x['all_regular_members_verified']}
base='/home2/faculty/bpiotrowski/smallvit_runs/'
jobs=json.loads((OUT/'system_jobs.json').read_text(encoding='utf-8'))['stdout']
jobrows=[]
for line in jobs.splitlines():
 parts=line.split('|')
 if len(parts)>=6:jobrows.append(dict(zip(['id','name','state','start','end','workdir'],parts[:6])))
(OUT/'jobs.json').write_text(json.dumps(jobrows,ensure_ascii=False,indent=2),encoding='utf-8')
def jobstatus(path):
 if '/.cache/' in path:return 'brak aktywnych jobów; cache użytkownika'
 run=path.split('/smallvit_runs/')[-1].split('/')[0]
 js=[j for j in jobrows if '/'+run in j['workdir']]
 if not js:
  known={'2026-10-06_vit_sd_pelny_latent_filmy':'1823172 COMPLETED, poprzednie FAILED; STATUS complete 20000', '2026-10-06_vit_konwerter_dino_filmy':'1823170/1823174 COMPLETED', '2026-10-06_vit_sd_od_zera_k8000_filmy':'1823302 FAILED, 1823430 COMPLETED; FINAL plateau, stan recovery chroniony','2026-10-07_vit_sd_k8000_ulepszenia':'1823544 COMPLETED'}
  return known.get(run,'brak aktywnych; dokładne powiązanie joba nieustalone')
 last=max(js,key=lambda j:j['end'])
 states=','.join(sorted(set(j['state'] for j in js)))
 return last['id']+' '+last['state']+'; historia: '+states
def category(f):
 p=f['path'];n=Path(p).name
 if '/.cache/pip/' in p:return 'cache-pip','TAK_KANDYDAT','niski','Cache pobrań/wheels instalatora, odtwarzalny; nie jest środowiskiem ani modelem.', 'wysoka'
 if '/.cache/' in p:return 'cache-modeli','NIE','średni','Cache Torch/HF obejmuje modele i kod; zachować do analizy zależności, offline może uniemożliwić odtworzenie.', 'wysoka'
 if '/__pycache__/' in p and n.endswith('.pyc'):return 'bytecode','TAK_KANDYDAT','niski','Odtwarzalny bytecode; zachować źródła, nie usuwać deps ani runtime.', 'wysoka'
 if '2026-10-07_maly_vit_sily_czastek' in p and n.endswith(('.npy','.npz')):return 'sily-latenty','NIE','wysoki','Źródłowe Z i siły są potrzebne analizie DMDc; lokalny raport nie zastępuje pełnych tablic.', 'wysoka'
 if '2026-10-03_maly_vit_gesta_paczka' in p or '2026-10-01_maly_vit_ruch_czastek_gesta_historia' in p:
  return 'historia-adamw','NIE','wysoki','Gęsta historia i pełne stany AdamW potrzebne badaniu dynamiki; nie kwalifikować po wieku ani zakończeniu joba.', 'wysoka'
 if n.endswith(('.tar','.tar.gz','.zip')):
  if p in deploy_ok:return 'deploy-zweryfikowany','TAK_KANDYDAT','niski','Każdy regularny człon archiwum ma identyczny SHA256 w pozostawionym rozpakowanym drzewie; zweryfikowano stabilność i właściciela. Usunięcie samego archiwum pozostawia kod i wejścia.', 'wysoka'
  if n=='deploy_bundle.tar.gz' or n=='code_bundle_vit8z256.tar.gz':return 'deploy','WARUNKOWO','średni','Pakiet wdrożeniowy: przed kwalifikacją sprawdzić każdy człon względem rozpakowanych wejść i kodu; SHA całego archiwum niezweryfikowane.', 'średnia'
  return 'archiwum-zrodlowe','NIE','wysoki','Może zawierać unikalne obrazy, filmy, etykiety lub provenance; input/context tar nie jest automatycznie zbędnym deploy.', 'wysoka'
 if '/wyniki/smoke/' in p or '/wyniki/smoke_v2/' in p or '/GPU_SMOKE_' in p:
  if n.endswith(('.pt','.pth')):return 'smoke-checkpoint','WARUNKOWO','średni','Zakończony test techniczny; kandydat po potwierdzeniu zbędności tego stanu, kopii i zachowaniu kodu, metryk oraz logów. Ma momenty AdamW lub stan modelu.', 'średnia'
 if '2026-10-04_vit_optymalizacja_latentu128/results/r' in p and n=='TARGETS_ORACLE.npz' and p in verified:
  return 'oracle-kopia-archiwalna','WARUNKOWO','średni','Pełna lokalna kopia NPZ potwierdzona SHA256; historyczne zadania COMPLETED. Można rozważyć pozostawienie tylko kopii lokalnej po wykluczeniu zależności zdalnych, z zachowaniem metryk/provenance i bez zmiany jedynej kopii.', 'wysoka co do kopii, średnia co do retencji'
 if n.endswith(('.pt','.pth','.safetensors','.bin')):return 'modele-recovery','NIE','wysoki','Chronione modele/best, SD assets i recovery. Kopia samych wag nie zastępuje pełnego optymalizatora; identyczny SHA nie stanowi zgody na usunięcie.', 'wysoka'
 if n.endswith(('.npy','.npz')):return 'latenty-wyniki','NIE','wysoki','Artefakt pomiarowy/oracle/historyczny. Zachować; archiwizacja wymaga osobnej oceny wykorzystania nawet przy zgodnym SHA.', 'wysoka'
 return 'kod-logi-provenance','NIE','średni','Zachować kod, konfiguracje, wyniki, logi i provenance; niska korzyść lub nieustalona rola.', 'wysoka'
inode=collections.defaultdict(list)
for f in files:inode[(f['dev'],f['ino'])].append(f)
hardlinks=[dict(dev=k[0],ino=k[1],nlink=v[0]['nlink'],observed=len(v),blocks=v[0]['blocks'],paths=[f['path'] for f in v]) for k,v in inode.items() if v[0]['nlink']>1]
(OUT/'hardlinks.json').write_text(json.dumps(hardlinks,ensure_ascii=False,indent=2),encoding='utf-8')
rows=[];done=set();safe_paths=set();conditional_paths=set()
for f in sorted(files,key=lambda x:x['path']):
 cat,eligible,risk,reason,confidence=category(f)
 key=(f['dev'],f['ino']); h=verified.get(f['path'])
 local='brak potwierdzonej kopii w sprawdzonym zakresie'
 if h:local='SHA256 zgodny: '+ '; '.join(h['verified_local'])
 elif f['path'] in checked:local='sprawdzone kandydatury lokalne mają inne SHA256; NIE są zgodną kopią (verified_copies.json)'
 if cat in ['cache-pip','bytecode']:local='nie wymaga kopii; odtwarzalny cache'
 if cat=='deploy-zweryfikowany':local='lokalna niepotwierdzona; pełna rozpakowana kopia na Edenie, każdy człon SHA zgodny (deploy_verified.json)'
 if eligible=='TAK_KANDYDAT':safe_paths.add(f['path'])
 if eligible=='WARUNKOWO':conditional_paths.add(f['path'])
 potential=0
 if eligible!='NIE' and key not in done:
  linked=inode[key]
  if len(linked)==f['nlink'] and all(category(x)[1]!='NIE' for x in linked):potential=f['blocks'];done.add(key)
  elif f['nlink']>1:reason+=' Hardlink: pozostają chronione lub nieodnalezione nazwy, odzysk 0.'
 rows.append(dict(path=f['path'],sizeGB=round(f['size']/1e9,9),statusjob=jobstatus(f['path']),localkopia=local,reason=reason,risk=risk,eligible=eligible,bytes_reclaimable=potential,confidence=confidence,allocated_bytes=f['blocks'],realpath=f['realpath'],device=f['dev'],inode=f['ino'],nlink=f['nlink'],category=cat,sha256=h.get('sha256','') if h else ''))
def reclaim(paths):
 return sum(v[0]['blocks'] for v in inode.values() if len(v)==v[0]['nlink'] and all(f['path'] in paths for f in v))
safe=reclaim(safe_paths);conditional=reclaim(safe_paths|conditional_paths)-safe
fieldnames=list(rows[0])
with (OUT/'TABELA_PLIKOW.csv').open('w',encoding='utf-8-sig',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
(OUT/'classification.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
sizes=(OUT/'sizes_final.txt').read_text(encoding='utf-8')
du={}
for line in sizes.splitlines():
 m=re.match(r'^(\d+)\s+(/home2/faculty/bpiotrowski/(?:smallvit_runs|\.cache).*)$',line)
 if m:du[m[2]]=int(m[1])
final=(OUT/'final_state.txt').read_text(encoding='utf-8-sig')
free=int(next(line.split()[3] for line in final.splitlines() if line.startswith('10.0.11.12@tcp:')))
summary=dict(free_bytes=free,safe_reclaimable_bytes=safe,conditional_extra_bytes=conditional,smallvit_runs_du_bytes=du.get(base.rstrip('/')),full_fp32_snapshot_bytes=1500*600*768*4,verified_copies=len(verified),file_count=len(files),hardlink_groups=len(hardlinks))
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
def table(rs,fields):
 lines=['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
 for r in rs:lines.append('| '+' | '.join(str(r.get(k,'')).replace('|',' / ').replace('\n',' ') for k in fields)+' |')
 return '\n'.join(lines)
groups=[]
for cat in sorted(set(r['category'] for r in rows)):
 rs=[r for r in rows if r['category']==cat]
 blocks=sum({(r['device'],r['inode']):r['allocated_bytes'] for r in rs}.values())
 groups.append(dict(category=cat,allocatedGB=round(blocks/1e9,3),bytes_reclaimable=sum(r['bytes_reclaimable'] for r in rs),eligible=','.join(sorted(set(r['eligible'] for r in rs)))))
big=[r for r in rows if r['sizeGB']>=0.4 or r['category'].startswith('deploy')]
force_bytes=sum(f['size'] for f in files if '/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane/' in f['path'] and f['path'].endswith('.npy'))
report=f'''# Audyt miejsca Eden - 8 października 2026, tylko odczyt

Na wybranym systemie plików pozostało {free/1e9:.3f} GB. Bez naruszania modeli, źródeł, pełnych stanów AdamW i istniejących pomiarów można wskazać {safe/1e9:.3f} GB odtwarzalnego cache/bytecode i zweryfikowanych pakietów deploy jako kandydatów o niskim ryzyku. Dalsze {conditional/1e9:.3f} GB to warunkowe smoke i trzy zweryfikowane lokalne kopie historycznego oracle, wymagające decyzji retencji i wykluczenia zależności. Nie ma podstaw do obiecania dziesiątek GB bez archiwizacji ważnych wyników lub zmiany retencji.

Nic nie usunięto, nie zatrzymano ani nie zmieniono na Edenie. Nie zgłoszono jobów. Zapis lokalny ograniczono do tego nowego katalogu raportu. SSH używał BatchMode i StrictHostKeyChecking=yes. Python zdalny działał z wejścia standardowego, bez plików roboczych. Odczyt może aktualizować atime zgodnie z ustawieniami systemu plików; audyt nie wykonywał jawnych operacji zapisu.

## Miejsce, quota i zakres

- `smallvit_runs` ma {du.get(base.rstrip('/'),0)/1e9:.3f} GB przydzielonych bloków wg jednego `du -x` bez podążania za symlinkami.
- Realpath: `/home2/faculty/bpiotrowski/smallvit_runs` = `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs`. Nie są to dwie kopie.
- `df -B1`: pojemność 214748364800 B, wolne {free} B. To wynik systemu plików dla tej ścieżki, nie gwarancja rezerwacji dla eksperymentu.
- `quota` nie jest zainstalowane. `lfs quota -u bpiotrowski /mnt/evafs`: użycie 189178136 KiB, limity bloków i plików 0 (brak ustawionego limitu w tym odczycie), 116077 plików. Quota nie pokazuje dodatkowej dostępnej przestrzeni. Ogranicza obecny `df`, z uwzględnieniem równoległych zmian.
- Własny `.cache` zajmuje 1.034 GB, z czego pip 0.504 GB, HF 0.436 GB, Torch 0.094 GB. Modele HF/Torch zachowano.
- Metadane odczytywano wyłącznie w swoich `smallvit_runs` i wskazanych cache. Katalogi obrazów, wejść, filmów, etykiet i datasets były pomijane podczas inwentaryzacji plikowej. Agregat `du` w swoim katalogu nie analizował zawartości źródeł. Pełna lista pominięć jest w `inventory.json`. Nie badano cudzych katalogów, innych mountów, scratch węzłów ani całych datasets.
- Odczyt własnego home ograniczono do jego pierwszego poziomu. Nie znaleziono własnego `~/tmp`; środowiska `venvs`, `.conda`, `smallvit_data` i `grant_tft` nie były skanowane ani kwalifikowane. Audyt nie obejmuje całego użycia wykazanego przez quota.
- Kopie lokalne sprawdzano w artefaktach `assets`, `outputs`, `experiments`, `results` i archiwum `Eden/EdenBackup/=/smallvit_runs`, pomijając źródła danych oraz symlinki. Brak trafienia oznacza brak potwierdzenia w tym zakresie, nie dowód nieistnienia kopii gdzie indziej.

## Zadania i recovery

`squeue -u bpiotrowski` na początku i końcu audytu było puste. Końcowy ponowny odczyt kolejki, df i quota: `final_state.txt`, {final.splitlines()[0]}. `sacct` obejmuje 2026-09-14 do chwili odczytu; surowe dowody w `system_jobs.json`, rozpisanie w `jobs.json`. Nie utożsamiam COMPLETED z naukowo udanym ani zbędnym biegiem.

- K1000/1500: brak nowego biegu w odczytanej kolejce i brak odpowiadającego katalogu w `smallvit_runs` w chwili inwentaryzacji. To nie wyklucza zgłoszenia przez drugiego agenta po odczycie.
- `2026-10-08_vit_k8000_podmiana_1z256`: joby 1824697 i agregator 1824702 COMPLETED; FINAL wskazuje complete/500. STATUS zachował training/500, więc sam STATUS nie jest dowodem aktywnego joba.
- `2026-10-08_vit_k8000_podmiana_8z256`: 1824755 COMPLETED, zachować wyniki i checkpointy.
- `2026-10-06_vit_sd_pelny_latent_filmy`: 1823172 COMPLETED, STATUS complete 20000/20000 po wcześniejszych FAILED. Zachować best i ostatni pełny stan.
- `2026-10-06_vit_sd_od_zera_k8000_filmy`: 1823302 FAILED, późniejszy 1823430 COMPLETED; FINAL opisuje plateau i quality=false przy 19000. `latest_checkpoint.pt` zawiera optimizer, exp_avg, exp_avg_sq i RNG; nie usuwać jako rzekomo starego nieudanego zadania.
- `2026-10-02_nndmd_1500_expandedstate_multigpu`: 1819574 CANCELLED. Zachować latest, final_refit, inner_train, snapshot, cache potrzebny wznowieniu, kod i logi. Brak aktywnego joba nie znosi potrzeby recovery.
- `2026-10-07_maly_vit_sily_czastek`: 1823539 COMPLETED, dopasowanie 1823541 COMPLETED. Pełne Z i siły są źródłem bieżącego DMDc, nie zbędnym temp.

## Ochrona i sprawdzenie kopii

Zachować filmy, ZIP-y filmów, źródłowe obrazy i etykiety, `tracking_static36.tar`, `context_train.tar`, wszystkie K1000/K8000, SD best/base/EMA i provenance. Wszystkie pełne stany AdamW produkcji i historii są poza rekomendacją usuwania.

Bezpiecznie sprawdzono nagłówki ZIP i instrukcje pickle w wybranych checkpointach, bez deserializacji lub wykonania kodu: `details.json`. `ostatni.pt` z 30 września (13.754 GB), pełne `pelny/k1000.pt` i `latest_checkpoint.pt` mają momenty optymalizatora. Smoke również mogą mieć optimizer; oddzielenie ich od produkcji wymaga sprawdzenia celu testu.

Aktualne SHA-256 obu stron sprawdzono dla {len(copies)} wybranych zdalnych plików; zgodne kopie znaleziono dla {len(verified)}. Cztery kandydatury zbieżne rozmiarem mają inne SHA (B_mixed, KONWERTER smoke_v2, checkpoint 19750 i 20000), więc nie są duplikatami sprawdzonych lokalnych modeli. Stabilność rozmiaru/mtime/inode kontrolowano przed i po zdalnym hashowaniu. SHA lokalny pochodzi z odczytu w tym audycie; przed późniejszym usunięciem ponownie sprawdzić istnienie i niezmienność kopii lokalnej. Szczegóły, lokalne ścieżki i SHA są w `verified_copies.json`. Dopasowanie nazwy lub rozmiaru bez SHA nie jest uznawane za kopię. Nie uruchamiano ponownie inferencji ani treningu.

Chronione główne modele mają aktualnie potwierdzone SHA: K1000 `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`, K8000 s47 `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`, SD A_control `1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41`. Lokalne PROVENANCE.json wskazują zgodne ścieżki źródeł; zachowano modele mimo potwierdzonych kopii.

## Kandydaci i rekomendacje

1. **Wysoka pewność, niskie ryzyko:** własny cache pip, odtwarzalny `__pycache__/*.pyc` z zachowaniem kodu i dwa zweryfikowane pakiety deploy. Potencjał razem {safe} B ({safe/1e9:.3f} GB). Dla `deploy_bundle.tar.gz` sprawdzono 791 regularnych członów, dla `code_bundle_vit8z256.tar.gz` 7; wszystkie SHA zgodne z pozostawionym rozpakowanym drzewem na Edenie, nie z kopią lokalną. W przyszłej operacji najpierw powtórzyć kontrolę kolejki i ścieżek; usunięcie cache oznacza ewentualne ponowne pobranie/kompilację.
2. **Średnia pewność, warunkowo:** dokładne checkpointy `wyniki/smoke`, `wyniki/smoke_v2` oraz `GPU_SMOKE_*` wskazane w CSV, a także trzy `2026-10-04_vit_optymalizacja_latentu128/results/r*/TARGETS_ORACLE.npz` (łącznie około 1.306 GB, identyczne kopie lokalne SHA). Zachować logi, konfiguracje, raporty i dowód testu. Przed kwalifikacją wykluczyć zdalnych konsumentów i potrzebę danego stanu AdamW oraz potwierdzić kopię lub odtwarzalność. Potencjał dodatkowy {conditional} B ({conditional/1e9:.3f} GB), nie jest częścią potwierdzonego niskiego ryzyka.
3. **Wysoka pewność, zachować:** {force_bytes/1e9:.3f} GB pełnych tablic `Z`, `S_nowa`, `S_ped`, `S_zan`, `S_buf`, `S_grupy` z pomiaru sił. Starszy lokalny WYNIKI.md podaje tylko metadane i wyniki dopasowania, a aktualny audyt kopii pokazuje także nowsze trafienia (szczegóły i SHA w CSV/verified_copies.json). Bieżący `experiments/vit_dmdc_paczki_20261008/PLAN.md` odwołuje się do tych sił i ich zgodnego Z. Nawet zgodne kopie potrzebnych tablic nie są automatycznie eligible.
4. **Wysoka pewność, zachować:** 13.754 GB `2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ostatni.pt` i SD/EMA. To pełny stan, nie duplikat checkpointa EMA. Zysk duży, ale ryzyko utraty wznowienia i zasobów SD niedopuszczalne w tym audycie.
5. **Dalsze istotne miejsce wymaga archiwizacji:** kompletna kopia poza Edenem wraz z SHA, kodem, manifestem i optimizerem, a następnie osobna decyzja retencji. Ten audyt nie wykonał kopiowania dużych plików ani usuwania. Nie rekomenduję użycia chronionych źródeł jako budżetu nowego biegu.

## Budżet pełnych trajektorii

`1500 * 600 * 768 * 4 = 2764800000 B = 2.7648 GB = 2.574920654 GiB` na jeden snapshot FP32 całej bazy. Dla 1000 obrazów: 1.8432 GB. Liczba 1000 w K1000 oznacza checkpoint treningu; dokładna baza eksperymentu ma 1500 obrazów.

| Snapshoty 1500 FP32 | Same latenty GB | Ocena |
| --- | --- | --- |
| 1 | 2.7648 | mieści się obecnie |
| 7 | 19.3536 | teoretycznie mieści się, mały zapas |
| 10 | 27.648 | za mało obecnego miejsca |
| 21 | 58.0608 | wymaga znacznego dodatkowego miejsca |
| 101 | 279.2448 | przekracza całą pojemność tego systemu plików |
| 1001 | 2767.5648 | około 2.77 TB, niemożliwe tutaj |

Obecne wolne miejsce wystarcza na {free//2764800000} pełnych snapshotów bez pozostałych artefaktów. Po samych kandydatach niskiego ryzyka: {(free+safe)//2764800000}; po wszystkich warunkowych: {(free+safe+conditional)//2764800000}. Zapas operacyjny 5 GB oraz dodatkowe `before/after/delta`, checkpointy, optymalizatory i atomowy zapis obniżą te liczby. Potwierdzone mało ryzykowne porządki nie rozwiązują problemu długiej trajektorii.

Lokalny PLAN oracle zapisuje `before/after/delta` FP32 i historię FP16 co 100 kroków oraz końcowy krok. To odmienny budżet niż pełna trajektoria FP32. Trzy tablice FP32 dla całej bazy to dodatkowe 8.2944 GB; trzy zestawy stanu latentu/AdamW, jeżeli równocześnie zapisane, również mnożą budżet. Nie zakładać kompresji bez pomiaru. Dobór częstotliwości/formatu lub innego miejsca jest osobną decyzją, nie został zmieniony przez audyt.

## Jak liczone są bajty

Rozmiar `sizeGB` to długość logiczna dzielona przez 10^9. `allocated_bytes` i `bytes_reclaimable` bazują na `st_blocks * 512`, a nie na sumie długości plików. Inode liczone po parze device/inode, jednokrotnie. Gdy część hardlinków jest chroniona lub poza zakresem, odzysk jest 0. Dla wszystkich nazw kwalifikowanego inode blok przypisano tylko raz w tabeli. Identyczne bajty na różnych inode zajmują oddzielne bloki, ale nie są automatycznie eligible.

Wykryto {len(hardlinks)} grup inode z hardlinkami ({sum(1 for f in files if f['nlink']>1)} nazw). `hardlinks.json` pokazuje wszystkie zaobserwowane nazwy i nlink. Linki katalogów `results_view3`, `results_k4_view` i snapshoty HF nie zostały dereferencjonowane; usunięcie symlinka nie odzyskuje bloków celu. `du` przypisuje współdzielone inode pierwszemu odwiedzonemu katalogowi, więc rozmiary pojedynczych rodzin zależą od kolejności i nie są niezależnymi prognozami odzysku.

Nie badano otwartych deskryptorów na węzłach ani serwerowych kopii/snapshotów; nawet policzone bloki mogą zostać zwolnione z opóźnieniem. To analiza kandydatów, nie autoryzacja kasowania. `eligible=TAK_KANDYDAT` oznacza rekomendację do osobnej operacji, `WARUNKOWO` oznacza brak wystarczających dowodów teraz, `NIE` ochronę.

## Zestawienie klas (nie sumować ponownie z tabelą plików)

{table(groups,['category','allocatedGB','eligible','bytes_reclaimable'])}

## Większe pliki i pakiety

Pełna wymagana tabela wszystkich {len(rows)} plików: `TABELA_PLIKOW.csv` (kolumny path, sizeGB, statusjob, localkopia, reason, risk, eligible, bytes_reclaimable oraz dodatkowe pola dowodowe). Poniższa tabela jest jej wyborem, nie dodatkowym budżetem. Nie sumować z klasami lub rozmiarami katalogów.

{table(big,['path','sizeGB','statusjob','localkopia','reason','risk','eligible','bytes_reclaimable'])}

## Własne rodziny: agregat du

{table([dict(path=p,sizeGB=round(n/1e9,3),statusjob=jobstatus(p),eligible='NIE: nie usuwać całej rodziny',bytes_reclaimable=0) for p,n in sorted(du.items(),key=lambda x:x[1],reverse=True) if p.startswith(base)],['path','sizeGB','statusjob','eligible','bytes_reclaimable'])}

## Dowody i ograniczenia

`system_jobs.json`, `jobs.json`, `sizes_final.txt`, `inventory.json`, `details.json`, `local_candidates.json`, `remote_hashes.json`, `verified_copies.json`, `deploy_verified.json`, `hardlinks.json`, `classification.json`, `SUMMARY.json`. Błędy SSH odróżniono od banera w plikach stderr. Brak błędów inwentaryzacji: {d['errors']}. Zachowano kod zbierania i generowania raportu wyłącznie w tym katalogu. Odczyt ma charakter punktowy; przed późniejszą decyzją konieczna aktualizacja bieżącego stanu, bo inne zadania i agenci mogą działać równolegle.
'''
(OUT/'RAPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,indent=2));print('report rows',len(rows),'big rows',len(big))
