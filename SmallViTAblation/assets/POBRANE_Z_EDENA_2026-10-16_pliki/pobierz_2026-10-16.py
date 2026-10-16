#!/usr/bin/env python3
"""Pobieranie z Edena (2026-10-16). Nic nie jest kasowane ani zmieniane po stronie Edena.
Sposob: lista plikow z Edena (find), SHA-256 policzone NA EDENIE, plik juz lokalnie (ta sama sciezka lub ta sama
tresc >= 5 MB gdziekolwiek w assets/experiments/outputs) jest pomijany, reszta idzie strumieniem tar przez ssh
w paczkach, po kazdej paczce kontrola SHA-256 po stronie lokalnej (niezgodne pobierane drugi raz).
uzycie: python pobierz.py plan|run [czesc ...]
"""
import sys, os, re, subprocess, hashlib, json, time, fnmatch, pickle
from concurrent.futures import ThreadPoolExecutor

SSH = ['ssh', '-o', 'BatchMode=yes', 'eden-cluster']
SM = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation'
R = SM + '/assets'
SZ = R + '/RuchCzasteczek_ViT_szklo_2026-10'
IK = None  # I: nie uzywac
ER = '~/smallvit_runs'
SCR = os.path.dirname(os.path.abspath(__file__))
STAT = SCR + '/status.json'
MIN_DEDUP = 5 * 1024 * 1024
PACZKA = 4 * 1024 ** 3

JUNK = ['*__pycache__*', '*.pytest_cache*', '*/.venv/*', '*/.pip-cache/*', '*/.mpl-cache/*', '*.pyc']
BIG5 = ['bezpoz_lr_5e-4_rp', 'cnn_lr_5e-4_rp_zostaw', 'lr_2e-3_rp_zostaw', 'lr_5e-4_rp_z48_zostaw', 'lr_5e-4_rp_z49_zostaw']
DEIT_MAIN = ['s47_obciecie', 's47_bez_obciecia', 'maly_s47_obciecie']
DEIT_CTRL = ['maly_s48_obciecie', 'maly_s47_wd0', 'maly_s47_losowe_etykiety']


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(SCR + '/pobieranie.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def ssh_out(cmd, stdin_bytes=None):
    p = subprocess.run(SSH + [cmd], input=stdin_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout.decode('utf-8', 'replace')


def lista_eden(base, roots):
    cmd = "cd %s && find %s -type f -printf '%%s\\t%%p\\n'" % (base, ' '.join(roots))
    out = ssh_out(cmd)
    res = {}
    for ln in out.splitlines():
        m = re.match(r'^(\d+)\t(.+)$', ln)
        if m:
            p = m.group(2)
            if p.startswith('./'):
                p = p[2:]
            res[p] = int(m.group(1))
    return res


def sha_eden(base, rels):
    """SHA-256 plikow policzone na Edenie; pliki podawane przez stdin."""
    wynik = {}
    # paczkami, zeby pojedyncze polaczenie nie trwalo bez konca
    for i in range(0, len(rels), 3000):
        ch = rels[i:i + 3000]
        out = ssh_out("cd %s && xargs -d '\\n' -P4 -n 8 sha256sum" % base, ('\n'.join(ch) + '\n').encode())
        for ln in out.splitlines():
            m = re.match(r'^([0-9a-f]{64})  (.+)$', ln)
            if m:
                wynik[m.group(2)] = m.group(1)
    return wynik


def sha_lokalny(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_idx = None
_hcache = {}


def indeks_lokalny():
    global _idx
    if _idx is not None:
        return _idx
    pk = SCR + '/indeks_lokalny.pkl'
    if os.path.exists(pk) and time.time() - os.path.getmtime(pk) < 6 * 3600:
        _idx = pickle.load(open(pk, 'rb'))
        return _idx
    log('buduje indeks lokalnych plikow >= 5 MB (assets, experiments, outputs)')
    idx = {}
    for root in (R, SM + '/experiments', SM + '/outputs'):
        for dp, dn, fn in os.walk(root):
            for n in fn:
                p = os.path.join(dp, n)
                try:
                    s = os.stat(p).st_size
                except OSError:
                    continue
                if s >= MIN_DEDUP:
                    idx.setdefault(s, []).append(p.replace('\\', '/'))
    pickle.dump(idx, open(pk, 'wb'))
    _idx = idx
    log('indeks: %d rozmiarow, %d plikow' % (len(idx), sum(len(v) for v in idx.values())))
    return idx


def lokalna_kopia(size, sha):
    if size < MIN_DEDUP:
        return None
    for p in indeks_lokalny().get(size, [])[:40]:
        if p not in _hcache:
            try:
                _hcache[p] = sha_lokalny(p)
            except OSError:
                _hcache[p] = None
        if _hcache[p] == sha:
            return p
    return None


def czesc(nazwa, base, roots, excl, dest, filt=None, cmp_map=None, opis=''):
    return dict(nazwa=nazwa, base=base, roots=roots, excl=excl + JUNK, dest=dest, filt=filt, cmp_map=cmp_map or {}, opis=opis)


def deit_wagi_ok(rel):
    m = re.search(r'/wagi/k(\d+)\.pt$', rel)
    if not m:
        return True
    k = int(m.group(1))
    return k <= 1000 or k % 2000 == 0 or k in (1563,)


def budowa():
    P = []
    P.append(czesc('fdt_koopman', ER, ['2026-10-13_fdt_temperatura', '2026-10-13_koopman_pulapki'], [],
                   SZ + '/10_pomiary_pulapek_eden_2026-10-13'))
    P.append(czesc('k1000_wzorce_korekta', ER, ['vit_k1000_wzorce_pydmd_deeptime_20261009', 'vit_k1000_wspolna_korekta_patch_20261009'],
                   ['*/smoke*'], SZ + '/09_k1000_biegi_eden'))
    P.append(czesc('k1000_reszta', ER, ['2026-10-08_k1000_podmiana150_sd', 'vit_k1000_1500_pelne_latenty_90_20261008', 'vit_k1000_replay900k_co5_20261009'],
                   ['*/smoke/*', '*/wyniki/proba/*', '*/deps/*'], SZ + '/09_k1000_biegi_eden'))
    P.append(czesc('koloidalne_reszta', ER + '/2026-10-10_szklo_koloidalne', ['kod', 'logi', 'wyniki'],
                   ['wyniki/proba/*', 'wyniki/pomiar_pelny60_3000tr_1000val/*'], SZ + '/04_szklo_koloidalne_BD/eden_symulacje'))
    P.append(czesc('pomiar60_kontrola', ER + '/2026-10-10_szklo_koloidalne/wyniki', ['pomiar_pelny60_3000tr_1000val'], [],
                   SZ + '/03_pomiar_naturalny_60_epok_eden'))
    ex_pk = ['wyniki/klas_vit_od_zera_60ep_lr_5e-4/*']
    for r in BIG5:
        ex_pk += ['wyniki/%s/gesty/pca32_*' % r, 'wyniki/%s/gesty/pelne768_*' % r]
    P.append(czesc('pokretla_male', ER + '/2026-10-10_szklo_pokretla', ['.'], ex_pk, SZ + '/05_biegi_pokretla_eden'))
    lekki = lambda rel: (rel.startswith('kod/') or rel.startswith('logi/') or
                         bool(re.match(r'^wyniki/[^/]+/dane/', rel)) or
                         bool(re.match(r'^wyniki/[^/]+/gesty/(BAZA_PCA32\.npz|PACZKI\.npz|SONDY\.json)$', rel)) or
                         rel.startswith('dane/'))
    P.append(czesc('deit_lekki', ER + '/2026-10-13_deit_cifar10', ['kod', 'logi', 'dane', 'wyniki'], ['wyniki/proba/*'],
                   R + '/DeiTTiny_CIFAR10_2026-10-13', filt=lekki,
                   cmp_map={'dane/cifar10_uint8.npz': R + '/Dane_CIFAR10/cifar10_uint8.npz'}))
    P.append(czesc('deit_kontrole', ER + '/2026-10-13_deit_cifar10', ['wyniki/' + r for r in DEIT_CTRL], [],
                   R + '/DeiTTiny_CIFAR10_2026-10-13', filt=lambda rel: (not '/wagi/' in rel) or int(re.search(r'k(\d+)\.pt$', rel).group(1)) <= 100))
    P.append(czesc('deit_glowne', ER + '/2026-10-13_deit_cifar10', ['wyniki/' + r for r in DEIT_MAIN], [],
                   R + '/DeiTTiny_CIFAR10_2026-10-13', filt=deit_wagi_ok))
    ex_big = []
    P.append(czesc('pokretla_gesty', ER + '/2026-10-10_szklo_pokretla',
                   ['wyniki/' + r + '/gesty' for r in BIG5] + ['wyniki/klas_vit_od_zera_60ep_lr_5e-4'], ex_big,
                   SZ + '/05_biegi_pokretla_eden',
                   filt=lambda rel: ('cnn_lr_5e-4_rp_zostaw' not in rel) and not re.match(r'^wyniki/(%s)/gesty/(BAZA_PCA32\.npz|PACZKI\.npz|SONDY\.json)$' % '|'.join(map(re.escape, BIG5)), rel)))
    P.append(czesc('stare_male', ER, ['2026-09-14_nano_smoke', '2026-09-14_tracking_nano', '2026-09-21_koopman_langevin',
                   '2026-09-22_dobor_cech_adaptacyjny', '2026-09-22_dobor_klatek_stan_sieci', '2026-09-23_fastgan', '2026-09-24_koopman_fastgan_petla',
                   '2026-09-24_koopman_gesta_historia', '2026-09-25_kontynuacja_dobor', '2026-09-28_maly_vit_od_zera', '2026-09-28_maly_vit_sila',
                   '2026-09-28_maly_vit_wektor', '2026-09-29_dyfuzja_narzedzie', '2026-09-29_maly_vit_sila_it2', '2026-09-29_rownanie_latentu',
                   '2026-09-30_dyfuzja_sd_filmy', '2026-10-01_maly_vit_ruch_czastek_gesta_historia', '2026-10-01_petla43_sd_prowadzenie',
                   '2026-10-06_maly_vit_rozne_latenty_5000', '2026-10-06_vit_konwerter_dino_filmy', '2026-10-09_maly_vit_od_zera_35000_21epok',
                   'outputs', 'experiments'],
                   ['*/deps/*', '*/smoke*'], R + '/Eden_stare_male_przebiegi_2026-10-16'))
    P.append(czesc('imagenette', '~/smallvit_data/imagenette', ['.'], ['pobieranie.log'], R + '/Dane_Imagenette'))
    P.append(czesc('cataract_male', '~/smallvit_data/cataract_lmm', ['sd576_train','sd576_train_proba2','sd576_val_co10','sd576_val_co10_proba2'], [], R + '/Eden_cataract_sd576_wskazniki'))
    return P


def plan_czesci(c):
    L = lista_eden(c['base'], c['roots'])
    sel = {}
    for rel, s in L.items():
        if any(fnmatch.fnmatch(rel, pat) for pat in c['excl']):
            continue
        if c['filt'] and not c['filt'](rel):
            continue
        sel[rel] = s
    return sel


def klasyfikuj(c, sel, hashe):
    dest = c['dest']
    pull, juz, rozne, loc = [], [], [], []
    for rel, s in sorted(sel.items()):
        sha = hashe.get(rel)
        lp = c['cmp_map'].get(rel)
        cand = lp if lp else dest + '/' + rel
        if os.path.exists(cand):
            if os.path.getsize(cand) == s and sha_lokalny(cand) == sha:
                juz.append((rel, s, cand))
            else:
                rozne.append((rel, s, cand))
            continue
        k = lokalna_kopia(s, sha) if sha else None
        if k:
            loc.append((rel, s, k))
        else:
            pull.append((rel, s))
    return pull, juz, rozne, loc


def fmt(b):
    return '%.2f GB' % (b / 1024 ** 3) if b > 1e9 else '%.1f MB' % (b / 1024 ** 2)


def pobierz_paczke(c, rels):
    dest = c['dest']
    os.makedirs(dest, exist_ok=True)
    p1 = subprocess.Popen(SSH + ["cd %s && tar cf - -T -" % c['base']], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(['tar', 'xf', '-'], stdin=p1.stdout, cwd=dest, stderr=subprocess.PIPE)
    p1.stdout.close()
    p1.stdin.write(('\n'.join(rels) + '\n').encode())
    p1.stdin.close()
    err = p2.communicate()[1]
    p1.wait()
    if p2.returncode:
        log('tar lokalny blad: %s' % err.decode('utf-8', 'replace')[:300])


def uruchom(c):
    n = c['nazwa']
    log('=== %s: lista z Edena (%s : %s)' % (n, c['base'], ' '.join(c['roots'])[:150]))
    sel = plan_czesci(c)
    log('%s: %d plikow, %s' % (n, len(sel), fmt(sum(sel.values()))))
    t0 = time.time()
    hashe = sha_eden(c['base'], sorted(sel))
    log('%s: SHA-256 z Edena policzone (%d/%d) w %.0f s' % (n, len(hashe), len(sel), time.time() - t0))
    brak = [r for r in sel if r not in hashe]
    pull, juz, rozne, loc = klasyfikuj(c, sel, hashe)
    log('%s: do pobrania %d (%s); juz w dest %d; lokalnie gdzie indziej %d; ROZNE %d' %
        (n, len(pull), fmt(sum(s for _, s in pull)), len(juz), len(loc), len(rozne)))
    os.makedirs(c['dest'], exist_ok=True)
    tag = c['dest'].rstrip('/').split('/')[-1]
    with open('%s/POMIJANE_%s_2026-10-16.tsv' % (SCR, n), 'w', encoding='utf-8') as f:
        f.write('rel\tbajty\tpowod\twskazanie\n')
        for rel, s, p in juz: f.write('%s\t%d\tjuz_w_celu\t%s\n' % (rel, s, p))
        for rel, s, p in loc: f.write('%s\t%d\tjuz_lokalnie\t%s\n' % (rel, s, p))
        for rel, s, p in rozne: f.write('%s\t%d\tROZNE\t%s\n' % (rel, s, p))
        for rel in brak: f.write('%s\t%d\tBRAK_SHA_EDEN\t\n' % (rel, sel[rel]))
    # paczki
    paczki, cur, cs = [], [], 0
    for rel, s in pull:
        if rel in brak:
            continue
        cur.append((rel, s)); cs += s
        if cs >= PACZKA or len(cur) >= 4000:
            paczki.append(cur); cur, cs = [], 0
    if cur: paczki.append(cur)
    zgodne, bledne = {}, []
    total = sum(s for _, s in pull); done = 0
    for i, pk in enumerate(paczki, 1):
        import shutil
        if shutil.disk_usage('F:/').free < 20*1024**3 + sum(s for _, s in pk):
            log('%s: STOP, brak miejsca (limit 20 GB wolnego na F:)' % n); break
        for proba in (1, 2):
            t1 = time.time()
            do = [r for r, s in pk if r not in zgodne]
            if not do: break
            pobierz_paczke(c, do)
            def spr(rel):
                p = c['dest'] + '/' + rel
                return rel, (os.path.exists(p) and sha_lokalny(p) == hashe[rel])
            with ThreadPoolExecutor(4) as ex:
                for rel, ok in ex.map(spr, do):
                    if ok: zgodne[rel] = hashe[rel]
            if all(r in zgodne for r, _ in pk): break
            log('%s: paczka %d proba %d: niezgodnych %d' % (n, i, proba, sum(1 for r, _ in pk if r not in zgodne)))
        b = sum(s for _, s in pk); done += b
        log('%s: paczka %d/%d %s, %.0f MB/s, razem %s z %s' % (n, i, len(paczki), fmt(b), b / 1048576 / max(1, time.time() - t1), fmt(done), fmt(total)))
        json.dump({'czesc': n, 'paczka': i, 'z': len(paczki), 'pobrano_bajty': done, 'razem_bajty': total, 'czas': time.strftime('%H:%M:%S')}, open(STAT, 'w'))
    bledne = [r for r, _ in pull if r not in zgodne]
    with open('%s/SHA256SUMS_EDEN_2026-10-16_%s.txt' % (c['dest'], n), 'w', encoding='utf-8') as f:
        for rel in sorted(zgodne): f.write('%s  %s\n' % (zgodne[rel], rel))
        for rel, s, p in juz: f.write('%s  %s\n' % (hashe[rel], rel))
    wynik = dict(czesc=n, dest=c['dest'], plikow_eden=len(sel), bajty_eden=sum(sel.values()), pobrano=len(zgodne),
                 bajty_pobrane=sum(sel[r] for r in zgodne), juz_w_celu=len(juz), juz_lokalnie=len(loc), rozne=len(rozne), bledne=bledne[:50], n_bledne=len(bledne))
    json.dump(wynik, open('%s/WYNIK_%s.json' % (SCR, n), 'w'), indent=1)
    log('%s: KONIEC pobrano %d plikow zgodnych z Edenem (%s), bledne %d, rozne %d' % (n, len(zgodne), fmt(wynik['bajty_pobrane']), len(bledne), len(rozne)))


def plan(c):
    sel = plan_czesci(c)
    s = sum(sel.values())
    log('PLAN %s: %d plikow %s -> %s' % (c['nazwa'], len(sel), fmt(s), c['dest']))
    top = {}
    for rel, b in sel.items():
        k = '/'.join(rel.split('/')[:3])
        top[k] = top.get(k, 0) + b
    for k, b in sorted(top.items(), key=lambda x: -x[1])[:8]:
        log('     %s  %s' % (fmt(b), k))


if __name__ == '__main__':
    tryb = sys.argv[1]
    P = budowa()
    wyb = sys.argv[2:]
    for c in P:
        if wyb and c['nazwa'] not in wyb:
            continue
        (plan if tryb == 'plan' else uruchom)(c)
