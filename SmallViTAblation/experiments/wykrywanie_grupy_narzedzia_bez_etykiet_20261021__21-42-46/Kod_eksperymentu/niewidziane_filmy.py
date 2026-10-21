"""Ocena na filmach, których trening NIE widział: 25 filmów TEST podziału kampanii (split_manifest, rozłączne z 117 filmami
treningu). W trybie 'seedy' zero przestrzenne tylko na krokach 1000, 10000 i 65640 (czas). Metoda wykrywania grupy bez zmian (WYBRANA_KONFIGURACJA.json z etapu DEV).

Klatki: z każdego filmu `n` klatek z obrysem narzędzia, równomiernie w czasie (pierwsza i ostatnia włącznie), wyłącznie klatki
z ≥ 1 łatą o udziale ≥ 0,25. Na każdym checkpoincie i klatce:
  * grupa wykryta bez etykiet (TokenCut na aktywacjach MLP bloku 6, τ = 0,2, spójna) i mapa głowicy: Dice, precyzja, czułość;
  * etykieta z obrysu (narzędzie = udział ≥ 0,25, łaty mieszane usunięte): Fisher, sylwetka i kNN w pełnym 192D oraz Fisher
    po LDA (oś dopasowana na tej klatce) z zerem przestrzennym (ta sama maska przesunięta w tło; dla LDA oś od nowa).
Tryby: 'glowny' (seed 47 kroki 0…65640 i seed 3047 epoki 60…400, n = 20) albo 'seedy' (24 seedy × 12 checkpointów
kampanii, n = 10, bez LDA). Wynik: Wyniki/Dane/niewidziane_<tryb>.jsonl.
"""
import sys, json, hashlib, zipfile, time
from pathlib import Path
import numpy as np, cv2, torch
APP = Path(__file__).resolve().parents[3] / 'app'
EXP = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'Wyniki' / 'Dane'
sys.path.insert(0, str(APP)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lasery_dane as LD, rzuty_dane as RD, etykiety_narzedzia as EN, film as FI
import wykryj as WK

KAMP = LD.ROOT / 'outputs/kompensacja_funkcjonalna_dgx2_20261017/download/unpacked_20261018_035119'
SPLIT = json.loads((KAMP / 'input/split_manifest.json').read_text(encoding='utf-8'))['groups']
C200 = EXP / 'vit_kontynuacja_do_200epok_20261020__19-06-42/Wyniki/Dane/checkpoints'
C400 = EXP / 'vit_kontynuacja_do_400epok_20261020__23-50-57/Wyniki/Dane/checkpoints'
S47 = LD.ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
dev = RD._dev()
MIARY = ('sil', 'hit', 'marg', 'fisher')


def podzial(film):
    return SPLIT.get(film, 'poza_podzialem')


def klatki(n):
    tren = {r['film'] for r in LD.zbior_uczacy()}
    out = []
    for z in sorted(FI.ADNOTACJE.glob('*.zip')):
        f = z.stem
        if f in tren or podzial(f) != 'test':                          # tylko 25 filmów TEST podziału kampanii
            continue
        et = FI.etykiety(f)
        with zipfile.ZipFile(z) as zz:
            n_jpg = sum(x.endswith('.jpg') for x in zz.namelist())
        if len(et) != n_jpg or (et and max(et) != n_jpg):           # adnotacje nie pasują do obrazów (TR_0058, TR_0118): pomijamy
            print('pomijam', f, 'wierszy adnotacji', len(et), 'obrazów', n_jpg, flush=True)
            continue
        dobre = []
        for kl in sorted(et):
            e = EN.etykieta(f, kl)
            if e is not None and (e['udzial'] >= 0.25).any():
                dobre.append(e)
        if not dobre:
            continue
        for j in np.unique(np.linspace(0, len(dobre) - 1, n).round().astype(int)):
            e = dobre[j]
            out.append(dict(film=f, klatka=e['klatka'], podzial=podzial(f), udzial=e['udzial'],
                            gt=e['ramki'][e['kategorie'][0]], frame_id=f'{f}/{e["klatka"]}'))
    return out


def obraz(r):
    with zipfile.ZipFile(FI.ADNOTACJE / f"{r['film']}.zip") as z:
        raw = z.read(f"{r['film']}/{r['film']}_{r['klatka']:06d}.jpg")
    im = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA), hashlib.sha256(raw).hexdigest()


def separacja(X, u, z_lda=True):
    """192D (i LDA) z zerem przestrzennym; X (600, 192) torch."""
    narz = u >= 0.25; mies = (u > 0) & ~narz; keep = ~mies
    nt = int(narz.sum())
    if nt < 2:
        return None
    k = RD.k_sasiadow(nt)
    Xk = X[torch.as_tensor(keep, device=dev)]
    R = EN.przesuniecia(narz, u > 0)
    if len(R) < 19:
        return dict(n_narz=nt, n_polozen=len(R))
    M = torch.as_tensor(np.vstack([narz[None], R])[:, keep], device=dev)
    wyn = dict(n_narz=nt, n_polozen=len(R))
    widoki = (('192D', lambda Mb: Xk[None].expand(len(Mb), -1, -1)),) + \
             ((('LDA', lambda Mb: Xk[None] @ RD.baza_L_t(Xk, Mb)),) if z_lda else ())
    for nazwa, P in widoki:
        v = {q: [] for q in MIARY}
        for a in range(0, len(M), 64):
            o = RD.miary_separacji_t(P(M[a:a + 64]), M[a:a + 64], k)
            for q in MIARY:
                v[q].append(o[q].double().cpu().numpy())
        v = {q: np.concatenate(x) for q, x in v.items()}
        wyn[nazwa] = {q: float(v[q][0]) for q in MIARY}
        wyn[nazwa + '_p'] = {q: float((1 + (v[q][1:] >= v[q][0]).sum()) / len(v[q])) for q in MIARY}
        wyn[nazwa + '_tlo_med'] = {q: float(np.median(v[q][1:])) for q in MIARY}
    return wyn


def checkpointy(tryb):
    if tryb == 'glowny':
        dost = sorted(int(p.stem[1:]) for p in S47.glob('k*.pt'))
        c = {f's47_k{k:05d}': S47 / f'k{min(dost, key=lambda d: abs(d - k)):05d}.pt' for k in (0, 30, 100, 300, 1000, 3000, 10100, 28444, 65640)}
        c.update({'s3047_ep60': C200 / 'k065640.pt', 's3047_ep100': C200 / 'k109375.pt', 's3047_ep200': C200 / 'k218750.pt',
                  's3047_ep300': C400 / 'k328125.pt', 's3047_ep400': C400 / 'k437500.pt'})
        return c
    c = {}
    for sd in sorted((KAMP / 'replications').glob('s*'), key=lambda p: int(p.name[1:])):
        for p in sorted((sd / 'weights').glob('k*.pt')):
            c[f'{sd.name}_{p.stem}'] = p
    return c


if __name__ == '__main__':
    tryb = sys.argv[1]
    n = 20 if tryb == 'glowny' else 10
    t0 = time.time()
    R = klatki(n)
    ob, sha = zip(*[obraz(r) for r in R]); ob = np.stack(ob)
    print(tryb, 'klatek', len(R), {p: sum(r['podzial'] == p for r in R) for p in ('val', 'test', 'poza_podzialem')},
          'filmów', len({r['film'] for r in R}), round(time.time() - t0), 's', flush=True)
    wyb = json.loads((OUT / 'WYBRANA_KONFIGURACJA.json').read_text(encoding='utf-8'))['konfiguracja']
    wyb['id'] = json.dumps(wyb, sort_keys=True)
    konf = [k for k in WK.konfiguracje() if k['id'] in (wyb['id'], json.dumps({a: b for a, b in dict(wyb, spojna=False).items() if a != 'id'}, sort_keys=True))]
    siec = WK.Siec(); MOD = LD.modul_modelu()
    plik = (OUT / f'niewidziane_{tryb}.jsonl').open('w', encoding='utf-8', newline='\n')
    for nazwa, p in checkpointy(tryb).items():
        siec.wczytaj(p)
        for a in range(0, len(R), 32):
            with torch.no_grad():
                o, _ = siec.net(torch.from_numpy(ob[a:a + 32]).permute(0, 3, 1, 2).float().div(255).to(dev))
            cechy = {k: v.float() for k, v in siec.p.items()}
            mapa = o[..., 0].sigmoid().reshape(len(o), -1).float()
            bx, sc = MOD.dekoduj(o.float().cpu(), k=1)
            u = np.stack([r['udzial'] for r in R[a:a + 32]])
            G = WK.grupy(cechy, mapa, konf)
            m_wyb, m_gl = WK.miary(G[wyb['id']], u), WK.miary(G['glowica'], u)
            for j in range(len(u)):
                r = R[a + j]
                rec = dict(ckpt=nazwa, film=r['film'], klatka=r['klatka'], podzial=r['podzial'], sha=sha[a + j],
                           grupa=dict(zip(('dice', 'czulosc', 'precyzja', 'precyzja_sas1', 'n'), map(float, m_wyb[j]))),
                           glowica=dict(zip(('dice', 'czulosc', 'precyzja', 'precyzja_sas1', 'n'), map(float, m_gl[j]))),
                           iou=float(LD.iou(bx[j, 0].tolist(), r['gt'])), pew=float(sc[j, 0]),
                           sep=separacja(cechy['x6'][j], u[j], z_lda=(tryb == 'glowny'))
                           if tryb == 'glowny' or nazwa.endswith(('k01000', 'k10000', 'k65640')) else None)
                plik.write(json.dumps(rec, ensure_ascii=False) + '\n')
        plik.flush()
        print(nazwa, round(time.time() - t0), 's', flush=True)
    plik.close()
