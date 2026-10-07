"""Kontrolowane gałęzie uczenia małego ViT (oś time_axis = 'network_training'): odpowiedź na interwencję treningową
i wzrost małych zaburzeń.

Stan startowy w kroku k: wagi, bufory BatchNorm i momenty AdamW z powtórki gęstej historii (stan_uczenia.Odtworzenie).
Gałęzie startują z identycznej kopii tego stanu. Jedna dostaje interwencję w kroku k, potem wszystkie dostają te
same paczki i augmentacje (prawdziwe paczki biegu albo inna, ustalona ziarnem przyszła sekwencja). Bez kotwiczenia
do zapisanych wag (gałęzie są kontrfaktyczne).

Interwencje (jawna parametryzacja obrazu):
  waga        składnik straty obrazu w slocie p paczki kroku k mnożony przez 1 + ε (normy paczki bez zmian);
  podmiana    obraz w slocie p zastąpiony wybranym obrazem z jego etykietą (bez odbicia i bez zmiany jasności);
  zaburzenie  wagi gałęzi przed krokiem k: θ + ε ||θ|| ξ/||ξ|| w każdym tensorze parametrów (ξ gaussowskie, ziarno).
Gałąź kontrolna C jest drugą identyczną kopią bez interwencji: różnica A − C to numeryczny poziom odniesienia
(niedeterminizm GPU); wyniku poniżej tego poziomu nie interpretujemy jako efektu obrazu.

Wyjścia po h krokach od interwencji (h = 1 to sam krok z interwencją): zmiana latentu sond w trybie eval (pełny
latent i tokeny narzędzia), zmiana straty sond (gładkie wyjście), IoU najpewniejszej ramki wybranego obrazu.
Liniowość: odpowiedź na 2ε wobec 2 × odpowiedź na ε; superpozycja: sloty p i q razem wobec sumy osobnych.
Wzrost zaburzeń: d(h)/d(1) w tokenach sond; to miara wzrostu odchylenia reprezentacji, nie największy wykładnik
Lapunowa pełnego układu, a dodatni wzrost w krótkim oknie nie dowodzi chaosu.
"""
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import korekta_latentu as KL
from paczki_uczenia import odtworz_paczki
from pchniecie import strata_obrazow


class deterministycznie:
    """Deterministyczne obliczenia GPU na czas gałęzi: atencja SDPA w wersji MATH (domyślna memory-efficient ma
    niedeterministyczny krok wstecz), torch.use_deterministic_algorithms i cudnn.deterministic. Bez tego dwie identyczne
    gałęzie rozchodzą się (sprawdzone: po 5 krokach różnica rzędu odpowiedzi na obraz). Poprzednie ustawienia wracają
    po wyjściu. wlacz=False zostawia zwykłe (szybsze, niedeterministyczne) obliczenia, jak w prawdziwym biegu."""

    def __init__(self, wlacz=True):
        self.wlacz = wlacz

    def __enter__(self):
        if not self.wlacz:
            return self
        from torch.nn.attention import SDPBackend, sdpa_kernel
        self._stare = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                       torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        self._sdpa = sdpa_kernel(SDPBackend.MATH); self._sdpa.__enter__()
        return self

    def __exit__(self, *a):
        if not self.wlacz:
            return False
        self._sdpa.__exit__(*a)
        d, w, cd, cb = self._stare
        torch.use_deterministic_algorithms(d, warn_only=w)
        torch.backends.cudnn.deterministic = cd; torch.backends.cudnn.benchmark = cb
        return False


def bieg_gestej_historii():
    import analiza_uczenia as AU
    for b in AU.znajdz_biegi():
        if 'gesta_historia' in b['klucz']:
            return b
    raise FileNotFoundError('brak biegu gęstej historii (LATENT_GESTY.npy)')


class Galezie:
    def __init__(self, od, X_sond, ramki_sond, maski_sond, nazwy_sond):
        if not torch.cuda.is_available():
            raise RuntimeError('gałęzie uczenia wymagają GPU (uczenie biegu w bf16 autocast cuda)')
        self.od, self.M = od, od.M
        self.dev = 'cuda'
        self.Xs = torch.from_numpy(np.ascontiguousarray(X_sond)).permute(0, 3, 1, 2).float().to(self.dev) / 255.
        self.ramki = [np.asarray(r, np.float32).reshape(-1, 4) for r in ramki_sond]
        self.cele = [t.to(self.dev) for t in self.M.cele(self.ramki, 20, 30)]
        self.maski = torch.from_numpy(np.asarray(maski_sond, bool)).to(self.dev)
        self.nazwy = list(nazwy_sond)
        self._alt = {}

    # ------------------------------------------------------------ paczki
    def paczka(self, j, przyszlosc):
        if przyszlosc is None:
            return self.od.paczka(j)
        if przyszlosc not in self._alt:
            I, O, A = odtworz_paczki(self.od.kroki + 64, len(self.od.baza), 10_000 + int(przyszlosc))
            self._alt[przyszlosc] = SimpleNamespace(baza=self.od.baza, I=I, O=O, A=A)
        return type(self.od).paczka(self._alt[przyszlosc], j)

    # ------------------------------------------------------------ gałęzie
    def _klon(self, stan):
        u = self.od.T.Uczen(self.od.ziarno)
        u.wczytaj(copy.deepcopy(stan))
        return u

    def _krok(self, u, x, heat, ltrb, poz, wagi=None):
        u.net.train()
        for g in u.opt.param_groups:
            g['lr'] = u._lr()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = u.net(x)
        skl, _, _ = strata_obrazow(self.M, o, heat, ltrb, poz)
        L = (skl * wagi).sum() if wagi is not None else skl.sum()
        u.opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(u.net.parameters(), 1.0)
        u.opt.step()
        u.krok += 1

    @torch.no_grad()
    def pomiar(self, u):
        """Latent sond [S, 600, 768] float32 (eval, bufory gałęzi), składniki straty [S] i miary detekcji."""
        net = u.net
        net.eval(); net.zapisuj_latent(True)
        try:
            o, z = net(self.Xs)
            z = z.float().clone()
        finally:
            net.zapisuj_latent(False); net.train()
        cz = KL.skladniki_straty(o, *self.cele)
        from model_snapshot import dekoduj
        bx, sc = dekoduj(o)
        iou = []
        for i in range(len(self.Xs)):
            m = KL.ocena_obrazu(bx[i].cpu().numpy(), sc[i].cpu().numpy(), self.ramki[i])
            iou.append(m['iou_top'] if m['iou_top'] is not None else np.nan)
        return dict(z=z, strata=cz['strata'].cpu().numpy(), iou=np.array(iou))

    def _zaburz(self, u, eps, ziarno):
        g = torch.Generator(device='cpu').manual_seed(int(ziarno))
        with torch.no_grad():
            for p in u.net.parameters():
                xi = torch.randn(tuple(p.shape), generator=g).to(p.device)
                p.add_(xi * (eps * p.norm() / xi.norm().clamp(min=1e-12)))

    def galaz(self, stan0, k, H, horyzonty, przyszlosc, interwencja=None, przerwij=None):
        """Jedna gałąź od stanu kroku k przez H kroków; pomiary po h krokach (h w horyzonty). interwencja: None albo
        dict(rodzaj='waga'|'podmiana'|'zaburzenie', ...)."""
        u = self._klon(stan0)
        if interwencja is not None and interwencja['rodzaj'] == 'zaburzenie':
            self._zaburz(u, interwencja['eps'], interwencja.get('ziarno', 0))
        wyn = {}
        theta0 = None
        for j in range(H):
            if przerwij is not None and przerwij():
                return wyn, 'cancelled'
            x, heat, ltrb, poz, _ = self.od.paczka(k) if j == 0 else self.paczka(k + j, przyszlosc)
            wagi = None
            if j == 0 and interwencja is not None:
                r = interwencja['rodzaj']
                if r == 'waga':
                    wagi = torch.ones(len(x), device=self.dev)
                    for p, e in zip(interwencja['sloty'], interwencja['eps_slotow']):
                        wagi[p] += e
                elif r == 'podmiana':
                    p = interwencja['sloty'][0]
                    x = x.clone(); heat = heat.clone(); ltrb = ltrb.clone(); poz = poz.clone()
                    x[p] = interwencja['x']; heat[p] = interwencja['cel'][0][0]; ltrb[p] = interwencja['cel'][1][0]
                    poz[p] = interwencja['cel'][2][0]
            if wagi is None:
                wagi = torch.ones(len(x), device=self.dev)       # ta sama ścieżka kodu we wszystkich gałęziach
            self._krok(u, x, heat, ltrb, poz, wagi)
            h = j + 1
            if h in horyzonty:
                wyn[h] = self.pomiar(u)
                wyn[h]['theta'] = torch.cat([p.detach().flatten().float() for p in u.net.parameters()]).cpu()
        return wyn, 'complete'


def _roznice(a, b, maski):
    """Miary różnicy dwóch pomiarów (b − a): normy zmiany latentu [S] (pełny, tokeny narzędzia), zmiana straty [S],
    zmiana IoU [S], odległość wag."""
    dz = (b['z'] - a['z'])
    pel = dz.flatten(1).norm(dim=1).cpu().numpy()
    nar = torch.stack([dz[i][maski[i]].norm() for i in range(len(dz))]).cpu().numpy()
    return dict(dz=pel, dz_narzedzie=nar, dstrata=b['strata'] - a['strata'], diou=b['iou'] - a['iou'],
                dtheta=float((b['theta'] - a['theta']).norm()))


def licz_odpowiedz(G, k, interwencja, H=20, horyzonty=(1, 5, 20), przyszlosci=(None, 1, 2), liniowosc=True,
                   superpozycja=False, postep=None, przerwij=None, nazwa='galezie', deterministyczne=True):
    """Odpowiedź na interwencję treningową i poziom szumu dla kilku przyszłych sekwencji paczek. Zapisuje pomiar."""
    with deterministycznie(deterministyczne):
        return _licz_odpowiedz(G, k, interwencja, H, horyzonty, przyszlosci, liniowosc, superpozycja, postep, przerwij,
                               nazwa, deterministyczne)


def _licz_odpowiedz(G, k, interwencja, H, horyzonty, przyszlosci, liniowosc, superpozycja, postep, przerwij, nazwa,
                    deterministyczne):
    horyzonty = tuple(sorted({int(h) for h in horyzonty if 1 <= int(h) <= H}))
    t0 = time.time()
    stan0 = G.od.stan(k).stan()
    stan0 = copy.deepcopy(stan0)
    kat = KL.WYJSCIE / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{nazwa}_k{k:04d}"
    kat.mkdir(parents=True, exist_ok=True)
    torch.save(stan0, kat / 'STAN_STARTOWY.pt')
    przysz = {str(p): (None if p is None else odtworz_paczki(G.od.kroki + 64, len(G.od.baza), 10_000 + int(p)))
              for p in przyszlosci}
    np.savez(kat / 'PACZKI_PRZYSZLE.npz', **{f'{n}_{a}': v[i][k + 1:k + H]
                                            for n, v in przysz.items() if v is not None for i, a in enumerate('IOA')},
             prawdziwe_I=G.od.I[k:k + H], prawdziwe_O=G.od.O[k:k + H], prawdziwe_A=G.od.A[k:k + H])
    konf = dict(time_axis='network_training', krok_startu=k, H=H, horyzonty=list(horyzonty),
                przyszlosci=['prawdziwe paczki biegu' if p is None else f'inna sekwencja (ziarno {10_000 + int(p)})'
                             for p in przyszlosci],
                interwencja={a: (b if not torch.is_tensor(b) else 'tensor') for a, b in interwencja.items()
                             if a not in ('x', 'cel')},
                sondy=G.nazwy, bieg=str(G.od.bieg['dane']), ziarno_biegu=G.od.ziarno,
                stan_startowy='STAN_STARTOWY.pt (wagi, bufory BN, AdamW z powtórki gęstej historii)',
                paczki='PACZKI_PRZYSZLE.npz: indeksy bazy, odbicia, kontrast i jasność dla kroków k..k+H−1',
                deterministyczne=deterministyczne,
                uwaga='gałęzie bez kotwiczenia do zapisanych wag; krok k zawsze z prawdziwą paczką biegu')
    KL._json(kat / 'CONFIG.json', konf)
    KL._json(kat / 'STATUS.json', dict(status='running'))
    warianty = [('A', None), ('C', None), ('B', interwencja)]
    if liniowosc and interwencja['rodzaj'] in ('waga', 'zaburzenie'):
        i2 = dict(interwencja)
        if interwencja['rodzaj'] == 'waga':
            i2['eps_slotow'] = [2 * e for e in interwencja['eps_slotow']]
        else:
            i2['eps'] = 2 * interwencja['eps']
        warianty.append(('B2', i2))
    if superpozycja and interwencja['rodzaj'] == 'waga' and len(interwencja['sloty']) == 1:
        q = (interwencja['sloty'][0] + 1) % 32
        warianty.append(('Bq', dict(interwencja, sloty=[q])))
        warianty.append(('Bpq', dict(interwencja, sloty=[interwencja['sloty'][0], q],
                                     eps_slotow=interwencja['eps_slotow'] * 2)))
    wyniki = {}; status = 'complete'
    n_all = len(przyszlosci) * len(warianty); i_all = 0
    for p in przyszlosci:
        pom = {}
        for nazwa_w, iw in warianty:
            if postep:
                postep(int(100 * i_all / n_all), f'przyszłość {p if p is not None else "prawdziwa"}, gałąź {nazwa_w}')
            pom[nazwa_w], st = G.galaz(stan0, k, H, horyzonty, p, iw, przerwij)
            i_all += 1
            if st == 'cancelled':
                status = 'cancelled'; break
        if status == 'cancelled':
            break
        w = {}
        for h in horyzonty:
            A = pom['A'][h]
            w[h] = dict(szum=_roznice(A, pom['C'][h], G.maski), odpowiedz=_roznice(A, pom['B'][h], G.maski),
                        strata_A=A['strata'], iou_A=A['iou'])
            if 'B2' in pom:
                r1 = (pom['B'][h]['z'] - A['z']).flatten(); r2 = (pom['B2'][h]['z'] - A['z']).flatten()
                w[h]['liniowosc'] = float((r2 - 2 * r1).norm() / (2 * r1).norm().clamp(min=1e-30))
            if 'Bpq' in pom:
                rp = (pom['B'][h]['z'] - A['z']).flatten(); rq = (pom['Bq'][h]['z'] - A['z']).flatten()
                rpq = (pom['Bpq'][h]['z'] - A['z']).flatten()
                w[h]['superpozycja'] = float((rpq - rp - rq).norm() / rpq.norm().clamp(min=1e-30))
        wyniki['prawdziwa' if p is None else f'ziarno_{10_000 + int(p)}'] = w
        del pom
        torch.cuda.empty_cache()
    zb = podsumuj(wyniki, horyzonty)
    out = dict(konfiguracja=konf, status=status, wyniki=wyniki, podsumowanie=zb, czas_s=time.time() - t0, kat=str(kat))
    KL._json(kat / 'WYNIK.json', {a: b for a, b in out.items()})
    KL._json(kat / 'STATUS.json', dict(status=status, czas_s=time.time() - t0,
                                       pamiec_gpu_max_MiB=torch.cuda.max_memory_allocated() / 2 ** 20))
    return out


def podsumuj(wyniki, horyzonty):
    """Średnia i rozrzut po przyszłych sekwencjach: odpowiedź i szum na każdej sondzie i horyzoncie."""
    zb = {}
    for h in horyzonty:
        R = [w[h] for w in wyniki.values() if h in w]
        if not R:
            continue
        def st(sek, klucz):
            a = np.stack([r[sek][klucz] for r in R])
            return dict(srednia=a.mean(0), min=a.min(0), max=a.max(0), n=len(a))
        zb[h] = dict(odpowiedz_dz=st('odpowiedz', 'dz'), odpowiedz_dz_narzedzie=st('odpowiedz', 'dz_narzedzie'),
                     odpowiedz_dstrata=st('odpowiedz', 'dstrata'), szum_dz=st('szum', 'dz'),
                     szum_dstrata=st('szum', 'dstrata'), odpowiedz_diou=st('odpowiedz', 'diou'))
        for k in ('liniowosc', 'superpozycja'):
            v = [r[k] for r in R if k in r]
            if v:
                zb[h][k] = dict(srednia=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))
    return zb


def licz_wzrost(G, k, skale=(1e-4, 1e-3, 1e-2), H=20, horyzonty=(1, 2, 5, 10, 20), przyszlosci=(None, 1),
                ziarno=0, postep=None, przerwij=None, deterministyczne=True):
    with deterministycznie(deterministyczne):
        return _licz_wzrost(G, k, skale, H, horyzonty, przyszlosci, ziarno, postep, przerwij, deterministyczne)


def _licz_wzrost(G, k, skale, H, horyzonty, przyszlosci, ziarno, postep, przerwij, deterministyczne):
    """Wzrost małych zaburzeń wag: dla każdej skali ε osobna gałąź z θ + ε||θ||ξ, te same paczki co kontrola.
    d(h) = odległość latentu sond od kontroli; wzrost = d(h)/d(1) (po kroku wprowadzającym zaburzenie)."""
    horyzonty = tuple(sorted({int(h) for h in horyzonty if 1 <= int(h) <= H}))
    t0 = time.time()
    stan0 = copy.deepcopy(G.od.stan(k).stan())
    kat = KL.WYJSCIE / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_wzrost_zaburzen_k{k:04d}"
    kat.mkdir(parents=True, exist_ok=True)
    torch.save(stan0, kat / 'STAN_STARTOWY.pt')
    konf = dict(time_axis='network_training', krok_startu=k, H=H, horyzonty=list(horyzonty), skale=list(skale),
                ziarno_zaburzenia=ziarno, przyszlosci=['prawdziwe' if p is None else 10_000 + int(p) for p in przyszlosci],
                sondy=G.nazwy, deterministyczne=deterministyczne,
                uwaga='wzrost odchylenia reprezentacji w tokenach sond; nie wykładnik Lapunowa')
    KL._json(kat / 'CONFIG.json', konf)
    wyn = {}; status = 'complete'
    warianty = [('A', None), ('C', None)] + [(f'e{e:g}', dict(rodzaj='zaburzenie', eps=e, ziarno=ziarno)) for e in skale]
    n_all = len(przyszlosci) * len(warianty); i_all = 0
    for p in przyszlosci:
        pom = {}
        for nazwa_w, iw in warianty:
            if postep:
                postep(int(100 * i_all / n_all), f'przyszłość {p if p is not None else "prawdziwa"}, gałąź {nazwa_w}')
            pom[nazwa_w], st = G.galaz(stan0, k, H, horyzonty, p, iw, przerwij)
            i_all += 1
            if st == 'cancelled':
                status = 'cancelled'; break
        if status == 'cancelled':
            break
        w = {}
        for nazwa_w in pom:
            if nazwa_w == 'A':
                continue
            d = {h: _roznice(pom['A'][h], pom[nazwa_w][h], G.maski) for h in horyzonty}
            w[nazwa_w] = dict(d={h: float(np.linalg.norm(d[h]['dz'])) for h in horyzonty},
                              d_narzedzie={h: float(np.linalg.norm(d[h]['dz_narzedzie'])) for h in horyzonty},
                              dtheta={h: d[h]['dtheta'] for h in horyzonty})
            d1 = w[nazwa_w]['d'][horyzonty[0]]
            w[nazwa_w]['wzrost'] = {h: (w[nazwa_w]['d'][h] / d1 if d1 > 0 else None) for h in horyzonty}
        wyn['prawdziwa' if p is None else f'ziarno_{10_000 + int(p)}'] = w
        del pom
        torch.cuda.empty_cache()
    out = dict(konfiguracja=konf, status=status, wyniki=wyn, czas_s=time.time() - t0, kat=str(kat))
    KL._json(kat / 'WYNIK.json', out)
    KL._json(kat / 'STATUS.json', dict(status=status, czas_s=time.time() - t0))
    return out


def sondy_gestej_historii(bieg):
    """Obrazy 12 klatek próbnych, ramki z obrysu cystotomu (480 x 320), maski tokenów narzędzia i opisy."""
    import analiza_uczenia as AU
    op = AU.opis_biegu(bieg)
    ramki = []
    for o in AU._json(Path(bieg['dane']) / 'images.json', []):
        r = []
        seg = o.get('cystotome_seg') or []
        if isinstance(seg, str):
            seg = json.loads(seg)
        for poly in seg:
            p = np.asarray(poly, np.float32).reshape(-1, 2) * (480 / 720)
            r.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
        ramki.append(np.asarray(r, np.float32).reshape(-1, 4))
    return np.asarray(op['obrazy']), ramki, np.asarray(op['narzedzie'], bool), op['opisy']
