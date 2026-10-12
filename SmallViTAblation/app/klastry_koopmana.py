"""Dobór osi Koopmana pod klastry cząsteczek z możliwie małą liczbą outlierów.

Wejście to próbka stanów Hankela cząsteczek z karty (ruch_gpu.ChmuraRuchuGPU.zadaj_probke_koopmana): h (M × 4d),
lewe wektory własne W, wariancje funkcji własnych i wartości własne. Kandydaci na oś to pojedyncze składowe ψ_j:
Re ψ_j, Im ψ_j i moduł |ψ_j| modów zespolonych (z pary sprzężonej bierzemy mod o Im λ > 0, drugi daje to samo z
przeciwnym znakiem Im). Każdą parę (X, Y), a potem trójkę z osią Z, ocenia na karcie (torch, wsadowo dla wszystkich
kandydatów naraz) mieszanina Gaussów z osobną wariancją każdej osi w każdym klastrze (EM, start z k-means++), dla
k = 2..k_max i kilku startów.

Ocena układu (osie standaryzowane odchyleniem po przycięciu percentyli 1-99, w każdym zapisie osobno):
- d_c(x) = odległość punktu od środka klastra c w σ tego klastra (√Σ ((x - μ_c) / σ_c)²), przydział jak w mieszaninie;
- a = d do własnego klastra, b = najmniejsze d do innego klastra; sylwetka s = (b - a) / max(a, b). Odległość w σ
  klastra sprawia, że wąskie, wydłużone, ale rozdzielone klastry mają wysoką sylwetkę;
- outlier: punkt z a > próg (w σ: dla gaussowskiego klastra próg 3 odcina 1,1 % punktów w 2D i 2,9 % w 3D) albo
  punkt w klastrze mniejszym niż min_udzial próbki (taki „klaster” to też garść odstających punktów);
- liczba klastrów k: najmniejszy BIC mieszaniny dla k = 1..k_max (k = 1 wygrywa: w tej parze nie ma klastrów);
- surowy wynik = średnia sylwetka punktów nieodstających - waga_out × udział outlierów (co najmniej 2 klastry);
- tło: surowy wynik układu zerowego przy tym samym k (próbka gaussowska o tej samej średniej i kowariancji, po
  przycięciu ogonów). Mieszanina podzieli też jedną chmurę i dostanie sylwetkę około 0,3-0,45. Klastry są istotne,
  gdy surowy wynik przewyższa tło o co najmniej MARGINES; ranking: istotne po surowym wyniku, potem nieistotne.
Najpierw najlepsza para (X, Y) ze wszystkich par, potem Z: składowa, z którą klastry w 3D wypadają najlepiej.
Model najlepszego układu (środki, σ, wagi, standaryzacja) trafia do shadera: kolor „klaster Koopmana”.
"""
from __future__ import annotations

import itertools

import numpy as np

TOL_IM = 1e-7
MARGINES = 0.15                                      # o ile surowy wynik musi przewyższyć tło
MIN_SIGMA = 0.02                                     # dolna granica σ klastra (standaryzowane jednostki)


def funkcje_wlasne(H, W, war, prog):
    """ψ (M × D, zespolone) dla próbki stanów H (M × D), skala jak w shaderze: 1 / √max(wariancja, próg)."""
    skala = 1.0 / np.sqrt(np.maximum(np.asarray(war, np.float64), prog))
    return np.asarray(H, np.float64) @ (np.asarray(W) * skala[None, :])


def kandydaci(vals, zdeg=()):
    """Kandydaci na oś: lista osi {'skladowe': [(j, 're'|'im')], 'kombinacja': 0|1} w kolejności bliskości λ do 1."""
    vals = np.asarray(vals, complex)
    zdeg = set(int(j) for j in zdeg)
    wynik = []
    for j in np.argsort(np.abs(vals - 1.0)):
        j = int(j)
        if j in zdeg or vals[j].imag < -TOL_IM * max(1.0, abs(vals[j])):
            continue
        wynik.append(dict(skladowe=[(j, 're')], kombinacja=0))
        if vals[j].imag > TOL_IM * max(1.0, abs(vals[j])):
            wynik.append(dict(skladowe=[(j, 'im')], kombinacja=0))
            wynik.append(dict(skladowe=[(j, 're'), (j, 'im')], kombinacja=1))
    return wynik


def wspolrzedna(psi, os_):
    """Wartość osi (jak os_koop w shaderze) dla funkcji własnych psi (M × D)."""
    if not os_['skladowe']:
        return np.zeros(len(psi))
    v = np.stack([(np.real if c == 're' else np.imag)(psi[:, j]) for j, c in os_['skladowe']], 1)
    return np.sqrt((v * v).sum(1)) if os_['kombinacja'] == 1 else v.sum(1)


def nazwa_osi(os_, vals):
    cz = [f'{c.capitalize()} ψ(λ = {vals[j].real:+.3f}{vals[j].imag:+.3f}i)' for j, c in os_['skladowe']]
    if not cz:
        return '0'
    if os_['kombinacja'] == 1:
        return f'|ψ(λ = {vals[os_["skladowe"][0][0]].real:+.3f}{vals[os_["skladowe"][0][0]].imag:+.3f}i)|' \
            if len({j for j, _ in os_['skladowe']}) == 1 else '√(' + '² + '.join(cz) + '²)'
    return ' + '.join(cz)


def _standaryzuj(X):
    """Standaryzacja kolumn: (x - mediana) / odchylenie standardowe po przycięciu do percentyli 1-99. Mediana i MAD
    zawodzą przy kilku klastrach (MAD bierze rozrzut wewnątrz największego klastra i rozciąga oś), a zwykłe odchylenie
    przy ciężkich ogonach (kilka dalekich punktów ściska resztę w punkt). Zwraca (Z, ok, mediana, skala)."""
    med = np.median(X, 0)
    lo, hi = np.percentile(X, [1, 99], axis=0)
    skala = np.clip(X, lo, hi).std(0)
    ok = skala > 1e-12
    skala = np.where(ok, skala, 1.0)
    return (X - med) / skala, ok, med, skala


def _zerowy(uklady, ziarno=1):
    """Układy zerowe: próbki gaussowskie o średniej i kowariancji danych (po przycięciu do percentyli 1-99)."""
    rng = np.random.default_rng(ziarno)
    out = np.empty_like(uklady)
    for c, Z in enumerate(uklady):
        lo, hi = np.percentile(Z, [1, 99], axis=0)
        Zc = np.clip(Z, lo, hi)
        cov = np.atleast_2d(np.cov(Zc, rowvar=False)) + 1e-9 * np.eye(Z.shape[1])
        out[c] = Zc.mean(0) + rng.normal(size=Z.shape) @ np.linalg.cholesky(cov).T
    return out


def _kmeans(Z, k, iters, gen):
    """k-means++ i Lloyd wsadowo. Z: B × M × p. Zwraca (odległości do środków B × M × k, środki B × k × p)."""
    import torch
    B, M, p = Z.shape
    ar = torch.arange(B, device=Z.device)
    i0 = torch.randint(M, (B,), generator=gen, device=Z.device)
    C = Z[ar, i0][:, None, :]
    d2 = ((Z - C) ** 2).sum(-1)
    for _ in range(1, k):
        nast = torch.multinomial(d2 + 1e-12, 1, generator=gen).squeeze(1)
        c = Z[ar, nast][:, None, :]
        C = torch.cat([C, c], 1)
        d2 = torch.minimum(d2, ((Z - c) ** 2).sum(-1))
    for _ in range(iters):
        lab = torch.cdist(Z, C).argmin(-1)
        oh = torch.nn.functional.one_hot(lab, k).to(Z.dtype)
        n = oh.sum(1)
        S = oh.transpose(1, 2) @ Z
        C = torch.where(n[..., None] > 0, S / n.clamp(min=1)[..., None], C)
    return torch.cdist(Z, C), C


def _gmm(Z, k, gen, iters_km=10, iters_em=15):
    """Mieszanina Gaussów z diagonalną kowariancją, wsadowo. Z: B × M × p. Zwraca (μ B×k×p, σ B×k×p, log π B×k)."""
    import torch
    B, M, p = Z.shape
    D, mu = _kmeans(Z, k, iters_km, gen)
    R = torch.nn.functional.one_hot(D.argmin(-1), k).to(Z.dtype)
    Z2 = Z * Z
    for it in range(iters_em + 1):
        Nk = R.sum(1) + 1e-6                                                      # B × k
        mu = (R.transpose(1, 2) @ Z) / Nk[..., None]
        var = ((R.transpose(1, 2) @ Z2) / Nk[..., None] - mu * mu).clamp(min=MIN_SIGMA ** 2)
        logpi = torch.log(Nk / M)
        if it == iters_em:
            break
        R = torch.softmax(_logp(Z, mu, var, logpi), -1)
    return mu, var.sqrt(), logpi


def _logp(Z, mu, var, logpi):
    """log π_c + log N(x | μ_c, diag σ_c²) bez stałej, B × M × k (w kawałkach po osiach: bez tensora B×M×k×p)."""
    acc = logpi[:, None, :] - 0.5 * torch_log(var).sum(-1)[:, None, :]
    for j in range(Z.shape[2]):
        acc = acc - 0.5 * (Z[:, :, j, None] - mu[:, None, :, j]) ** 2 / var[:, None, :, j]
    return acc


def torch_log(x):
    import torch
    return torch.log(x)


def _odleglosci(Z, mu, sigma):
    """d_c(x) w σ klastra c: B × M × k."""
    acc = 0.0
    for j in range(Z.shape[2]):
        acc = acc + ((Z[:, :, j, None] - mu[:, None, :, j]) / sigma[:, None, :, j]) ** 2
    return acc.sqrt()


def _ocen_wsad(Z, k, prog, min_udzial, waga_out, gen):
    """Metryki układów B × M × p dla k klastrów: wynik, sylwetka, udział outlierów, liczba klastrów, etykiety (-1 =
    outlier) i model: środki, σ, log π, czy klaster duży."""
    import torch
    mu, sigma, logpi = _gmm(Z, k, gen)
    lp = _logp(Z, mu, sigma * sigma, logpi)
    lab = lp.argmax(-1)
    d = _odleglosci(Z, mu, sigma)
    a = d.gather(2, lab[..., None]).squeeze(-1)
    b = d.scatter(2, lab[..., None], float('inf')).min(-1).values
    s = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    B, M = a.shape
    n = torch.nn.functional.one_hot(lab, k).sum(1)                                     # B × k
    duzy = n >= max(2, int(np.ceil(min_udzial * M)))
    out = (a > prog) | ~duzy.gather(1, lab)
    f_out = out.float().mean(1)
    w = (~out).float()
    s_in = (s * w).sum(1) / w.sum(1).clamp(min=1)
    k_eff = duzy.sum(1)
    wynik = torch.where(k_eff >= 2, s_in - waga_out * f_out, torch.full_like(s_in, -1e9))
    ll = torch.logsumexp(lp, -1).sum(1)                                                # log-wiarygodność (bez stałej 2π)
    return wynik, s_in, f_out, k_eff, torch.where(out, torch.full_like(lab, -1), lab), (mu, sigma, logpi, duzy), ll


def _bic(ll, k, p, M):
    return -2.0 * ll + (k * 2 * p + k - 1) * np.log(M)


def _ocen_surowo(uklady, k_max, starty, prog, min_udzial, waga_out, dev, ziarno, wsad_elem, z_modelem):
    """Dla każdego układu: liczba klastrów k z najmniejszego BIC mieszaniny (k = 1..k_max; najlepszy z kilku startów
    według wiarygodności), potem metryki przy tym k. BIC wybiera k = 1: brak klastrów (wynik -1e9)."""
    import torch
    gen = torch.Generator(device=dev); gen.manual_seed(ziarno)
    Zc = torch.as_tensor(np.asarray(uklady, np.float32), device=dev)
    C, M, p = Zc.shape
    wsad_elem = max(1, wsad_elem // max(p, 1))
    war = Zc.var(1, unbiased=False).clamp(min=MIN_SIGMA ** 2)                          # k = 1: jeden Gauss
    ll1 = (-0.5 * M * torch.log(war).sum(1) - 0.5 * M * p).cpu().numpy()
    naj = [dict(wynik=-1e9, sylwetka=0.0, outliery=0.0, k=1, k_mieszaniny=1, bic=_bic(float(ll1[c]), 1, p, M),
                wg_k={}) for c in range(C)]
    for k in range(2, max(2, k_max) + 1):
        wsad = max(1, wsad_elem // (M * k * starty))
        for c0 in range(0, C, wsad):
            Z = Zc[c0:c0 + wsad].repeat_interleave(starty, 0)
            w, s_in, f_out, k_eff, lab, (srodki, sigma, logpi, duzy), ll = \
                _ocen_wsad(Z, k, prog, min_udzial, waga_out, gen)
            ll = ll.view(-1, starty)
            best = ll.argmax(1)
            for q in range(ll.shape[0]):
                r = q * starty + int(best[q])
                bic = _bic(float(ll[q, best[q]]), k, p, M)
                naj[c0 + q]['wg_k'][k] = float(w[r])                    # wynik przy wymuszonym k (dla tła)
                if bic < naj[c0 + q]['bic']:
                    naj[c0 + q] = dict(wynik=float(w[r]), sylwetka=float(s_in[r]), outliery=float(f_out[r]),
                                       k=int(k_eff[r]), k_mieszaniny=k, bic=bic, wg_k=naj[c0 + q]['wg_k'])
                    if z_modelem:
                        naj[c0 + q].update(etykiety=lab[r].cpu().numpy(), srodki=srodki[r].cpu().numpy(),
                                           sigma=sigma[r].cpu().numpy(), logpi=logpi[r].cpu().numpy(),
                                           duzy=duzy[r].cpu().numpy())
    return naj


def ocen(uklady, k_max=6, starty=3, prog=3.0, min_udzial=0.02, waga_out=4.0, urzadzenie=None, ziarno=0,
         wsad_elem=6_000_000, zerowy=True):
    """Ocena układów. uklady: C × M × p (numpy, już standaryzowane). Zwraca listę słowników (najlepsze k każdego):
    wynik (netto, nad układem zerowym), surowy, tlo, sylwetka, outliery, k, etykiety, srodki, promienie."""
    import torch
    dev = torch.device(urzadzenie or ('cuda' if torch.cuda.is_available() else 'cpu'))
    uklady = np.asarray(uklady, np.float32)
    arg = (k_max, starty, prog, min_udzial, waga_out, dev, ziarno, wsad_elem)
    naj = _ocen_surowo(uklady, *arg, z_modelem=True)
    tlo = _ocen_surowo(_zerowy(uklady).astype(np.float32), *arg, z_modelem=False) if zerowy else None
    for c, o in enumerate(naj):
        o['surowy'] = o['wynik']
        # tło: ten sam podział (to samo k mieszaniny) gaussowskiego układu zerowego; k = 1 w danych: brak klastrów
        o['tlo'] = max(tlo[c]['wg_k'].get(o['k_mieszaniny'], 0.0), 0.0) if tlo else 0.0
        o.pop('wg_k', None)
        o['nadwyzka'] = max(o['surowy'] - o['tlo'], -1.0)
        o['istotne'] = bool(o['k'] >= 2 and o['nadwyzka'] >= MARGINES)
        # ranking: istotne klastry po surowym wyniku (sylwetka - kara), nieistotne zawsze niżej, brak klastrów na końcu
        o['wynik'] = o['surowy'] if o['istotne'] else (o['surowy'] - 2.0 if o['k'] >= 2 else -3.0)
    return naj


def _ocen_zestawy(X, zestawy, grupy=None, grupa_modelu=None, **kw):
    """Ocena zestawów kolumn X (surowe współrzędne osi, standaryzowane tutaj). Z grupami (zapis osi czasu każdego wiersza próbki) każdy zapis jest oceniany osobno
    i wynik, sylwetka, outliery są średnimi po zapisach: klastry muszą być grupami cząsteczek w każdym zapisie, a nie
    rozdzielać wczesny trening od późnego. Model (środki, etykiety, k) pochodzi z zapisu grupa_modelu."""
    if grupy is None:
        Xs = _standaryzuj(X)[0]
        return ocen(np.stack([Xs[:, list(z)] for z in zestawy]), **kw)
    grupy = np.asarray(grupy)
    ids = list(np.unique(grupy))
    idx = [np.flatnonzero(grupy == g) for g in ids]
    m = min(len(i) for i in idx)
    Xg = [_standaryzuj(X[i[:m]])[0] for i in idx]                  # każdy zapis standaryzowany osobno
    G = len(idx)
    oc = ocen(np.stack([x[:, list(z)] for z in zestawy for x in Xg]), **kw)
    g0 = ids.index(grupa_modelu) if grupa_modelu in ids else len(ids) - 1
    wynik = []
    for c in range(len(zestawy)):
        r = oc[c * G:(c + 1) * G]
        o = dict(r[g0])
        for klucz in ('wynik', 'surowy', 'tlo', 'nadwyzka', 'sylwetka', 'outliery'):
            o[klucz] = float(np.mean([max(q[klucz], -3.0) for q in r]))
        o['istotne'] = bool(np.mean([q['istotne'] for q in r]) >= 0.5)       # klastry w większości zapisów
        o['k_zapisy'] = [q['k'] for q in r]
        wynik.append(o)
    return wynik


def dobierz(H, W, war, prog_war, vals, zdeg=(), z_tez=True, osie_obecne=None, k_max=6, prog=3.0, waga_out=4.0,
            min_udzial=0.02, urzadzenie=None, postep=None, grupy=None, grupa_modelu=None):
    """Najlepsze osie pod klastry. grupy: zapis osi czasu każdego wiersza H (ocena osobno w każdym zapisie i średnia),
    grupa_modelu: zapis, z którego model idzie do shadera. Zwraca dict(osie, ocena, ocena_xy, ocena_obecnych, ranking,
    model, ...)."""
    kw = dict(k_max=k_max, prog=prog, waga_out=waga_out, min_udzial=min_udzial, urzadzenie=urzadzenie)
    vals = np.asarray(vals, complex)
    psi = funkcje_wlasne(H, W, war, prog_war)
    kand = kandydaci(vals, zdeg)
    if len(kand) < 2:
        raise ValueError('za mało modów Koopmana do wyboru osi (potrzebne co najmniej 2 składowe)')
    F = np.stack([wspolrzedna(psi, o) for o in kand], 1)              # surowe współrzędne kandydatów
    wiersze = np.ones(len(F), bool) if grupy is None else (np.asarray(grupy) == (
        grupa_modelu if grupa_modelu in set(np.asarray(grupy).tolist()) else np.max(grupy)))
    _, ok, med, skala = _standaryzuj(F[wiersze])                  # standaryzacja zapisu modelu (do shadera)
    if grupy is not None:
        for g in np.unique(grupy):                                # składowa stała w którymkolwiek zapisie odpada
            ok &= _standaryzuj(F[np.asarray(grupy) == g])[1]
    kand = [o for o, t in zip(kand, ok) if t]
    X, med, skala = F[:, ok], med[ok], skala[ok]
    if X.shape[1] < 2:
        raise ValueError('składowe ψ w próbce są stałe: brak osi do rozdzielenia klastrów')
    pary = list(itertools.combinations(range(X.shape[1]), 2))
    if postep:
        postep(f'Klastry: {len(pary)} par osi X, Y i ich układy zerowe (k = 2..{k_max})…', 0.1)
    oc = _ocen_zestawy(X, pary, grupy, grupa_modelu, **kw)
    kolej = sorted(range(len(pary)), key=lambda i: -oc[i]['wynik'])
    najlepsze = [(pary[i], oc[i]) for i in kolej]
    (a, b), ocena_xy = najlepsze[0]
    pomin = ('etykiety', 'srodki', 'sigma', 'logpi', 'duzy', 'k_zapisy')
    ranking = [dict(osie=[nazwa_osi(kand[i], vals) for i in w], **{k: v for k, v in o.items() if k not in pomin})
               for w, o in najlepsze[:5]]
    if 'srodki' not in ocena_xy:                      # BIC: jedna chmura w każdej parze, osie bez zmian
        return dict(osie=None, ocena=ocena_xy, ocena_xy=ocena_xy, ocena_obecnych=None, ranking=ranking, z_tez=False,
                    kandydaci=len(kand), probka=len(H), model=None, zapisy=None if grupy is None else len(set(grupy)))
    osie = dict(x=kand[a], y=kand[b], z=(osie_obecne or {}).get('z', dict(skladowe=[], kombinacja=0)))
    ocena, kolumny = ocena_xy, [a, b]
    if z_tez and X.shape[1] >= 3:                     # X, Y zostają; Z = składowa, z którą klastry w 3D są najlepsze
        reszta = [c for c in range(X.shape[1]) if c not in (a, b)]
        if postep:
            postep(f'Klastry: {len(reszta)} kandydatów na oś Z…', 0.6)
        oc3 = _ocen_zestawy(X, [(a, b, c) for c in reszta], grupy, grupa_modelu, **kw)
        z_ok = [i for i in range(len(reszta)) if 'srodki' in oc3[i]]
        if z_ok:                                      # bez klastrów w 3D: Z bez zmian, model z płaszczyzny X, Y
            i3 = max(z_ok, key=lambda i: oc3[i]['wynik'])
            osie['z'], ocena = kand[reszta[i3]], oc3[i3]
            kolumny = [a, b, reszta[i3]]
        else:
            z_tez = False
    # model do kolorowania w shaderze: współrzędne osi -> standaryzacja -> klaster o największym log π + log N,
    # outlier gdy d > próg albo klaster mały (oś pominięta: σ = 1, środek 0)
    kk, pw = len(ocena['srodki']), len(kolumny)
    srodki, sigma = np.zeros((kk, 3)), np.ones((kk, 3))
    srodki[:, :pw], sigma[:, :pw] = ocena['srodki'], ocena['sigma']
    model = dict(osie=osie, wym=pw, srodki=srodki, sigma=sigma, logpi=np.asarray(ocena['logpi'], float),
                 duzy=np.asarray(ocena['duzy'], bool), prog=float(prog),
                 mediana=np.r_[med[kolumny], np.zeros(3 - pw)], skala=np.r_[skala[kolumny], np.ones(3 - pw)])
    ocena_obecnych = None
    if osie_obecne:                                           # te same kryteria dla obecnych osi (porównanie)
        wym = ('x', 'y', 'z') if z_tez else ('x', 'y')
        Xo = np.stack([wspolrzedna(psi, osie_obecne[a_]) for a_ in wym], 1)
        oko = _standaryzuj(Xo)[1]
        if oko.sum() >= 2:
            ocena_obecnych = _ocen_zestawy(Xo, [tuple(np.flatnonzero(oko))], grupy, grupa_modelu, **kw)[0]
    return dict(osie=osie, ocena=ocena, ocena_xy=ocena_xy, ocena_obecnych=ocena_obecnych, ranking=ranking, z_tez=z_tez,
                kandydaci=len(kand), probka=len(H), model=model, zapisy=None if grupy is None else len(set(grupy)))


def _krotko(o):
    return (f'{o["k"]} klastry, outliery {100 * o["outliery"]:.1f} %, sylwetka {o["sylwetka"]:.2f}, '
            f'nadwyżka nad tłem {o["nadwyzka"]:+.2f}' + ('' if o.get('istotne') else ', NIEISTOTNE'))


def opis(wynik, vals):
    o = wynik['ocena']
    if wynik.get('model') is None:
        return (f'Osie pod klastry (próbka {wynik["probka"]:,} cząsteczek, {wynik["kandydaci"]} składowych ψ): w żadnej '
                'parze składowych mieszanina Gaussów nie znalazła więcej niż jednej chmury (BIC woli k = 1). Osie '
                'zostają bez zmian.')
    xy = wynik.get('ocena_xy', o)
    tekst = (f'Osie pod klastry (próbka {wynik["probka"]:,} cząsteczek, {wynik["kandydaci"]} składowych ψ'
             + (f', średnie po {wynik["zapisy"]} zapisach, k z bieżącego' if wynik.get('zapisy') else '')
             + f'). Płaszczyzna X, Y: {_krotko(xy)}')
    if wynik.get('z_tez'):
        tekst += f'; z osią Z: {_krotko(o)}'
    ob = wynik.get('ocena_obecnych')
    if ob and ob['k'] >= 2:
        tekst += f'. Poprzednie osie: {_krotko(ob)}'
    elif ob:
        tekst += '. Poprzednie osie: bez co najmniej dwóch klastrów'
    if not xy.get('istotne'):
        tekst += ('.\nUWAGA: nadwyżka nad gaussowskim tłem jest bliska zera, więc w tej próbce nie ma wyraźnych '
                  'klastrów w żadnej parze składowych; podział k-means jest wtedy podziałem jednej chmury')
    tekst += '.\nNastępne pary X, Y w rankingu:\n' + '\n'.join(
        f'  {i + 2}. ' + ' | '.join(r['osie']) + f' — {r["k"]} kl., out. {100 * r["outliery"]:.1f} %, '
        f'syl. {r["sylwetka"]:.2f}, nadwyżka {r["nadwyzka"]:+.2f}'
        for i, r in enumerate(wynik['ranking'][1:4]))
    return tekst
