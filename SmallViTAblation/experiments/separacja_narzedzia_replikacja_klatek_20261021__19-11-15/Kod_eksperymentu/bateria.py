"""Bateria weryfikacyjna separacji narzędzia od tła (blok 6, stan x̃ przed MLP) dla kilku klatek z historią 806 checkpointów.

Dla każdej klatki i wybranych checkpointów:
  * 192D, PCA3 chmury (bez etykiet), LDA ze skurczem (λ = 0,1, osie 2–3 LOL) dopasowana na bieżącym checkpoincie;
  * zero etykietowe: 18 (n) losowych tokenów, PCA3 ze stałymi osiami, LDA z ponownym dopasowaniem (DiProPerm);
  * zero przestrzenne: ramka tego samego kształtu (wiersze × kolumny) w każdym położeniu siatki 20 × 30 rozłącznym z prawdziwą
    ramką; PCA3 (osie stałe, bo bez etykiet) i LDA z ponownym dopasowaniem; raportujemy percentyl prawdziwej ramki wśród położeń;
  * na końcu: LDA z puli 100 ostatnich checkpointów i walidacja blokowana (5 bloków).
Obliczenia na GPU w float32 (TF32), jak w aplikacji. Wynik: Wyniki/Dane/bateria_<klatka>.json + wspólny CSV.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
APP = Path(__file__).resolve().parents[3] / 'app'
OUT = Path(__file__).resolve().parents[1] / 'Wyniki' / 'Dane'
sys.path.insert(0, str(APP))
import lasery_dane as LD, rzuty_dane as RD

KLATKI = {'A_TR0050_k1629': '140854f7a6dbcbe5', 'B_TR0077_k1113': '2cca2743e1c74b14', 'C_nieustalona': 'c584ef4647149c7c'}
CHECK = [0, 10, 28, 56, 100, 200, 400, 600, 805]
dev = RD._dev()
MIARY = ('sil', 'hit', 'marg', 'fisher')


def pca3(X):
    Xc = X - X.mean(0, keepdim=True)
    _, U = torch.linalg.eigh(Xc.T @ Xc)
    return Xc @ U[:, -3:]


def miary(P, M):
    o = RD.miary_separacji_t(P, M)
    return {k: v.double().cpu().numpy() for k, v in o.items()}


def ramki(narz, W=LD.W, H=LD.TOK // LD.W):
    idx = np.flatnonzero(narz); r, c = np.divmod(idx, W)
    h, w = r.max() - r.min() + 1, c.max() - c.min() + 1
    pelna = len(idx) == h * w
    out = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            m = np.zeros(LD.TOK, bool)
            rr, cc = np.meshgrid(np.arange(r0, r0 + h), np.arange(c0, c0 + w), indexing='ij')
            m[(rr * W + cc).ravel()] = True
            if not (m & narz).any():
                out.append(m)
    return np.array(out), (int(h), int(w), bool(pelna))


def pct(prawda, rozklad):
    """Odsetek położeń tła z wartością ≥ prawdziwej (p empiryczne, z +1)."""
    rozklad = np.asarray(rozklad)
    return float((1 + (rozklad >= prawda).sum()) / (1 + len(rozklad)))


def losowe(n_narz, n=200, ziarno=0):
    g = torch.Generator().manual_seed(ziarno)
    perm = torch.argsort(torch.rand(n, LD.TOK, generator=g), 1)[:, :n_narz]
    return torch.zeros(n, LD.TOK, dtype=torch.bool).scatter_(1, perm, True)


def bateria(nazwa, hid):
    h = LD.Historia(APP / 'cache_uczenia/lasery' / hid)
    d = RD.wczytaj(h)
    assert d is not None, f'brak rzutów dla {hid}'
    narz = np.asarray(h.narz, bool)
    rdzen = RD.rdzen_narzedzia(narz, 1)
    R, ksztalt = ramki(narz)
    Mr = torch.as_tensor(R, device=dev)
    Ml = losowe(int(narz.sum())).to(dev)
    wyn = dict(klatka=nazwa, hist=hid, n_narz=int(narz.sum()), n_rdzen=int(rdzen.sum()), ksztalt=ksztalt, n_polozen=len(R),
               kroki=[int(h.kroki[i]) for i in CHECK], punkty=[])
    for i in CHECK:
        X = torch.as_tensor(np.asarray(d.X[i], np.float32), device=dev)
        rec = dict(i=i, krok=int(h.kroki[i]))
        for etyk, m in (('ramka', narz), ('rdzen', rdzen)):
            M1 = torch.as_tensor(m, device=dev)[None]
            rec[etyk] = {'192D': miary(X[None], M1), 'PCA3': miary(pca3(X)[None], M1),
                         'LDA': miary(X[None] @ RD.baza_L_t(X, M1), M1)}
            rec[etyk] = {w: {k: float(v[0]) for k, v in s.items()} for w, s in rec[etyk].items()}
        P3 = pca3(X)
        # zero etykietowe
        zl_p = miary(P3[None].expand(len(Ml), -1, -1), Ml)
        zl_l = {k: [] for k in MIARY}
        for a in range(0, len(Ml), 50):
            B = RD.baza_L_t(X, Ml[a:a + 50]); o = miary(X[None] @ B, Ml[a:a + 50])
            for k in MIARY: zl_l[k].append(o[k])
        zl_l = {k: np.concatenate(v) for k, v in zl_l.items()}
        rec['zero_etyk'] = {'PCA3': {k: float(np.percentile(v, 95)) for k, v in zl_p.items()},
                            'LDA_refit': {k: float(np.percentile(v, 95)) for k, v in zl_l.items()}}
        # zero przestrzenne
        zp_p = {k: [] for k in MIARY}; zp_l = {k: [] for k in MIARY}; zp_d = {k: [] for k in MIARY}
        for a in range(0, len(Mr), 50):
            Mb = Mr[a:a + 50]
            o = miary(P3[None].expand(len(Mb), -1, -1), Mb); [zp_p[k].append(o[k]) for k in MIARY]
            o = miary(X[None] @ RD.baza_L_t(X, Mb), Mb); [zp_l[k].append(o[k]) for k in MIARY]
            o = miary(X[None].expand(len(Mb), -1, -1), Mb); [zp_d[k].append(o[k]) for k in MIARY]
        zp = {'PCA3': {k: np.concatenate(v) for k, v in zp_p.items()}, 'LDA_refit': {k: np.concatenate(v) for k, v in zp_l.items()},
              '192D': {k: np.concatenate(v) for k, v in zp_d.items()}}
        rec['zero_przestrz'] = {w: {k: dict(p95=float(np.percentile(v, 95)), max=float(np.max(v)), mediana=float(np.median(v)),
                                            p_prawdziwej=pct(rec['ramka'][w if w != 'LDA_refit' else 'LDA'][k], v))
                                    for k, v in s.items()} for w, s in zp.items()}
        wyn['punkty'].append(rec)
        print(nazwa, i, 'PCA3', {k: round(rec['ramka']['PCA3'][k], 2) for k in ('sil', 'hit')},
              'p_przestrz', {k: round(rec['zero_przestrz']['PCA3'][k]['p_prawdziwej'], 3) for k in ('sil', 'hit')},
              'LDA', {k: round(rec['ramka']['LDA'][k], 2) for k in ('sil', 'hit')},
              'p_przestrz', {k: round(rec['zero_przestrz']['LDA_refit'][k]['p_prawdziwej'], 3) for k in ('sil', 'hit')}, flush=True)
    # pula 100 ostatnich i walidacja blokowana
    ks = np.arange(d.K - 100, d.K)
    Xk = np.asarray(d.X[ks], np.float32)
    for etyk, m in (('ramka', narz), ('rdzen', rdzen)):
        Xt = torch.as_tensor(Xk, device=dev).reshape(-1, Xk.shape[-1])
        B = RD.baza_L_t(Xt, torch.as_tensor(np.tile(m, len(ks)), device=dev)[None])[0]
        Xl = torch.as_tensor(Xk[-1], device=dev)
        o = miary((Xl @ B)[None], torch.as_tensor(m, device=dev)[None])
        cv, _ = RD.cv_blokowane_gpu(Xk, m, 0.1, 0, 5)
        wyn['pula100_' + etyk] = dict(LDA={k: float(v[0]) for k, v in o.items()}, cv=cv,
                                     zero_refit=RD.zero_z_refitem_gpu(Xk[-1], m, n=200))
    return wyn


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    wybor = sys.argv[1:] or list(KLATKI)
    for n in wybor:
        t0 = time.time()
        w = bateria(n, KLATKI[n])
        w['czas_s'] = round(time.time() - t0, 1)
        (OUT / f'bateria_{n}.json').write_text(json.dumps(w, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
        print(n, 'zapisano', w['czas_s'], 's', flush=True)
