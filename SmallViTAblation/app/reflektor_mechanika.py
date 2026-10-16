"""Mechanika reflektora c bloku 6 (okienko 13 zakładki „Lasery i koloidy”): odrzut wiązek i wykonany ruch igły, celownika i progu.

Reflektor c to neuron c MLP bloku 6. Ma jedną igłę pchnięcia v_c (kolumna fc2), jeden celownik w_c (wiersz fc1) i jeden próg b_c,
wspólne dla wszystkich tokenów: 600 wiązek to linki przyczepione do tej samej piasty, a nie osobne lusterka.

Siła pola celu na koloid j to F_j = −δ_j, δ_j = ∂L/∂x_j (x_j = wyjście bloku 6; L = strata uczenia, model.strata). Odrzut przenoszą
wiązki proporcjonalnie do jasności, więc na reflektor działa dokładnie:
    igła      f_v = −∂L/∂v_c = Σ_j z_jc F_j                         (Z22)
    celownik  f_w = −∂L/∂w_c = Σ_j GELU′(u_jc) (v_c · F_j) LN₂(x̃_j)   (skręca ku koloidom, którym pchnięcie pomaga)
    próg      f_b = −∂L/∂b_c = Σ_j GELU′(u_jc) (v_c · F_j)
Sprężyny powrotne (weight decay AdamW 0,05) ciągną v_c i w_c do zera (próg ich nie ma: trening nie stosuje wd do biasów).
Wykonany ruch = różnica wag między checkpointem i a następnym (co 10 kroków do kroku 1000, potem co 100): suma wielu kroków AdamW.
Stanu AdamW (bezwładność, tłumienie) checkpointy nie zawierają: widać tylko jego skutek, czyli różnicę kierunku odrzutu i ruchu.

Źródła odrzutu: 'sonda' = oglądane zdjęcie historii (także linki tokenów), 'zbior' = paczka 64 klatek zbioru uczącego (stałe
losowanie, ziarno 0), bez augmentacji, fp32, sieć w trybie eval (BatchNorm łodygi ze statystyk biegu) i bez obcinania gradientu.

Lokalny układ 3D (kąty prawdziwe): e₁ = kierunek igły, e₂ = kierunek wykonanego obrotu (część Δv prostopadła do v), e₃ = reszta
odrzutu prostopadła do e₁ i e₂. To samo dla celownika.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import lasery_dane as LD

WD = 0.05
N_ZBIOR = 64


def _uklad(a, d, f):
    """(e1, e2, e3) dla wektora a, jego zmiany d i siły f (numpy, float64)."""
    e1 = a / max(np.linalg.norm(a), 1e-30)
    dp = d - (d @ e1) * e1
    fp = f - (f @ e1) * e1
    e2 = dp if np.linalg.norm(dp) > 1e-12 * max(np.linalg.norm(a), 1e-30) else fp
    e2 = e2 / max(np.linalg.norm(e2), 1e-30)
    r = fp - (fp @ e2) * e2
    if np.linalg.norm(r) < 1e-12 * max(np.linalg.norm(fp), 1e-30):
        r = np.random.default_rng(0).normal(size=a.shape); r -= (r @ e1) * e1 + (r @ e2) * e2
    e3 = r / max(np.linalg.norm(r), 1e-30)
    return np.stack([e1, e2, e3])


def _czesc(a, d, f, sprezyna):
    """Rozkład jednej części (igła albo celownik) w jej układzie lokalnym."""
    E = _uklad(a, d, f)
    dp = d - (d @ E[0]) * E[0]; fp = f - (f @ E[0]) * E[0]
    na = float(np.linalg.norm(a))
    return dict(E=E, norma=na, kat=math.degrees(math.atan2(float(np.linalg.norm(dp)), na + float(d @ E[0]))),
                dl=float(np.linalg.norm(a + d) - na), d3=E @ d, f3=E @ f, s3=E @ (-WD * a) if sprezyna else np.zeros(3),
                cos=float(dp @ fp / max(np.linalg.norm(dp) * np.linalg.norm(fp), 1e-30)),
                cos_pelny=float(d @ f / max(np.linalg.norm(d) * np.linalg.norm(f), 1e-30)))


class Liczydlo:
    def __init__(self, h, urzadzenie='cuda'):
        import torch
        self.torch = torch
        self.h = h
        self.dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
        self.M = LD.modul_modelu()
        self.net = self.M.MalyViT().to(self.dev).eval()
        self._wagi = {}
        self._grad = {}
        self._zbior = None
        self.blad_zbioru = ''
        rgb = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255)
        self.sonda = (rgb, [h.gt] if h.gt is not None else None)

    def wagi(self, i):
        """Wagi MLP bloku 6 w checkpoincie i: W1 (768, 192), b1 (768,), W2 (192, 768) jako float64 numpy (pamięć 8)."""
        i = int(i)
        if i not in self._wagi:
            st = self.torch.load(Path(self.h.wagi) / f'k{int(self.h.kroki[i]):05d}.pt', map_location='cpu', weights_only=True)
            st = st.get('net', st)
            p = 'bloki.5.mlp.'
            self._wagi[i] = dict(st=st, W1=st[p + 'fc1.weight'].double().numpy(), b1=st[p + 'fc1.bias'].double().numpy(),
                                 W2=st[p + 'fc2.weight'].double().numpy())
            if len(self._wagi) > 8:
                self._wagi.pop(next(iter(self._wagi)))
        return self._wagi[i]

    def _paczka_zbioru(self):
        """64 klatki zbioru uczącego (stałe losowanie) i ich ramki; raz na historię."""
        if self._zbior is None:
            zb = LD.zbior_uczacy()
            nr = np.random.default_rng(0).choice(len(zb), N_ZBIOR, replace=False)
            X, R = [], []
            for n in nr:
                im, _ = LD.wczytaj_zdjecie(f'{LD.PREFIKS_UCZACY}{int(n)}')
                X.append(im); R.append(zb[int(n)]['ramka'])
            x = self.torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).float().div(255)
            self._zbior = (x, R)
        return self._zbior

    def gradienty(self, i, zrodlo):
        """Gradienty straty po wagach MLP bloku 6 (i dla 'sonda' także z, u, LN₂ x̃, δ tokenów zdjęcia). Pamięć ostatnich 16."""
        torch = self.torch
        klucz = (int(i), zrodlo)
        if klucz in self._grad:
            return self._grad[klucz]
        if zrodlo == 'zbior':
            x, ramki = self._paczka_zbioru()
        else:
            x, ramki = self.sonda[0], self.sonda[1]
            if ramki is None:
                raise ValueError('zdjęcie bez ramki: brak pola celu (wybierz klatkę zbioru uczącego albo źródło „zbiór”)')
        self.net.load_state_dict(self.wagi(i)['st'], strict=True)
        self.net.zero_grad(set_to_none=True)
        blk = self.net.bloki[-1]; pam = {}

        def wyjscie(m, a, o):                                         # hak nic nie zwraca (inaczej podmieniłby wyjście bloku)
            o.retain_grad(); pam['x'] = o
        haki = [blk.mlp.fc1.register_forward_hook(lambda m, a, o: pam.__setitem__('u', o)),
                blk.mlp.act.register_forward_hook(lambda m, a, o: pam.__setitem__('z', o)),
                blk.norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('xn', o)),
                blk.register_forward_hook(wyjscie)]
        try:
            heat, ltrb, poz = self.M.cele([[r] for r in ramki], LD.H, LD.W)
            o, _ = self.net(x.to(self.dev))
            L, _ = self.M.strata(o, heat.to(self.dev), ltrb.to(self.dev), poz.to(self.dev))
            L.backward()
        finally:
            for k in haki:
                k.remove()
        m = blk.mlp
        cpu = lambda t: t.detach().double().cpu().numpy()
        g = dict(gW1=cpu(m.fc1.weight.grad), gb1=cpu(m.fc1.bias.grad), gW2=cpu(m.fc2.weight.grad), L=float(L))
        if zrodlo == 'sonda':
            g.update(u=cpu(pam['u'][0]), z=cpu(pam['z'][0]), xn=cpu(pam['xn'][0]), F=-cpu(pam['x'].grad[0]))
        if len(self._grad) >= 16:
            self._grad.pop(next(iter(self._grad)))
        self._grad[klucz] = g
        return g

    def policz(self, i, c, zrodlo='sonda', tok=0, n_linek=20):
        """Reflektor c w checkpoincie i: wykonany ruch do następnego checkpointu, odrzut ze źródła, linki tokenów zdjęcia."""
        h = self.h; i, c, tok = int(i), int(c), int(tok)
        j = i + 1 if i + 1 < h.K else i - 1
        A, B = self.wagi(i), self.wagi(j)
        zn = 1.0 if j > i else -1.0                                  # ostatni checkpoint: ruch, który do niego doprowadził
        dk = abs(int(h.kroki[j]) - int(h.kroki[i]))
        v, w, b = A['W2'][:, c], A['W1'][c], float(A['b1'][c])
        dv, dw, db = zn * (B['W2'][:, c] - v), zn * (B['W1'][c] - w), zn * float(B['b1'][c] - b)
        g = self.gradienty(i, zrodlo)
        fv, fw, fb = -g['gW2'][:, c], -g['gW1'][c], -float(g['gb1'][c])
        igla, cel = _czesc(v, dv, fv, True), _czesc(w, dw, fw, True)
        # linki: wkład każdego tokenu zdjęcia (sonda) w odrzut igły i celownika, w układach lokalnych
        s = g if zrodlo == 'sonda' else self.gradienty(i, 'sonda') if h.gt is not None else None
        linki = None
        if s is not None:
            u = s['u'][:, c]
            gp = 0.5 * (1 + np.vectorize(math.erf)(u / math.sqrt(2))) + u * np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
            Cv = s['z'][:, c, None] * s['F']                              # (600, 192) siła linki tokenu na igłę
            kw = gp * (s['F'] @ v)                                        # (600,) ile pchnięcie tej wiązki pomaga tokenowi
            Cw = kw[:, None] * s['xn']                                    # (600, 192) siła linki tokenu na celownik
            sv, sw = Cv @ igla['E'].T, Cw @ cel['E'].T                    # (600, 3)
            nv = np.linalg.norm(sv, axis=1)
            kol = list(np.argsort(-nv)[:n_linek])
            if tok not in kol:
                kol.append(tok)
            narz = h.narz.astype(bool)
            tot = Cv.sum(0)
            udz = lambda m_: float((Cv[m_].sum(0) @ tot) / max(tot @ tot, 1e-30))
            linki = dict(tokeny=np.array(kol), v3=sv[kol], w3=sw[kol], z=s['z'][kol, c], narz=narz[kol],
                         udzial_narz=udz(narz), udzial_tla=udz(~narz),
                         konflikt=float(np.linalg.norm(tot) / max(np.linalg.norm(Cv, axis=1).sum(), 1e-30)),
                         sila_tok=float(nv[tok]), ranga_tok=int((nv > nv[tok]).sum()) + 1,
                         jasnosc_tok=float(s['z'][tok, c]))
            if zrodlo != 'sonda':                                         # zgodność odrzutu zdjęcia z odrzutem paczki
                linki['cos_sonda_zbior'] = float(tot @ fv / max(np.linalg.norm(tot) * np.linalg.norm(fv), 1e-30))
        return dict(i=i, j=j, c=c, tok=tok, zrodlo=zrodlo, krok=int(h.kroki[i]), dk=dk, ostatni=j < i, L=g['L'],
                    igla=igla, cel=cel, prog=dict(b=b, db=db, fb=fb, cos=float(np.sign(db * fb)) if db * fb != 0 else 0.0),
                    linki=linki, lr=5e-4 * min(1.0, (int(h.kroki[i]) + 1) / 300))


def opis(d):
    pl = lambda x, k=2: f'{x:.{k}f}'.replace('.', ',')
    ig, ce, pr, li = d['igla'], d['cel'], d['prog'], d['linki']
    zr = 'to zdjęcie (sonda)' if d['zrodlo'] == 'sonda' else f'paczka {N_ZBIOR} klatek zbioru uczącego'
    t = (f'Reflektor {d["c"]}, krok {d["krok"]} → {"poprzedni" if d["ostatni"] else "następny"} checkpoint ({d["dk"]} kroków).  '
         f'Odrzut: {zr}.  Igła: obrót {pl(ig["kat"], 3)}°, moc {ig["dl"]:+.2e}; zgodność odrzutu z obrotem cos {pl(ig["cos"])}.  '
         f'Celownik: obrót {pl(ce["kat"], 3)}°, cos {pl(ce["cos"])}.  Próg: {pl(pr["b"], 3)} ({pr["db"]:+.1e}).')
    if li is not None:
        t += (f'  Odrzut zdjęcia: narzędzie {li["udzial_narz"]:+.0%}, tło {li["udzial_tla"]:+.0%}, zgodność linek '
              f'{pl(li["konflikt"])} (1 = ciągną razem, 0 = znoszą się).  Token {d["tok"]}: jasność {pl(li["jasnosc_tok"])}, '
              f'siła nr {li["ranga_tok"]} z 600.')
        if 'cos_sonda_zbior' in li:
            t += f'  Zgodność odrzutu zdjęcia z paczką: cos {pl(li["cos_sonda_zbior"])}.'
    return t
