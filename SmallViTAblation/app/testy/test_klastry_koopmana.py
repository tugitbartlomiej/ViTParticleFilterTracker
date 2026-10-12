"""Dobór osi Koopmana pod klastry: syntetyczne funkcje własne o znanym podziale (bez karty graficznej OpenGL)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
import klastry_koopmana as K  # noqa: E402


def _uklad(M=1500, ziarno=0):
    """Pięć rzeczywistych modów, H = jednostkowe (ψ_j = h_j). Czyste klastry są tylko w parze (mod 3, mod 1);
    mod 0 ma ciężkie ogony (dużo outlierów), mod 2 i 4 to szum gaussowski."""
    rng = np.random.default_rng(ziarno)
    lab = rng.integers(0, 3, M)
    srodki = np.array([[0, 0], [1.2, 3.6], [3.6, 1.2]])      # w każdej osi osobno grupy się zlewają
    H = np.empty((M, 5))
    H[:, 0] = rng.standard_cauchy(M)
    H[:, [3, 1]] = srodki[lab] + rng.normal(size=(M, 2)) * .4
    H[:, 2] = rng.normal(size=M)
    H[:, 4] = rng.normal(size=M)
    vals = np.array([.99, .95, .9, .8, .5], complex)
    return H, np.eye(5, dtype=complex), np.ones(5), 1e-12, vals, lab


def test_kandydaci_bez_sprzezonych_i_zdegenerowanych():
    vals = np.array([1.0, .9 + .1j, .9 - .1j, .5, .2])
    k = K.kandydaci(vals, zdeg={4})
    sk = [tuple(o['skladowe']) for o in k]
    assert ((1, 're'),) in sk and ((1, 'im'),) in sk and ((1, 're'), (1, 'im')) in sk
    assert all(j not in (2, 4) for o in k for j, _ in o['skladowe'])


def test_wybiera_pare_z_klastrami():
    H, W, war, prog, vals, lab = _uklad()
    w = K.dobierz(H, W, war, prog, vals, z_tez=False, urzadzenie='cpu')
    xy = {w['osie']['x']['skladowe'][0][0], w['osie']['y']['skladowe'][0][0]}
    assert xy == {1, 3}, w['ranking']
    o = w['ocena']
    assert o['k'] == 3 and o['outliery'] < .03 and o['sylwetka'] > .6
    # etykiety klastrów zgodne z prawdziwym podziałem (bez outlierów)
    e = o['etykiety']
    ok = e >= 0
    zgodne = sum(np.bincount(lab[ok & (e == c)], minlength=3).max() for c in range(3))
    assert zgodne / ok.sum() > .98


def test_os_z_i_porownanie_z_obecnymi():
    H, W, war, prog, vals, _ = _uklad(ziarno=1)
    obecne = dict(x=dict(skladowe=[(0, 're')], kombinacja=0), y=dict(skladowe=[(2, 're')], kombinacja=0),
                  z=dict(skladowe=[(4, 're')], kombinacja=0))
    w = K.dobierz(H, W, war, prog, vals, z_tez=True, osie_obecne=obecne, urzadzenie='cpu')
    xy = {w['osie']['x']['skladowe'][0][0], w['osie']['y']['skladowe'][0][0]}
    assert xy == {1, 3}
    assert w['osie']['z']['skladowe'][0][0] not in xy
    assert w['ocena']['wynik'] > w['ocena_obecnych']['wynik'] and w['ocena_xy']['k'] == 3
    assert 'Osie pod klastry' in K.opis(w, vals)


def test_jedna_chmura_to_nie_klastry():
    """Wydłużona gaussowska chmura: k-means ją podzieli, ale nadwyżka nad układem zerowym jest bliska zera,
    a prawdziwe klastry mają nadwyżkę wyraźnie dodatnią; model ma środki i promienie dla shadera."""
    rng = np.random.default_rng(3)
    M = 1500
    H = rng.normal(size=(M, 3)) @ np.array([[3, 1, 0], [0, 1, 0], [0, .2, .5]])
    vals = np.array([.99, .9, .8], complex)
    w = K.dobierz(H, np.eye(3, dtype=complex), np.ones(3), 1e-12, vals, urzadzenie='cpu')
    assert not w['ocena_xy']['istotne'] and w['ocena_xy']['nadwyzka'] < .15 and 'UWAGA' in K.opis(w, vals)
    H, W, war, prog, vals, _ = _uklad(ziarno=2)
    w = K.dobierz(H, W, war, prog, vals, urzadzenie='cpu')
    assert w['ocena_xy']['istotne'] and w['ocena_xy']['nadwyzka'] > .3
    m = w['model']
    assert m['wym'] == 3 and m['srodki'].shape == m['sigma'].shape == (len(m['logpi']), 3)
    # model odtwarza etykiety z oceny regułą shadera: max log π - Σ log σ - d²/2, outlier gdy d > próg albo mały
    psi = K.funkcje_wlasne(H, W, war, prog)
    Z = (np.stack([K.wspolrzedna(psi, m['osie'][a]) for a in 'xyz'], 1) - m['mediana']) / m['skala']
    d = np.sqrt((((Z[:, None, :] - m['srodki'][None]) / m['sigma'][None]) ** 2).sum(2))
    c = (m['logpi'][None] - np.log(m['sigma']).sum(1)[None] - .5 * d * d).argmax(1)
    dc = d[np.arange(len(d)), c]
    lab = np.where(~m['duzy'][c] | (dc > m['prog']), -1, c)
    assert (lab == w['ocena']['etykiety']).mean() > .995


def test_caly_ruch_nie_myli_dryfu_z_klastrami():
    """Dwa zapisy: w modach 0, 1 cała chmura przesuwa się między zapisami (dryf w czasie, w każdym zapisie jedna
    chmura), w modach 2, 3 cząsteczki tworzą 2 grupy w obu zapisach. Z grupami wygrywają klastry, nie dryf."""
    rng = np.random.default_rng(5)
    M = 800
    lab = rng.integers(0, 2, M)
    zapisy = []
    for t in range(2):
        H = rng.normal(size=(M, 4)) * .5
        H[:, :2] += 12.0 * t                                    # dryf całej chmury
        H[:, 2] += 3.0 * lab                                    # grupy cząsteczek (te same w obu zapisach)
        zapisy.append(H)
    H = np.concatenate(zapisy)
    grupy = np.repeat([0, 1], M)
    vals = np.array([.99, .95, .9, .8], complex)
    arg = (H, np.eye(4, dtype=complex), np.ones(4), 1e-12, vals)
    z = K.dobierz(*arg, z_tez=False, urzadzenie='cpu', grupy=grupy, grupa_modelu=1)
    assert 2 in {z['osie']['x']['skladowe'][0][0], z['osie']['y']['skladowe'][0][0]}
    assert z['ocena']['k_zapisy'] == [2, 2] and len(z['ocena']['etykiety']) == M and z['zapisy'] == 2


def test_brak_klastrow_osie_bez_zmian():
    """Niezależny szum gaussowski we wszystkich modach: BIC woli jedną chmurę, osie zostają, opis to mówi."""
    rng = np.random.default_rng(7)
    H = rng.normal(size=(1200, 3))
    vals = np.array([.99, .9, .8], complex)
    w = K.dobierz(H, np.eye(3, dtype=complex), np.ones(3), 1e-12, vals, urzadzenie='cpu')
    assert (w['model'] is None and w['osie'] is None and 'bez zmian' in K.opis(w, vals)) or \
        not w['ocena_xy']['istotne']
