"""Testy kryteriów akceptacji stanowiska korekcji latentu (SPECYFIKACJA, sekcja 10).

Uruchomienie z katalogu SmallViTAblation:  F:/Python/python.exe experiments/stanowisko_korekcji_latentu_gui/testy/test_stanowisko.py
Pomiary testowe trafiają do katalogu tymczasowego (KOREKTA_WYJSCIE), raport do testy/RAPORT_TESTOW.json.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

TU = Path(__file__).resolve().parent
APP = TU.parents[2] / 'app'
os.environ['KOREKTA_WYJSCIE'] = tempfile.mkdtemp(prefix='korekta_testy_')
sys.path.insert(0, str(APP))

import numpy as np
import torch

import korekta_latentu as KL
import dynamika_korekty as DK
import rownanie_czastki as RC

WYNIKI = {}


def test(nazwa):
    def dek(f):
        t = time.time()
        try:
            out = f()
            WYNIKI[nazwa] = dict(ok=True, s=round(time.time() - t, 2), **(out or {}))
            print(f'OK   {nazwa}  {out or ""}')
        except AssertionError as e:
            WYNIKI[nazwa] = dict(ok=False, blad=str(e)); print(f'FAIL {nazwa}: {e}')
        except Exception as e:
            import traceback
            WYNIKI[nazwa] = dict(ok=False, blad=traceback.format_exc()[-1500:]); print(f'ERR  {nazwa}: {e}')
        return f
    return dek


# ---------------------------------------------------------------------------------------------------- ewaluator
@test('IoU: identyczne 1, rozłączne 0')
def _():
    b = np.array([10., 10, 50, 40])
    assert abs(KL.iou(b[None], b)[0] - 1) < 1e-6
    assert KL.iou(np.array([[100., 100, 120, 120]]), b)[0] == 0


@test('AP: idealna detekcja, brak detekcji, duplikaty, wiele GT, obraz bez obiektu, zbiór bez GT')
def _():
    g = [np.array([[10., 10, 50, 40]]), np.array([[100., 100, 150, 140]])]
    det_ok = [(g[0], np.array([.9])), (g[1], np.array([.8]))]
    assert KL.ap(det_ok, g) == 1.0
    assert KL.ap([(np.zeros((0, 4)), np.zeros(0))] * 2, g) == 0.0                     # brak detekcji
    dup = [(np.stack([g[0][0], g[0][0]]), np.array([.9, .85])), (g[1], np.array([.8]))]
    a_dup = KL.ap(dup, g)                     # duplikat (fałszywa) przed trafieniem innego obrazu obniża AP
    assert abs(a_dup - (51 + 50 * 2 / 3) / 101) < 1e-9, a_dup
    dup2 = [(np.stack([g[0][0], g[0][0]]), np.array([.9, .7])), (g[1], np.array([.8]))]
    assert KL.ap(dup2, g) == 1.0                                                        # duplikat na końcu rankingu
    zly = [(np.stack([np.array([200., 200, 220, 220]), g[0][0]]), np.array([.95, .9])), (g[1], np.array([.8]))]
    a_zly = KL.ap(zly, g)
    assert a_zly < 1.0, a_zly
    multi = [np.array([[10., 10, 50, 40], [100., 100, 150, 140]])]
    assert KL.ap([(multi[0], np.array([.9, .8]))], multi) == 1.0                       # dwa GT na jednym obrazie
    assert abs(KL.ap([(multi[0][:1], np.array([.9]))], multi) - 51 / 101) < 1e-9       # połowa recall
    bez = [g[0], np.zeros((0, 4))]
    a_bez = KL.ap([(g[0], np.array([.5])), (np.array([[1., 1, 9, 9]]), np.array([.9]))], bez)
    assert a_bez < 1.0                                                                  # detekcja na obrazie bez obiektu
    assert np.isnan(KL.ap([(g[0], np.array([.9]))], [np.zeros((0, 4))]))               # zbiór bez GT: nieokreślone
    return dict(ap_duplikat_nizej=a_dup, ap_fp_wyzej=a_zly, ap_obraz_bez_obiektu=a_bez)


@test('protokół obrazu: TP, FP, pominięcia, ramka wybrana bez GT')
def _():
    gt = np.array([[10., 10, 50, 40]])
    bx = np.array([[200., 200, 220, 220], [10, 10, 50, 40], [12, 10, 50, 40]])
    m = KL.ocena_obrazu(bx, np.array([.9, .8, .6]), gt, tau=.5)
    assert m['ramka_top'] == bx[0].tolist() and m['iou_top'] == 0          # najpewniejsza = predykcja, nie GT
    assert abs(m['iou_najlepsze_diag'] - 1.0) < 1e-6                       # diagnostyczna osobno
    assert (m['tp'], m['fp'], m['pominiete']) == (1, 2, 0), m
    m0 = KL.ocena_obrazu(bx, np.array([.9, .8, .6]), np.zeros((0, 4)))
    assert m0['iou_top'] is None and m0['fp'] == 3


@test('błędna adnotacja daje jasny błąd')
def _():
    for zla in ([[10, 10, 5, 40]], [[600, 10, 700, 40]], [[np.nan, 1, 2, 3]]):
        try:
            KL.sprawdz_ramki(zla)
        except ValueError:
            continue
        raise AssertionError(f'nie odrzucono {zla}')


@test('AP historyczne z DETECTIONS_FULL400 = RESULT.json')
def _():
    a = KL.ap_historyczne()
    for n in ('baseline', 'optimized', 'random'):
        for k in ('AP50', 'AP50_95'):
            assert abs(a['policzone'][n][k] - a['zapisane'][n][k]) < 1e-6, (n, k)
    return {n: round(a['policzone'][n]['AP50'], 6) for n in a['policzone']}


# ---------------------------------------------------------------------------------------------------- model
ST = {}


def st(dev=None):
    k = dev or 'domyslne'
    if k not in ST:
        ST[k] = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k1000'], dev)
    return ST[k]


@test('zerowa interwencja = pełny forward (każdy tryb), wagi bez zmian')
def _():
    s = st(); rows = KL.manifest(); bl = 0.
    for i in (0, 199, 399):
        rgb, ram = KL.obraz_manifestu(rows[i])
        baza = s.przebieg(rgb)
        assert baza['blad_ogona'] <= 2e-5
        for tryb in KL.TRYBY:
            sp = s.przygotuj_spec(dict(tryb=tryb, kanaly=KL.kanaly_j128(), tokeny=0), ram)
            with torch.no_grad():
                _, _, o = s.wyjscie(torch.zeros(s.wymiar(sp), device=s.dev), baza, sp)
            bl = max(bl, float((o[0] - baza['o']).abs().max()))
    assert bl <= 2e-5, bl
    assert s.zgodny()
    return dict(max_blad=bl)


@test('tryby wejścia MLP = prawdziwa sieć z hakiem (gałąź: wejście norm2; wspólne: residual)')
def _():
    s = st(); rgb, ram = KL.obraz_manifestu(KL.manifest()[7])
    baza = s.przebieg(rgb)
    g = torch.Generator().manual_seed(0)
    for tryb in ('wejscie_galaz', 'wejscie_wspolne'):
        sp = s.przygotuj_spec(dict(tryb=tryb, tokeny=2), ram)
        e = (torch.randn(s.wymiar(sp), generator=g) * .3).to(s.dev)
        with torch.no_grad():
            _, _, o = s.wyjscie(e, baza, sp)
        E = torch.zeros(600, 192, device=s.dev); E[sp['_T']] = e
        b = s.blok
        if tryb == 'wejscie_galaz':
            h = b.norm2.register_forward_pre_hook(lambda m, a: (a[0] + E[None],))
        else:
            h = b.drop_path1.register_forward_hook(lambda m, a, out: out + E[None])
        try:
            with torch.no_grad():
                x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(s.dev) / 255
                o2, _ = s.net(x)
        finally:
            h.remove()
        assert float((o - o2).abs().max()) < 2e-5, (tryb, float((o - o2).abs().max()))
    assert s.zgodny()


@test('tryby latentu = prawdziwa sieć z hakiem na GELU')
def _():
    s = st(); rgb, ram = KL.obraz_manifestu(KL.manifest()[3])
    baza = s.przebieg(rgb)
    sp = s.przygotuj_spec(dict(tryb='swobodne', kanaly=KL.kanaly_j128(), tokeny=3), ram)
    d = torch.randn(s.wymiar(sp), generator=torch.Generator().manual_seed(1)).to(s.dev)
    with torch.no_grad():
        zp, _, o = s.wyjscie(d, baza, sp)
    h = s.blok.mlp.act.register_forward_hook(lambda m, a, out: zp)
    try:
        with torch.no_grad():
            o2, _ = s.net(torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(s.dev) / 255)
    finally:
        h.remove()
    assert float((o - o2).abs().max()) < 2e-5


@test('mała perturbacja z jądra fc2 nie zmienia wyjścia końcówki')
def _():
    s = st(); rgb, _ = KL.obraz_manifestu(KL.manifest()[11])
    baza = s.przebieg(rgb)
    assert s.fc2['rzad'] == 192 and s.fc2['jadro'] == 576, s.fc2['rzad']
    g = torch.randn(600, 768, generator=torch.Generator().manual_seed(2)).to(s.dev)
    inv = g - (g @ s.Vr) @ s.Vr.T
    inv = inv / inv.norm() * baza['z'].norm()                 # ta sama norma co cały latent obrazu
    with torch.no_grad():
        o1 = s.ogon(baza['z'][None], baza['r'][None]); o2 = s.ogon((baza['z'] + inv)[None], baza['r'][None])
        vis = (g @ s.Vr) @ s.Vr.T; vis = vis / vis.norm() * baza['z'].norm()
        o3 = s.ogon((baza['z'] + vis)[None], baza['r'][None])
    dj, dv = float((o2 - o1).abs().max()), float((o3 - o1).abs().max())
    assert dj < 1e-3 and dv > 100 * dj, (dj, dv)
    pf = s.podzial_fc2(inv)
    assert pf['udzial_niewidocznej'] > .999
    assert s.podzial_fc2(torch.zeros_like(inv))['udzial_niewidocznej'] is None      # zero: udział nieokreślony
    return dict(zmiana_od_jadra=dj, zmiana_od_przestrzeni_wierszowej=dv, rzad=s.fc2['rzad'], tolerancja=s.fc2['tol'])


@test('korekta: ograniczenia, nośnik, kontrole o tej samej normie, zapis, przerwanie, ponowne wczytanie')
def _():
    s = st(); rgb, ram = KL.obraz_manifestu(KL.manifest()[5])
    out = {}
    for tryb, norma in (('zakresy', 'brak'), ('dol_gelu', 'calosc'), ('swobodne', 'token')):
        sp = dict(tryb=tryb, kanaly=KL.kanaly_j128(), tokeny=2, cel='lokalizacja', norma=norma, rho=20., kroki=40,
                  lr=.3, zapis_co=5, ziarno=0)
        w = KL.oblicz_korekte(s, rgb, ram, sp, dict(id='test'))
        sp_ = w['sprawdzenia']
        assert w['status'] == 'complete' and sp_['zerowa_interwencja_blad'] <= 2e-5 and sp_['wagi_bez_zmian']             and sp_['poza_interwencja_bez_zmian'], sp_
        kn = np.load(Path(w['kat']) / 'KONIEC.npz')
        z = kn['z_baza']; T = kn['tokeny']; C = kn['kanaly']; d = kn['d_najlepszy']
        zp = z[T][:, C] + d
        if tryb == 'zakresy':
            assert (zp >= s.h_min.cpu().numpy()[C] - 1e-5).all() and (zp <= s.h_max.cpu().numpy()[C] + 1e-5).all()
        if tryb == 'dol_gelu':
            assert zp.min() >= KL.GELU_MIN - 1e-5 and np.linalg.norm(d) <= 20 + 1e-3
        if tryb == 'swobodne':
            assert np.linalg.norm(d, axis=1).max() <= 20 + 1e-3
        kr, D = KL.wczytaj_trajektorie(w['kat'])
        assert list(kr) == list(range(0, 41, 5)), kr                         # rzeczywiste indeksy kroków
        for c in w['kontrole']:
            assert c['norma_d'] <= c['norma_przed_rzutem'] + 1e-3            # faktyczna norma po rzucie zmierzona
        w2 = KL.wczytaj_pomiar(w['kat'])
        assert w2['status'] == 'complete' and len(w2['metryki']) == 41
        out[tryb] = dict(strata=(round(w['przed']['strata'], 4), round(w['najlepszy']['strata'], 4)))
    licz = [0]

    def przerwij():
        licz[0] += 1
        return licz[0] > 10
    w = KL.oblicz_korekte(s, rgb, ram, dict(tryb='zakresy', kanaly=KL.kanaly_j128(), tokeny=0, kroki=500, zapis_co=1),
                          dict(id='test_przerwanie'), przerwij=przerwij)
    st_ = json.loads((Path(w['kat']) / 'STATUS.json').read_text(encoding='utf-8'))
    assert w['status'] == 'cancelled' and st_['status'] == 'cancelled'
    kr, _ = KL.wczytaj_trajektorie(w['kat']); assert len(kr) >= 10      # częściowe wyniki zachowane
    assert s.zgodny()
    return out


@test('brak GPU: model i zerowa interwencja na CPU')
def _():
    s = st('cpu'); rgb, ram = KL.obraz_manifestu(KL.manifest()[0])
    b = s.przebieg(rgb)
    assert b['blad_ogona'] <= 2e-5
    w = KL.oblicz_korekte(s, rgb, ram, dict(tryb='zakresy', kanaly=KL.kanaly_j128(), tokeny=2, kroki=3), dict(id='cpu'))
    assert w['status'] == 'complete'


@test('brak checkpointu i brak etykiety: jasny błąd, bez podstawiania')
def _():
    try:
        KL.Stanowisko('F:/nie/ma/takiego.pt')
        raise AssertionError('brak błędu')
    except FileNotFoundError:
        pass
    s = st()
    try:
        s.przygotuj_spec(dict(tryb='zakresy', kanaly=KL.kanaly_j128(), tokeny=2), None)
        raise AssertionError('brak błędu')
    except ValueError:
        pass


@test('import historyczny: zgodny checkpoint bez ostrzeżeń, inny checkpoint z ostrzeżeniem')
def _():
    h = KL.import_historyczny(st(), 5)
    assert not h['ostrzezenia'], h['ostrzezenia']
    s5 = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k0500'])
    h5 = KL.import_historyczny(s5, 5)
    assert any('NIEZGODNY CHECKPOINT' in o for o in h5['ostrzezenia'])
    return dict(naruszenia_zakresow_historyczne=h['naruszenia']['ulamek'])


# ---------------------------------------------------------------------------------------------------- dynamika
@test('GLE z sum Grama = rownanie_czastki model 1')
def _():
    X = np.cumsum(np.random.default_rng(0).standard_normal((120, 30)), 0)
    m1 = RC.dopasuj(X, 1, od=6, do=80, okno=20, L=5); m2 = DK.gle_dopasuj(X, 81, L=5, okno=20, od=6)
    assert np.allclose(np.r_[m1['a'], m1['kappa'], m1['c0']], m2['w'], rtol=1e-8, atol=1e-10)


@test('prognozy przyczynowe: zmiana przyszłości nie zmienia prognozy ze startu s')
def _():
    X = np.cumsum(np.random.default_rng(1).standard_normal((100, 12)), 0)
    m = DK.gle_dopasuj(X, 60, L=3, okno=10)
    p1 = DK.gle_prognoza(X, m, [70], 5)
    X2 = X.copy(); X2[71:] += 100
    assert np.allclose(p1, DK.gle_prognoza(X2, m, [70], 5))
    r = DK.ocen_modele(X, hs=(1, 5, 60))
    assert 'brak' in r['tabela']['GLE'][60] and abs(r['tabela']['persystencja'][1]['rel'] - 1) < 1e-12


@test('model z wejściem: krok Adama z gradientów przewiduje swobodną korektę (bez rzutu) prawie dokładnie')
def _():
    s = st(); rgb, ram = KL.obraz_manifestu(KL.manifest()[5])
    w = KL.oblicz_korekte(s, rgb, ram, dict(tryb='swobodne', kanaly=KL.kanaly_j128(), tokeny=2, kroki=80, lr=.1,
                                            zapis_co=1), dict(id='wejscie'))
    _, D = KL.wczytaj_trajektorie(w['kat'], 'd'); _, G = KL.wczytaj_trajektorie(w['kat'], 'g')
    X = D.reshape(len(D), -1); U = DK.krok_adama(G.reshape(len(G), -1), .1)
    r = DK.porownaj_wejscie(X, U, Ls=(0,))
    assert r['L=0 + wejście']['rel'] < 1e-3, r
    return dict(rel_bez=r['L=0']['rel'], rel_z_wejsciem=r['L=0 + wejście']['rel'])


@test('obserwator: pełny pomiar = rzut, otwarta pętla bez obserwacji, start nie zna przyszłości')
def _():
    rng = np.random.default_rng(3)
    X = np.cumsum(rng.standard_normal((200, 10, 8)) * .1, 0) + rng.standard_normal((1, 10, 8))
    r = DK.obserwator(X, r=4, ulamek_tokenow=1.0)
    assert np.allclose(r['blad_modow']['posterior'], 0, atol=1e-6)
    r2 = DK.obserwator(X, r=4, ulamek_tokenow=.3, co_ile=4)
    assert r2['koszt']['lacznie'] < .1 and r2['obserwowalnosc']['rzad'] <= 8
    X3 = X.copy(); X3[150:] += 5
    r3 = DK.obserwator(X3, r=4, ulamek_tokenow=.3, co_ile=4)
    assert r3['do'] == 120 and np.allclose(r2['blad_modow']['posterior'][:30], r3['blad_modow']['posterior'][:30])


# ---------------------------------------------------------------------------------------------------- gałęzie
@test('gałęzie: identyczne gałęzie dają 0 w trybie deterministycznym, stan startowy i paczki zapisane')
def _():
    if not torch.cuda.is_available():
        return dict(pominiete='brak GPU')
    import galezie_uczenia as GU
    from stan_uczenia import Odtworzenie
    b = GU.bieg_gestej_historii(); od = Odtworzenie(b); od.przygotuj()
    X, R, Mk, N = GU.sondy_gestej_historii(b)
    G = GU.Galezie(od, X, R, Mk, N)
    w = GU.licz_odpowiedz(G, 300, dict(rodzaj='waga', sloty=[3], eps_slotow=[.5]), H=3, horyzonty=(1, 3),
                          przyszlosci=(None, 1), liniowosc=True)
    for h in (1, 3):
        assert float(np.max(w['podsumowanie'][h]['szum_dz']['max'])) == 0.0
        assert float(np.min(w['podsumowanie'][h]['odpowiedz_dz']['srednia'])) > 0
    kat = Path(w['kat'])
    assert (kat / 'STAN_STARTOWY.pt').exists() and (kat / 'PACZKI_PRZYSZLE.npz').exists()
    wn = GU.licz_wzrost(G, 300, skale=(1e-3,), H=3, horyzonty=(1, 3), przyszlosci=(None,), deterministyczne=False)
    return dict(odpowiedz_h3=float(np.mean(w['podsumowanie'][3]['odpowiedz_dz']['srednia'])),
                liniowosc_h1=w['podsumowanie'][1]['liniowosc']['srednia'],
                szum_bez_determinizmu_h3=wn['wyniki']['prawdziwa']['C']['d'][3])


if __name__ == '__main__':
    n_ok = sum(v['ok'] for v in WYNIKI.values())
    print(f'\n{n_ok}/{len(WYNIKI)} testów OK')
    (TU / 'RAPORT_TESTOW.json').write_text(json.dumps(dict(
        czas=time.strftime('%Y-%m-%d %H:%M:%S'), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        wyniki=WYNIKI), indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    import shutil
    shutil.rmtree(os.environ['KOREKTA_WYJSCIE'], ignore_errors=True)
    sys.exit(0 if n_ok == len(WYNIKI) else 1)
