"""Podział laser_app/lasers_tab/tab.py: wątki → workers.py, widżety zakładki → widgets.py, ogólne widżety → ui/widgets.py,
stałe → shared.py, metody LasersTab → mixiny w osobnych plikach; _buduj → metody sekcji. Importy przycinane do użytych nazw."""
import ast
import builtins
import re
from pathlib import Path

APP = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp')
LT = APP / 'laser_app' / 'lasers_tab'
SRC = LT / 'tab.py'
s = SRC.read_text(encoding='utf-8').replace('\r\n', '\n')
L = s.split('\n')
T = ast.parse(s)


def seg(n):
    """Źródło węzła z dekoratorami i komentarzami tuż nad nim (wiersze 1-indeksowane)."""
    a = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])])
    while a > 1 and L[a - 2].strip().startswith('#'):
        a -= 1
    return '\n'.join(L[a - 1:n.end_lineno])


# ------------------------------------------------------------------ nagłówek: importy i stałe
importy = [n for n in T.body if isinstance(n, (ast.Import, ast.ImportFrom))]
DOSTAWCY = {}                                                         # nazwa lokalna → instrukcja importu (tekst)
for n in importy:
    if isinstance(n, ast.ImportFrom) and n.module == '__future__':
        continue
    for a in n.names:
        loc = a.asname or a.name.split('.')[0]
        if isinstance(n, ast.ImportFrom):
            DOSTAWCY[loc] = ('from', n.module, a.name, a.asname)
        else:
            DOSTAWCY[loc] = ('import', a.name, None, a.asname)

klasy = {n.name: n for n in T.body if isinstance(n, ast.ClassDef)}
funkcje = {n.name: n for n in T.body if isinstance(n, ast.FunctionDef)}
stale = [n for n in T.body if isinstance(n, ast.Assign)]

WORKERS = ['HistoryWorker', 'ImageSetWorker', 'FocalTrapWorker', 'TrapMapWorker', 'SubstepFieldWorker', 'ValleyWorker',
           'MechanicsWorker', 'TrapMeasurementsWorker', 'SubstepsWorker', 'AllPassagesWorker']
WORKER_FUN = ['plaski_podkrokow', 'statystyki_podkrokow']
TAB_WIDGETS = ['View2D', 'NetworkDiagram', 'ABSlider']
UI_WIDGETS_K = ['CollapsibleSection']
UI_WIDGETS_F = ['_staly_uklad', 'lista_liczb', '_spin']
for n in WORKERS + TAB_WIDGETS + UI_WIDGETS_K:
    assert n in klasy, n
for n in WORKER_FUN + UI_WIDGETS_F:
    assert n in funkcje, n
assert set(klasy) == set(WORKERS + TAB_WIDGETS + UI_WIDGETS_K + ['LasersTab']), set(klasy)
assert set(funkcje) == set(WORKER_FUN + UI_WIDGETS_F), set(funkcje)

# nazwy definiowane w nowych modułach (do importów między nimi)
NOWE = {}
for n in WORKERS + WORKER_FUN:
    NOWE[n] = 'laser_app.lasers_tab.workers'
for n in TAB_WIDGETS:
    NOWE[n] = 'laser_app.lasers_tab.widgets'
for n in UI_WIDGETS_K + UI_WIDGETS_F:
    NOWE[n] = 'laser_app.ui.widgets'
for n in stale:
    for t in n.targets:
        NOWE[t.id] = 'laser_app.lasers_tab.shared'

# ------------------------------------------------------------------ LasersTab: metody → mixiny
LT_ = klasy['LasersTab']
metody = {m.name: m for m in LT_.body if isinstance(m, ast.FunctionDef)}
atrybuty_klasy = [m for m in LT_.body if isinstance(m, ast.Assign)]

MIXINY = {   # plik: (klasa, opis, metody)
    'layout.py': ('LayoutMixin', 'Okienka-doki zakładki: tworzenie, widoczność (liczymy tylko to, co widać), układy domyślne i '
                                 'gotowe, sloty, zapis układu w ustawieniach, panel parametrów i okienka ustawień pod prawym klikiem.',
                  ['_z_napisem', '_okno_2d', '_dok', 'widac', '_odslonieto', '_po_odslonieciu', '_dok_sterowania', '_buduj_panel',
                   'menu_okienka', '_wykres', '_uklad_domyslny', 'showEvent', '_rozmiary_domyslne', '_uklad_gotowy', '_slot',
                   '_komunikat', 'uklad_okienek', 'ustaw_uklad_okienek', '_zastosuj_uklad', '_przywroc_uklad']),
    'data_view.py': ('DataMixin', 'Dane zakładki: wybór historii checkpointów i zdjęcia, liczenie albo wczytanie historii '
                                  '(history.policz w wątku), wykresy 6–10.',
                     ['_buduj_sek_dane', '_lista_historii', '_inny_katalog', '_lista_zdjec', '_klatki_filmu', '_plik_zdjecia', '_wybor',
                      '_po_wyborze', '_wczytaj_z_cache', '_licz', '_przerwij', '_postep', '_blad', '_gotowe', '_wykresy']),
    'image_set_view.py': ('ImageSetMixin', 'Kilka zdjęć naraz na sferze reflektorów (strona „Kilka zdjęć”, limit 1–32).',
                          ['_buduj_zestaw', '_zestaw_odswiez_liste', '_zestaw_dodaj', '_zestaw_usun', '_zestaw_czysc', '_zestaw_licz',
                           '_zestaw_gotowe', '_zestaw_aktywne', '_zestaw_idx', '_h_sfery', '_tok_sfery', '_zestaw_na_sferze']),
    'labels_view.py': ('LabelsMixin', 'Etykieta narzędzia (tryby, podgląd łat), obszary oka, wybór cząsteczek i kolory cząsteczek.',
                       ['_buduj_wybor', '_buduj_etykiety', '_etykieta_zmiana', '_opis_etykiety', '_zaznacz_obszar', '_kolory_czastek',
                        '_kolory_tryb', '_kolory_sfery']),
    'substeps_view.py': ('SubstepsMixin', 'Podkroki sieci: przejście sygnału przez 6 bloków (okienko P, pasek podkroków), zapis przejść '
                                          'na dysk, animacja i opis podkroków.',
                         ['_buduj_sek_zapis', '_buduj_przejscie', '_widac_p', '_p_widoczne', '_opcje_przejscia', '_zlec_podkroki',
                          '_pamiec_pod', '_prz_arg', '_prz_opis', '_prz_wszystkie', '_prz_baza', '_prz_przerwij', '_z_wyprzedzeniem',
                          '_podkroki_gotowe', '_plaski_podkrokow', '_zastosuj_podkroki', '_dopasuj_podkroki', '_tryb_podkrokow',
                          '_statystyki_podkrokow', '_uklad', '_kadruj_przejscie', '_plynnie', 'ustaw_podkrok', '_graj_podkroki',
                          '_klatka_podkrokow', '_opis_podkroku', '_klik_przejscie']),
    'trap_view.py': ('TrapMixin', 'Okienko 11: pułapka optyczna wybranej cząsteczki (focal_trap), jej wiązki, światło, obrys 3D, '
                                  'pole w podkrokach, gotowe pułapki z dysku i pomiar w podkrokach dla całej historii.',
                     ['_buduj_sek_pulapka', '_og_klucz_teraz', '_og_zlec', '_og_przerwij', '_og_postep', '_og_blad', '_dokladnosc',
                      '_og_lista_gotowych', '_og_wybierz_gotowa', '_og_gotowe', '_n_tok', '_og_menu_policzone', '_og_zastosuj',
                      '_og_kamera_refl', '_og_opcje', '_og_opis_obrysu', '_og_anim_tik', '_og_czas_zmiana', '_og_tytul', '_og_tryb_pod',
                      '_og_podkrok', '_pomiary_start', '_pomiary_stop', '_og_pole_podkroku', '_pole_pod_gotowe', '_og_przenikanie',
                      '_og_czas']),
    'trap_map_view.py': ('TrapMapMixin', 'Okienko 14: mapa pułapek wszystkich 600 cząsteczek (trap_map) i nakładka na sferze.',
                         ['_buduj_sek_mapa_pul', '_mp_widac', '_mp_zlec', '_mp_gotowe', '_mp_czas']),
    'valley_view.py': ('ValleyMixin', 'Okienko 12: dolina pewności i IoU wybranej cząsteczki (valley) i ta sama dolina na sferze.',
                       ['_buduj_sek_dolina', '_dol_klucz_teraz', '_dol_potrzebna', '_dol_zlec', '_dol_czas', '_dol_gotowe', '_dol_progi',
                        '_dol_wyglad', '_dol_przeksztalcenie']),
    'mechanics_view.py': ('MechanicsMixin', 'Okienko 13: mechanika wybranego reflektora (reflector_mechanics, reflector_drawing).',
                          ['_buduj_sek_mech', '_mech_zlec', '_mech_gotowe', '_mech_rysuj']),
    'playback.py': ('PlaybackMixin', 'Oś czasu: odtwarzanie, punkty A–B, tempo, synchronizacja z podkrokami, klawisze i kliknięcia '
                                     'w widoki (wybór cząsteczki).',
                    ['_ustaw_ab', '_zakres_ab', '_graj', '_dt', '_nastepny', '_klatka_sync', '_nowy_checkpoint', '_start_przejscia',
                     '_ustaw_przejscie', '_klatka', 'keyPressEvent', '_klik_wykres', '_wybierz_tok', '_klik_2d', '_klik_sfera']),
}
# atrybuty klasy (SEKCJE_OKIENEK, STRONY) do LayoutMixin
ATRYBUTY_DO = 'layout.py'

# ------------------------------------------------------------------ _buduj → metody sekcji
B = metody['_buduj']
GRANICE = [(690, '_buduj_sek_dane'), (724, '_buduj_sek_rzuty'), (728, '_buduj_sek_plaskie'), (744, '_buduj_sek_sfera'),
           (854, '_buduj_sek_zapis'), (877, '_buduj_sek_pulapka'), (1085, '_buduj_sek_mapa_pul'), (1129, '_buduj_sek_mapy'),
           (1144, '_buduj_sek_dolina'), (1194, '_buduj_sek_mech'), (1228, '_buduj_okienka')]
for g, _ in GRANICE:
    assert L[g - 1].startswith('        # '), (g, L[g - 1])
OPIS_SEKCJI = {'_buduj_sek_dane': 'Sekcja „Dane”: historia checkpointów, zakres kroków, film i zdjęcie, liczenie.',
               '_buduj_sek_rzuty': 'Sekcja rzutów z 192 wymiarów (okienka R0–R12, projections_view).',
               '_buduj_sek_plaskie': 'Sekcja okienek 1–3 (rzut płaski: ruch globalny, studnie, drgania).',
               '_buduj_sek_sfera': 'Sekcja sfery reflektorów: wiązki, koloidy, wektory, ścieżki, wygładzanie.',
               '_buduj_sek_zapis': 'Sekcja zapisu przejść przez sieć na dysk.',
               '_buduj_sek_pulapka': 'Sekcja pułapki optycznej (okienko 11).',
               '_buduj_sek_mapa_pul': 'Sekcja mapy pułapek wszystkich cząsteczek (okienko 14).',
               '_buduj_sek_mapy': 'Sekcja okienek 4–5: mapy na zdjęciu.',
               '_buduj_sek_dolina': 'Sekcja doliny pewności i IoU (okienko 12).',
               '_buduj_sek_mech': 'Sekcja mechaniki reflektora (okienko 13).',
               '_buduj_okienka': 'Pasek osi czasu, okienka-doki z widokami (1–14, sfera, R0–R12, P), panel i menu „Okienka”.'}
b_start = B.body[0].end_lineno + 1 if isinstance(B.body[0], ast.Expr) else B.body[0].lineno   # po docstringu
naglowek_b = L[B.lineno - 1:b_start - 1]                                                        # def + docstring
poczatek = L[b_start - 1:GRANICE[0][0] - 1]                                                    # układ, stereo
sekcje = {}
for (g, nazwa), (g2, _) in zip(GRANICE, GRANICE[1:] + [(B.end_lineno + 1, None)]):
    tresc = L[g - 1:g2 - 1]
    while tresc and not tresc[-1].strip():
        tresc.pop()
    sekcje[nazwa] = tresc
# _buduj_okienka korzysta z pl_ (główny układ zakładki)
sekcje['_buduj_okienka'] = [l.replace('pl_.', 'self.layout().') for l in sekcje['_buduj_okienka']]
sekcje['_buduj_okienka'] = [l.replace('self._pl = pl_', 'self._pl = self.layout()') for l in sekcje['_buduj_okienka']]
for nazwa, tresc in sekcje.items():
    assert not any(re.search(r'\bpl_\b', l) for l in tresc), nazwa


def metoda_sekcji(nazwa, tresc):
    return '\n'.join([f'    def {nazwa}(self):', f'        """{OPIS_SEKCJI[nazwa]}"""'] + tresc)


nowy_buduj = naglowek_b + poczatek + [''] + [f'        self.{n}()' for _, n in GRANICE]
nowy_buduj = '\n'.join(nowy_buduj)
# pl_ zostaje w części początkowej (stereo) tylko jako alias układu: usuwamy nieużywany alias
nowy_buduj = nowy_buduj.replace('        pl_ = lay\n', '')
SEK_W_TAB = ['_buduj_sek_rzuty', '_buduj_sek_plaskie', '_buduj_sek_sfera', '_buduj_sek_mapy', '_buduj_okienka']

# ------------------------------------------------------------------ składanie plików
def pliki_kodu():
    out = {}
    wszystkie_mixiny = {m for _, (_, _, ms) in MIXINY.items() for m in ms}
    for nazwa_m in metody:
        if nazwa_m in ('__init__', '_buduj'):
            continue
        if nazwa_m not in wszystkie_mixiny and nazwa_m not in ('_opcje', '_wybrane_refl', '_wybrane_tok', '_vmax_oswietlenia',
                                                               '_tok_narzedzia', '_refl_pokaz', 'ustaw_czas'):
            raise SystemExit(f'metoda bez miejsca: {nazwa_m}')
    for plik, (kl, opis, ms) in MIXINY.items():
        czesci = []
        if plik == ATRYBUTY_DO:
            czesci += [seg(a) for a in atrybuty_klasy]
        for m in ms:
            if m in sekcje:
                czesci.append(metoda_sekcji(m, sekcje[m]))
            else:
                czesci.append(seg(metody[m]))
        cialo = f'class {kl}:\n    """{opis}\n    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""\n\n' + '\n\n'.join(czesci)
        out[LT / plik] = (f'"""{opis}"""', cialo)
    # workers
    cialo = '\n\n\n'.join(seg(funkcje[n]) for n in WORKER_FUN) + '\n\n\n' + '\n\n\n'.join(seg(klasy[n]) for n in WORKERS)
    out[LT / 'workers.py'] = ('"""Wątki zakładki: liczenie historii, zestawu zdjęć, pułapki, mapy pułapek, pola w podkrokach, doliny, mechaniki, '
                              'pomiarów pułapki,\nprzejść przez sieć; oraz rzuty płaskie podkroków (plaski_podkrokow, statystyki_podkrokow)."""', cialo)
    cialo = '\n\n\n'.join(seg(klasy[n]) for n in TAB_WIDGETS)
    out[LT / 'widgets.py'] = ('"""Widżety zakładki: widok 2D okienek 1–3 (View2D), schemat sieci (NetworkDiagram), suwak osi czasu z punktami A–B (ABSlider)."""', cialo)
    cialo = '\n\n\n'.join([seg(klasy[n]) for n in UI_WIDGETS_K] + [seg(funkcje[n]) for n in UI_WIDGETS_F])
    out[APP / 'laser_app' / 'ui' / 'widgets.py'] = ('"""Proste widżety wspólne dla zakładki i rzutów: zwijana sekcja panelu (CollapsibleSection), pole liczbowe (_spin),\n'
                                                     'lista liczb z tekstu (lista_liczb), napis o stałym układzie (_staly_uklad)."""', cialo)
    cialo = '\n'.join(seg(n) for n in stale)
    out[LT / 'shared.py'] = ('"""Stałe zakładki: kolor tła widoków, kroków na epokę, format liczb z przecinkiem."""', cialo)
    # tab.py
    mixiny = ', '.join(kl for kl, _, _ in MIXINY.values())
    reszta = [seg(metody['__init__']), nowy_buduj] + [metoda_sekcji(n, sekcje[n]) for n in SEK_W_TAB] + \
             [seg(metody[n]) for n in ('_opcje', '_wybrane_refl', '_wybrane_tok', '_vmax_oswietlenia', '_tok_narzedzia', '_refl_pokaz',
                                       'ustaw_czas')]
    cialo = (f'class LasersTab({mixiny}, QWidget):\n    """Zakładka „Lasery i koloidy”: okno aplikacji bez menu. Metody są pogrupowane w mixinach '
             f'(pliki obok);\n    tu: budowa (sekcje panelu i okienka), wspólne opcje rysowania i ustawienie chwili osi czasu we wszystkich '
             f'widocznych okienkach."""\n\n' + '\n\n'.join(reszta))
    doc = ast.get_docstring(T)
    out[LT / 'tab.py'] = (f'"""{doc}\n"""', cialo)
    return out


def uzyte_nazwy(kod):
    t = ast.parse(kod)
    u = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Name):
            u.add(n.id)
    return u


def zdefiniowane(kod):
    t = ast.parse(kod)
    d = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            d.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            d.add(n.name)
        elif isinstance(n, ast.arg):
            d.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                d.add(a.asname or a.name.split('.')[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            d.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            d.update(n.names)
    return d


def importy_dla(kod, wlasny_modul):
    t_ = ast.parse(kod)
    modulowe = {n.name for n in t_.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    potrzebne = uzyte_nazwy(kod) - modulowe - set(dir(builtins)) - {'self', 'cls'}
    grupy, linie_nowe = {}, {}
    brak = []
    for n in sorted(potrzebne):
        if n in DOSTAWCY:
            k = DOSTAWCY[n]
            if k[0] == 'from':
                grupy.setdefault(k[1], []).append((k[2], k[3]))
            else:
                grupy.setdefault(('import', k[1], k[3]), [])
        elif n in NOWE and NOWE[n] != wlasny_modul:
            grupy.setdefault(NOWE[n], []).append((n, None))
        else:
            brak.append(n)
    linie_std, linie_obce, linie_app = [], [], []
    for mod, nazwy in sorted(grupy.items(), key=lambda x: str(x[0])):
        if isinstance(mod, tuple):
            _, m, al = mod
            linia = f'import {m}' + (f' as {al}' if al else '')
            (linie_obce if m in ('numpy', 'pyqtgraph') else linie_std).append(linia)
            continue
        el = ', '.join(sorted({(a + (f' as {b}' if b else '')) for a, b in nazwy}, key=lambda x: x.lower()))
        linia = f'from {mod} import {el}'
        if len(linia) > 125:
            czesci = el.split(', ')
            wiersze, akt = [], f'from {mod} import ('
            for c in czesci:
                if len(akt) + len(c) + 2 > 125:
                    wiersze.append(akt.rstrip()); akt = ' ' * (len(f'from {mod} import ('))
                akt += c + ', '
            wiersze.append(akt.rstrip(', ') + ')')
            linia = '\n'.join(wiersze)
        if mod.startswith('laser_app'):
            linie_app.append(linia)
        elif mod.split('.')[0] in ('PyQt6', 'numpy', 'pyqtgraph'):
            linie_obce.append(linia)
        else:
            linie_std.append(linia)
    bloki = [b for b in (sorted(linie_std), sorted(linie_obce), sorted(linie_app)) if b]
    return '\n\n'.join('\n'.join(b) for b in bloki), brak


def main():
    out = pliki_kodu()
    for p, (doc, cialo) in out.items():
        mod = 'laser_app.' + '.'.join(p.relative_to(APP / 'laser_app').with_suffix('').parts)
        imp, brak = importy_dla(cialo, mod)
        if brak:
            print('NIEZDEFINIOWANE', p.name, brak)
        tekst = f'{doc}\nfrom __future__ import annotations\n\n{imp}\n\n\n{cialo.rstrip()}\n'
        ast.parse(tekst)
        p.write_text(tekst, encoding='utf-8', newline='\r\n' if p.name == 'tab.py' else '\n')
        print(p.relative_to(APP), len(tekst.split('\n')), 'wierszy')


if __name__ == '__main__':
    main()
