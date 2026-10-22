"""Podział laser_app/ui/projections_view.py: ProjectionWindows zostaje w nim (dane, czas, budowa), reszta metod w mixinach
pakietu laser_app/ui/projection_windows/; wątki, sceny i stałe w osobnych modułach tego pakietu."""
import ast
import builtins
import sys
from pathlib import Path

APP = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp')
SRC = APP / 'laser_app' / 'ui' / 'projections_view.py'
PK = APP / 'laser_app' / 'ui' / 'projection_windows'
s = SRC.read_text(encoding='utf-8').replace('\r\n', '\n')
L = s.split('\n')
T = ast.parse(s)
STD = set(sys.stdlib_module_names)


def seg(n):
    a = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])])
    while a > 1 and L[a - 2].strip().startswith('#'):
        a -= 1
    return '\n'.join(L[a - 1:n.end_lineno])


DOSTAWCY = {}
for n in T.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == '__future__'):
        for a in n.names:
            loc = a.asname or a.name.split('.')[0]
            DOSTAWCY[loc] = ('from', n.module, a.name, a.asname) if isinstance(n, ast.ImportFrom) else ('import', a.name, None, a.asname)

klasy = {n.name: n for n in T.body if isinstance(n, ast.ClassDef)}
WSPOLNE = [n for n in T.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
NOWE = {}
for n in WSPOLNE:
    for x in ([n.name] if isinstance(n, ast.FunctionDef) else [y.id for t in n.targets for y in ast.walk(t) if isinstance(y, ast.Name)]):
        NOWE[x] = 'laser_app.ui.projection_windows.shared'
for k in ('ProjectionWorker', 'MetricsWorker', 'FrameCWorker'):
    NOWE[k] = 'laser_app.ui.projection_windows.workers'
for k in ('LocalSections', 'Scene'):
    NOWE[k] = 'laser_app.ui.projection_windows.scene'

PW = klasy['ProjectionWindows']
metody = {m.name: m for m in PW.body if isinstance(m, ast.FunctionDef)}
MIXINY = {
    'photo_lda.py': ('PhotoLDAMixin', 'Okienka R0 (zdjęcie z patchami, ramki GT i modelu) oraz R9–R10 (separacja narzędzia od tła, LDA na żywo).',
                     ['_buduj_zdjecie', '_krawedzie', '_rysuj_zdjecie', '_buduj_lda', '_wyklucz', '_rysuj_lda', '_opis_zr']),
    'metrics.py': ('MetricsMixin', 'Okienko R11: miary separacji w czasie uczenia (liczone w tle, zapisywane obok stanów 192D).',
                   ['_buduj_miary', '_rysuj_miary', '_miary_gotowe']),
    'knn.py': ('KnnMixin', 'Okienko R12: kNN na chmurze 3D (wspólna baza PCA albo wyrównana Procrustesem, interpolacja, metryka).',
               ['_buduj_knn', '_osie12', '_sasiedzi12', '_rysuj_knn']),
    'panel.py': ('PanelMixin', 'Okienko R8: panel bez rzutu (wierność widoków w czasie, separacja, cos, pary wiązek, widmo J).',
                 ['_buduj_panel', '_panel_wiernosc', '_rysuj_panel', '_panel_widmo']),
    'frames.py': ('FramesMixin', 'Okienka R4 (rama siły B) i R5 (rama Schura C, liczona w tle).',
                  ['_rama_B', '_rysuj_B', '_klucz_C', '_zlec_C', '_C_gotowe', '_rysuj_C']),
    'bases.py': ('BasesMixin', 'Bazy rzutu okienek (A, A_t, S, W, D, L, Fano) i etykiety klas (narzędzie, tło, łaty mieszane).',
                 ['_baza_okienka', '_grupa', '_etykieta', '_klucz_etyk', '_etykieta_osi', '_opis_etyk', '_baza_L', '_baza_D']),
    'drawing.py': ('DrawingMixin', 'Rysowanie okienek R1–R7 (cząsteczki, ślady, siły, wiązki, tory zerowe) i ich opisy (wierność, separacja, ruch).',
                   ['_rysuj', '_rysuj_globalny', '_kolory_wiazek', '_wiazki_lokalne', '_wklady', '_zerowe', '_tory_zerowe', '_opis',
                    '_opis_separacji_podst', '_opis_separacji', '_opis_ruchu', '_ekran', '_wygladz', '_tor_wzgl']),
}
W_RDZENIU = [m for m in metody if not any(m in v[2] for v in MIXINY.values())]
KLASY_MIX = {f: kl for f, (kl, _, _) in MIXINY.items()}
for f, kl in KLASY_MIX.items():
    NOWE[kl] = 'laser_app.ui.projection_windows.' + f[:-3]


def importy(kod, wlasny):
    t_ = ast.parse(kod)
    modulowe = set()
    for n in t_.body:
        if hasattr(n, 'name'):
            modulowe.add(n.name)
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                modulowe |= {x.id for x in ast.walk(tg) if isinstance(x, ast.Name)}
    uzyte = {n.id for n in ast.walk(t_) if isinstance(n, ast.Name)}
    potrzebne = uzyte - modulowe - set(dir(builtins)) - {'self', 'cls'}
    std, obce, app, zw = {}, {}, {}, {'std': [], 'obce': [], 'app': []}
    for nm in sorted(potrzebne):
        if nm in DOSTAWCY:
            k = DOSTAWCY[nm]
            mod = k[1]
            gr = 'app' if mod.startswith('laser_app') else ('std' if mod.split('.')[0] in STD else 'obce')
            if k[0] == 'import':
                zw[gr].append(f'import {k[1]}' + (f' as {k[3]}' if k[3] else ''))
            else:
                {'std': std, 'obce': obce, 'app': app}[gr].setdefault(mod, []).append(k[2] + (f' as {k[3]}' if k[3] else ''))
        elif nm in NOWE and NOWE[nm] != wlasny:
            app.setdefault(NOWE[nm], []).append(nm)

    def blok(d, z):
        out = sorted(z, key=str.lower)
        for m, v in sorted(d.items()):
            el = sorted(dict.fromkeys(v), key=lambda x: x.lstrip('_').lower())
            linia = f'from {m} import {", ".join(el)}'
            if len(linia) > 125:
                wc = ' ' * len(f'from {m} import ('); akt = f'from {m} import ('; w = []
                for c in el:
                    if len(akt) + len(c) + 2 > 125:
                        w.append(akt.rstrip()); akt = wc
                    akt += c + ', '
                w.append(akt.rstrip(', ') + ')'); linia = '\n'.join(w)
            out.append(linia)
        return out
    return '\n\n'.join('\n'.join(b) for b in (blok(std, zw['std']), blok(obce, zw['obce']), blok(app, zw['app'])) if b)


def zapisz(p, doc, cialo, mod):
    tekst = f'"""{doc}"""\nfrom __future__ import annotations\n\n{importy(cialo, mod)}\n\n\n{cialo.rstrip()}\n'
    ast.parse(tekst)
    p.write_text(tekst, encoding='utf-8', newline='\n')
    print(p.relative_to(APP), len(tekst.split('\n')))


def main():
    PK.mkdir(exist_ok=True)
    (PK / '__init__.py').write_text('"""Części okienek rzutów R0–R12 (ProjectionWindows w laser_app/ui/projections_view.py): wątki, sceny 3D, stałe\n'
                                    'i mixiny poszczególnych okienek."""\n', encoding='utf-8', newline='\n')
    zapisz(PK / 'shared.py', 'Stałe okienek rzutów: kolory, lista okienek R0–R12, formatowanie liczb, strzałki.',
           '\n'.join(seg(n) for n in WSPOLNE), 'laser_app.ui.projection_windows.shared')
    zapisz(PK / 'workers.py', 'Wątki okienek rzutów: stany 192D i bazy (ProjectionWorker), miary w czasie (MetricsWorker), rama Schura (FrameCWorker).',
           '\n\n\n'.join(seg(klasy[k]) for k in ('ProjectionWorker', 'MetricsWorker', 'FrameCWorker')), 'laser_app.ui.projection_windows.workers')
    zapisz(PK / 'scene.py', 'Scena 3D jednego rzutu (Scene) i lokalne sekcje okienek (LocalSections).',
           '\n\n\n'.join(seg(klasy[k]) for k in ('LocalSections', 'Scene')), 'laser_app.ui.projection_windows.scene')
    for f, (kl, opis, ms) in MIXINY.items():
        cialo = f'class {kl}:\n    """{opis}\n    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""\n\n' + \
                '\n\n'.join(seg(metody[m]) for m in ms)
        zapisz(PK / f, opis, cialo, 'laser_app.ui.projection_windows.' + f[:-3])
    doc_klasy = ast.get_docstring(PW)
    rdzen = [seg(metody[m]) for m in W_RDZENIU]
    baza = ', '.join(KLASY_MIX.values()) + ', QWidget'
    cialo = f'class ProjectionWindows({baza}):\n    """{doc_klasy}"""\n\n' + '\n\n'.join(rdzen)
    zapisz(SRC, ast.get_docstring(T), cialo, 'laser_app.ui.projections_view')


if __name__ == '__main__':
    main()
