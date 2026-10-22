"""Migracja LaserLatentSpaceApp do pakietu laser_app: nowe ścieżki modułów, angielskie nazwy plików, aliasów i klas.

Zmienia tylko tokeny kodu (NAME) i odwołania w napisach/komentarzach do plików i klas (`stary.py`, `stary.funkcja`, `Klasa`),
nigdy napisów GUI. Tryb --sprawdz: tylko raport kolizji. Bez flagi: zapis plików w miejscu (końce linii zachowane),
potem `git mv` wykonuje osobny krok."""
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

APP = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp')

# stary moduł → (nowy pakiet, nowa nazwa modułu, nowy alias dla `import stary as X`)
MOD = {
    'app': ('laser_app', 'main_window', 'main_window'),
    'ustawienia': ('laser_app', 'settings', 'settings'),
    'styl': ('laser_app.ui', 'style', 'style'),
    'wygoda': ('laser_app.ui', 'input_comfort', 'input_comfort'),
    'okna': ('laser_app.ui', 'dock_windows', 'dock_windows'),
    'uklad': ('laser_app.ui', 'layout_widgets', 'layout_widgets'),
    'widok_gl': ('laser_app.ui', 'gl_view', 'gl_view'),
    'panel_ustawien': ('laser_app.ui', 'settings_panel', 'settings_panel'),
    'dymki': ('laser_app.ui', 'tooltips', 'tooltips'),
    'grupy_parametrow': ('laser_app.ui', 'parameter_groups', 'parameter_groups'),
    'podglad_etykiety': ('laser_app.ui', 'label_preview', 'label_preview'),
    'rzuty_okienka': ('laser_app.ui', 'projections_view', 'projections_view'),
    'zakladka_laserow': ('laser_app.lasers_tab', 'tab', 'lasers_tab'),
    'lasery_gpu': ('laser_app.gpu', 'lasers_gpu', 'lasers_gpu'),
    'rzuty_gpu': ('laser_app.gpu', 'projections_gpu', 'projections_gpu'),
    'lasery_dane': ('laser_app.compute', 'history', 'hist'),
    'ogniskowa': ('laser_app.compute', 'focal_trap', 'focal_trap'),
    'mapa_pulapek': ('laser_app.compute', 'trap_map', 'trap_map'),
    'pole_podkrokow': ('laser_app.compute', 'substep_field', 'substep_field'),
    'pulapka_obliczenia': ('laser_app.compute', 'trap_measurements', 'trap_measurements'),
    'przejscia_zapis': ('laser_app.compute', 'passage_store', 'passage_store'),
    'rzuty_dane': ('laser_app.compute', 'projections', 'projections'),
    'etykiety_narzedzia': ('laser_app.compute', 'tool_labels', 'tool_labels'),
    'obszary_oka': ('laser_app.compute', 'eye_regions', 'eye_regions'),
    'dolina': ('laser_app.compute', 'valley', 'valley'),
    'reflektor_mechanika': ('laser_app.compute', 'reflector_mechanics', 'reflector_mechanics'),
    'reflektor_rysunek': ('laser_app.compute', 'reflector_drawing', 'reflector_drawing'),
    'historie_checkpointow': ('laser_app.compute', 'checkpoint_histories', 'checkpoint_histories'),
    'film': ('laser_app.compute', 'video', 'video'),
    'detekcja': ('laser_app.compute', 'detection', 'detection'),
    'silnik': ('laser_app.compute', 'sd_engine', 'sd_engine'),
    'wspolne': ('laser_app.compute', 'common', 'common'),
    '_wspolne_gui': ('', '_gui_common', 'gui'),                 # testy: moduł obok testów
}

# klasy: stary moduł → {stara nazwa: nowa}
KLASY = {
    'app': {'Okno': 'MainWindow'},
    'detekcja': {'Detektor': 'Detector'},
    'dolina': {'Liczydlo': 'ValleySolver'},
    'grupy_parametrow': {'NaglowekGrupy': 'GroupHeader'},
    'historie_checkpointow': {'Historia': 'CheckpointHistory'},
    'lasery_dane': {'Przerwano': 'Interrupted', 'Historia': 'History', 'ZZestawu': 'ImageSetZ', 'HistoriaZestawu': 'ImageSetHistory'},
    'lasery_gpu': {'_ElementGPU': '_GPUElement', 'RuchPlaski': 'PlanarMotion', 'SferaReflektorow': 'ReflectorSphere',
                   '_LiczObrys': '_OutlineWorker', 'PoleOgniskowej': 'FocalField', 'MapyNaZdjeciu': 'ImageMaps',
                   'PrzejscieGPU': 'NetworkPassageGPU', 'DolinaGPU': 'ValleyGPU', 'SchematGPU': 'SchematicGPU'},
    'mapa_pulapek': {'MapaGL': 'TrapMapGL'},
    'ogniskowa': {'Przerwano': 'Interrupted', '_Parametry': '_Block6Params'},
    'okna': {'_Oslona': '_LayoutShield', 'PasekDoku': 'DockTitleBar', 'Przewijane': 'ScrollWrapper'},
    'panel_ustawien': {'PanelParametrow': 'ParameterPanel', '_Wypozyczenie': '_SectionLoan', 'PopupUstawien': 'SettingsPopup',
                       '_FiltrMenu': '_ContextMenuFilter'},
    'podglad_etykiety': {'PodgladEtykiety': 'LabelPreview'},
    'reflektor_mechanika': {'Liczydlo': 'MechanicsSolver'},
    'reflektor_rysunek': {'Schemat': 'ReflectorSchematic'},
    'rzuty_dane': {'Przerwano': 'Interrupted', 'DaneRzutow': 'ProjectionData'},
    'rzuty_gpu': {'RzutGPU': 'ProjectionGPU'},
    'rzuty_okienka': {'LiczenieRzutow': 'ProjectionWorker', 'MiaryWatek': 'MetricsWorker', 'RamaCWatek': 'FrameCWorker',
                      'Lokalne': 'LocalSections', 'Scena': 'Scene', 'Rzuty': 'ProjectionWindows'},
    'uklad': {'Przeplyw': 'FlowLayout', 'PanelOpcji': 'OptionsPanel', 'WaskiCombo': 'NarrowCombo', '_OknoOdpiete': '_DetachedWindow',
              'Odpinanie': 'Detacher', 'RamaPanelu': 'PanelFrame'},
    'widok_gl': {'WidokGL': 'GLView', 'ObrazPrzezroczysty': 'TransparentImage', 'Widok3D': 'View3D'},
    'wygoda': {'LeniwyModul': 'LazyModule', '_KolkoTylkoZFokusem': '_WheelOnlyWithFocus'},
    'zakladka_laserow': {'Widok2D': 'View2D', 'Liczenie': 'HistoryWorker', 'ZestawLiczenie': 'ImageSetWorker',
                         'OgniskowaWatek': 'FocalTrapWorker', 'MapaPulapekWatek': 'TrapMapWorker',
                         'PolePodkrokowWatek': 'SubstepFieldWorker', 'DolinaWatek': 'ValleyWorker', 'MechanikaWatek': 'MechanicsWorker',
                         'PomiaryPulapkiWatek': 'TrapMeasurementsWorker', 'Zwijana': 'CollapsibleSection',
                         'PodkrokiWatek': 'SubstepsWorker', 'WszystkiePrzejsciaWatek': 'AllPassagesWorker',
                         'DiagramSieci': 'NetworkDiagram', 'SuwakAB': 'ABSlider', 'ZakladkaLaserow': 'LasersTab'},
}
# nazwy klas, które są też zwykłymi polskimi słowami: w napisach zmieniamy je tylko w `…` albo przed „.” i „(”
SLOWA = {'Historia', 'Schemat', 'Scena', 'Rzuty', 'Lokalne', 'Liczenie', 'Okno', 'Przerwano', 'Liczydlo', 'Zwijana'}
PLIKI_TESTOW = {'test_moduly': 'test_modules', 'test_etykiety': 'test_tool_labels', 'test_rzuty_dane': 'test_projections',
                'test_dymki': 'test_tooltips', 'test_gui': 'test_gui', '_wspolne_gui': '_gui_common', 'conftest': 'conftest'}


def nowa_sciezka(mod):
    pak, nazwa, _ = MOD[mod]
    return f'{pak}.{nazwa}' if pak else nazwa


def kontekst(drzewo, wlasny):
    """alias → moduł (dla `import X as A` i `import X`), nazwa lokalna → moduł (from-importy i własne klasy)."""
    aliasy, gole = {}, {}
    for n in ast.walk(drzewo):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in MOD:
                    aliasy[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module in MOD:
            for a in n.names:
                gole[a.asname or a.name] = (n.module, a.name)
    if wlasny in KLASY:
        for k in KLASY[wlasny]:
            gole.setdefault(k, (wlasny, k))
    return aliasy, gole


def nowy_alias(mod, alias):
    return MOD[mod][2]


def przetworz(sciezka, wlasny, raport):
    tekst = open(sciezka, encoding='utf-8', newline='').read()
    crlf = '\r\n' in tekst
    s = tekst.replace('\r\n', '\n')
    drzewo = ast.parse(s)
    aliasy, gole = kontekst(drzewo, wlasny)
    linie_importow = set()
    for n in ast.walk(drzewo):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            linie_importow.update(range(n.lineno, n.end_lineno + 1))
    toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
    wszystkie_nazwy = {t.string for t in toks if t.type == tokenize.NAME}
    # kolizje: nowy alias już używany w pliku jako inna nazwa
    for a, m in aliasy.items():
        na = nowy_alias(m, a)
        if na != a and na in wszystkie_nazwy:
            raport.append(f'{sciezka.name}: alias {a} → {na}: „{na}” już jest w pliku')
    zmiany = []                                                          # (start, koniec, nowy tekst) w (wiersz, kolumna)

    def zmien(t, nowy):
        if nowy != t.string:
            zmiany.append((t.start, t.end, nowy))

    def napis(txt, fstring):
        # odwołania do plików i funkcji modułów: stary.py, stary.nazwa
        for m in MOD:
            nb = MOD[m][1] if MOD[m][0] else MOD[m][1]
            txt = re.sub(rf'(?<![\w/]){re.escape(m)}\.py\b', f'{nb}.py', txt)
            txt = re.sub(rf'(?<![\w.]){re.escape(m)}\.(?=[A-Za-z_])', f'{nb}.', txt)
        if fstring:                                                       # kod w {…} f-stringa: aliasy
            for a, m in aliasy.items():
                txt = re.sub(rf'(?<![\w.]){re.escape(a)}\.', f'{nowy_alias(m, a)}.', txt)
        for m, mapa in KLASY.items():
            for k, nk in mapa.items():
                if k in SLOWA:
                    txt = re.sub(rf'`{k}\b', f'`{nk}', txt)
                    txt = re.sub(rf'(?<![\w]){k}(?=\.\w|\()', nk, txt)
                else:
                    txt = re.sub(rf'(?<![\w]){k}\b', nk, txt)
        return txt

    for i, t in enumerate(toks):
        if t.type == tokenize.NAME:
            nast = toks[i + 1] if i + 1 < len(toks) else None
            pop = toks[i - 1] if i else None
            pop2 = toks[i - 2] if i > 1 else None
            w_imporcie = t.start[0] in linie_importow
            if t.string in aliasy and nast is not None and nast.string == '.' and not (pop is not None and pop.string == '.') and not w_imporcie:
                zmien(t, nowy_alias(aliasy[t.string], t.string))
            elif pop is not None and pop.string == '.' and pop2 is not None and pop2.type == tokenize.NAME and pop2.string in aliasy:
                m = aliasy[pop2.string]
                if t.string in KLASY.get(m, {}):
                    zmien(t, KLASY[m][t.string])
            elif not (pop is not None and pop.string == '.') and t.string in gole:
                m, oryg = gole[t.string]
                if oryg == t.string and t.string in KLASY.get(m, {}):
                    zmien(t, KLASY[m][t.string])
                elif w_imporcie and oryg in KLASY.get(m, {}):        # from X import Stara as Lokalna: zmiana importowanej nazwy
                    zmien(t, KLASY[m][oryg]) if t.string == oryg else None
            elif t.string in aliasy and not w_imporcie and not (pop is not None and pop.string == '.'):
                zmien(t, nowy_alias(aliasy[t.string], t.string))
                raport.append(f'{sciezka.name}:{t.start[0]}: alias {t.string} bez kropki → {nowy_alias(aliasy[t.string], t.string)}')
        elif t.type == tokenize.STRING:
            f = re.match(r'^[rbuRBU]*[fF]', t.string) is not None
            zmien(t, napis(t.string, f))
        elif t.type == tokenize.COMMENT:
            zmien(t, napis(t.string, False))
    # zastosowanie zmian od końca
    wiersze = s.split('\n')
    poz = [0]
    for w in wiersze:
        poz.append(poz[-1] + len(w) + 1)
    off = lambda rc: poz[rc[0] - 1] + rc[1]
    for a, b, nowy in sorted(zmiany, key=lambda z: off(z[0]), reverse=True):
        s = s[:off(a)] + nowy + s[off(b):]
    # importy (wiersz po wierszu, także wcięte w funkcjach)
    out = []
    for w in s.split('\n'):
        m = re.match(r'^(\s*)import (\w+) as (\w+)(.*)$', w)
        if m and m.group(2) in MOD:
            mod = m.group(2); pak, nazwa, al = MOD[mod]
            w = f'{m.group(1)}from {pak} import {nazwa} as {al}{m.group(4)}' if pak else f'{m.group(1)}import {nazwa} as {al}{m.group(4)}'
            w = w.replace(f'import {nazwa} as {nazwa}', f'import {nazwa}')
        else:
            m = re.match(r'^(\s*)import (\w+)(\s*(#.*)?)$', w)
            if m and m.group(2) in MOD:
                pak, nazwa, _ = MOD[m.group(2)]
                w = f'{m.group(1)}from {pak} import {nazwa}{m.group(3)}' if pak else f'{m.group(1)}import {nazwa}{m.group(3)}'
            else:
                m = re.match(r'^(\s*)from (\w+) import (.*)$', w)
                if m and m.group(2) in MOD:
                    w = f'{m.group(1)}from {nowa_sciezka(m.group(2))} import {m.group(3)}'
        out.append(w)
    s = '\n'.join(out)
    # odwołania do aliasów bez kropki (np. przekazany moduł) – nazwa modułu bez aliasu: `import dymki` → `tooltips`
    for a, m in aliasy.items():
        if a == m:                                                        # import bez aliasu: dymki.x → tooltips.x już zrobione
            pass
    return s, crlf


def main():
    sprawdz = '--sprawdz' in sys.argv
    raport = []
    pliki = [p for p in APP.glob('*.py')] + [p for p in (APP / 'testy').glob('*.py')]
    for p in pliki:
        s, crlf = przetworz(p, p.stem, raport)
        if not sprawdz:
            open(p, 'w', encoding='utf-8', newline='\r\n' if crlf else '\n').write(s)
    print('\n'.join(raport) or 'bez uwag')


if __name__ == '__main__':
    main()
