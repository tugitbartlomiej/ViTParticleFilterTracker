"""Podział laser_app/gpu/lasers_gpu.py na moduły elementów GL (base + po jednym pliku na element z jego shaderami)."""
import ast
import builtins
from pathlib import Path

APP = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp')
GPU = APP / 'laser_app' / 'gpu'
s = (GPU / 'lasers_gpu.py').read_text(encoding='utf-8').replace('\r\n', '\n')
L = s.split('\n')
T = ast.parse(s)
DOC = ast.get_docstring(T)

# (plik, opis, pierwszy wiersz, ostatni wiersz) – zakresy z mapy węzłów najwyższego poziomu
PLIKI = [
    ('base.py', 'Wspólna część elementów OpenGL 4.3: stałe siatki tokenów i kanałów, kompilacja programów shaderów, palety (LUT) i '
                'klasa bazowa _GPUElement (bufory SSBO, wysyłka dużych tablic porcjami w tle).', 19, 237),
    ('planar_gl.py', 'Okienka 1–3: rzut płaski ruchu (globalny, studnie i strzałki od adresów, drgania) w shaderach (PlanarMotion).', 241, 390),
    ('sphere_gl.py', 'Sfera reflektorów: 768 reflektorów, wiązki do 600 koloidów (albo kilku zdjęć), koloidy i ich ślady (ReflectorSphere).',
     394, 889),
    ('focal_field_gl.py', 'Okienko 11: pułapka optyczna wokół koloidu – mgła i strzałki pola, wiązki zbiegające się w cząsteczce, światło '
                          'ogniska, obrys 3D (liczony w tle), reflektory (FocalField).', 893, 1692),
    ('image_maps_gl.py', 'Okienka 4–5: mapy na zdjęciu (centra, drgania, atencja, natężenie reflektora) (ImageMaps).', 1696, 1811),
    ('network_passage_gl.py', 'Okienko P: przejście sygnału przez 6 bloków – siły uwagi, lasery MLP, tory (NetworkPassageGPU).', 1815, 1967),
    ('valley_gl.py', 'Okienko 12: dolina pewności i IoU wokół cząsteczki – mgła siatki i obrys (ValleyGPU).', 1971, 2103),
    ('schematic_gl.py', 'Okienko 13: rysunek techniczny reflektora (odcinki, punkty) (SchematicGPU).', 2107, 2179),
]
naglowek_importy = [n for n in T.body if isinstance(n, (ast.Import, ast.ImportFrom))]
DOSTAWCY = {}
for n in naglowek_importy:
    for a in n.names:
        loc = a.asname or a.name.split('.')[0]
        DOSTAWCY[loc] = ('from', n.module, a.name, a.asname) if isinstance(n, ast.ImportFrom) else ('import', a.name, None, a.asname)

# nazwy najwyższego poziomu → plik
GDZIE = {}
for n in T.body:
    if isinstance(n, (ast.Import, ast.ImportFrom, ast.Expr)):
        continue
    for plik, _, a, b in PLIKI:
        if a <= n.lineno <= b:
            nazwy = [n.name] if hasattr(n, 'name') else []
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    nazwy += [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
            if isinstance(n, ast.Try):                               # klasa _OutlineWorker w try/except ImportError
                nazwy += [x.name for x in ast.walk(n) if isinstance(x, ast.ClassDef)]
            for x in nazwy:
                GDZIE[x] = plik
            break
    else:
        raise SystemExit(f'węzeł poza zakresami: {n.lineno}')


def uzyte(kod):
    return {n.id for n in ast.walk(ast.parse(kod)) if isinstance(n, ast.Name)}


def importy(kod, plik):
    t = ast.parse(kod)
    modulowe = set()
    for n in t.body:
        if hasattr(n, 'name'):
            modulowe.add(n.name)
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                modulowe |= {x.id for x in ast.walk(tg) if isinstance(x, ast.Name)}
        if isinstance(n, ast.Try):
            modulowe |= {x.name for x in ast.walk(n) if isinstance(x, ast.ClassDef)}
    potrzebne = uzyte(kod) - modulowe - set(dir(builtins)) - {'self', 'cls'}
    std, obce, app = [], [], {}
    for nm in sorted(potrzebne):
        if nm in DOSTAWCY:
            k = DOSTAWCY[nm]
            if k[0] == 'import':
                (std if k[1] == 'ctypes' else obce).append(f'import {k[1]}' + (f' as {k[3]}' if k[3] else ''))
            else:
                obce.append(f'from {k[1]} import {k[2]}' + (f' as {k[3]}' if k[3] else ''))
        elif nm in GDZIE and GDZIE[nm] != plik:
            app.setdefault(GDZIE[nm][:-3], []).append(nm)
    # scal from-importy z tego samego modułu
    scal = {}
    for l in obce:
        if l.startswith('from '):
            m, r = l[5:].split(' import ')
            scal.setdefault(m, []).append(r)
    obce = sorted({l for l in obce if l.startswith('import ')} | {f'from {m} import {", ".join(sorted(set(r)))}' for m, r in scal.items()},
                  key=lambda x: x.split()[1].lower())
    app_l = [f'from laser_app.gpu.{m} import {", ".join(sorted(v, key=str.lower))}' for m, v in sorted(app.items())]
    return '\n\n'.join('\n'.join(b) for b in (sorted(std), obce, app_l) if b)


def main():
    for plik, opis, a, b in PLIKI:
        kod = '\n'.join(L[a - 1:b]).strip('\n')
        imp = importy(kod, plik)
        tekst = f'"""{opis}"""\n{imp}\n\n{kod}\n'
        ast.parse(tekst)
        (GPU / plik).write_text(tekst, encoding='utf-8', newline='\n')
        print(plik, len(tekst.split('\n')))
    (GPU / '__init__.py').write_text(f'"""{DOC}\n\nModuły: base (wspólne), sphere_gl, focal_field_gl, planar_gl, image_maps_gl, network_passage_gl, '
                                     f'valley_gl, schematic_gl, trap_map_gl, projections_gl."""\n', encoding='utf-8', newline='\n')
    import json
    print(json.dumps({k: v for k, v in sorted(GDZIE.items()) if not k.startswith(('VS_', 'FS_', '_SSBO'))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
