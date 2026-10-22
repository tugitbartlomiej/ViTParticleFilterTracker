"""Domknięcie importów lokalnych modułów app/ zaczynając od podanych modułów."""
import ast
import sys
from pathlib import Path

APP = Path(sys.argv[1])
lokalne = {p.stem for p in APP.glob('*.py')}


def importy(mod):
    drzewo = ast.parse((APP / f'{mod}.py').read_text(encoding='utf-8'))
    wynik = set()
    for w in ast.walk(drzewo):
        if isinstance(w, ast.Import):
            for a in w.names:
                wynik.add(a.name.split('.')[0])
        elif isinstance(w, ast.ImportFrom) and w.module and w.level == 0:
            wynik.add(w.module.split('.')[0])
        elif isinstance(w, ast.Call) and getattr(w.func, 'attr', '') == 'import_module' and w.args \
                and isinstance(w.args[0], ast.Constant):
            wynik.add(str(w.args[0].value).split('.')[0])
    return wynik & lokalne


start = sys.argv[2:]
widziane, stos = set(), list(start)
krawedzie = {}
while stos:
    m = stos.pop()
    if m in widziane:
        continue
    widziane.add(m)
    krawedzie[m] = sorted(importy(m))
    stos.extend(krawedzie[m])
for m in sorted(widziane):
    print(f'{m}: {", ".join(krawedzie[m])}')
print('\nRAZEM', len(widziane))
# kto z NIE-domknięcia importuje moduły domknięcia (współdzielone)
print('\nUżywane też poza domknięciem:')
for m in sorted(lokalne - widziane):
    wsp = importy(m) & widziane
    if wsp:
        print(f'  {m} -> {", ".join(sorted(wsp))}')
