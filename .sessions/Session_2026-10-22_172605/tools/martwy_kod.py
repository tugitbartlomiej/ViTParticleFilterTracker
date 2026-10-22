"""Definicje najwyższego poziomu (funkcje, klasy, stałe) w laser_app, do których nikt się nie odwołuje (AST: Name, Attribute,
importy; także w f-stringach). Wypisuje kandydatów; z --usun usuwa je (powtarza, aż nic nie zostanie)."""
import ast, glob, re, sys
from collections import Counter

def zbierz():
    pliki = glob.glob('laser_app/**/*.py', recursive=True) + glob.glob('tests/*.py') + ['run.py']
    ref = Counter()
    for p in pliki:
        t = ast.parse(open(p, encoding='utf-8').read())
        for n in ast.walk(t):
            if isinstance(n, ast.Name): ref[n.id] += 1
            elif isinstance(n, ast.Attribute): ref[n.attr] += 1
            elif isinstance(n, ast.ImportFrom):
                for a in n.names: ref[a.name] += 1
    kand = []
    for p in glob.glob('laser_app/**/*.py', recursive=True):
        t = ast.parse(open(p, encoding='utf-8').read())
        for n in t.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                nazwa = n.name
                wlasne = sum(1 for x in ast.walk(n) if isinstance(x, ast.Name) and x.id == nazwa)
                if ref[nazwa] - wlasne <= 0:
                    kand.append((p, nazwa, n.lineno, n.end_lineno))
    return kand

def usun(kand):
    po_pliku = {}
    for p, nazwa, a, b in kand:
        po_pliku.setdefault(p, []).append((a, b, nazwa))
    for p, zak in po_pliku.items():
        s = open(p, encoding='utf-8', newline='').read(); crlf = '\r\n' in s
        L = s.replace('\r\n', '\n').split('\n')
        for a, b, nazwa in sorted(zak, reverse=True):
            t = ast.parse('\n'.join(L))
            n = next(x for x in t.body if getattr(x, 'name', None) == nazwa)
            a = min([n.lineno] + [d.lineno for d in n.decorator_list])
            while a > 1 and L[a - 2].strip().startswith('#'): a -= 1
            del L[a - 1:n.end_lineno]
        s = re.sub(r'\n{4,}', '\n\n\n', '\n'.join(L))
        open(p, 'w', encoding='utf-8', newline='\r\n' if crlf else '\n').write(s)

if '--usun' in sys.argv:
    while True:
        k = zbierz()
        if not k: break
        for x in k: print('usuwam', x[0], x[1], x[3] - x[2] + 1, 'l.')
        usun(k)
else:
    for x in zbierz(): print(x[0], x[1], x[3] - x[2] + 1, 'l.')
