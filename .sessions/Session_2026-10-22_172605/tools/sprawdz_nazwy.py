"""Nazwy czytane, a nigdzie w pliku niezdefiniowane (brak importu po przenosinach) w pakiecie, testach i run.py."""
import ast, builtins, glob
for p in glob.glob('laser_app/**/*.py', recursive=True) + glob.glob('tests/*.py') + ['run.py']:
    t = ast.parse(open(p, encoding='utf-8').read())
    d, u = set(dir(builtins)) | {'__file__', '__name__'}, {}
    for n in ast.walk(t):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, (ast.Store, ast.Del)): d.add(n.id)
            else: u.setdefault(n.id, n.lineno)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)): d.add(n.name)
        elif isinstance(n, ast.arg): d.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: d.add(a.asname or a.name.split('.')[0])
        elif isinstance(n, ast.ExceptHandler) and n.name: d.add(n.name)
    brak = {k: v for k, v in u.items() if k not in d}
    if brak: print(p, brak)
print('sprawdzone')
