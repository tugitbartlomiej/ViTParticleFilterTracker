"""Stany 192D (rzuty_dane.policz) dla historii klatek B i C; wynik trafia do cache aplikacji (h.katalog/rzuty), jak w oknie R9."""
import sys, time
from pathlib import Path
APP = Path(__file__).resolve().parents[3] / 'app'
sys.path.insert(0, str(APP))
import lasery_dane as LD, rzuty_dane as RD
for h_id in sys.argv[1:]:
    h = LD.Historia(APP / 'cache_uczenia/lasery' / h_id)
    t0 = time.time()
    RD.policz(h, postep=lambda i, n, o: print(f'{h_id} {i}/{n} {o}', flush=True) if i % 100 == 0 else None)
    print(h_id, 'gotowe', round(time.time() - t0), 's', flush=True)
