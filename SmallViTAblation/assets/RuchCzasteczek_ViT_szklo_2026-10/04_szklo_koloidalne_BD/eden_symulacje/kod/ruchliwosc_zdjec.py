"""Ruchliwość każdego zdjęcia w każdym checkpoincie (Eden, CPU): średnia amplituda drgań tokenów wokół węzła |x − c|, c = średnia w oknie
±300 kroków, x = token rp128 względem średniej zdjęcia. Wynik: ruchliwosc.npz (kroki, amp (K,100), oceny (K,100,2), split).
Uruchomienie: python ruchliwosc_zdjec.py <eksport> <out.npz>"""
import json
import sys

import numpy as np

EKS, OUT = sys.argv[1], sys.argv[2]
META = json.load(open(f'{EKS}/META.json', encoding='utf-8'))
kroki = np.array(META['kroki']); K = len(kroki)
rp = np.load(f'{EKS}/rp128.npy', mmap_mode='r')
bufor, amp = {}, np.zeros((K, rp.shape[1]))
for t in range(K):
    okno = [u for u in range(K) if abs(kroki[u] - kroki[t]) <= 300]
    for u in okno:
        if u not in bufor:
            X = np.asarray(rp[u]).astype(np.float32); bufor[u] = X - X.mean(1, keepdims=True)
    for u in [u for u in bufor if u not in okno]:
        del bufor[u]
    c = np.mean([bufor[u] for u in okno], 0)
    amp[t] = np.linalg.norm(bufor[t] - c, axis=-1).mean(1)
    if t % 100 == 0:
        print('checkpoint', t, flush=True)
np.savez_compressed(OUT, kroki=kroki, amp=amp, oceny=np.load(f'{EKS}/oceny.npy'), split=np.array([z['split'] for z in META['zdjecia']]))
print('zapisano', OUT)
