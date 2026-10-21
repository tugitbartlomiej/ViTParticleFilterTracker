"""Print trap-field topology over training (P3 slice at the colloid) and how often a particle keeps its class between checkpoints."""
from pathlib import Path
import csv,sys
import numpy as np

D=Path(__file__).resolve().parents[1]/'Dane';sys.stdout.reconfigure(encoding='utf-8')
rows=list(csv.DictReader(open(D/'POLA_PULAPEK.csv',encoding='utf-8')))
print('epoka | narzędzie: przyciąga/odpycha/siodło, wir | tło: przyciąga/odpycha/siodło, wir | losowe: wir, siodło (T/B) | 192D: zespolone, Re>0, rot/strain (T/B)')
for r in rows:
    g=lambda k:float(r[k])
    print(f"{g('epoch'):5.1f} | {g('tool_P3_attract'):.2f}/{g('tool_P3_repel'):.2f}/{g('tool_P3_saddle'):.2f}, {g('tool_P3_vortex'):.2f} | "
          f"{g('bg_P3_attract'):.2f}/{g('bg_P3_repel'):.2f}/{g('bg_P3_saddle'):.2f}, {g('bg_P3_vortex'):.2f} | "
          f"{g('tool_rand_vortex'):.2f}/{g('bg_rand_vortex'):.2f}, {g('tool_rand_saddle'):.2f}/{g('bg_rand_saddle'):.2f} | "
          f"{g('tool_full_complex_share'):.2f}/{g('bg_full_complex_share'):.2f}, {g('tool_full_repel_share'):.2f}/{g('bg_full_repel_share'):.2f}, "
          f"{g('tool_rotation_to_strain'):.2f}/{g('bg_rotation_to_strain'):.2f}")
d=np.load(D/'POLA_PULAPEK_CZASTKI.npz');k=d['klasy'];ep=d['steps']*32/35000;tool=d['tool'].astype(bool)
lab=k[:,1]*2+k[:,0]                                  # sign class x vortex flag
print('\nTrwałość charakteru pola wokół tej samej cząstki między kolejnymi checkpointami (klasa znaku + wir):')
for i in range(1,len(ep)):
    same=lab[i]==lab[i-1]
    print(f"{ep[i-1]:6.1f}->{ep[i]:6.1f}: narzędzie {same[tool].mean():.2f}, tło {same[~tool].mean():.2f}")
