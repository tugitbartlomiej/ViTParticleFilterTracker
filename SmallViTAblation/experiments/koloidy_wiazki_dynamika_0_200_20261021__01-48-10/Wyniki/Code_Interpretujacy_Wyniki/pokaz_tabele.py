"""Print the measured tables (checkpoint metrics, motion between checkpoints, MSD) in a compact form."""
import csv,sys
from pathlib import Path

D=Path(__file__).resolve().parents[1]/'Dane'
def f(x):return float(x) if x not in ('',None,'None') else float('nan')
def rows(n):return list(csv.DictReader(open(D/n,encoding='utf-8')))

sys.stdout.reconfigure(encoding='utf-8')
cols=['epoch','R','token_norm','sep','tool_outside','sep_visible_P3','cloud_dims','kick_rel','coherence','sort_tool','sort_bg',
      'restore_tool','restore_bg','active_beams','pull_power_share','div_tool','div_bg','neg_dirs_tool','neg_dirs_bg',
      'stiff_neg_tool','stiff_neg_bg','beam_power_mean','beam_power_top10_share','beam_bundle_dims','beam_rotation_deg_per_epoch']
print(' '.join(c[:9].rjust(9) for c in cols))
for x in rows('METRYKI_CHECKPOINTY.csv'):print(' '.join(f'{f(x[c]):9.3f}' for c in cols))
print('\nfrom->to | tool raw | bg raw | tool aligned | bg aligned (per epoch) | knn kept')
for x in rows('RUCH_MIEDZY_CHECKPOINTAMI.csv'):
    print(f"{f(x['from_epoch']):6.1f}->{f(x['to_epoch']):6.1f} | {f(x['speed_tool_raw_per_epoch']):.4f} | {f(x['speed_bg_raw_per_epoch']):.4f} | "
          f"{f(x['speed_tool_aligned_per_epoch']):.4f} | {f(x['speed_bg_aligned_per_epoch']):.4f} | {f(x['knn_kept']):.3f}")
print()
for x in rows('MSD_60_200.csv'):print(x)
