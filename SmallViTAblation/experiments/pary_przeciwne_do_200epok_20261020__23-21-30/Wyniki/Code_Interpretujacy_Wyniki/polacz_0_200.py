"""Join parent counts (epochs 0-60) with continuation counts (60-200) into one table."""
from pathlib import Path
import csv,hashlib,json,sys

EXP=Path(__file__).resolve().parents[2]
PARENT=EXP.parent/'vit_wiazkiprzeciwstawne_20261020__18-10-00/Wyniki/Dane/opposite_pair_counts_by_epoch'
NEW=EXP/'Wyniki/Dane/pary_epoki_60_200'
OUT=EXP/'Wyniki/Dane/polaczone_0_200'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

done=json.loads((NEW/'DONE.json').read_text(encoding='utf-8'))
assert sha(NEW/'COUNTS_60_200.json')==done['files']['COUNTS_60_200.json']
assert json.loads((NEW/'QC_65640_VS_PARENT.json').read_text(encoding='utf-8'))['status']=='PASS','Step 65640 differs from parent'
parent_done=json.loads((PARENT/'DONE.json').read_text(encoding='utf-8'))
assert sha(PARENT/'COUNTS_BY_EPOCH.json')==parent_done['files']['COUNTS_BY_EPOCH.json']
old=json.loads((PARENT/'COUNTS_BY_EPOCH.json').read_text(encoding='utf-8'))
new=json.loads((NEW/'COUNTS_60_200.json').read_text(encoding='utf-8'))
# Step 65640 is in both; the parent's row stays, the new one was only the consistency check.
rows=[dict(r,source='vit_wiazkiprzeciwstawne') for r in old]+[dict(r,source='pary_przeciwne_do_200epok') for r in new if r['step']!=65640]
for r in rows:r['maximum_observed_pair']=json.dumps(r['maximum_observed_pair'])
rows.sort(key=lambda r:(r['step'],r['block'],r['threshold_deg']))
assert len({(r['step'],r['block'],r['threshold_deg']) for r in rows})==len(rows)
csvwrite('COUNTS_0_200.csv',rows)

summary=[]
for step in sorted({r['step'] for r in rows}):
    for th in sorted({r['threshold_deg'] for r in rows}):
        rr=[r for r in rows if r['step']==step and r['threshold_deg']==th];assert len(rr)==6
        summary.append(dict(step=step,epoch_equivalent=step*32/35000,threshold_deg=th,
            unique_pairs_all_blocks=sum(r['same_token_observed_unique_pairs'] for r in rr),
            pair_token_count_all_blocks=sum(r['same_token_observed_pair_token_count'] for r in rr),
            active05_unique_pairs_all_blocks=sum(r['same_token_active05_unique_pairs'] for r in rr),
            weight_only_unique_pairs_all_blocks=sum(r['weight_only_unique_pairs'] for r in rr),
            possible_signed_unique_pairs_all_blocks=sum(r['possible_signed_unique_pairs'] for r in rr),
            maximum_observed_signed_angle=max(r['maximum_observed_signed_angle'] for r in rr),
            **{f'unique_pairs_b{r["block"]}':r['same_token_observed_unique_pairs'] for r in rr},
            **{f'max_observed_angle_b{r["block"]}':r['maximum_observed_signed_angle'] for r in rr}))
csvwrite('SUMA_BLOKOW_0_200.csv',summary)
(OUT/'MANIFEST.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),
    parent_counts_sha256=sha(PARENT/'COUNTS_BY_EPOCH.json'),new_counts_sha256=sha(NEW/'COUNTS_60_200.json'),
    steps=sorted({r['step'] for r in rows}),rows=len(rows),
    files={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))}),indent=2),encoding='utf-8')

if '--print' in sys.argv:
    sys.stdout.reconfigure(encoding='utf-8')
    print('epoch  >=150  >=160  >=170  >=175 max_angle | >=150 per block b1..b6 | pair-tokens>=150 | possible>=150')
    for step in sorted({r['step'] for r in summary}):
        s={r['threshold_deg']:r for r in summary if r['step']==step}
        a=s[150]
        print(f"{a['epoch_equivalent']:5.0f} {a['unique_pairs_all_blocks']:6d} {s[160]['unique_pairs_all_blocks']:6d} {s[170]['unique_pairs_all_blocks']:6d} {s[175]['unique_pairs_all_blocks']:6d} {a['maximum_observed_signed_angle']:9.2f} | "
              +' '.join(f"{a[f'unique_pairs_b{b}']:3d}" for b in range(1,7))+f" | {a['pair_token_count_all_blocks']:8d} | {a['possible_signed_unique_pairs_all_blocks']:5d}")
