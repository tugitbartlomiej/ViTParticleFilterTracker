"""Sprawdza ścieżki i agregaty audytu; brak dowodu/linku wymaga poprawienia raportu."""
import json,pathlib,hashlib,re,datetime
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_k1000_analiza_rezimow_astra_20261009'
OLD=ROOT/'outputs/vit_k1000_1500_pelne_latenty_90_20261008'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
d=load(OUT/'DIAGNOSTYKA_EDEN.json');loc=load(OUT/'DIAGNOSTYKA_LOKALNA.json')
meta=load(ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009/data/metadata.json')['images']
summary={
 'model':'gpt-6-astra','reasoning_effort':'xhigh','model_evidence':'turn_context bieżącej sesji 01a11dfa-6abb-7230-ba84-4d9671bf666f',
 'adam_max_abs_error':max(r['adam_replay_max_abs_error'] for r in d['records']),
 'sample_target_trajectories':sum(r['target_in_panel'] for r in d['records']),
 'turns_by_phase':{k:{'eligible':sum(r['step_direction_cosine'][k]['cosine']['n'] for r in d['records']),'negative':sum(r['step_direction_cosine'][k]['negative'] for r in d['records'])} for k in ['fit','min_norm','switch']},
 'ridge_control_energy_upper_bound':sum(v['sum_centered_squares'] for v in d['control_train_scales'].values()),
 'ridge_lambda_lower_bound':1e-5*48000/65,
 'dmd_min_norm_relative_rmse_reduction':1-d['existing_dmd_model_metrics_by_phase']['heldout_min_norm']['rmse_reduced_state']['dmd']/d['existing_dmd_model_metrics_by_phase']['heldout_min_norm']['rmse_reduced_state']['persistence'],
 'full_z_f32_TB':1500*501*600*768*4/1e12,'full_z_f32_TiB':1500*501*600*768*4/2**40,
 'success_failure_norm_statistics':loc['endpoint_phase_split'],
 'input_hashes':[], 'snapshot_hashes_valid':True,
 'panel_npz_hashes_match_verified_projection_metadata':all(r['npz_sha256']==meta[r['image']]['source_npz_sha256'] for r in d['records']),
}
paths=[OLD/'eden_results/movement/MOVEMENT_ARRAYS.npz',OLD/'eden_results/regimes/MEAN_DELTA_ALL.npz',OLD/'eden_results/regimes/REGIME_MODEL_ALL.npz',OLD/'eden_results/regimes/REGIME_DIAGNOSTICS.json',OLD/'eden_results/dmdc/DMDC_MODEL.npz',OLD/'eden_results/dmdc/DMDC_REPORT.json',OLD/'provenance/CONTROL_PANEL.json',OLD/'provenance/MANIFEST_1500.jsonl',ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009/data/metadata.json',ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009/data/projection.npz']
for p in paths:summary['input_hashes'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for item in load(OUT/'PROVENANCE_ODCZYTU.json')['files']:
 if hashlib.sha256((OUT/'snapshot_eden'/item['relative']).read_bytes()).hexdigest()!=item['sha256']:summary['snapshot_hashes_valid']=False
links=re.findall(r'\]\((F:/[^)]+)\)',(OUT/'RAPORT.md').read_text(encoding='utf-8'))
invalid=[]
for link in links:
 m=re.match(r'^(.*?)(?::(\d+))?$',link);p=pathlib.Path(m[1]);line=int(m[2]) if m[2] else None
 if not p.exists():invalid.append({'link':link,'reason':'missing'})
 elif line and len(p.read_text(encoding='utf-8').splitlines())<line:invalid.append({'link':link,'reason':'line out of range'})
summary['report_local_links']={'count':len(links),'invalid':invalid}
(OUT/'PODSUMOWANIE_DIAGNOSTYK.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['input_hashes','success_failure_norm_statistics']},ensure_ascii=False,indent=2))
assert not invalid and summary['snapshot_hashes_valid']
