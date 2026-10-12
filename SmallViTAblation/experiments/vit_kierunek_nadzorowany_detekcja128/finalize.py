"""Domyka raport pilota; nie zmienia korekty ani wyboru na podstawie testu."""
import json,hashlib
from datetime import datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parents[1]/'outputs'/HERE.name
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
p=read('PROGRESS.json');assert p['stage']=='complete'
v=read('VERIFICATION.json');assert v['weights_and_BN_buffers_unchanged'] and v['checkpoint_unchanged']
assert read('SOURCE_AUDIT.json')['pass_all']
freeze=read('FREEZE.json');history=read('HISTORY.json');assert len(history)==1200
if not (OUT/'COST_RUNNER.json').exists():write('COST_RUNNER.json',read('COST.json'))
cost=read('COST_RUNNER.json')
launch=read('LAUNCH.json');first=read('FAILED_FIRST.json');second=read('FAILED_RELOAD.json')
end=datetime.fromtimestamp((OUT/'PROGRESS.json').stat().st_mtime).astimezone();beg=datetime.fromisoformat(launch['started'])
cost.update(end_to_end_wall_seconds=(end-beg).total_seconds(),completed_at=end.isoformat(),started_at=launch['started'],runner_seconds_excluding_imports_total=first['elapsed_seconds']+second['elapsed_seconds']+p['elapsed_seconds'],optimization_and_validation_selection_seconds=freeze['train_seconds'],source_audit_seconds=read('SOURCE_AUDIT.json')['seconds'],training_pid=launch['pid'],evaluation_pid=p['pid'],optimization_repeated=False,optimization_steps=1200,initializations=3,learning_rates=2,repair_count=2,repairs=['logging serialization of numpy AP arrays; evaluation resumed from frozen selection','saved scalar alpha reload promoted numpy vector to float64; explicitly loaded float32'],note='Wall time includes imports, source audit overlap and repair delays. Runner time excludes imports. Energy and billed GPU cost not measured.')
write('COST.json',cost)
cost['peak_cuda_memory_scope']='final evaluation/reload process only; training-process peak was not persisted before technical failure'
write('COST.json',cost)
m=read('METRICS.json');b=read('BOOTSTRAP.json');test=m['test'];val=m['val']
text='\n\n## Interpretacja końcowa i audyt\n\n'
text+=f"Walidacja: AP50 {val['baseline']['AP50']:.6f} → {val['shared']['AP50']:.6f}. Test: AP50 {test['baseline']['AP50']:.6f} → {test['shared']['AP50']:.6f}, AP50:95 {test['baseline']['AP50_95']:.6f} → {test['shared']['AP50_95']:.6f}. Różnice na teście wynoszą {100*b['test']['delta_AP50']:.4f} i {100*b['test']['delta_AP50_95']:.4f} punktu procentowego.\n\n"
text+=f"95-procentowy przedział bootstrap różnicy AP50 to [{100*b['test']['CI95'][0][0]:.4f}, {100*b['test']['CI95'][1][0]:.4f}] punktu procentowego; AP50:95 to [{100*b['test']['CI95'][0][1]:.4f}, {100*b['test']['CI95'][1][1]:.4f}]. Oba obejmują zero. Pilot nie potwierdza poprawy AP na teście. Kontrola random11 osiąga wyższe testowe AP50 ({test['random11']['AP50']:.6f}) niż wspólny kierunek. Wspólny kierunek ma najwyższy punktowy AP50:95 spośród zbadanych wariantów, ale bez rozstrzygającego przedziału względem baseline. Walidacja służyła selekcji, więc jej przedział nie jest niezależną oceną skuteczności.\n\n"
text+='Audyt źródłowy SOURCE_AUDIT.json: 400/400 JPEG zgodnych byte-for-byte z archiwami, ramki i poligony zgodne z właściwymi rekordami frame_number źródłowych annotation.csv; 116 filmów. Nie używano poligonów do wyboru tokenów w modelu.\n\n'
text+=f"Trening wykonał PID {launch['pid']}; ocenę i poprawny świeży reload zakończył PID {p['pid']}. Dwa błędy techniczne naprawiono bez ponownej optymalizacji i bez zmiany FREEZE.json lub DIRECTION.npz: serializacja logu oraz dtype korekty podczas reloadu. Zachowano FAILED_FIRST.json, FAILED_RELOAD.json i logi. Łączny czas runnerów bez importów: {cost['runner_seconds_excluding_imports_total']:.2f} s; optymalizacja i walidacyjna selekcja: {freeze['train_seconds']:.2f} s; czas od pierwszego uruchomienia do zakończenia z importami i naprawami: {cost['end_to_end_wall_seconds']:.2f} s. Koszt i peak GPU w COST.json.\n"
for path in [OUT/'REPORT.md',HERE/'WYNIKI_PILOT.md']:
    original=path.read_text(encoding='utf-8').split('\n\n## Interpretacja końcowa i audyt')[0]
    path.write_text(original+text,encoding='utf-8')
write('SOURCE_HASHES.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.iterdir() if f.is_file()})
write('ARTIFACT_HASHES.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name not in ['ARTIFACT_HASHES.json','RUN.lock']})
print(json.dumps(dict(test=test['shared'],baseline=test['baseline'],cost=cost),ensure_ascii=False))
