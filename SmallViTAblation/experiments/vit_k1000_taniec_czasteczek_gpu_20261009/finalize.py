"""Zapisuje kartę wyników po weryfikacji GPU: status, sumy plików kodu i wyników."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'outputs' / HERE.name


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


proof = json.loads((OUT / 'WERYFIKACJA_GPU.json').read_text(encoding='utf-8'))
if proof['status'] != 'passed':
    raise SystemExit('Weryfikacja GPU nie przeszła; karta nie zostanie zapisana.')
code = sorted([HERE / 'serve.py', *(HERE / 'web').glob('*')])
artifacts = sorted([OUT / 'README.md', OUT / 'URUCHOM_WIZUALIZACJE.ps1', OUT / 'WERYFIKACJA_GPU.json', *(OUT / 'klatki').glob('*.png')])
card = json.loads((HERE / 'WYNIKI.json').read_text(encoding='utf-8'))
card.update({
    'status': 'complete', 'tag': 'webgpu_compute_wspolna_PCA3', 'ziarno': 20261009,
    'source_npz_sha256': proof['source_npz_sha256'],
    'verification': {'cases_passed': proof['cases_passed'], 'cases_total': proof['cases_total'],
                     'gpu': proof['gpu']['adapter'], 'verified_utc': proof['verified_utc'],
                     'kin_milestone_900000_ms_per_dispatch_wall': proof['timings']['kin_milestone_900000_ms_per_dispatch_wall'],
                     'timing_scope': proof['timings']['scope']},
    'url': 'http://127.0.0.1:8775/',
    'code': [{'path': p.relative_to(HERE).as_posix(), 'sha256': sha(p)} for p in code],
    'artifacts': [{'filename': p.relative_to(OUT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in artifacts],
    'limitations': ['Rzut 3D pomija pozostale wymiary', 'Odcinki miedzy stanami 0/100/500 nie sa zmierzona trasa',
                    'Norma 768D z eksportu; surowe latenty tylko na Edenie', 'Oracle nie mierzy generalizacji'],
    'completed_utc': datetime.now(timezone.utc).isoformat(),
})
(OUT / 'WYNIKI.json').write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"Zapisano WYNIKI.json: {proof['cases_passed']}/{proof['cases_total']} przypadków, {len(artifacts)} plików wyników.")
