"""Wysyła kod na Eden, podpina wejścia twardymi dowiązaniami i sprawdza środowisko. Zgłasza zadanie tylko z --submit."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / HERE.name
B = '/home2/faculty/bpiotrowski'
REMOTE = f'{B}/smallvit_runs/2026-10-06_vit_konwerter_dino_filmy'
ZRODLO = f'{B}/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wejscia'
PY = f'{B}/venvs/rfdetr_nano_clean_1_10_1/bin/python'
DEPS = f'{B}/smallvit_runs/2026-09-28_maly_vit_od_zera/deps'
CK = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
V2 = '--v2' in sys.argv  # wersja 2 w osobnym katalogu kodu; kod wersji 1 zostaje nietknięty
KOD = 'kod_v2' if V2 else 'kod'
SLURM = 'trening_v2.slurm' if V2 else 'trening.slurm'
PLIKI = ['run.py', 'dane.py', 'model.py', 'model_vit.py', 'config_v2.json' if V2 else 'config.json', SLURM]


def call(args):
    p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', timeout=300)
    if p.returncode:
        raise RuntimeError(p.stderr + '\n' + p.stdout)
    return p.stdout.strip()


def ssh(cmd):
    return call(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'eden-cluster', cmd])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
ssh(f'mkdir -p {REMOTE}/{KOD} {REMOTE}/wejscia {REMOTE}/logi {REMOTE}/wyniki')
call(['scp', *[str(HERE / n) for n in PLIKI], f'eden-cluster:{REMOTE}/{KOD}/'])
# Twarde dowiązania: te same bajty co wejścia biegu SD, przetrwają ewentualne sprzątanie tamtego katalogu.
ssh(f'cd {REMOTE}/wejscia && for f in FILMY.json MODEL_VIT_K1000.pt; do [ -e $f ] || ln {ZRODLO}/$f $f; done')
zdalny_ck = ssh(f'sha256sum {REMOTE}/wejscia/MODEL_VIT_K1000.pt').split()[0]
if zdalny_ck != sha(CK):
    raise RuntimeError('Checkpoint ViT na Edenie różni się od lokalnego')
zdalny_kod = {Path(l.split()[1]).name: l.split()[0] for l in ssh(f'cd {REMOTE}/{KOD} && sha256sum {" ".join(PLIKI)}').splitlines()}
if any(zdalny_kod[n] != sha(HERE / n) for n in PLIKI):
    raise RuntimeError('Kod na Edenie różni się od lokalnego')
srodowisko = ssh(f'cd {REMOTE}/{KOD} && PYTHONPATH={DEPS} {PY} -c "import torch,timm,av,cv2,torchvision,PIL,json;'
                 f'import dane,model;r=dane.wczytaj_manifest(\'{REMOTE}/wejscia/FILMY.json\');'
                 f'print(json.dumps(dict(torch=torch.__version__,timm=timm.__version__,av=av.__version__,'
                 f'train=sum(x[\'split\']==\'train\' for x in r),val=sum(x[\'split\']==\'val\' for x in r))))"')
wynik = dict(remote=REMOTE, checkpoint_sha256=zdalny_ck, kod_sha256=zdalny_kod, srodowisko=json.loads(srodowisko.splitlines()[-1]))
if '--submit' in sys.argv:
    if (OUT / ('SUBMISSION_v2.json' if V2 else 'SUBMISSION.json')).exists():
        raise RuntimeError('SUBMISSION.json istnieje: sprawdź poprzednie zgłoszenie zamiast puszczać duplikat')
    wynik['job_id'] = ssh(f'sbatch --parsable {REMOTE}/{KOD}/{SLURM}')
    (OUT / ('SUBMISSION_v2.json' if V2 else 'SUBMISSION.json')).write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
else:
    (OUT / ('DEPLOYMENT_v2.json' if V2 else 'DEPLOYMENT.json')).write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(wynik, indent=2, ensure_ascii=False))
