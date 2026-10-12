"""Rejestruje faktyczny stan Slurm bez udawania ukończonego eksperymentu."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--status',required=True);p.add_argument('--report-job');a=p.parse_args()
print(zapisz_wynik(__file__,'eden_20261004',dict(ziarno=20261004,status=a.status,smoke_job=1822385,array_job=1822386,report_job=a.report_job,remote='eden-cluster:smallvit_runs/2026-10-04_vit_optymalizacja_latentu128',not_completed=a.status!='complete')))
