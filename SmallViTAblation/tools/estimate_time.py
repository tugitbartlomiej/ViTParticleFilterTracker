#!/usr/bin/env python3
"""
Szacunek czasu calego eksperymentu, liczony z listy biegow.

Jednostka niezalezna od sprzetu to OBRAZO-EPOKA: suma po biegach z
(budzet obrazow x liczba epok). Czas = obrazo-epoki / przepustowosc.

Przepustowosci nie zgadujemy w artykule ani w planie — mierzy ja bieg
kalibracyjny (docs/03, sekcja 2). Do czasu pomiaru pokazujemy WIDELKI.

Uzycie:
    py -3.11 tools/estimate_time.py
    py -3.11 tools/estimate_time.py --throughput 55 --gpus 8
    py -3.11 tools/estimate_time.py --epochs 40 --fullpool-images 71000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runs() -> list[dict]:
    import json
    out = ROOT / "_runs_tmp.jsonl"
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_run_manifest.py"),
                    "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    runs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    out.unlink(missing_ok=True)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60, help="stala liczba epok na bieg")
    ap.add_argument("--proxy-epochs", type=int, default=10, help="epoki modelu proxy EL2N")
    ap.add_argument("--proxy-runs", type=int, default=8, help="ile modeli proxy (1 na ziarno)")
    ap.add_argument("--fullpool-images", type=int, default=71000,
                    help="klatki w filmach treningowych (do policzenia, docs/05 O2)")
    ap.add_argument("--throughput", type=float, default=None,
                    help="obrazow/s na 1 GPU; brak = pokaz widelki")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--eval-overhead", type=float, default=0.15,
                    help="narzut na ewaluacje, zapis, kolejkowanie (ulamek)")
    args = ap.parse_args()

    runs = load_runs()

    img_epochs = 0
    for r in runs:
        b = r["budget"]
        if b == "full_train_pool":
            b = args.fullpool_images
        img_epochs += b * args.epochs

    # modele proxy dla EL2N: trenowane na pelnej puli treningowej, krotko
    proxy_img_epochs = args.proxy_runs * args.fullpool_images * args.proxy_epochs
    total = img_epochs + proxy_img_epochs

    print(f"Biegow: {len(runs)}")
    print(f"Obrazo-epoki (treningi):    {img_epochs:,}")
    print(f"Obrazo-epoki (proxy EL2N):  {proxy_img_epochs:,}")
    print(f"Obrazo-epoki RAZEM:         {total:,}")
    print(f"(epok na bieg: {args.epochs}, pula treningowa: {args.fullpool_images:,})")
    print()

    tputs = [args.throughput] if args.throughput else [30, 40, 60, 80, 120]
    print(f"{'img/s na GPU':>13} | {'GPU-godziny':>12} | {f'doba na {args.gpus} GPU':>16}")
    print("-" * 48)
    for t in tputs:
        gpu_h = total / t / 3600 * (1 + args.eval_overhead)
        wall_days = gpu_h / args.gpus / 24
        print(f"{t:>13.0f} | {gpu_h:>12,.0f} | {wall_days:>16.2f}")
    print()
    print("Narzut na ewaluacje/zapis/kolejke wliczony:"
          f" +{args.eval_overhead:.0%}")
    print("Widelki znikaja po biegu kalibracyjnym - wtedy --throughput <zmierzone>.")


if __name__ == "__main__":
    main()
