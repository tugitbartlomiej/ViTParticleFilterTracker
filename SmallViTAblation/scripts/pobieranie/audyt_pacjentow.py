#!/usr/bin/env python3
"""Sprawdza, czy jeden klip Cataract-LMM to na pewno jeden pacjent.

Problem: nazwy klipow sledzenia (TR_0001_S1_P03) NIE zawieraja identyfikatora
nagrania zrodlowego. Segmentacja go ma (SE_indeks_GLOBALID_S1), sledzenie nie.
Z samej nazwy nie da sie wiec powiedziec, czy dwa klipy pochodza z tej samej
operacji. A jesli pochodza, to podzial na uczenie i test po nazwie klipu
przecieka - ten sam pacjent trafia po obu stronach.

Dwa niezalezne testy:

  1. ODCISK DOKLADNY - skrot z trzech klatek. Wylapuje pliki bajtowo identyczne,
     czyli ten sam material zapisany dwa razy pod dwiema nazwami.

  2. ODCISK PERCEPCYJNY - male szare miniatury kilkunastu klatek, porownywane
     korelacja. Rekompresja JPEG zmienia bajty, wiec test pierwszy przegapilby
     to samo nagranie zakodowane ponownie. Ten test tego nie przegapi.

Trzeci dowod, wypalony w obrazie stempel czasowy mikroskopu, zbiera osobny
skrypt stemple_czasowe.py - tam odczyt jest wzrokowy, bo nie ma OCR.

Uzycie:
    py -3.11 audyt_pacjentow.py
    py -3.11 audyt_pacjentow.py --prog 0.85 --klatek 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DOMYSLNY_ZBIOR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                      r"\Dataset\Cataract-LMM")
BOK = 32


def odciski(ann: Path, klatek: int):
    """Dla kazdego klipu: skrot dokladny i odcisk percepcyjny."""
    dokladne, percepcyjne, dlugosci = {}, {}, {}
    zs = sorted(ann.glob("*.zip"))
    for j, z in enumerate(zs, 1):
        try:
            with zipfile.ZipFile(z) as f:
                jp = sorted(n for n in f.namelist() if n.endswith(".jpg"))
                if not jp:
                    continue
                dlugosci[z.stem] = len(jp)

                h = hashlib.sha1()
                for n in (jp[0], jp[len(jp) // 2], jp[-1]):
                    h.update(f.read(n))
                dokladne[z.stem] = (len(jp), h.hexdigest())

                mini = []
                for i in np.linspace(0, len(jp) - 1, klatek).astype(int):
                    img = cv2.imdecode(np.frombuffer(f.read(jp[i]), np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    m = cv2.resize(img, (BOK, BOK)).astype(np.float32)
                    mini.append((m - m.mean()) / (m.std() + 1e-6))
                if mini:
                    percepcyjne[z.stem] = np.stack(mini)
        except Exception as e:
            print("  pominieto", z.name, e)
        if j % 40 == 0:
            print(f"  {j}/{len(zs)} klipow", flush=True)
    return dokladne, percepcyjne, dlugosci


def pary_podobne(percepcyjne: dict, prog: float):
    nazwy = sorted(percepcyjne)
    wynik = []
    for i in range(len(nazwy)):
        A = percepcyjne[nazwy[i]].reshape(len(percepcyjne[nazwy[i]]), -1)
        for k in range(i + 1, len(nazwy)):
            B = percepcyjne[nazwy[k]].reshape(len(percepcyjne[nazwy[k]]), -1)
            sim = (A @ B.T) / (BOK * BOK)
            s = float(sim.max(axis=1).mean())
            if s >= prog:
                wynik.append((s, nazwy[i], nazwy[k]))
    wynik.sort(reverse=True)
    return wynik


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zbior", type=Path, default=DOMYSLNY_ZBIOR)
    ap.add_argument("--klatek", type=int, default=12,
                    help="ile klatek na odcisk percepcyjny")
    ap.add_argument("--prog", type=float, default=0.75,
                    help="od jakiego podobienstwa raportowac pare")
    args = ap.parse_args()

    ann = args.zbior / "3_Object_Tracking/annotations"
    if not ann.exists():
        raise SystemExit(f"Brak katalogu: {ann}")

    print("Licze odciski klipow...")
    dokladne, percepcyjne, dlugosci = odciski(ann, args.klatek)

    grupy = defaultdict(list)
    for k, v in dokladne.items():
        grupy[v].append(k)
    duplikaty = sorted(sorted(ks) for ks in grupy.values() if len(ks) > 1)

    print(f"\nplikow:              {len(dokladne)}")
    print(f"DUPLIKATY (ten sam material pod dwiema nazwami): {len(duplikaty)}")
    for ks in duplikaty:
        print(f"  {dlugosci[ks[0]]:>5} klatek: {', '.join(ks)}")

    print(f"\nSzukam nagran tej samej operacji (prog {args.prog})...")
    pary = pary_podobne(percepcyjne, args.prog)
    w_duplikatach = {tuple(sorted(ks[:2])) for ks in duplikaty}
    pozostale = [(s, a, b) for s, a, b in pary
                 if tuple(sorted((a, b))) not in w_duplikatach]

    print(f"par powyzej progu (bez duplikatow): {len(pozostale)}")
    for s, a, b in pozostale[:20]:
        znak = "  <-- TA SAMA OPERACJA" if s > 0.9 else ""
        print(f"  {s:.3f}  {a}  <->  {b}{znak}")

    # ile realnie roznych operacji
    scal = {n: n for n in dokladne}

    def znajdz(x):
        while scal[x] != x:
            scal[x] = scal[scal[x]]
            x = scal[x]
        return x

    for ks in duplikaty:
        for k in ks[1:]:
            scal[znajdz(k)] = znajdz(ks[0])
    for s, a, b in pozostale:
        if s > 0.9:
            scal[znajdz(b)] = znajdz(a)
    operacji = len({znajdz(n) for n in dokladne})

    print(f"\nPLIKOW: {len(dokladne)}   ROZNYCH OPERACJI: {operacji}")
    print("Do podzialu na uczenie/walidacje/test uzywaj grup operacji,")
    print("nie nazw plikow - inaczej ten sam pacjent trafi po obu stronach.")

    out = args.zbior / "_audyt_pacjentow.json"
    grupy_operacji = defaultdict(list)
    for n in sorted(dokladne):
        grupy_operacji[znajdz(n)].append(n)
    out.write_text(json.dumps({
        "plikow": len(dokladne),
        "roznych_operacji": operacji,
        "duplikaty": duplikaty,
        "pary_podobne": [{"podobienstwo": round(s, 4), "a": a, "b": b}
                         for s, a, b in pozostale],
        "grupy_operacji": {k: v for k, v in sorted(grupy_operacji.items())},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nzapisano: {out}")


if __name__ == "__main__":
    main()
