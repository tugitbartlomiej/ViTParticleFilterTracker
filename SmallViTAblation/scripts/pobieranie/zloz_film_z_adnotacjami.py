#!/usr/bin/env python3
"""Sklada caly klip Cataract-LMM w film z naniesionymi adnotacjami narzedzia.

Po co: probki pojedynczych klatek pokazaly okolo 90% trafnych adnotacji, ale
z pojedynczej klatki nie widac, KIEDY adnotacja sie psuje. Na filmie widac to
od razu: znacznik odrywa sie od narzedzia, ramka zostaje w tyle za ruchem albo
adnotacja znika na kilkadziesiat klatek.

Co rysuje:
  - maske narzedzia (polprzezroczysta),
  - ramke narzedzia z podanym rozmiarem w pikselach,
  - koncowke jako krzyzyk w kolku,
  - slad koncowki z ostatnich N klatek (widac drgania i przeskoki),
  - pasek u gory: numer klatki, czas, co jest oznaczone,
  - pasek u dolu: os czasu klipu z zaznaczonymi lukami w adnotacji.

Uzycie:
    py -3.11 zloz_film_z_adnotacjami.py TR_0011_S1_P03
    py -3.11 zloz_film_z_adnotacjami.py TR_0011_S1_P03 --anatomia --fps 25
    py -3.11 zloz_film_z_adnotacjami.py --lista          # jakie klipy sa na dysku
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from ast import literal_eval
from collections import deque
from pathlib import Path

import cv2
import numpy as np

DOMYSLNY_ZBIOR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                      r"\Dataset\Cataract-LMM")

# BGR: anatomia chlodna, narzedzie ciepłe, koncowka czerwona
KOLOR = {
    "cornea": (255, 200, 0),
    "pupil": (255, 120, 60),
    "cystotome": (0, 220, 60),
    "tip": (0, 0, 255),
    "slad": (0, 160, 255),
}
OPIS = {"cornea": "Rogowka", "pupil": "Zrenica", "cystotome": "Narzedzie"}


def podpis(img, tekst, x, y, kol):
    (w, h), _ = cv2.getTextSize(tekst, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(y, h + 6)
    x = min(x, img.shape[1] - w - 8)
    cv2.rectangle(img, (x, y - h - 5), (x + w + 6, y + 3), kol, -1)
    cv2.putText(img, tekst, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)


def wczytaj_wiersze(zip_klipu: Path) -> list[dict]:
    with zipfile.ZipFile(zip_klipu) as f:
        nazwa = [n for n in f.namelist() if n.endswith("annotation.csv")][0]
        rows = list(csv.DictReader(io.StringIO(
            f.read(nazwa).decode("utf-8", "replace"))))
    rows.sort(key=lambda r: int(r["frame_number"]))
    return rows


def pasek_osi_czasu(szer: int, wys: int, rows: list[dict], idx: int) -> np.ndarray:
    """Pasek pokazujacy, na ktorych klatkach klipu jest koncowka."""
    pas = np.full((wys, szer, 3), 40, np.uint8)
    n = len(rows)
    for i, r in enumerate(rows):
        x = int(i / max(1, n - 1) * (szer - 1))
        ma_tip = bool(r.get("cystotome_tip", "").strip())
        pas[6:wys - 6, x] = (60, 200, 60) if ma_tip else (60, 60, 200)
    x = int(idx / max(1, n - 1) * (szer - 1))
    cv2.line(pas, (x, 0), (x, wys), (255, 255, 255), 2)
    return pas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("klip", nargs="?", help="np. TR_0011_S1_P03")
    ap.add_argument("--zbior", type=Path, default=DOMYSLNY_ZBIOR)
    ap.add_argument("--wyjscie", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--anatomia", action="store_true",
                    help="rysuj takze rogowke i zrenice")
    ap.add_argument("--slad", type=int, default=30,
                    help="ile ostatnich pozycji koncowki rysowac (0 = bez sladu)")
    ap.add_argument("--od", type=int, default=0, help="pierwsza klatka")
    ap.add_argument("--do", type=int, default=0, help="ostatnia klatka (0 = do konca)")
    ap.add_argument("--lista", action="store_true", help="wypisz dostepne klipy")
    args = ap.parse_args()

    ann = args.zbior / "3_Object_Tracking/annotations"
    if args.lista or not args.klip:
        zs = sorted(ann.glob("*.zip"))
        print(f"klipow na dysku: {len(zs)}\n")
        for z in zs[:40]:
            print(" ", z.stem)
        if len(zs) > 40:
            print(f"  ... i {len(zs) - 40} wiecej")
        return

    z = ann / f"{args.klip}.zip"
    if not z.exists():
        raise SystemExit(f"Brak klipu: {z}\nUruchom z --lista, zeby zobaczyc dostepne.")

    rows = wczytaj_wiersze(z)
    a, b = args.od, (args.do or len(rows))
    rows = rows[a:b]
    if not rows:
        raise SystemExit("Wybrany zakres klatek jest pusty.")

    out = args.wyjscie or (args.zbior / "_filmy" / f"{args.klip}_adnotacje.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    pola = [("cystotome", True)]
    if args.anatomia:
        pola = [("cornea", False), ("pupil", False), ("cystotome", True)]

    slad: deque[tuple[int, int]] = deque(maxlen=max(0, args.slad))
    pisarz = None
    n_tip = n_bbox = 0

    with zipfile.ZipFile(z) as f:
        nazwy = set(f.namelist())
        for i, r in enumerate(rows):
            sciezka = f"{args.klip}/{r['filename']}"
            if sciezka not in nazwy:
                continue
            img = cv2.imdecode(np.frombuffer(f.read(sciezka), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue

            # maski
            nakl = img.copy()
            for pole, _ in pola:
                seg = r.get(f"{pole}_seg", "").strip()
                if not seg:
                    continue
                try:
                    for wielokat in literal_eval(seg):
                        pts = np.array(wielokat, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(nakl, [pts], KOLOR[pole])
                except Exception:
                    pass
            img = cv2.addWeighted(nakl, 0.3, img, 0.7, 0)

            # ramki
            for pole, glowne in pola:
                bb = r.get(f"{pole}_bbox", "").strip()
                if not bb:
                    continue
                try:
                    x, y, w, h = [int(round(v)) for v in literal_eval(bb)]
                except Exception:
                    continue
                cv2.rectangle(img, (x, y), (x + w, y + h), KOLOR[pole],
                              2 if glowne else 1)
                podpis(img, f"{OPIS[pole]} {w}x{h}px", x, y, KOLOR[pole])
                if glowne:
                    n_bbox += 1

            # slad koncowki - rysowany PRZED biezaca koncowka, zeby jej nie zaslonic
            tip = r.get("cystotome_tip", "").strip()
            if tip:
                try:
                    tx, ty = [int(round(v)) for v in literal_eval(tip)]
                    slad.append((tx, ty))
                except Exception:
                    tip = ""
            if len(slad) > 1:
                for k in range(1, len(slad)):
                    waga = k / len(slad)
                    cv2.line(img, slad[k - 1], slad[k], KOLOR["slad"],
                             1 + int(waga * 2), cv2.LINE_AA)

            if tip:
                tx, ty = slad[-1]
                cv2.drawMarker(img, (tx, ty), KOLOR["tip"], cv2.MARKER_CROSS, 24, 2)
                cv2.circle(img, (tx, ty), 12, KOLOR["tip"], 2)
                n_tip += 1

            # paski informacyjne
            szer = img.shape[1]
            gora = np.zeros((26, szer, 3), np.uint8)
            czas = int(r["frame_number"]) / args.fps
            stan = []
            if tip:
                stan.append("koncowka")
            if r.get("cystotome_bbox", "").strip():
                stan.append("ramka")
            if not stan:
                stan.append("BRAK ADNOTACJI")
            cv2.putText(gora, f'{args.klip}  klatka {r["frame_number"]}  '
                              f'{czas//60:02.0f}:{czas%60:04.1f}  |  {", ".join(stan)}',
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255) if tip else (80, 80, 255), 1, cv2.LINE_AA)
            dol = pasek_osi_czasu(szer, 22, rows, i)
            kadr = np.vstack([gora, img, dol])

            if pisarz is None:
                pisarz = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (kadr.shape[1], kadr.shape[0]))
                if not pisarz.isOpened():
                    raise SystemExit("Nie udalo sie otworzyc pliku wyjsciowego.")
            pisarz.write(kadr)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(rows)} klatek", flush=True)

    if pisarz is None:
        raise SystemExit("Nie zapisano ani jednej klatki - sprawdz nazwe klipu.")
    pisarz.release()

    n = len(rows)
    print(f"\nzapisano: {out}")
    print(f"klatek:          {n:,}")
    print(f"z koncowka:      {n_tip:,}  ({n_tip / n * 100:.1f}%)")
    print(f"z ramka:         {n_bbox:,}  ({n_bbox / n * 100:.1f}%)")
    print(f"dlugosc:         {n / args.fps / 60:.1f} min przy {args.fps:g} kl/s")
    print("\nPasek na dole: zielony = klatka ma koncowke, czerwony = nie ma.")


if __name__ == "__main__":
    main()
