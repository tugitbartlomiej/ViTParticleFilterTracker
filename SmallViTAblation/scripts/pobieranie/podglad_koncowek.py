#!/usr/bin/env python3
"""Podglad koncowek narzedzia w 3_Object_Tracking Cataract-LMM.

Robi trzy rzeczy:
  1. siatka wycinkow wokol koncowki z ROZNYCH klipow - jak wyglada adnotacja
     u roznych pacjentow,
  2. pasek kolejnych klatek z jednego klipu - pokazuje, jak bardzo sasiednie
     klatki sa do siebie podobne (to jest nadmiarowosc, ktora kuracja usuwa),
  3. klatka z klipu oznaczonego najgorzej - zeby zobaczyc, co tam nie zagralo.

Uzycie:
    py -3.11 podglad_koncowek.py [katalog_zbioru] [katalog_wyjsciowy]
"""
from __future__ import annotations

import csv
import io
import random
import sys
import zipfile
from ast import literal_eval
from pathlib import Path

import cv2
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else
         r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else D / "_podglad")
OUT.mkdir(parents=True, exist_ok=True)
ANN = D / "3_Object_Tracking/annotations"
CZERWONY = (0, 0, 255)


def wiersze(z: Path):
    with zipfile.ZipFile(z) as f:
        nz = [n for n in f.namelist() if n.endswith("annotation.csv")][0]
        return list(csv.DictReader(io.StringIO(
            f.read(nz).decode("utf-8", "replace"))))


def klatka(z: Path, nazwa: str):
    with zipfile.ZipFile(z) as f:
        p = f"{z.stem}/{nazwa}"
        if p not in f.namelist():
            return None
        return cv2.imdecode(np.frombuffer(f.read(p), np.uint8), cv2.IMREAD_COLOR)


def znacznik(img, tx, ty, promien=18):
    cv2.drawMarker(img, (tx, ty), CZERWONY, cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (tx, ty), promien, CZERWONY, 2)


def siatka_wycinkow(n_klipow: int = 12, bok: int = 220) -> None:
    """Wycinki wokol koncowki, po jednym z kazdego z n roznych klipow."""
    zs = sorted(ANN.glob("*.zip"))
    random.seed(42)
    wybor = random.sample(zs, min(n_klipow, len(zs)))
    kafle = []
    for z in sorted(wybor):
        try:
            rows = [r for r in wiersze(z) if r.get("cystotome_tip", "").strip()]
            if not rows:
                continue
            r = rows[len(rows) // 2]              # srodek klipu
            img = klatka(z, r["filename"])
            if img is None:
                continue
            tx, ty = [int(round(v)) for v in literal_eval(r["cystotome_tip"])]
            h, w = img.shape[:2]
            x0, y0 = max(0, tx - bok // 2), max(0, ty - bok // 2)
            x0, y0 = min(x0, max(0, w - bok)), min(y0, max(0, h - bok))
            wyc = img[y0:y0 + bok, x0:x0 + bok].copy()
            if wyc.shape[0] < bok or wyc.shape[1] < bok:
                wyc = cv2.copyMakeBorder(wyc, 0, bok - wyc.shape[0], 0,
                                         bok - wyc.shape[1], cv2.BORDER_CONSTANT)
            znacznik(wyc, tx - x0, ty - y0)
            cv2.rectangle(wyc, (0, 0), (bok - 1, 18), (0, 0, 0), -1)
            cv2.putText(wyc, f'{z.stem[:12]} kl.{r["frame_number"]}', (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            kafle.append(wyc)
        except Exception as e:
            print("  pominieto", z.name, e)

    if not kafle:
        return
    kol = 4
    wiersz_list = [np.hstack(kafle[i:i + kol]) for i in range(0, len(kafle), kol)
                   if len(kafle[i:i + kol]) == kol]
    if not wiersz_list:
        return
    siatka = np.vstack(wiersz_list)
    pas = np.zeros((30, siatka.shape[1], 3), np.uint8)
    cv2.putText(pas, "KONCOWKI NARZEDZIA - po jednej z roznych klipow (wycinek 220x220 px)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    f = OUT / "koncowki_siatka.jpg"
    cv2.imwrite(str(f), np.vstack([pas, siatka]))
    print("zapisano:", f.name)


def pasek_kolejnych(klip: str = "TR_0001_S1_P03", krok: int = 1, ile: int = 6) -> None:
    """Kolejne klatki obok siebie - ile realnie zmienia sie miedzy nimi."""
    z = ANN / f"{klip}.zip"
    if not z.exists():
        return
    rows = [r for r in wiersze(z) if r.get("cystotome_tip", "").strip()]
    if len(rows) < ile * krok:
        return
    start = len(rows) // 2
    kafle = []
    for i in range(ile):
        r = rows[start + i * krok]
        img = klatka(z, r["filename"])
        if img is None:
            continue
        tx, ty = [int(round(v)) for v in literal_eval(r["cystotome_tip"])]
        znacznik(img, tx, ty)
        img = cv2.resize(img, (360, int(360 * img.shape[0] / img.shape[1])))
        cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(img, f'klatka {r["frame_number"]}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        kafle.append(img)
    if len(kafle) < 2:
        return
    kol = 3
    ws = [np.hstack(kafle[i:i + kol]) for i in range(0, len(kafle), kol)
          if len(kafle[i:i + kol]) == kol]
    siatka = np.vstack(ws)
    pas = np.zeros((30, siatka.shape[1], 3), np.uint8)
    cv2.putText(pas, f"KOLEJNE KLATKI z {klip} - co {krok} klatke. Widac, jak malo sie zmienia.",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    f = OUT / "koncowki_kolejne_klatki.jpg"
    cv2.imwrite(str(f), np.vstack([pas, siatka]))
    print("zapisano:", f.name)


def klip_bez_adnotacji(klip: str = "TR_0162_S1_P03") -> None:
    """Klip oznaczony w zero procent - co tam widac."""
    z = ANN / f"{klip}.zip"
    if not z.exists():
        return
    rows = wiersze(z)
    r = rows[len(rows) // 2]
    img = klatka(z, r["filename"])
    if img is None:
        return
    for pole, kol in (("cornea", (255, 200, 0)), ("pupil", (255, 120, 60)),
                      ("cystotome", (0, 220, 60))):
        bb = r.get(f"{pole}_bbox", "").strip()
        if bb:
            try:
                x, y, w, h = [int(round(v)) for v in literal_eval(bb)]
                cv2.rectangle(img, (x, y), (x + w, y + h), kol, 2)
                cv2.putText(img, pole, (x + 3, max(y - 5, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, kol, 1, cv2.LINE_AA)
            except Exception:
                pass
    tip = r.get("cystotome_tip", "").strip()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f'{klip} klatka {r["frame_number"]} - koncowka: '
                     f'{"JEST" if tip else "BRAK"} (klip oznaczony w 0%)',
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    f = OUT / "koncowki_klip_bez_adnotacji.jpg"
    cv2.imwrite(str(f), img)
    print("zapisano:", f.name)


if __name__ == "__main__":
    print(f"Wyjscie: {OUT}\n")
    siatka_wycinkow()
    pasek_kolejnych()
    klip_bez_adnotacji()
