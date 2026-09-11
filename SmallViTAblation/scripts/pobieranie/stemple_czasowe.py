#!/usr/bin/env python3
"""Wyciaga wypalony stempel czasowy z klatek Cataract-LMM i sklada w arkusze.

Po co: nazwy klipow sledzenia (TR_0001_S1_P03) NIE zawieraja identyfikatora
pacjenta. Segmentacja go ma (SE_indeks_GLOBALID_S1), sledzenie nie. Nie da sie
wiec z samej nazwy powiedziec, czy dwa klipy pochodza od tej samej operacji,
ani czy klip sledzenia i klatki segmentacji to ten sam pacjent.

Jedyny dowod, jaki zostaje w danych, to napis wypalony przez mikroskop:
nazwa sali i DATA + GODZINA operacji. Dwa nagrania o tej samej dacie i
godzinie to to samo wejscie na sale, czyli ten sam pacjent.

Skrypt wycina pasek z napisem i sklada arkusze do odczytania wzrokowo. Z kazdego
nagrania bierze kilka klatek i wybiera te o najwyzszym kontrascie w pasie
napisu, bo na przeswietlonych klatkach napis znika.

Uzycie:
    py -3.11 stemple_czasowe.py --zrodlo trk      # 170 klipow sledzenia
    py -3.11 stemple_czasowe.py --zrodlo seg      # nagrania segmentacji
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DOMYSLNY_ZBIOR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                      r"\Dataset\Cataract-LMM")
SZER_KAFLA = 560


def pas_napisu(img: np.ndarray) -> np.ndarray:
    """Dolny lewy rog kadru - tam mikroskop wypala nazwe sali i czas."""
    h, w = img.shape[:2]
    return img[int(h * 0.86):h, 0:int(w * 0.55)]


def kontrast(wyc: np.ndarray) -> float:
    """Odchylenie standardowe jasnosci - na przeswietlonej klatce spada do zera."""
    return float(cv2.cvtColor(wyc, cv2.COLOR_BGR2GRAY).std())


def kafel(wyc: np.ndarray, etykieta: str) -> np.ndarray:
    s = SZER_KAFLA / wyc.shape[1]
    wyc = cv2.resize(wyc, (SZER_KAFLA, max(1, int(wyc.shape[0] * s))),
                     interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(wyc, (0, 0), (wyc.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(wyc, etykieta, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
    return wyc


def najlepsza_klatka(obrazy) -> np.ndarray | None:
    """Z kilku kandydatow bierze te z najczytelniejszym napisem."""
    najlepszy, najlepszy_k = None, -1.0
    for img in obrazy:
        if img is None:
            continue
        wyc = pas_napisu(img)
        k = kontrast(wyc)
        if k > najlepszy_k:
            najlepszy, najlepszy_k = wyc, k
    return najlepszy


def ze_sledzenia(zbior: Path, prob: int):
    zs = sorted((zbior / "3_Object_Tracking/annotations").glob("*.zip"))
    for z in zs:
        try:
            with zipfile.ZipFile(z) as f:
                jp = sorted(n for n in f.namelist() if n.endswith(".jpg"))
                if not jp:
                    continue
                krok = max(1, len(jp) // prob)
                kand = [cv2.imdecode(np.frombuffer(f.read(n), np.uint8),
                                     cv2.IMREAD_COLOR)
                        for n in jp[::krok][:prob]]
                wyc = najlepsza_klatka(kand)
                if wyc is not None:
                    yield z.stem, wyc
        except Exception as e:
            print("  pominieto", z.name, e)


def z_segmentacji(zbior: Path, prob: int):
    """Dla segmentacji grupujemy klatki po nagraniu (SE_indeks_GLOBALID_S1)."""
    root = zbior / "2_Instance_Segmentation/annotations/coco_json/images/images/images"
    wg_nagrania: dict[str, list[Path]] = defaultdict(list)
    for p in root.glob("*.jpg"):
        m = re.match(r"(SE_\d+_\d+_S\d)", p.name)
        if m:
            wg_nagrania[m.group(1)].append(p)
    for nazwa in sorted(wg_nagrania):
        pliki = sorted(wg_nagrania[nazwa])
        krok = max(1, len(pliki) // prob)
        kand = [cv2.imread(str(p)) for p in pliki[::krok][:prob]]
        wyc = najlepsza_klatka(kand)
        if wyc is not None:
            yield f"{nazwa}  ({len(pliki)} klatek)", wyc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zrodlo", choices=["trk", "seg"], required=True)
    ap.add_argument("--zbior", type=Path, default=DOMYSLNY_ZBIOR)
    ap.add_argument("--prob", type=int, default=5,
                    help="ile klatek sprawdzic na nagranie")
    ap.add_argument("--na-arkusz", type=int, default=20)
    args = ap.parse_args()

    out = args.zbior / f"_stemple_{args.zrodlo}"
    out.mkdir(parents=True, exist_ok=True)

    zrodlo = ze_sledzenia if args.zrodlo == "trk" else z_segmentacji
    kafle, nazwy = [], []
    for i, (nazwa, wyc) in enumerate(zrodlo(args.zbior, args.prob), 1):
        kafle.append(kafel(wyc, f"{i:03d}  {nazwa}"))
        nazwy.append(nazwa)
        if i % 20 == 0:
            print(f"  {i} nagran", flush=True)

    if not kafle:
        raise SystemExit("Nic nie wyciagnieto.")

    ark = 0
    for i in range(0, len(kafle), args.na_arkusz):
        grupa = kafle[i:i + args.na_arkusz]
        h = min(k.shape[0] for k in grupa)
        siatka = np.vstack([k[:h] for k in grupa])
        ark += 1
        f = out / f"arkusz_{ark:02d}.jpg"
        cv2.imwrite(str(f), siatka)
        print("zapisano:", f.name)

    (out / "nagrania.json").write_text(
        json.dumps(nazwy, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nnagran: {len(nazwy)}   arkuszy: {ark}")
    print("Odczytaj z arkuszy DATE i GODZINE. Ta sama para = ta sama operacja.")


if __name__ == "__main__":
    main()
