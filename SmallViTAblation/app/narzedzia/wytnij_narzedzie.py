"""Sekwencja klatek z samym narzędziem na czarnym tle (z obrysu cystotome_seg z etykiet Cataract-LMM).

Każda klatka: piksele wewnątrz obrysu narzędzia (wypełniony wielokąt plus linia obwodu) mają oryginalny kolor,
cała reszta jest dokładnie czarna (0, 0, 0). Klatki to JPG z zipa adnotacji, czyli te same, na których robiono
etykiety, w natywnej rozdzielczości 720x480. Obok: maski (białe narzędzie na czarnym) i MANIFEST.json.

Użycie:  python wytnij_narzedzie.py [film] [od_sekundy] [ile_klatek]
         python wytnij_narzedzie.py TR_0001_S1_P03 20 400
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import film as F  # noqa: E402

FPS = 29.97
CEL = Path(__file__).resolve().parents[1] / 'narzedzie_czarne_tlo'


def maska_narzedzia(obrysy, ksztalt):
    """Wielokąty obrysu [n, 2] w pikselach -> maska uint8 0/255 (wnętrze i linia obwodu)."""
    m = np.zeros(ksztalt, np.uint8)
    w = [np.round(p).astype(np.int32) for p in obrysy]
    if w:
        cv2.fillPoly(m, w, 255)
        cv2.polylines(m, w, True, 255, 1)
    return m


def main():
    film = sys.argv[1] if len(sys.argv) > 1 else 'TR_0001_S1_P03'
    od_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    e = F.etykiety(film)
    i0 = int(round(od_s * FPS))                        # indeks klatki od 0; frame_number = i + 1
    indeksy = list(range(i0, i0 + n))
    katalog = CEL / f'{film}_{F.mmss(od_s).replace(":", "m")}s_{n}kl'
    (katalog / 'maski').mkdir(parents=True, exist_ok=True)
    wpisy, bez = [], 0
    for k, (i, rgb) in enumerate(F.czytaj_klatki_jpg(film, indeksy)):
        et = F.etykieta_klatki(e.get(i + 1), skala=(1, 1))
        m = maska_narzedzia(et['obrys'], rgb.shape[:2])
        bez += not m.any()
        wynik = np.where(m[..., None] > 0, rgb, 0).astype(np.uint8)
        nazwa = f'klatka_{k + 1:04d}.png'
        Image.fromarray(wynik).save(katalog / nazwa)
        Image.fromarray(m).save(katalog / 'maski' / nazwa)
        wpisy.append(dict(plik=nazwa, frame_number=i + 1, czas_s=round(i / FPS, 3),
                          pikseli_narzedzia=int((m > 0).sum()), wielokatow=len(et['obrys']),
                          czubek=et['czubek'], ramka_xyxy=et['ramka']))
        if (k + 1) % 50 == 0:
            print(f'{k + 1}/{n}', flush=True)
    opis = dict(film=film, od=F.mmss(od_s), do=F.mmss((i0 + n - 1) / FPS), klatek=n, fps=FPS,
                rozdzielczosc=[int(rgb.shape[1]), int(rgb.shape[0])],
                zrodlo='JPG z Dataset/Cataract-LMM/3_Object_Tracking/annotations/<film>.zip, maska z cystotome_seg',
                tlo='dokładnie czarne (0, 0, 0) poza obrysem narzędzia', klatek_bez_obrysu=bez, klatki=wpisy)
    (katalog / 'MANIFEST.json').write_text(json.dumps(opis, ensure_ascii=False, indent=1), encoding='utf-8')
    print(katalog, f'klatek {n}, bez obrysu {bez}')


if __name__ == '__main__':
    main()
