"""Wycina kilka klatek z filmów Cataract-LMM do zdjeciatestowe/.

Klatki pochodzą z zipów adnotacji 3_Object_Tracking (te same klatki co w filmach
MP4, zapisane jako JPG). Wybierane są klatki z widocznym narzędziem
(niepuste cystotome_bbox): początek, środek i koniec zakresu występowania.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

ZRODLO = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM\3_Object_Tracking\annotations')
MANIFEST_FILMOW = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_sd_pelny_latent_filmy_eden_20261006\FILMY.json')
CEL = Path(__file__).resolve().parents[1] / 'zdjeciatestowe'

FILMY = ['TR_0001_S1_P03', 'TR_0025_S1_P03', 'TR_0050_S1_P03',
         'TR_0077_S1_P03', 'TR_0127_S1_P03', 'TR_0140_S1_P03']


def main():
    CEL.mkdir(exist_ok=True)
    splity = {}
    if MANIFEST_FILMOW.exists():
        for wpis in json.loads(MANIFEST_FILMOW.read_text(encoding='utf-8-sig')):
            splity[wpis['video_id']] = wpis['split']
    zapisane = []
    for film in FILMY:
        sciezka = ZRODLO/f'{film}.zip'
        if not sciezka.exists():
            print(f'{film}: brak zipa, pomijam')
            continue
        with zipfile.ZipFile(sciezka) as z:
            wiersze = list(csv.DictReader(io.StringIO(z.read(f'{film}/annotation.csv').decode('utf-8-sig'))))
            z_narzedziem = [int(w['frame_number']) for w in wiersze if w['cystotome_bbox'].strip()]
            if not z_narzedziem:
                print(f'{film}: brak klatek z narzędziem, pomijam')
                continue
            wybrane = sorted({z_narzedziem[0], z_narzedziem[len(z_narzedziem)//2], z_narzedziem[-1]})
            for klatka in wybrane:
                member = f'{film}/{film}_{klatka:06d}.jpg'
                nazwa = f'{film}_klatka_{klatka:06d}.jpg'
                (CEL/nazwa).write_bytes(z.read(member))
                zapisane.append(dict(plik=nazwa, film=film, klatka=klatka,
                                     split=splity.get(film, '?'),
                                     klatek_z_narzedziem=len(z_narzedziem)))
                print(f'{film}: klatka {klatka} -> {nazwa}')
    (CEL/'MANIFEST.json').write_text(json.dumps(zapisane, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Zapisano {len(zapisane)} klatek do {CEL}')


if __name__ == '__main__':
    main()
