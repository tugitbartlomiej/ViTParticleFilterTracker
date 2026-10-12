"""Skleja Scena1..Scena9 w jeden film i sprawdza wynik: Full HD, 24 kl./s, brak dźwięku, cały strumień dekodowalny.

    python zloz.py  ->  videos/Tozsamosc_czasteczki_pary400.mp4, MANIFEST_FILMU.json, kontrola/*.png
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / 'videos'
OUT = V / 'Tozsamosc_czasteczki_pary400.mp4'
SCENY = [V / f'Scena{i}.mp4' for i in range(1, 10)]


def probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type,width,height,r_frame_rate',
                        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    video = [s for s in d['streams'] if s['codec_type'] == 'video'][0]
    return dict(sekundy=float(d['format']['duration']), szerokosc=video['width'], wysokosc=video['height'],
                kl_s=video['r_frame_rate'], audio=any(s['codec_type'] == 'audio' for s in d['streams']))


def main():
    for p in SCENY:
        assert p.exists(), p
    lista = HERE / 'concat.txt'
    lista.write_text(''.join(f"file '{p.as_posix()}'\n" for p in SCENY), encoding='utf-8')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lista),
                    '-c', 'copy', '-movflags', '+faststart', str(OUT)], check=True)
    sceny = {p.name: probe(p) for p in SCENY}
    calosc = probe(OUT)
    dekod = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(OUT), '-f', 'null', '-'], capture_output=True, text=True)
    suma = sum(s['sekundy'] for s in sceny.values())
    assert calosc['szerokosc'] == 1920 and calosc['wysokosc'] == 1080, calosc
    assert abs(calosc['sekundy'] - suma) < 1.0, (calosc['sekundy'], suma)
    assert not dekod.stderr.strip(), dekod.stderr[:500]
    K = HERE / 'kontrola'
    K.mkdir(exist_ok=True)
    start = 0.0
    for p in SCENY:                      # klatka z końca każdej sceny, wycięta z filmu zbiorczego
        start += sceny[p.name]['sekundy']
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{start - .3:.2f}', '-i', str(OUT),
                        '-frames:v', '1', str(K / f'{p.stem}_koniec.png')], check=True)
    manifest = dict(film=OUT.name, calosc=calosc, sceny=sceny, dekodowanie_bez_bledow=True, glos=False)
    (HERE / 'MANIFEST_FILMU.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"{OUT.name}: {calosc['sekundy']:.1f} s, {calosc['szerokosc']}x{calosc['wysokosc']}, {calosc['kl_s']} kl./s, audio={calosc['audio']}")


if __name__ == '__main__':
    main()
