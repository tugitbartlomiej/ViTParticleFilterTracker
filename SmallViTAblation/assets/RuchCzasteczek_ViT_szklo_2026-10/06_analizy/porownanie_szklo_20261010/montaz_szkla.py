"""Montaż filmu „Ruch tokenów ViT jak w szkle”: karty tytułowe (PIL) + klipy z aplikacji (A2 względem populacji jako kontrast,
G1–G3 układ względem klatki, G4–G5 symulowane szkło koloidalne). Każdy klip osobno przeskalowany do 1920×1080 (letterbox),
jednakowo zakodowany, potem bezstratny concat. Klipy, których nie ma na dysku, są pomijane.
Uruchomienie: F:/Python/3.13/python.exe montaz_szkla.py"""
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

E = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(E, 'animacje')
A2 = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/animacje_ruchu_20261010/A2_drgania_w_klatce_jeden_obraz.mp4'
FONT = r'C:\Windows\Fonts\segoeui.ttf'
FPS = 30


def karta_txt():
    try:
        u = json.load(open(os.path.join(E, 'UKLAD_SZKLA.json'), encoding='utf-8'))
        k = u['karty']
        s1 = u['etapy']['S1']
        return (f"układ klatki: {k['S1']['spelnione']} z {k['S1']['z']} odcisków szkła (populacja: {k['S0']['spelnione']}), "
                f"bez pędu i w czasie termicznym: {k['S4']['spelnione']} · rozmiar klatki {s1['lindemann_shell']:.2f} odstępu sąsiadów, "
                f"antykorelacja {s1['c_plateau']:.2f}")
    except Exception:
        return 'karta zgodności w UKLAD_SZKLA.json'


KLIPY = [
    ('karta', 'Ruch cząsteczek ViT przeniesiony do szkła', 'tokeny ViT w układzie względem klatki (minus ruch 6 sąsiadów) i symulowane szkło koloidalne\n'
     'z tym samym zestawem odcisków palca: MSD, antykorelacja, α₂, Fs(q,τ), ucieczki z klatki, starzenie', 4.0),
    ('klip', A2, 'przed: względem populacji'),
    ('klip', os.path.join(AN, 'G1_szklo_klatka_jedno_zdjecie.mp4'), 'po: względem klatki'),
    ('klip', os.path.join(AN, 'G2_szklo_klatka_cala_chmura.mp4'), ''),
    ('klip', os.path.join(AN, 'G3_szklo_klatka_temperatura.mp4'), ''),
    ('klip', os.path.join(AN, 'G4_szklo_koloidalne_BD.mp4'), ''),
    ('klip', os.path.join(AN, 'G5_szklo_koloidalne_BD.mp4'), ''),
    ('karta', 'Ile w tym szkła', karta_txt() + '\nciecz przechłodzona z pędem, która sama stygnie: klatka luźna, antykorelacja jak u Weeksa, ogony wykładnicze,\n'
     'starzenie przy stałym lr; po usunięciu pędu Adama krótkie czasy stają się dyfuzyjne jak w koloidzie', 7.0),
]


def ease_out(t):
    return 1 - (1 - t) ** 3


def karta(tytul, podtytul, sek, plik):
    n = int(sek * FPS)
    f1 = ImageFont.truetype(FONT, 60); f2 = ImageFont.truetype(FONT, 28)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080',
                             '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-r', str(FPS),
                             '-video_track_timescale', '15360', plik], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        im = Image.new('RGB', (1920, 1080), (16, 23, 28)); d = ImageDraw.Draw(im)
        a = ease_out(min(1.0, t / 1.2))
        d.text((120, 420 + (1 - a) * 60), tytul, font=f1, fill=(int(232 * a) + 16, int(232 * a) + 23, int(232 * a) + 28))
        b = ease_out(max(0.0, min(1.0, (t - 0.8) / 1.0)))
        if b > 0:
            d.multiline_text((120, 520), podtytul, font=f2, fill=(int(170 * b) + 16, int(170 * b) + 23, int(170 * b) + 28), spacing=10)
        d.rectangle((120, 400, 120 + int(a * 420), 404), fill=(255, 209, 102))
        proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close(); proc.wait()
    assert proc.returncode == 0, 'ffmpeg karta'


def segment(zrodlo, plik, etykieta=''):
    vf = 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#10171c,fps=30'
    # etykieta pomijana: ta kompilacja ffmpeg nie ma filtru drawtext, a klipy mają już wpalone napisy
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', zrodlo, '-vf', vf, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-an',
                    '-video_track_timescale', '15360', plik], check=True)


seg = os.path.join(E, 'animacje', 'segmenty'); os.makedirs(seg, exist_ok=True)
lista = []
for i, k in enumerate(KLIPY):
    plik = os.path.join(seg, f'{i:02d}.mp4')
    if k[0] == 'karta':
        karta(k[1], k[2], k[3], plik)
    else:
        if not os.path.isfile(k[1]):
            print('pominięty (brak):', os.path.basename(k[1]), flush=True); continue
        segment(k[1], plik, k[2])
    lista.append(plik); print('segment', i, os.path.basename(k[1]) if k[0] == 'klip' else 'karta', flush=True)
with open(os.path.join(seg, 'concat.txt'), 'w', encoding='utf-8') as f:
    for p in lista:
        f.write(f"file '{p.replace(chr(92), '/')}'\n")
final = os.path.join(E, 'animacje', 'szklo_final.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(seg, 'concat.txt'), '-c', 'copy',
                '-movflags', '+faststart', final], check=True)
out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=width,height', '-of', 'csv=p=0', final],
                     capture_output=True, text=True).stdout
print('szklo_final.mp4:', out.strip().replace('\n', ' '), flush=True)
