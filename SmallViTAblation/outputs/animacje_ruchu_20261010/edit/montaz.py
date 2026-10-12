"""Montaż final.mp4: karty tytułowe (PIL) + klipy A1–A6 z aplikacji + sloty animacji z danych.
Każdy klip osobno przeskalowany do 1920×1080 (letterbox) i zakodowany jednakowo, potem bezstratny concat (-c copy).
Uruchomienie: F:/Python/3.13/python.exe montaz.py"""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

E = os.path.dirname(os.path.abspath(__file__))
A = os.path.dirname(E)
FONT = r'C:\Windows\Fonts\segoeui.ttf'
FPS = 30
KLIPY = [
    ('karta', 'Ruch cząsteczek w latencie ViT podczas uczenia', 'sześć typów ruchu nagranych z aplikacji (900 000 cząsteczek na karcie)\n'
     'i dwie animacje z gęstego zapisu co 1 krok · historia K1000 i bieg na 35 000 klatkach', 4.0),
    ('klip', os.path.join(A, 'A1_rozgrzewka_przeplyw_kolektywny.mp4')),
    ('klip', os.path.join(A, 'A3_rozgrzewka_na_siatce_zdjecia.mp4')),
    ('klip', os.path.join(A, 'A2_drgania_w_klatce_jeden_obraz.mp4')),
    ('klip', os.path.join(E, 'animations', 'slot_1', 'render.mp4')),
    ('klip', os.path.join(A, 'A4_temperatura_narzedzie_vs_tlo.mp4')),
    ('klip', os.path.join(A, 'A5_atencja_i_IoU_rozpoznanie_narzedzia.mp4')),
    ('klip', os.path.join(A, 'A6_epoki_kondensacja_chmury.mp4')),
    ('klip', os.path.join(E, 'animations', 'slot_2', 'render.mp4')),
    ('karta', 'Skąd wiemy, że sieć nauczyła się narzędzia', 'IoU i pewność rosną do plateau · atencja skupia się na narzędziu · temperatura narzędzia\n'
     'rozdziela je od tła (AUC 0,88) · chmura kondensuje się do ~8. epoki, potem mAP walidacji już nie rośnie', 6.0),
]


def ease_out(t):
    return 1 - (1 - t) ** 3


def karta(tytul, podtytul, sek, plik):
    """Karta tytułowa: tytuł wjeżdża z dołu (ease-out), podtytuł pojawia się po 0,8 s; ostatnia klatka trzymana."""
    n = int(sek * FPS)
    f1 = ImageFont.truetype(FONT, 64); f2 = ImageFont.truetype(FONT, 30)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080',
                             '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-r', str(FPS),
                             '-video_track_timescale', '15360', plik], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        im = Image.new('RGB', (1920, 1080), (16, 23, 28)); d = ImageDraw.Draw(im)
        a = ease_out(min(1.0, t / 1.2))
        y = 440 + (1 - a) * 60
        d.text((120, y), tytul, font=f1, fill=(int(232 * a) + 16, int(232 * a) + 23, int(232 * a) + 28))
        b = ease_out(max(0.0, min(1.0, (t - 0.8) / 1.0)))
        if b > 0:
            d.multiline_text((120, 540), podtytul, font=f2, fill=(int(170 * b) + 16, int(170 * b) + 23, int(170 * b) + 28), spacing=10)
        d.rectangle((120, 420, 120 + int(a * 420), 424), fill=(255, 209, 102))
        proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close(); proc.wait()
    assert proc.returncode == 0, 'ffmpeg karta'


def segment(zrodlo, plik):
    """Skalowanie do 1920×1080 z letterboxem, jednolity kodek (per-segment extract)."""
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', zrodlo, '-vf',
                    'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#10171c,fps=30',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-an', '-video_track_timescale', '15360', plik], check=True)


seg = os.path.join(E, 'segmenty'); os.makedirs(seg, exist_ok=True)
lista = []
for i, k in enumerate(KLIPY):
    plik = os.path.join(seg, f'{i:02d}.mp4')
    if k[0] == 'karta':
        karta(k[1], k[2], k[3], plik)
    else:
        assert os.path.isfile(k[1]), k[1]
        segment(k[1], plik)
    lista.append(plik); print('segment', i, os.path.basename(k[1]) if k[0] == 'klip' else 'karta', flush=True)
with open(os.path.join(seg, 'concat.txt'), 'w', encoding='utf-8') as f:
    for p in lista:
        f.write(f"file '{p.replace(chr(92), '/')}'\n")
final = os.path.join(E, 'final.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(seg, 'concat.txt'),
                '-c', 'copy', '-movflags', '+faststart', final], check=True)
out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=width,height,r_frame_rate',
                      '-of', 'csv=p=0', final], capture_output=True, text=True).stdout
print('final.mp4:', out.strip().replace('\n', ' '), flush=True)
