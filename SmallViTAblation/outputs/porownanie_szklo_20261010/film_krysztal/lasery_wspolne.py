"""Wspólne dla filmów laserowych (film_lasery_1_zapalanie, film_lasery_2_ruch, film_lasery_3_uczenie): dane z szescian.npz, Prokrustes 2D,
kodowanie pozycji MalyViT (sincos_2d jak w model.py), światło studni, zapis ffmpeg."""
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EP, H, W, TOK = 1094, 20, 30, 600
ZIELONY = np.array([0.22, 1.0, 0.48]); BIALOZIEL = np.array([0.75, 1.0, 0.85])


def dane(J=0):
    d = np.load(HERE / 'szescian.npz')
    return dict(kroki=d['kroki'], g3=d['g'], A3=d[f'atomy_{J}'].astype(np.float64), Q3=d[f'wezly_{J}'].astype(np.float64),
                iou=d[f'iou_{J}'], narz=d[f'narzedzie_{J}'], split=str(d[f'split_{J}']))


def prokrustes2d(Q, g):
    """Dla każdego checkpointu: przesunięcie t, kąt (stopnie), skala s mapujące węzły Q[t] na siatkę idealną g; zwraca też Q, A po przekształceniu."""
    K = len(Q); par = np.zeros((K, 4)); mapy = []
    for t in range(K):
        mq, mg = Q[t].mean(0), g.mean(0); qc, gc = Q[t] - mq, g - mg
        U_, S_, Vt = np.linalg.svd(qc.T @ gc); dd = np.sign(np.linalg.det(U_ @ Vt)); Dg = np.diag([1, dd]); R = U_ @ Dg @ Vt
        sk = float((S_ * np.diag(Dg)).sum() / (qc ** 2).sum())
        par[t] = [*(mq - mg), np.degrees(np.arctan2(R[1, 0], R[0, 0])), 1 / sk]
        mapy.append((mq, R, sk, mg))
    return par, mapy


def zastosuj(P, mapa):
    mq, R, sk, mg = mapa
    return (P - mq) @ R * sk + mg


def sincos_2d(h=H, w=W, dim=192):
    y, x = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing='ij')
    d = dim // 4; om = 1.0 / (10000 ** (np.arange(d) / d))
    ox, oy = x.reshape(-1, 1) * om, y.reshape(-1, 1) * om
    return np.concatenate([np.sin(ox), np.cos(ox), np.sin(oy), np.cos(oy)], 1), om


class Swiatlo:
    """Obraz RGB studni światła (gaussy w węzłach) na siatce pikseli; współrzędne w komórkach, środek siatki w 0."""

    def __init__(self, XL, YL, PX=22, SIG=0.26):
        self.XL, self.YL, self.PX, self.SIG = XL, YL, PX, SIG
        self.NX, self.NY = int(2 * XL * PX), int(2 * YL * PX)
        self.R = int(3 * SIG * PX) + 1
        self.ky, self.kx = np.mgrid[-self.R:self.R + 1, -self.R:self.R + 1] / PX
        self.xs = (np.arange(self.NX) + 0.5) / PX - XL; self.ys = (np.arange(self.NY) + 0.5) / PX - YL

    def studnie(self, q, jasnosc=None):
        I = np.zeros((self.NY, self.NX)); R, PX = self.R, self.PX
        for n, (x_, y_) in enumerate(q):
            cx, cy = int((x_ + self.XL) * PX), int((y_ + self.YL) * PX)
            x0, x1, y0, y1 = cx - R, cx + R + 1, cy - R, cy + R + 1
            if x0 < 0 or y0 < 0 or x1 > self.NX or y1 > self.NY:
                continue
            ox, oy = (x_ + self.XL) * PX - cx - 0.5, (y_ + self.YL) * PX - cy - 0.5
            I[y0:y1, x0:x1] += (1 if jasnosc is None else jasnosc[n]) * np.exp(-((self.kx - ox / PX) ** 2 + (self.ky - oy / PX) ** 2) / (2 * self.SIG ** 2))
        return np.clip(I, 0, 1.4)

    def rgb(self, I, tlo=None, kolor=ZIELONY):
        L = (0.55 * I)[..., None] * kolor + (0.25 * I ** 3)[..., None] * BIALOZIEL
        if tlo is not None:
            L = L + tlo
        return np.clip(L, 0, 1)


def ffmpeg(sciezka, fps=30):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '1920x1080', '-r', str(fps), '-i', '-',
                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-maxrate', '12M', '-bufsize', '24M', '-preset', 'medium', str(sciezka)],
                            stdin=subprocess.PIPE)


def kolor_fali(lam, lam_min=6.28, lam_max=400.0):
    """Kolor wiązki zależny od długości fali kodowania (krótka = fiolet, długa = czerwień), jak widmo światła."""
    import matplotlib.pyplot as plt
    u = np.clip((np.log(lam) - np.log(lam_min)) / (np.log(lam_max) - np.log(lam_min)), 0, 1)
    return np.array(plt.cm.turbo(0.08 + 0.87 * u)[:3])


def oko(sciezka=HERE / 'zdjecie_train0.jpg'):
    import cv2
    im = cv2.cvtColor(cv2.imread(str(sciezka)), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (W * 16, H * 16), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.
