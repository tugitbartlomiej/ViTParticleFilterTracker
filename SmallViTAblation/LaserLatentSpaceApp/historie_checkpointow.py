"""Historie checkpointów małego ViT do pomiaru naturalnego i detekcji: lokalne katalogi kNNNN.pt z kolejnych kroków.

Każda historia to katalog wag (`k<krok>.pt` = {'net', 'krok'}), logi (LOG.json, OCENY.json) i opcjonalnie długość
epoki w krokach. Dla historii z epokami krok jest opisywany numerem epoki, a wybór „od epoki / do epoki” zamienia się
na kroki 1094 · e. Lista kroków pochodzi z nazw plików na dysku: nic nie jest zgadywane ani dopisywane.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Historia:
    klucz: str
    nazwa: str
    wagi: Path
    dane: Path
    na_epoke: int | None = None          # kroków na epokę (None: historia bez epok, opis krokami)
    opis: str = ''

    def kroki(self):
        """Kroki zapisanych checkpointów, rosnąco (puste, gdy katalogu nie ma)."""
        if not self.wagi.is_dir():
            return []
        out = []
        for p in self.wagi.iterdir():
            m = re.fullmatch(r'k([0-9]+)\.pt', p.name)
            if m and p.is_file():
                out.append(int(m[1]))
        return sorted(out)

    def sciezka(self, krok):
        for p in self.wagi.glob('k*.pt'):
            m = re.fullmatch(r'k([0-9]+)\.pt', p.name)
            if m and int(m[1]) == krok:
                return p
        raise FileNotFoundError(f'brak checkpointu kroku {krok} w {self.wagi}')

    def epoki(self):
        """Numery epok z zapisanym końcem epoki (0 = stan początkowy), gdy historia ma epoki."""
        if not self.na_epoke:
            return []
        return [k // self.na_epoke for k in self.kroki() if k % self.na_epoke == 0]

    def opis_kroku(self, krok):
        if self.na_epoke and krok % self.na_epoke == 0:
            return f'epoka {krok // self.na_epoke} (krok {krok})'
        if self.na_epoke:
            return f'krok {krok} (epoka {krok / self.na_epoke:.2f})'
        return f'krok {krok}'


HISTORIE = (
    Historia('k1000', 'K1000: gęsta historia, kroki 0–1000 co 10',
             ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi',
             ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane',
             opis='Uczenie od zera na bazie 1500 klatek, 101 checkpointów (pobrane z Edena 2026-10-01).'),
    Historia('35k', '35 000 klatek od zera: 60 epok (kroki co 10, co 100 i końce epok)',
             ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi',
             ROOT / 'assets/MalyViT_35000_od_zera_s47/dane', na_epoke=1094,
             opis='Uczenie od zera na 35 000 klatkach TRAIN, 60 epok po 1094 kroki (Eden 2026-10-09). Wszystkie '
                  '806 checkpointów w assets/MalyViT_35000_od_zera_s47 (co 10 kroków do 1000, co 100 dalej, końce epok).'),
)


# Pojedyncze modele z assets zgodne z MalyViT (600 × 768, wczytanie strict=True): (klucz, nazwa, plik, krok, opis).
MODELE = (
    ('k1000_sd', 'K1000: model warunkujący SD (krok 1000)', ROOT / 'assets/MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt',
     1000, 'Detektor po 1000 aktualizacjach na bazie 1500 klatek; jego latenty warunkowały Stable Diffusion.'),
    ('k8000_s47', 'K8000: ziarno 47, dobór losowy (krok 8000, mAP 0,611)',
     ROOT / 'assets/MalyViT_K8000_s47_losowe/MALY_VIT_K8000_s47_losowe.pth', 8000,
     'Pętla 43, bieg petla43b_s47, wariant losowe; zapisany bez numeru kroku (sam state_dict).'),
)


def historia(klucz):
    for h in HISTORIE:
        if h.klucz == klucz:
            return h
    raise KeyError(klucz)
