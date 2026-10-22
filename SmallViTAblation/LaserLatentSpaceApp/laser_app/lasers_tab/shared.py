"""Stałe zakładki: kolor tła widoków, kroków na epokę, format liczb z przecinkiem."""
from __future__ import annotations

from laser_app.compute import history as hist


TLO = '#03060a'
EP = hist.NA_EPOKE
pl = lambda x, n=2: f'{x:.{n}f}'.replace('.', ',')
