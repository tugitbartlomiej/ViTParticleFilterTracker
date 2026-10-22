"""Lasery i koloidy (LaserLatentSpaceApp): oś czasu uczenia jednego zdjęcia przez checkpointy małego ViT.

Pakiety:
- `compute`: obliczenia bez Qt (historia uczenia, pułapka optyczna, mapa pułapek, rzuty, etykiety, dolina, mechanika);
- `gpu`: elementy OpenGL 4.3 (shadery GLSL, bufory SSBO) rysowane w widokach pyqtgraph;
- `ui`: widżety, okienka-doki, panel parametrów, dymki, widok rzutów R0–R12;
- `lasers_tab`: główna zakładka (okienka 1–14, sfera, przejście przez sieć) i jej wątki;
- `main_window`, `settings`, `paths`: okno główne, zapis ustawień, ścieżki.
Start: `python run.py` (okienko ładowania) albo `python -m laser_app`.
"""
import os

import OpenGL

# PyOpenGL domyślnie woła glGetError po KAŻDYM wywołaniu GL i każde czeka na kartę (synchronizacja CPU–GPU); w odtwarzaniu
# to było ~50 tys. wywołań i połowa czasu klatki. Ustawione tu, przed pierwszym importem OpenGL.GL w pakiecie.
# LASERY_GL_DEBUG=1 włącza sprawdzanie z powrotem.
OpenGL.ERROR_CHECKING = os.environ.get('LASERY_GL_DEBUG') == '1'
